import java.util.*;
import java.util.concurrent.*;
import java.io.*;

/**
 * Tests whether RL-Sync can beat FairMutex under the workload it was
 * actually designed to adapt to: bursty, phase-varying traffic
 * (LIGHT -> BURST -> STEADY -> IDLE -> SPIKE -> STEADY), rather than
 * static uniform contention. If RL-Sync's adaptivity has ANY genuine
 * edge over a fair mutex, this is the most favorable place to find it.
 *
 * Compares:
 *   - RL-Sync-Guarded   (trained at nT=12, guard enabled)
 *   - RL-Sync-AlwaysAdmit (action=1 forced always -- the theoretical
 *     ceiling for any policy over this action space, per the structural
 *     argument that backoff cannot help against a fair FIFO semaphore)
 *   - FairMutex
 * at nT = {4, 8, 16}, across the full WORKLOAD_PHASES sequence, SEEDS=10.
 */
public class PhaseSweep {
    static final int[] THREADS = {4, 8, 16};
    static final int SEEDS = 16;
    static final int REQS_PER_PHASE = 40;
    static final int TRAIN_AT_NT = 12;

    public static void main(String[] args) throws Exception {
        System.out.println("Training policy once at nT=" + TRAIN_AT_NT + " (static workload, as before)...");
        RLSyncBenchmarkFixed.QTable qtBase = RLSyncBenchmarkFixed.trainOffline(TRAIN_AT_NT);
        System.out.println("Training complete.\n");

        List<String[]> outRows = new ArrayList<>();
        outRows.add(new String[]{"threads","method","mean_ms","ci95_ms","p99_ms","zdr_pct"});

        for (int nT : THREADS) {
            System.out.println("=== nT=" + nT + " (phase-varying workload) ===");
            double[][] gdD = new double[SEEDS][], aaD = new double[SEEDS][], mxD = new double[SEEDS][];

            for (int s = 0; s < SEEDS; s++) {
                // Randomize per-seed run order to rule out a systematic
                // "run last on a 2-core sandbox = penalized by GC/thread-pool
                // churn" ordering confound. Each of the 3 methods gets an
                // equal chance to run in any of the 3 slots.
                int[] order = {0, 1, 2};
                for (int i = order.length - 1; i > 0; i--) {
                    int j = ThreadLocalRandom.current().nextInt(i + 1);
                    int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
                }
                double[] gdRow = null, aaRow = null, mxRow = null;
                for (int slot : order) {
                    if (slot == 0) {
                        RLSyncBenchmarkFixed.QTable qt1 = new RLSyncBenchmarkFixed.QTable(qtBase);
                        RLSyncBenchmarkFixed.RLArbiter rlGd = new RLSyncBenchmarkFixed.RLArbiter(
                            qt1, 0, 0, false, "rlGd", true, nT);
                        runPhase(rlGd, nT);
                        gdRow = row(rlGd.metrics());
                    } else if (slot == 1) {
                        RLSyncBenchmarkFixed.QTable qt2 = new RLSyncBenchmarkFixed.QTable(qtBase);
                        RLSyncBenchmarkFixed.RLArbiter rlAA = new RLSyncBenchmarkFixed.RLArbiter(
                            qt2, 0, 0, false, "rlAA");
                        forceAlwaysAdmit(rlAA);
                        runPhase(rlAA, nT);
                        aaRow = row(rlAA.metrics());
                    } else {
                        RLSyncBenchmarkFixed.FairMutex mx = new RLSyncBenchmarkFixed.FairMutex("mx");
                        runPhase(mx, nT);
                        mxRow = row(mx.metrics());
                    }
                }
                gdD[s] = gdRow; aaD[s] = aaRow; mxD[s] = mxRow;

                System.out.print(".");
            }
            System.out.println();

            printSummary("RL-Sync-Guarded", nT, gdD, outRows);
            printSummary("RL-Sync-AlwaysAdmit", nT, aaD, outRows);
            printSummary("FairMutex", nT, mxD, outRows);

            // Welch's t on mean wait: Guarded vs FairMutex, AlwaysAdmit vs FairMutex
            double[] gdMean = col(gdD,0), aaMean = col(aaD,0), mxMean = col(mxD,0);
            System.out.printf("  Guarded vs Mutex:      t=%.3f d=%.3f%n",
                welchT(gdMean, mxMean), cohensD(gdMean, mxMean));
            System.out.printf("  AlwaysAdmit vs Mutex:  t=%.3f d=%.3f%n",
                welchT(aaMean, mxMean), cohensD(aaMean, mxMean));
            System.out.println();
        }

        String outPath = args.length > 0 ? args[0] : "phase_sweep.csv";
        try (PrintWriter pw = new PrintWriter(new FileWriter(outPath))) {
            for (String[] r : outRows) pw.println(String.join(",", r));
        }
        System.out.println("Written to " + outPath);
    }

    // Reflection-free trick: RLArbiter has no public "force admit" flag,
    // so we simply rely on guardEnabled=false + learning=false + eps=0,
    // which means action = qt.best(s) from a copy of the SAME trained
    // Q-table as Guarded. To get a true "always admit" baseline we
    // instead run with a QTable whose every state's action-1 value is
    // forced higher than action-0. Simplest: overwrite the table in place.
    static void forceAlwaysAdmit(RLSyncBenchmarkFixed.RLArbiter arb) {
        RLSyncBenchmarkFixed.QTable qt = arb.qt;
        for (int s = 0; s < qt.Q.length; s++) {
            qt.Q[s][1] = 1.0;   // admit
            qt.Q[s][0] = -1.0;  // backoff always dominated
        }
    }

    static void runPhase(RLSyncBenchmarkFixed.CSLock lk, int nT) throws Exception {
        RLSyncBenchmarkFixed.MEChecker chk = new RLSyncBenchmarkFixed.MEChecker();
        CountDownLatch latch = new CountDownLatch(nT);
        ExecutorService ex = Executors.newFixedThreadPool(nT);
        List<RLSyncBenchmarkFixed.Phase> seq = Arrays.asList(RLSyncBenchmarkFixed.WORKLOAD_PHASES);
        for (int t = 0; t < nT; t++)
            ex.submit(new RLSyncBenchmarkFixed.PhaseWorker(lk, seq, REQS_PER_PHASE, latch, true, chk));
        latch.await(120, TimeUnit.SECONDS);
        ex.shutdownNow();
        ex.awaitTermination(5, TimeUnit.SECONDS);
        if (chk.violated.get()) {
            System.out.println("\n  !!! VIOLATION detected in phase run !!!");
        }
    }

    static double[] row(RLSyncBenchmarkFixed.Metrics m) {
        return new double[]{m.meanMs(), 0, m.pctMs(99), m.zdr()};
    }

    static void printSummary(String label, int nT, double[][] d, List<String[]> outRows) {
        double[] mean = col(d,0), p99 = col(d,2), zdr = col(d,3);
        double m = mean(mean), c = ci95(mean);
        System.out.printf("  %-22s mean=%.3fms +/-%.3f  p99=%.3fms  zdr=%.1f%%%n",
            label, m, c, mean(p99), mean(zdr));
        outRows.add(new String[]{
            String.valueOf(nT), label,
            String.format("%.6f", m), String.format("%.6f", c),
            String.format("%.6f", mean(p99)), String.format("%.4f", mean(zdr))
        });
    }

    static double[] col(double[][] d, int c) {
        double[] out = new double[d.length];
        for (int i=0;i<d.length;i++) out[i]=d[i][c];
        return out;
    }
    static double mean(double[] a){double s=0;for(double v:a)s+=v;return s/a.length;}
    static double var(double[] a){double m=mean(a),s=0;for(double v:a)s+=(v-m)*(v-m);return s/Math.max(1,a.length-1);}
    static double std(double[] a){return Math.sqrt(var(a));}
    static double ci95(double[] a){
        double[] tc={0,12.706,4.303,3.182,2.776,2.571,2.447,2.365,2.306,2.262,2.228};
        int df=Math.min(a.length-1,tc.length-1);
        return (df>0?tc[df]:2.0)*std(a)/Math.sqrt(a.length);
    }
    static double welchT(double[] a, double[] b){
        double va=var(a),vb=var(b);
        double den=Math.sqrt(va/a.length+vb/b.length);
        return den<1e-12?0:(mean(a)-mean(b))/den;
    }
    static double cohensD(double[] a, double[] b){
        double p=Math.sqrt((var(a)+var(b))/2.0);
        return p<1e-12?0:(mean(a)-mean(b))/p;
    }
}
