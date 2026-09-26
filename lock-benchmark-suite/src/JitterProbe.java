import java.util.*;
import java.util.concurrent.*;

/**
 * Tests whether RL-Sync-AlwaysAdmit's apparent "win" over FairMutex under
 * the phase-varying workload is actually caused by RL-Sync's bookkeeping
 * overhead (atomic counters + Q-table state lookup executed before every
 * sem.acquire() call) acting as unintentional per-thread jitter that
 * desynchronizes threads which would otherwise arrive in lockstep at
 * phase boundaries (all PhaseWorkers share identical per-phase timing
 * distributions).
 *
 * If a "FairMutexJittered" (plain ReentrantLock(true) preceded by a tiny
 * amount of harmless dummy CPU work, calibrated to match RL-Sync's real
 * per-acquire overhead) ALSO beats plain FairMutex by a similar margin,
 * that proves the effect is a jitter/desynchronization artifact of this
 * specific synthetic workload generator -- not a genuine property of
 * RL-Sync's decision-making.
 */
public class JitterProbe {
    static final int[] THREADS = {8, 16};
    static final int SEEDS = 16;
    static final int REQS_PER_PHASE = 40;

    // Fair lock preceded by a small amount of dummy bookkeeping work,
    // meant to approximate RLArbiter's per-acquire overhead (atomic
    // increments + a handful of arithmetic ops) WITHOUT any actual
    // decision-making or backoff.
    static class FairMutexJittered implements RLSyncBenchmarkFixed.CSLock {
        final java.util.concurrent.locks.ReentrantLock lk = new java.util.concurrent.locks.ReentrantLock(true);
        final RLSyncBenchmarkFixed.Metrics met;
        final java.util.concurrent.atomic.AtomicInteger nWaiting = new java.util.concurrent.atomic.AtomicInteger(0);
        FairMutexJittered(String t) { met = new RLSyncBenchmarkFixed.Metrics(t); }
        @Override
        public long acquire(boolean rec, RLSyncBenchmarkFixed.Phase p) throws InterruptedException {
            boolean c = lk.hasQueuedThreads();
            long t0 = System.nanoTime();
            // Dummy bookkeeping to mimic RLArbiter's per-call overhead:
            // a few atomic ops + arithmetic, no sleeping, no branching on
            // outcome -- purely CPU jitter, same order of magnitude of
            // work as qt.sid()/nWaiting bookkeeping.
            int nw = nWaiting.incrementAndGet();
            double dummy = 0;
            for (int i = 0; i < 40; i++) dummy += Math.sqrt(i * nw + 1);
            if (dummy < -1) System.out.print(""); // prevent dead-code elimination
            nWaiting.decrementAndGet();

            lk.lockInterruptibly();
            long w = System.nanoTime() - t0;
            if (rec) met.record(w, c, p);
            return w;
        }
        @Override public void release() { lk.unlock(); }
        @Override public RLSyncBenchmarkFixed.Metrics metrics() { return met; }
    }

    public static void main(String[] args) throws Exception {
        for (int nT : THREADS) {
            System.out.println("=== nT=" + nT + " ===");
            double[][] plainD = new double[SEEDS][], jitD = new double[SEEDS][];
            for (int s = 0; s < SEEDS; s++) {
                int[] order = {0, 1};
                if (ThreadLocalRandom.current().nextBoolean()) { order[0]=1; order[1]=0; }
                double[] pRow = null, jRow = null;
                for (int slot : order) {
                    if (slot == 0) {
                        RLSyncBenchmarkFixed.FairMutex mx = new RLSyncBenchmarkFixed.FairMutex("mx");
                        runPhase(mx, nT);
                        pRow = row(mx.metrics());
                    } else {
                        FairMutexJittered mxj = new FairMutexJittered("mxj");
                        runPhase(mxj, nT);
                        jRow = row(mxj.metrics());
                    }
                }
                plainD[s] = pRow; jitD[s] = jRow;
                System.out.print(".");
            }
            System.out.println();
            double[] pMean = col(plainD,0), jMean = col(jitD,0);
            System.out.printf("  FairMutex-Plain    mean=%.3fms +/-%.3f%n", mean(pMean), ci95(pMean));
            System.out.printf("  FairMutex-Jittered mean=%.3fms +/-%.3f%n", mean(jMean), ci95(jMean));
            System.out.printf("  Jittered vs Plain: t=%.3f d=%.3f%n", welchT(jMean,pMean), cohensD(jMean,pMean));
            System.out.println();
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
    }

    static double[] row(RLSyncBenchmarkFixed.Metrics m) { return new double[]{m.meanMs()}; }
    static double[] col(double[][] d, int c){double[] o=new double[d.length];for(int i=0;i<d.length;i++)o[i]=d[i][c];return o;}
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
