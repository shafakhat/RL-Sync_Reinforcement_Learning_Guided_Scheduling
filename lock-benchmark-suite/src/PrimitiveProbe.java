import java.util.*;
import java.util.concurrent.*;

/**
 * Isolates whether the "RL-Sync beats FairMutex under phase-varying
 * workload" effect is actually just a difference between the raw
 * underlying primitives -- Semaphore(1,true) (what RLArbiter uses) vs
 * ReentrantLock(true) (what FairMutex uses) -- with ZERO RL policy,
 * bookkeeping, or backoff logic involved at all.
 */
public class PrimitiveProbe {
    static final int[] THREADS = {8, 16};
    static final int SEEDS = 16;
    static final int REQS_PER_PHASE = 40;

    static class PlainSemaphoreLock implements RLSyncBenchmarkFixed.CSLock {
        final Semaphore sem = new Semaphore(1, true);
        final RLSyncBenchmarkFixed.Metrics met;
        PlainSemaphoreLock(String t) { met = new RLSyncBenchmarkFixed.Metrics(t); }
        @Override
        public long acquire(boolean rec, RLSyncBenchmarkFixed.Phase p) throws InterruptedException {
            boolean c = sem.hasQueuedThreads();
            long t0 = System.nanoTime();
            sem.acquire();
            long w = System.nanoTime() - t0;
            if (rec) met.record(w, c, p);
            return w;
        }
        @Override public void release() { sem.release(); }
        @Override public RLSyncBenchmarkFixed.Metrics metrics() { return met; }
    }

    public static void main(String[] args) throws Exception {
        for (int nT : THREADS) {
            System.out.println("=== nT=" + nT + " ===");
            double[][] semD = new double[SEEDS][], mxD = new double[SEEDS][];
            for (int s = 0; s < SEEDS; s++) {
                int[] order = {0, 1};
                if (ThreadLocalRandom.current().nextBoolean()) { order[0]=1; order[1]=0; }
                double[] sRow = null, mRow = null;
                for (int slot : order) {
                    if (slot == 0) {
                        PlainSemaphoreLock sem = new PlainSemaphoreLock("sem");
                        runPhase(sem, nT);
                        sRow = row(sem.metrics());
                    } else {
                        RLSyncBenchmarkFixed.FairMutex mx = new RLSyncBenchmarkFixed.FairMutex("mx");
                        runPhase(mx, nT);
                        mRow = row(mx.metrics());
                    }
                }
                semD[s] = sRow; mxD[s] = mRow;
                System.out.print(".");
            }
            System.out.println();
            double[] sMean = col(semD,0), mMean = col(mxD,0);
            System.out.printf("  PlainSemaphore(1,true) mean=%.3fms +/-%.3f%n", mean(sMean), ci95(sMean));
            System.out.printf("  ReentrantLock(true)    mean=%.3fms +/-%.3f%n", mean(mMean), ci95(mMean));
            System.out.printf("  Semaphore vs Lock: t=%.3f d=%.3f%n", welchT(sMean,mMean), cohensD(sMean,mMean));
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
