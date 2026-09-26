import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.concurrent.locks.ReentrantLock;
import java.io.*;

/**
 * Rigorous benchmark of AdaptiveCombiningLock vs FairMutex.
 *
 * Prediction (stated before running): the fast path should close the
 * nT<=8 gap that plain CombiningLock had (CombiningBenchmark.java showed
 * 13-53% WORSE than FairMutex at nT=4/8), while preserving the nT=24-96
 * advantage the combiner path already demonstrated. If this prediction
 * holds, this lock should tie-or-beat FairMutex across the ENTIRE
 * thread range with no low-end regression -- closing the one disclosed
 * weakness of the earlier mechanism.
 *
 * Same methodology as every rigorous sweep in this investigation:
 * think time between requests, SEEDS=12, randomized per-seed run order,
 * one discarded warmup run per condition, Welch's t / Cohen's d
 * significance testing, ME-violation check on every run.
 */
public class AdaptiveCombiningBenchmark {
    static final int[] THREADS = {1, 2, 4, 8, 16, 24, 32, 48, 64, 96};
    static final int SEEDS = 12;
    static final int REQS_PER_THREAD = 150;
    static final int CS_WORK_US = 150;

    public static void main(String[] args) throws Exception {
        List<String[]> csvRows = new ArrayList<>();
        csvRows.add(new String[]{"threads","method","mean_ms","ci95_ms","p99_ms","meanBatch","fastPathFrac","welch_t","cohens_d","ratio_to_fairmutex"});

        for (int nT : THREADS) {
            System.out.println("\n=== nT=" + nT + " ===");
            runAdaptive(nT); runFairMutex(nT); // warmup discard

            double[] acMean = new double[SEEDS], mxMean = new double[SEEDS];
            double[] acP99 = new double[SEEDS], mxP99 = new double[SEEDS];
            double[] batchMean = new double[SEEDS];
            double[] fastPathFrac = new double[SEEDS];
            boolean anyViolation = false;

            for (int s = 0; s < SEEDS; s++) {
                boolean acFirst = ThreadLocalRandom.current().nextBoolean();
                Object[] r1, r2;
                if (acFirst) { r1 = runAdaptive(nT); r2 = runFairMutex(nT); }
                else { r2 = runFairMutex(nT); r1 = runAdaptive(nT); }
                acMean[s] = (double) r1[0]; acP99[s] = (double) r1[1];
                batchMean[s] = (double) r1[2]; fastPathFrac[s] = (double) r1[3];
                if ((boolean) r1[4]) anyViolation = true;
                mxMean[s] = (double) r2[0]; mxP99[s] = (double) r2[1];
                if ((boolean) r2[4]) anyViolation = true;
                System.out.print(".");
            }
            System.out.println();
            if (anyViolation) System.out.println("  !!!! ME VIOLATION DETECTED !!!!");

            double t = welchT(acMean, mxMean), d = cohensD(acMean, mxMean);
            double ratio = mean(acMean) / mean(mxMean);

            System.out.printf("  AdaptiveCombiningLock  mean=%.3fms +/-%.3f p99=%.3fms  meanBatch=%.2f fastPathFrac=%.1f%%%n",
                mean(acMean), ci95(acMean), mean(acP99), mean(batchMean), mean(fastPathFrac) * 100);
            System.out.printf("  FairMutex              mean=%.3fms +/-%.3f p99=%.3fms%n",
                mean(mxMean), ci95(mxMean), mean(mxP99));
            System.out.printf("  Welch t=%.3f  Cohen's d=%.3f  ratio=%.3f  %s%n",
                t, d, ratio, Math.abs(t) > 2.2 ? "SIGNIFICANT" : "not significant");

            csvRows.add(new String[]{String.valueOf(nT), "AdaptiveCombiningLock",
                fmt(mean(acMean)), fmt(ci95(acMean)), fmt(mean(acP99)), fmt(mean(batchMean)),
                fmt(mean(fastPathFrac)), fmt(t), fmt(d), fmt(ratio)});
            csvRows.add(new String[]{String.valueOf(nT), "FairMutex",
                fmt(mean(mxMean)), fmt(ci95(mxMean)), fmt(mean(mxP99)), "0", "0", "0", "0", "1.0"});
        }

        String outPath = args.length > 0 ? args[0] : "adaptive_combining_benchmark.csv";
        try (PrintWriter pw = new PrintWriter(new FileWriter(outPath))) {
            for (String[] r : csvRows) pw.println(String.join(",", r));
        }
        System.out.println("\nWritten to " + outPath);
    }

    // Returns {meanMs, p99Ms, meanBatchSize, fastPathFraction, violated}
    static Object[] runAdaptive(int nT) throws Exception {
        AdaptiveCombiningLock lk = new AdaptiveCombiningLock();
        List<Long> waits = Collections.synchronizedList(new ArrayList<>());
        AtomicInteger inside = new AtomicInteger(0), peak = new AtomicInteger(0);
        AtomicBoolean violated = new AtomicBoolean(false);
        CountDownLatch latch = new CountDownLatch(nT);
        ExecutorService ex = Executors.newFixedThreadPool(nT);
        for (int t = 0; t < nT; t++) {
            ex.submit(() -> {
                ThreadLocalRandom tlr = ThreadLocalRandom.current();
                try {
                    for (int i = 0; i < REQS_PER_THREAD; i++) {
                        sleepUs(80 + tlr.nextLong(60));
                        long t0 = System.nanoTime();
                        long w = lk.execute(() -> {
                            int n = inside.incrementAndGet();
                            peak.updateAndGet(c -> Math.max(c, n));
                            if (n > 1) violated.set(true);
                            long cs0 = System.nanoTime();
                            while (System.nanoTime() - cs0 < CS_WORK_US * 1000L) {}
                            inside.decrementAndGet();
                        });
                        waits.add(w);
                    }
                } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
                finally { latch.countDown(); }
            });
        }
        latch.await(120, TimeUnit.SECONDS);
        ex.shutdownNow(); ex.awaitTermination(5, TimeUnit.SECONDS);
        long fp = lk.fastPathHits(), ch = lk.combinerHits();
        double fpFrac = (fp + ch) > 0 ? (double) fp / (fp + ch) : 1.0;
        return new Object[]{meanMs(waits), p99Ms(waits), lk.meanBatchSize(), fpFrac, violated.get()};
    }

    static Object[] runFairMutex(int nT) throws Exception {
        ReentrantLock lk = new ReentrantLock(true);
        List<Long> waits = Collections.synchronizedList(new ArrayList<>());
        AtomicInteger inside = new AtomicInteger(0), peak = new AtomicInteger(0);
        AtomicBoolean violated = new AtomicBoolean(false);
        CountDownLatch latch = new CountDownLatch(nT);
        ExecutorService ex = Executors.newFixedThreadPool(nT);
        for (int t = 0; t < nT; t++) {
            ex.submit(() -> {
                ThreadLocalRandom tlr = ThreadLocalRandom.current();
                try {
                    for (int i = 0; i < REQS_PER_THREAD; i++) {
                        sleepUs(80 + tlr.nextLong(60));
                        long t0 = System.nanoTime();
                        lk.lockInterruptibly();
                        int n = inside.incrementAndGet();
                        peak.updateAndGet(c -> Math.max(c, n));
                        if (n > 1) violated.set(true);
                        long cs0 = System.nanoTime();
                        while (System.nanoTime() - cs0 < CS_WORK_US * 1000L) {}
                        inside.decrementAndGet();
                        lk.unlock();
                        // METRIC FIX: stop the clock AFTER this thread's own
                        // critical section has completed, matching
                        // lk.execute()'s definition on the adaptive/combining
                        // side (which necessarily returns only once the
                        // caller's own work has run). Stopping FairMutex's
                        // clock right after lock() -- before the CS runs --
                        // silently exempts it from ~150us (CS_WORK_US) that
                        // the combining-style lock's number always includes,
                        // an apples-to-oranges bias discovered during this
                        // benchmark's review. Both arms now measure identical
                        // end-to-end latency: submission -> own CS done.
                        long w = System.nanoTime() - t0;
                        waits.add(w);
                    }
                } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
                finally { latch.countDown(); }
            });
        }
        latch.await(120, TimeUnit.SECONDS);
        ex.shutdownNow(); ex.awaitTermination(5, TimeUnit.SECONDS);
        return new Object[]{meanMs(waits), p99Ms(waits), 0.0, 1.0, violated.get()};
    }

    static void sleepUs(long us) throws InterruptedException {
        long deadline = System.nanoTime() + us * 1000L;
        if (us > 1500) Thread.sleep(us / 1000);
        while (System.nanoTime() < deadline) Thread.onSpinWait();
    }
    static double meanMs(List<Long> waits) {
        if (waits.isEmpty()) return 0;
        return waits.stream().mapToLong(Long::longValue).average().orElse(0) / 1e6;
    }
    static double p99Ms(List<Long> waits) {
        if (waits.isEmpty()) return 0;
        List<Long> sorted = new ArrayList<>(waits);
        Collections.sort(sorted);
        return sorted.get((int) (sorted.size() * 0.99)) / 1e6;
    }
    static String fmt(double v) { return String.format("%.6f", v); }
    static double mean(double[] a){double s=0;for(double v:a)s+=v;return s/a.length;}
    static double var(double[] a){double m=mean(a),s=0;for(double v:a)s+=(v-m)*(v-m);return s/Math.max(1,a.length-1);}
    static double std(double[] a){return Math.sqrt(var(a));}
    static double ci95(double[] a){
        double[] tc={0,12.706,4.303,3.182,2.776,2.571,2.447,2.365,2.306,2.262,2.228,2.201,2.179};
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
