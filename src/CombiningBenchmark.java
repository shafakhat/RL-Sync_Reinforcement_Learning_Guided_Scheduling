import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.concurrent.locks.ReentrantLock;
import java.io.*;

/**
 * Rigorous benchmark of CombiningLock vs FairMutex (ReentrantLock(true)).
 *
 * Prediction being tested (stated up front, before running): flat
 * combining amortizes per-thread scheduling overhead across batched
 * requests, so its advantage should GROW with thread count / contention
 * -- the OPPOSITE scaling direction from the SJF+aging lock (which faded
 * under saturation). We test across a wide thread range, including well
 * past where every other mechanism in this investigation has topped out
 * (up to 64 threads on this 2-core sandbox), specifically to see whether
 * the advantage keeps growing as predicted, and report the mechanistic
 * evidence (batch size) alongside the timing, not just the headline
 * number.
 *
 * Methodology matches prior rigorous sweeps: think time between
 * requests (not zero-think saturation spam), SEEDS=12, randomized
 * per-seed run order, one discarded warmup run per condition, Welch's t
 * / Cohen's d significance testing, ME-violation check on every run.
 */
public class CombiningBenchmark {
    static final int[] THREADS = {4, 8, 16, 24, 32, 48, 64, 96, 128};
    static final int SEEDS = 12;
    static final int REQS_PER_THREAD = 150;
    static final int CS_WORK_US = 150; // same critical-section cost used throughout this investigation

    public static void main(String[] args) throws Exception {
        List<String[]> csvRows = new ArrayList<>();
        csvRows.add(new String[]{"threads","method","mean_ms","ci95_ms","p99_ms","meanBatch","maxBatch","welch_t","cohens_d","ratio_to_fairmutex"});

        for (int nT : THREADS) {
            System.out.println("\n=== nT=" + nT + " ===");

            // Discard one warmup run per lock type.
            runCombining(nT);
            runFairMutex(nT);

            double[] cbMean = new double[SEEDS], mxMean = new double[SEEDS];
            double[] cbP99 = new double[SEEDS], mxP99 = new double[SEEDS];
            double[] batchMean = new double[SEEDS];
            long[] batchMax = new long[SEEDS];
            boolean anyViolation = false;

            for (int s = 0; s < SEEDS; s++) {
                boolean cbFirst = ThreadLocalRandom.current().nextBoolean();
                Object[] r1, r2;
                if (cbFirst) {
                    r1 = runCombining(nT);
                    r2 = runFairMutex(nT);
                } else {
                    r2 = runFairMutex(nT);
                    r1 = runCombining(nT);
                }
                cbMean[s] = (double) r1[0]; cbP99[s] = (double) r1[1];
                batchMean[s] = (double) r1[2]; batchMax[s] = (long) r1[3];
                if ((boolean) r1[4]) anyViolation = true;
                mxMean[s] = (double) r2[0]; mxP99[s] = (double) r2[1];
                if ((boolean) r2[4]) anyViolation = true;
                System.out.print(".");
            }
            System.out.println();
            if (anyViolation) System.out.println("  !!!! ME VIOLATION DETECTED THIS ROUND !!!!");

            double t = welchT(cbMean, mxMean), d = cohensD(cbMean, mxMean);
            double ratio = mean(cbMean) / mean(mxMean);

            System.out.printf("  CombiningLock  mean=%.3fms +/-%.3f p99=%.3fms  meanBatch=%.2f maxBatch=%.0f%n",
                mean(cbMean), ci95(cbMean), mean(cbP99), mean(batchMean), (double) Arrays.stream(batchMax).max().orElse(0));
            System.out.printf("  FairMutex      mean=%.3fms +/-%.3f p99=%.3fms%n",
                mean(mxMean), ci95(mxMean), mean(mxP99));
            System.out.printf("  Welch t=%.3f  Cohen's d=%.3f  ratio=%.3f  %s%n",
                t, d, ratio, Math.abs(t) > 2.2 ? "SIGNIFICANT" : "not significant");

            csvRows.add(new String[]{String.valueOf(nT), "CombiningLock",
                fmt(mean(cbMean)), fmt(ci95(cbMean)), fmt(mean(cbP99)), fmt(mean(batchMean)),
                fmt((double) Arrays.stream(batchMax).max().orElse(0)), fmt(t), fmt(d), fmt(ratio)});
            csvRows.add(new String[]{String.valueOf(nT), "FairMutex",
                fmt(mean(mxMean)), fmt(ci95(mxMean)), fmt(mean(mxP99)), "0", "0", "0", "0", "1.0"});
        }

        String outPath = args.length > 0 ? args[0] : "combining_benchmark.csv";
        try (PrintWriter pw = new PrintWriter(new FileWriter(outPath))) {
            for (String[] r : csvRows) pw.println(String.join(",", r));
        }
        System.out.println("\nWritten to " + outPath);
    }

    // Returns {meanMs, p99Ms, meanBatchSize, maxBatchSize, violated}
    static Object[] runCombining(int nT) throws Exception {
        CombiningLock lk = new CombiningLock();
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
        return new Object[]{meanMs(waits), p99Ms(waits), lk.meanBatchSize(), lk.maxBatchSizeSeen(), violated.get()};
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
                        // METRIC FIX (post-hoc review): stop the clock AFTER
                        // this thread's own critical section completes, to
                        // match CombiningLock.execute()'s return semantics
                        // (which necessarily includes the caller's own CS
                        // time). The original version stopped FairMutex's
                        // clock right after lock() returned -- before its CS
                        // ran -- silently exempting it from the ~150us
                        // (CS_WORK_US) that every CombiningLock sample always
                        // included. That is an apples-to-oranges bias, found
                        // while reviewing AdaptiveCombiningLock's numbers.
                        // All combining-family results measured before this
                        // fix must be treated as invalid and are being
                        // rerun.
                        long w = System.nanoTime() - t0;
                        waits.add(w);
                    }
                } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
                finally { latch.countDown(); }
            });
        }
        latch.await(120, TimeUnit.SECONDS);
        ex.shutdownNow(); ex.awaitTermination(5, TimeUnit.SECONDS);
        return new Object[]{meanMs(waits), p99Ms(waits), 0.0, 0L, violated.get()};
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
