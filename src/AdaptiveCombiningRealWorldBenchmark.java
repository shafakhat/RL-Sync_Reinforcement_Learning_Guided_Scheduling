import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.concurrent.locks.ReentrantLock;
import java.io.*;

/**
 * Benchmarks AdaptiveCombiningLockV1, V2, and FairMutex across FIVE
 * critical-section durations anchored to real-world use cases, per user
 * request to cover "all aspects" rather than one arbitrarily chosen
 * duration. Anchors and sourcing (see conversation / ADAPTIVE_COMBINING
 * _FINDINGS.md for citations):
 *
 *   100ns   - in-memory data structure mutation (hash bucket, counter);
 *             a bare mutex lock/unlock cycle alone costs ~23.5ns on real
 *             hardware (Preshing), so 100ns represents a small amount of
 *             actual work atop that.
 *   1000ns  - heavier in-memory structure op (e.g. small tree/list
 *             rebalance step).
 *   10000ns - application-level shared-state/cache update (session
 *             state, connection-pool bookkeeping) -- production cache
 *             benchmarks show low tens of microseconds once real
 *             bookkeeping logic is included.
 *   100000ns  - DB-adjacent row/page-level latch hold (the actual
 *             mutation work, not the full transaction hold) --
 *             production guidance targets single/double-digit
 *             microseconds for this.
 *   500000ns - heavier in-lock bookkeeping stress point (matches the
 *             upper end already tested earlier in this investigation).
 *
 * No single value is treated as "the" standard -- all five are reported
 * so the reader can see exactly where each lock wins, ties, or loses,
 * per the standing rule against selectively choosing a workload range
 * that flatters one result.
 */
public class AdaptiveCombiningRealWorldBenchmark {
    static final int[] THREADS = {1, 2, 4, 8, 16, 24, 32, 48, 64, 96};
    static final long[] CS_DURATIONS_NS = {100, 1_000, 10_000, 100_000, 500_000};
    static final int SEEDS = 6;
    static final int REQS_PER_THREAD = 80;

    public static void main(String[] args) throws Exception {
        List<String[]> csvRows = new ArrayList<>();
        csvRows.add(new String[]{"cs_ns","threads","method","mean_ms","p99_ms","meanBatch","spinRetryFrac","welch_t","ratio_to_fairmutex","verdict"});

        for (long csNs : CS_DURATIONS_NS) {
            for (int nT : THREADS) {
                System.out.println("\n=== CS=" + csNs + "ns nT=" + nT + " ===");
                runV2(nT, csNs); runV1(nT, csNs); runFairMutex(nT, csNs); // warmup discard

                double[] v2Mean = new double[SEEDS], v1Mean = new double[SEEDS], mxMean = new double[SEEDS];
                double[] v2P99 = new double[SEEDS], v2Batch = new double[SEEDS], v2SpinFrac = new double[SEEDS];
                boolean anyViolation = false;

                for (int s = 0; s < SEEDS; s++) {
                    List<Integer> perm = new ArrayList<>(List.of(0,1,2));
                    Collections.shuffle(perm);
                    Object[] rv2=null, rv1=null, rmx=null;
                    for (int idx : perm) {
                        if (idx == 0) rv2 = runV2(nT, csNs);
                        else if (idx == 1) rv1 = runV1(nT, csNs);
                        else rmx = runFairMutex(nT, csNs);
                    }
                    v2Mean[s] = (double) rv2[0]; v2P99[s] = (double) rv2[1];
                    v2Batch[s] = (double) rv2[2]; v2SpinFrac[s] = (double) rv2[3];
                    if ((boolean) rv2[4]) anyViolation = true;
                    v1Mean[s] = (double) rv1[0];
                    if ((boolean) rv1[4]) anyViolation = true;
                    mxMean[s] = (double) rmx[0];
                    if ((boolean) rmx[4]) anyViolation = true;
                    System.out.print(".");
                }
                System.out.println();
                if (anyViolation) System.out.println("  !!!! ME VIOLATION DETECTED !!!!");

                double t = welchT(v2Mean, mxMean);
                double ratio = mean(v2Mean) / mean(mxMean);
                double ratioV1 = mean(v1Mean) / mean(mxMean);
                String verdict = Math.abs(t) <= 2.2 ? "TIE" : (ratio < 1.0 ? "V2 WINS" : "V2 LOSES");

                System.out.printf("  V2  mean=%.4fms p99=%.4fms batch=%.2f spin=%.1f%% ratio=%.3f  %s%n",
                    mean(v2Mean), mean(v2P99), mean(v2Batch), mean(v2SpinFrac)*100, ratio, verdict);
                System.out.printf("  V1  mean=%.4fms ratio=%.3f%n", mean(v1Mean), ratioV1);
                System.out.printf("  Fair mean=%.4fms%n", mean(mxMean));

                csvRows.add(new String[]{String.valueOf(csNs), String.valueOf(nT), "AdaptiveV2",
                    fmt(mean(v2Mean)), fmt(mean(v2P99)), fmt(mean(v2Batch)), fmt(mean(v2SpinFrac)),
                    fmt(t), fmt(ratio), verdict});
                csvRows.add(new String[]{String.valueOf(csNs), String.valueOf(nT), "AdaptiveV1",
                    fmt(mean(v1Mean)), "0","0","0","0", fmt(ratioV1), ""});
                csvRows.add(new String[]{String.valueOf(csNs), String.valueOf(nT), "FairMutex",
                    fmt(mean(mxMean)), "0","0","0","0","1.0",""});
            }
        }

        String outPath = args.length > 0 ? args[0] : "adaptive_combining_realworld.csv";
        try (PrintWriter pw = new PrintWriter(new FileWriter(outPath))) {
            for (String[] r : csvRows) pw.println(String.join(",", r));
        }
        System.out.println("\nWritten to " + outPath);
    }

    static Object[] runV2(int nT, long csNs) throws Exception {
        AdaptiveCombiningLockV2 lk = new AdaptiveCombiningLockV2();
        List<Long> waits = Collections.synchronizedList(new ArrayList<>());
        AtomicInteger inside = new AtomicInteger(0);
        AtomicBoolean violated = new AtomicBoolean(false);
        CountDownLatch latch = new CountDownLatch(nT);
        ExecutorService ex = Executors.newFixedThreadPool(nT);
        long thinkNs = Math.max(500, (long)(csNs * 0.7));
        for (int t = 0; t < nT; t++) {
            ex.submit(() -> {
                ThreadLocalRandom tlr = ThreadLocalRandom.current();
                try {
                    for (int i = 0; i < REQS_PER_THREAD; i++) {
                        sleepNs(thinkNs + tlr.nextLong(Math.max(1,thinkNs/2)));
                        long t0 = System.nanoTime();
                        lk.execute(() -> {
                            int n = inside.incrementAndGet();
                            if (n > 1) violated.set(true);
                            long cs0 = System.nanoTime();
                            while (System.nanoTime() - cs0 < csNs) {}
                            inside.decrementAndGet();
                        });
                        waits.add(System.nanoTime() - t0);
                    }
                } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
                finally { latch.countDown(); }
            });
        }
        latch.await(150, TimeUnit.SECONDS);
        ex.shutdownNow(); ex.awaitTermination(5, TimeUnit.SECONDS);
        long fp = lk.fastPathHits(), sp = lk.spinRetryHits(), ch = lk.combinerHits();
        double spinFrac = (fp+sp+ch) > 0 ? (double) sp / (fp+sp+ch) : 0;
        return new Object[]{meanMs(waits), p99Ms(waits), lk.meanBatchSize(), spinFrac, violated.get()};
    }

    static Object[] runV1(int nT, long csNs) throws Exception {
        AdaptiveCombiningLock lk = new AdaptiveCombiningLock();
        List<Long> waits = Collections.synchronizedList(new ArrayList<>());
        AtomicInteger inside = new AtomicInteger(0);
        AtomicBoolean violated = new AtomicBoolean(false);
        CountDownLatch latch = new CountDownLatch(nT);
        ExecutorService ex = Executors.newFixedThreadPool(nT);
        long thinkNs = Math.max(500, (long)(csNs * 0.7));
        for (int t = 0; t < nT; t++) {
            ex.submit(() -> {
                ThreadLocalRandom tlr = ThreadLocalRandom.current();
                try {
                    for (int i = 0; i < REQS_PER_THREAD; i++) {
                        sleepNs(thinkNs + tlr.nextLong(Math.max(1,thinkNs/2)));
                        long t0 = System.nanoTime();
                        lk.execute(() -> {
                            int n = inside.incrementAndGet();
                            if (n > 1) violated.set(true);
                            long cs0 = System.nanoTime();
                            while (System.nanoTime() - cs0 < csNs) {}
                            inside.decrementAndGet();
                        });
                        waits.add(System.nanoTime() - t0);
                    }
                } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
                finally { latch.countDown(); }
            });
        }
        latch.await(150, TimeUnit.SECONDS);
        ex.shutdownNow(); ex.awaitTermination(5, TimeUnit.SECONDS);
        return new Object[]{meanMs(waits), 0.0, 0.0, 0.0, violated.get()};
    }

    static Object[] runFairMutex(int nT, long csNs) throws Exception {
        ReentrantLock lk = new ReentrantLock(true);
        List<Long> waits = Collections.synchronizedList(new ArrayList<>());
        AtomicInteger inside = new AtomicInteger(0);
        AtomicBoolean violated = new AtomicBoolean(false);
        CountDownLatch latch = new CountDownLatch(nT);
        ExecutorService ex = Executors.newFixedThreadPool(nT);
        long thinkNs = Math.max(500, (long)(csNs * 0.7));
        for (int t = 0; t < nT; t++) {
            ex.submit(() -> {
                ThreadLocalRandom tlr = ThreadLocalRandom.current();
                try {
                    for (int i = 0; i < REQS_PER_THREAD; i++) {
                        sleepNs(thinkNs + tlr.nextLong(Math.max(1,thinkNs/2)));
                        long t0 = System.nanoTime();
                        lk.lockInterruptibly();
                        int n = inside.incrementAndGet();
                        if (n > 1) violated.set(true);
                        long cs0 = System.nanoTime();
                        while (System.nanoTime() - cs0 < csNs) {}
                        inside.decrementAndGet();
                        lk.unlock();
                        waits.add(System.nanoTime() - t0);
                    }
                } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
                finally { latch.countDown(); }
            });
        }
        latch.await(150, TimeUnit.SECONDS);
        ex.shutdownNow(); ex.awaitTermination(5, TimeUnit.SECONDS);
        return new Object[]{meanMs(waits), 0.0, 0.0, 0.0, violated.get()};
    }

    static void sleepNs(long ns) throws InterruptedException {
        long deadline = System.nanoTime() + ns;
        if (ns > 2_000_000) Thread.sleep(ns / 1_000_000);
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
    static double welchT(double[] a, double[] b){
        double va=var(a),vb=var(b);
        double den=Math.sqrt(va/a.length+vb/b.length);
        return den<1e-12?0:(mean(a)-mean(b))/den;
    }
}
