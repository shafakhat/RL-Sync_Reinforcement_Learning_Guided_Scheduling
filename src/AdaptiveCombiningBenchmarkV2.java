import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.concurrent.locks.ReentrantLock;
import java.io.*;

/**
 * Benchmarks AdaptiveCombiningLockV2 (bounded spin-retry before
 * escalation) against FairMutex AND against V1 (immediate escalation),
 * specifically targeting the nT=2 regime where V1 was found to lose.
 *
 * OVERFITTING GUARD: SPIN_LIMIT=40 in V2 is a fixed constant chosen from
 * general spin-then-block practice, not fit to this benchmark's timing.
 * To check it wasn't accidentally overfit to the one critical-section
 * duration (150us) used throughout this investigation, this harness
 * repeats the full thread sweep at THREE different CS durations: 30us
 * (much shorter), 150us (the standard one), and 500us (much longer). If
 * the fix only helps at 150us, that is a sign of overfitting and will be
 * reported as such rather than hidden.
 */
public class AdaptiveCombiningBenchmarkV2 {
    static final int[] THREADS = {1, 2, 4, 8, 16, 24, 32, 48, 64, 96};
    static final int[] CS_DURATIONS_US = {30, 150, 500};
    static final int SEEDS = 10;
    static final int REQS_PER_THREAD = 150;

    public static void main(String[] args) throws Exception {
        List<String[]> csvRows = new ArrayList<>();
        csvRows.add(new String[]{"cs_us","threads","method","mean_ms","ci95_ms","p99_ms","meanBatch","spinRetryFrac","welch_t","cohens_d","ratio_to_fairmutex"});

        for (int csUs : CS_DURATIONS_US) {
            for (int nT : THREADS) {
                System.out.println("\n=== CS=" + csUs + "us nT=" + nT + " ===");
                runV2(nT, csUs); runV1(nT, csUs); runFairMutex(nT, csUs); // warmup discard

                double[] v2Mean = new double[SEEDS], v1Mean = new double[SEEDS], mxMean = new double[SEEDS];
                double[] v2P99 = new double[SEEDS];
                double[] v2Batch = new double[SEEDS], v2SpinFrac = new double[SEEDS];
                boolean anyViolation = false;

                for (int s = 0; s < SEEDS; s++) {
                    int order = ThreadLocalRandom.current().nextInt(3);
                    Object[] rv2=null, rv1=null, rmx=null;
                    // randomize order across the three arms
                    List<Integer> perm = new ArrayList<>(List.of(0,1,2));
                    Collections.shuffle(perm);
                    for (int idx : perm) {
                        if (idx == 0) rv2 = runV2(nT, csUs);
                        else if (idx == 1) rv1 = runV1(nT, csUs);
                        else rmx = runFairMutex(nT, csUs);
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

                double t = welchT(v2Mean, mxMean), d = cohensD(v2Mean, mxMean);
                double ratio = mean(v2Mean) / mean(mxMean);
                double ratioV1 = mean(v1Mean) / mean(mxMean);

                System.out.printf("  V2(spin-retry)  mean=%.3fms p99=%.3fms meanBatch=%.2f spinFrac=%.1f%% ratioVsFairMutex=%.3f %s%n",
                    mean(v2Mean), mean(v2P99), mean(v2Batch), mean(v2SpinFrac)*100, ratio, Math.abs(t) > 2.2 ? "SIGNIFICANT" : "not significant");
                System.out.printf("  V1(no spin)     mean=%.3fms ratioVsFairMutex=%.3f%n", mean(v1Mean), ratioV1);
                System.out.printf("  FairMutex       mean=%.3fms%n", mean(mxMean));

                csvRows.add(new String[]{String.valueOf(csUs), String.valueOf(nT), "AdaptiveV2",
                    fmt(mean(v2Mean)), fmt(ci95(v2Mean)), fmt(mean(v2P99)), fmt(mean(v2Batch)),
                    fmt(mean(v2SpinFrac)), fmt(t), fmt(d), fmt(ratio)});
                csvRows.add(new String[]{String.valueOf(csUs), String.valueOf(nT), "AdaptiveV1",
                    fmt(mean(v1Mean)), "0","0","0","0","0","0", fmt(ratioV1)});
                csvRows.add(new String[]{String.valueOf(csUs), String.valueOf(nT), "FairMutex",
                    fmt(mean(mxMean)), "0","0","0","0","0","0","1.0"});
            }
        }

        String outPath = args.length > 0 ? args[0] : "adaptive_combining_v2_multidur.csv";
        try (PrintWriter pw = new PrintWriter(new FileWriter(outPath))) {
            for (String[] r : csvRows) pw.println(String.join(",", r));
        }
        System.out.println("\nWritten to " + outPath);
    }

    static Object[] runV2(int nT, int csUs) throws Exception {
        AdaptiveCombiningLockV2 lk = new AdaptiveCombiningLockV2();
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
                        sleepUs(csThink(csUs) + tlr.nextLong(Math.max(1,csThink(csUs)/2)));
                        long t0 = System.nanoTime();
                        lk.execute(() -> {
                            int n = inside.incrementAndGet();
                            peak.updateAndGet(c -> Math.max(c, n));
                            if (n > 1) violated.set(true);
                            long cs0 = System.nanoTime();
                            while (System.nanoTime() - cs0 < csUs * 1000L) {}
                            inside.decrementAndGet();
                        });
                        waits.add(System.nanoTime() - t0);
                    }
                } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
                finally { latch.countDown(); }
            });
        }
        latch.await(120, TimeUnit.SECONDS);
        ex.shutdownNow(); ex.awaitTermination(5, TimeUnit.SECONDS);
        long fp = lk.fastPathHits(), sp = lk.spinRetryHits(), ch = lk.combinerHits();
        double spinFrac = (fp+sp+ch) > 0 ? (double) sp / (fp+sp+ch) : 0;
        return new Object[]{meanMs(waits), p99Ms(waits), lk.meanBatchSize(), spinFrac, violated.get()};
    }

    static Object[] runV1(int nT, int csUs) throws Exception {
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
                        sleepUs(csThink(csUs) + tlr.nextLong(Math.max(1,csThink(csUs)/2)));
                        long t0 = System.nanoTime();
                        lk.execute(() -> {
                            int n = inside.incrementAndGet();
                            peak.updateAndGet(c -> Math.max(c, n));
                            if (n > 1) violated.set(true);
                            long cs0 = System.nanoTime();
                            while (System.nanoTime() - cs0 < csUs * 1000L) {}
                            inside.decrementAndGet();
                        });
                        waits.add(System.nanoTime() - t0);
                    }
                } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
                finally { latch.countDown(); }
            });
        }
        latch.await(120, TimeUnit.SECONDS);
        ex.shutdownNow(); ex.awaitTermination(5, TimeUnit.SECONDS);
        return new Object[]{meanMs(waits), 0.0, 0.0, 0.0, violated.get()};
    }

    static Object[] runFairMutex(int nT, int csUs) throws Exception {
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
                        sleepUs(csThink(csUs) + tlr.nextLong(Math.max(1,csThink(csUs)/2)));
                        long t0 = System.nanoTime();
                        lk.lockInterruptibly();
                        int n = inside.incrementAndGet();
                        peak.updateAndGet(c -> Math.max(c, n));
                        if (n > 1) violated.set(true);
                        long cs0 = System.nanoTime();
                        while (System.nanoTime() - cs0 < csUs * 1000L) {}
                        inside.decrementAndGet();
                        lk.unlock();
                        waits.add(System.nanoTime() - t0);
                    }
                } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
                finally { latch.countDown(); }
            });
        }
        latch.await(120, TimeUnit.SECONDS);
        ex.shutdownNow(); ex.awaitTermination(5, TimeUnit.SECONDS);
        return new Object[]{meanMs(waits), 0.0, 0.0, 0.0, violated.get()};
    }

    // Think time scales roughly with CS duration so relative contention
    // shape is comparable across the three CS_DURATIONS_US settings.
    static int csThink(int csUs) { return Math.max(20, (int)(csUs * 0.7)); }

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
