import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.concurrent.locks.ReentrantLock;
import java.io.*;

/**
 * FINAL, rigorous benchmark of AdaptivePriorityLockV2 (SJF+aging on an
 * AQS-style park/unpark primitive) vs FairMutex (ReentrantLock(true)).
 *
 * Two conditions, run back to back with the SAME code path, differing
 * only in the job-size mix -- this is the pre-registered falsification
 * test:
 *
 *   (A) HOMOGENEOUS sanity check: ALL requests cost the same (150us).
 *       Theory (M/M/1 queueing) says SJF cannot help here -- there is
 *       nothing to reorder by size. If AdaptivePriorityLockV2 wins here,
 *       that is a red flag of a rigged/broken benchmark, not a real
 *       result, and must be reported as such.
 *
 *   (B) HETEROGENEOUS: 80% SHORT (50us) / 20% LONG (2000us) requests,
 *       assigned randomly per-request (NOT correlated with thread id,
 *       so no thread is a priori "the slow one" -- every thread emits a
 *       realistic mix). This is where SJF has a textbook-justified,
 *       non-hardcoded lever: shortest-job-first minimizes MEAN wait in
 *       a non-preemptive single-server queue with heterogeneous service
 *       times, and should show up primarily as improved SHORT-job wait,
 *       not as an implausible uniform win everywhere.
 *
 * Includes: seeded repetition (SEEDS=12), one discarded JVM warmup run
 * per lock/condition (per WarmupCheck.java finding), Welch's t / Cohen's
 * d significance testing, and reports SHORT vs LONG wait breakdown
 * separately (the mechanistic prediction to check, not just an aggregate
 * number).
 */
public class PriorityLockBenchmark {
    static final int[] THREADS = {8, 16, 24};
    static final int SEEDS = 12;
    static final int REQS_PER_THREAD = 120;

    public static void main(String[] args) throws Exception {
        List<String[]> csvRows = new ArrayList<>();
        csvRows.add(new String[]{"condition","threads","method","mean_ms","ci95_ms","p99_ms","short_mean_ms","long_mean_ms","welch_t_vs_fairmutex","cohens_d_vs_fairmutex"});

        for (String condition : new String[]{"HOMOGENEOUS", "HETEROGENEOUS"}) {
            System.out.println("\n########## CONDITION: " + condition + " ##########");
            double longFrac = condition.equals("HOMOGENEOUS") ? 0.0 : 0.2;

            for (int nT : THREADS) {
                System.out.println("\n=== " + condition + " nT=" + nT + " ===");

                // Discard one warmup run per lock type (JIT/cold-start, per WarmupCheck.java).
                runPriority(nT, REQS_PER_THREAD, longFrac);
                runFairMutex(nT, REQS_PER_THREAD, longFrac);

                double[][] plRows = new double[SEEDS][];
                double[][] mxRows = new double[SEEDS][];
                double[][] plShortLong = new double[SEEDS][];
                double[][] mxShortLong = new double[SEEDS][];

                for (int s = 0; s < SEEDS; s++) {
                    boolean priorityFirst = ThreadLocalRandom.current().nextBoolean();
                    if (priorityFirst) {
                        Object[] r1 = runPriority(nT, REQS_PER_THREAD, longFrac);
                        Object[] r2 = runFairMutex(nT, REQS_PER_THREAD, longFrac);
                        plRows[s] = (double[]) r1[0]; plShortLong[s] = (double[]) r1[1];
                        mxRows[s] = (double[]) r2[0]; mxShortLong[s] = (double[]) r2[1];
                    } else {
                        Object[] r2 = runFairMutex(nT, REQS_PER_THREAD, longFrac);
                        Object[] r1 = runPriority(nT, REQS_PER_THREAD, longFrac);
                        plRows[s] = (double[]) r1[0]; plShortLong[s] = (double[]) r1[1];
                        mxRows[s] = (double[]) r2[0]; mxShortLong[s] = (double[]) r2[1];
                    }
                    System.out.print(".");
                }
                System.out.println();

                double[] plMean = col(plRows,0), mxMean = col(mxRows,0);
                double[] plP99  = col(plRows,1), mxP99  = col(mxRows,1);
                double[] plShort = col(plShortLong,0), plLong = col(plShortLong,1);
                double[] mxShort = col(mxShortLong,0), mxLong = col(mxShortLong,1);

                double t = welchT(plMean, mxMean), d = cohensD(plMean, mxMean);

                System.out.printf("  AdaptivePriorityLockV2  mean=%.3fms +/-%.3f p99=%.3fms  [short=%.3fms long=%.3fms]%n",
                    mean(plMean), ci95(plMean), mean(plP99), mean(plShort), mean(plLong));
                System.out.printf("  FairMutex               mean=%.3fms +/-%.3f p99=%.3fms  [short=%.3fms long=%.3fms]%n",
                    mean(mxMean), ci95(mxMean), mean(mxP99), mean(mxShort), mean(mxLong));
                System.out.printf("  Welch t=%.3f  Cohen's d=%.3f  %s%n",
                    t, d, Math.abs(t) > 2.2 ? "SIGNIFICANT (p<0.05 approx)" : "not significant");

                csvRows.add(new String[]{condition, String.valueOf(nT), "AdaptivePriorityLockV2",
                    fmt(mean(plMean)), fmt(ci95(plMean)), fmt(mean(plP99)), fmt(mean(plShort)), fmt(mean(plLong)),
                    fmt(t), fmt(d)});
                csvRows.add(new String[]{condition, String.valueOf(nT), "FairMutex",
                    fmt(mean(mxMean)), fmt(ci95(mxMean)), fmt(mean(mxP99)), fmt(mean(mxShort)), fmt(mean(mxLong)),
                    "0", "0"});
            }
        }

        String outPath = args.length > 0 ? args[0] : "priority_lock_benchmark.csv";
        try (PrintWriter pw = new PrintWriter(new FileWriter(outPath))) {
            for (String[] r : csvRows) pw.println(String.join(",", r));
        }
        System.out.println("\nWritten to " + outPath);
    }

    // Returns {double[]{meanMs, p99Ms}, double[]{shortMeanMs, longMeanMs}}
    static Object[] runPriority(int nT, int reqsPerThread, double longFrac) throws Exception {
        AdaptivePriorityLockV2 lk = new AdaptivePriorityLockV2();
        List<Long> allWaits = Collections.synchronizedList(new ArrayList<>());
        List<Long> shortWaits = Collections.synchronizedList(new ArrayList<>());
        List<Long> longWaits = Collections.synchronizedList(new ArrayList<>());
        AtomicInteger inside = new AtomicInteger(0), peak = new AtomicInteger(0);
        AtomicBoolean violated = new AtomicBoolean(false);
        CountDownLatch latch = new CountDownLatch(nT);
        ExecutorService ex = Executors.newFixedThreadPool(nT);
        for (int t = 0; t < nT; t++) {
            ex.submit(() -> {
                ThreadLocalRandom tlr = ThreadLocalRandom.current();
                try {
                    for (int i = 0; i < reqsPerThread; i++) {
                        sleepUs(80 + tlr.nextLong(60)); // think time between requests
                        boolean isLong = tlr.nextDouble() < longFrac;
                        AdaptivePriorityLockV2.JobClass jc = isLong
                            ? AdaptivePriorityLockV2.JobClass.LONG : AdaptivePriorityLockV2.JobClass.SHORT;
                        long csUs = isLong ? 2000 : 150;
                        long w = lk.acquire(jc);
                        allWaits.add(w);
                        (isLong ? longWaits : shortWaits).add(w);
                        int n = inside.incrementAndGet();
                        peak.updateAndGet(c -> Math.max(c, n));
                        if (n > 1) violated.set(true);
                        long t0 = System.nanoTime();
                        while (System.nanoTime() - t0 < csUs * 1000L) {}
                        inside.decrementAndGet();
                        lk.release(jc, System.nanoTime() - t0);
                    }
                } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
                finally { latch.countDown(); }
            });
        }
        latch.await(90, TimeUnit.SECONDS);
        ex.shutdownNow(); ex.awaitTermination(5, TimeUnit.SECONDS);
        if (violated.get()) System.out.println("  !!! ME VIOLATION in AdaptivePriorityLockV2 run !!!");
        return new Object[]{
            new double[]{meanMs(allWaits), p99Ms(allWaits)},
            new double[]{meanMs(shortWaits), meanMs(longWaits)}
        };
    }

    static Object[] runFairMutex(int nT, int reqsPerThread, double longFrac) throws Exception {
        ReentrantLock lk = new ReentrantLock(true);
        List<Long> allWaits = Collections.synchronizedList(new ArrayList<>());
        List<Long> shortWaits = Collections.synchronizedList(new ArrayList<>());
        List<Long> longWaits = Collections.synchronizedList(new ArrayList<>());
        AtomicInteger inside = new AtomicInteger(0), peak = new AtomicInteger(0);
        AtomicBoolean violated = new AtomicBoolean(false);
        CountDownLatch latch = new CountDownLatch(nT);
        ExecutorService ex = Executors.newFixedThreadPool(nT);
        for (int t = 0; t < nT; t++) {
            ex.submit(() -> {
                ThreadLocalRandom tlr = ThreadLocalRandom.current();
                try {
                    for (int i = 0; i < reqsPerThread; i++) {
                        sleepUs(80 + tlr.nextLong(60));
                        boolean isLong = tlr.nextDouble() < longFrac;
                        long csUs = isLong ? 2000 : 150;
                        long t0w = System.nanoTime();
                        lk.lockInterruptibly();
                        long w = System.nanoTime() - t0w;
                        allWaits.add(w);
                        (isLong ? longWaits : shortWaits).add(w);
                        int n = inside.incrementAndGet();
                        peak.updateAndGet(c -> Math.max(c, n));
                        if (n > 1) violated.set(true);
                        long t0 = System.nanoTime();
                        while (System.nanoTime() - t0 < csUs * 1000L) {}
                        inside.decrementAndGet();
                        lk.unlock();
                    }
                } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
                finally { latch.countDown(); }
            });
        }
        latch.await(90, TimeUnit.SECONDS);
        ex.shutdownNow(); ex.awaitTermination(5, TimeUnit.SECONDS);
        if (violated.get()) System.out.println("  !!! ME VIOLATION in FairMutex run !!!");
        return new Object[]{
            new double[]{meanMs(allWaits), p99Ms(allWaits)},
            new double[]{meanMs(shortWaits), meanMs(longWaits)}
        };
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
    static double[] col(double[][] d, int c){double[] o=new double[d.length];for(int i=0;i<d.length;i++)o[i]=d[i][c];return o;}
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
