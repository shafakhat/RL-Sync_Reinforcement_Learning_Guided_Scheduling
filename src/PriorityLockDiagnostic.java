import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

/**
 * Deeper diagnostic on why AdaptivePriorityLock showed 3-5x worse max wait
 * than FairMutex under saturation (StarvationCompare.java). Hypothesis:
 * the O(n) linear scan over the waiting set inside the synchronized
 * monitor on every release() (needed because the priority comparator is
 * time-varying due to aging) adds per-release overhead that scales with
 * queue depth, throttling overall throughput -- NOT just reordering
 * fairness. This would show up as lower total completions per wall-clock
 * second, not just as skewed LONG-vs-SHORT wait distributions.
 *
 * Measures: total wall time to drain a fixed number of requests, split
 * mean/max wait by job class, for both locks, same conditions as
 * StarvationCompare.
 */
public class PriorityLockDiagnostic {
    public static void main(String[] args) throws Exception {
        int nT = 24;
        int reqsPerThread = 150;

        System.out.println("=== AdaptivePriorityLock ===");
        runPriorityLock(nT, reqsPerThread);

        System.out.println("\n=== FairMutex ===");
        runFairMutex(nT, reqsPerThread);
    }

    static void runPriorityLock(int nT, int reqsPerThread) throws Exception {
        AdaptivePriorityLock lk = new AdaptivePriorityLock();
        List<Long> shortWaits = Collections.synchronizedList(new ArrayList<>());
        List<Long> longWaits = Collections.synchronizedList(new ArrayList<>());
        CountDownLatch latch = new CountDownLatch(nT);
        ExecutorService ex = Executors.newFixedThreadPool(nT);
        long wallT0 = System.nanoTime();
        for (int t = 0; t < nT; t++) {
            ex.submit(() -> {
                ThreadLocalRandom tlr = ThreadLocalRandom.current();
                try {
                    for (int i = 0; i < reqsPerThread; i++) {
                        AdaptivePriorityLock.JobClass jc = tlr.nextDouble() < 0.8
                            ? AdaptivePriorityLock.JobClass.SHORT : AdaptivePriorityLock.JobClass.LONG;
                        long csUs = jc == AdaptivePriorityLock.JobClass.SHORT ? 50 : 2000;
                        long w = lk.acquire(jc);
                        (jc == AdaptivePriorityLock.JobClass.SHORT ? shortWaits : longWaits).add(w);
                        long t0 = System.nanoTime();
                        while (System.nanoTime() - t0 < csUs * 1000L) {}
                        lk.release(jc, System.nanoTime() - t0);
                    }
                } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
                finally { latch.countDown(); }
            });
        }
        latch.await(90, TimeUnit.SECONDS);
        long wallMs = (System.nanoTime() - wallT0) / 1_000_000;
        ex.shutdownNow(); ex.awaitTermination(5, TimeUnit.SECONDS);
        report(wallMs, shortWaits, longWaits);
    }

    static void runFairMutex(int nT, int reqsPerThread) throws Exception {
        java.util.concurrent.locks.ReentrantLock lk = new java.util.concurrent.locks.ReentrantLock(true);
        List<Long> shortWaits = Collections.synchronizedList(new ArrayList<>());
        List<Long> longWaits = Collections.synchronizedList(new ArrayList<>());
        CountDownLatch latch = new CountDownLatch(nT);
        ExecutorService ex = Executors.newFixedThreadPool(nT);
        long wallT0 = System.nanoTime();
        for (int t = 0; t < nT; t++) {
            ex.submit(() -> {
                ThreadLocalRandom tlr = ThreadLocalRandom.current();
                try {
                    for (int i = 0; i < reqsPerThread; i++) {
                        boolean isLong = tlr.nextDouble() >= 0.8;
                        long csUs = isLong ? 2000 : 50;
                        long t0w = System.nanoTime();
                        lk.lockInterruptibly();
                        long w = System.nanoTime() - t0w;
                        (isLong ? longWaits : shortWaits).add(w);
                        long t0 = System.nanoTime();
                        while (System.nanoTime() - t0 < csUs * 1000L) {}
                        lk.unlock();
                    }
                } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
                finally { latch.countDown(); }
            });
        }
        latch.await(90, TimeUnit.SECONDS);
        long wallMs = (System.nanoTime() - wallT0) / 1_000_000;
        ex.shutdownNow(); ex.awaitTermination(5, TimeUnit.SECONDS);
        report(wallMs, shortWaits, longWaits);
    }

    static void report(long wallMs, List<Long> shortWaits, List<Long> longWaits) {
        System.out.printf("  wallMs=%d  nShort=%d nLong=%d%n", wallMs, shortWaits.size(), longWaits.size());
        printStats("SHORT", shortWaits);
        printStats("LONG ", longWaits);
    }

    static void printStats(String label, List<Long> waits) {
        if (waits.isEmpty()) { System.out.println("  " + label + ": no samples"); return; }
        double meanMs = waits.stream().mapToLong(Long::longValue).average().orElse(0) / 1e6;
        long maxNs = waits.stream().mapToLong(Long::longValue).max().orElse(0);
        List<Long> sorted = new ArrayList<>(waits);
        Collections.sort(sorted);
        double p99Ms = sorted.get((int) (sorted.size() * 0.99)) / 1e6;
        System.out.printf("  %s: meanMs=%.3f p99Ms=%.3f maxMs=%.3f%n", label, meanMs, p99Ms, maxNs / 1e6);
    }
}
