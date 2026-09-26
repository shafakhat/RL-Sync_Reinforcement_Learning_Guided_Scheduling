import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Sanity check: does the "starvation" observed in PriorityLockSafety at
 * nT=24/48 reflect a real defect in AdaptivePriorityLock's aging, or is
 * it simply what ANY lock (including plain FairMutex) produces when
 * nT threads hammer a single critical section with ZERO think time
 * (i.e. permanent, unthrottled saturation -- not a realistic workload)?
 *
 * Runs the SAME zero-think-time, 80/20 SHORT/LONG workload against:
 *   - AdaptivePriorityLock
 *   - Plain ReentrantLock(true)  (FairMutex)
 * and reports max observed wait for both. If FairMutex's max wait is
 * comparable (same order of magnitude), the "bound" in
 * PriorityLockSafety was simply too tight for a saturated workload, not
 * evidence of a real bug.
 */
public class StarvationCompare {
    public static void main(String[] args) throws Exception {
        int[] threadCounts = {24, 48};
        int reqsPerThread = 150;

        for (int nT : threadCounts) {
            System.out.println("=== nT=" + nT + " (zero think time, saturated) ===");
            runPriorityLock(nT, reqsPerThread);
            runFairMutex(nT, reqsPerThread);
            System.out.println();
        }
    }

    static void runPriorityLock(int nT, int reqsPerThread) throws Exception {
        AdaptivePriorityLock lk = new AdaptivePriorityLock();
        AtomicLong maxWaitNs = new AtomicLong(0);
        AtomicInteger inside = new AtomicInteger(0), peak = new AtomicInteger(0);
        AtomicBoolean violated = new AtomicBoolean(false);
        CountDownLatch latch = new CountDownLatch(nT);
        ExecutorService ex = Executors.newFixedThreadPool(nT);
        for (int t = 0; t < nT; t++) {
            ex.submit(() -> {
                ThreadLocalRandom tlr = ThreadLocalRandom.current();
                try {
                    for (int i = 0; i < reqsPerThread; i++) {
                        AdaptivePriorityLock.JobClass jc = tlr.nextDouble() < 0.8
                            ? AdaptivePriorityLock.JobClass.SHORT : AdaptivePriorityLock.JobClass.LONG;
                        long csUs = jc == AdaptivePriorityLock.JobClass.SHORT ? 50 : 2000;
                        long w = lk.acquire(jc);
                        maxWaitNs.accumulateAndGet(w, Math::max);
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
        System.out.printf("  AdaptivePriorityLock  maxWaitMs=%.3f peakInside=%d violated=%s%n",
            maxWaitNs.get()/1e6, peak.get(), violated.get());
    }

    static void runFairMutex(int nT, int reqsPerThread) throws Exception {
        ReentrantLock lk = new ReentrantLock(true);
        AtomicLong maxWaitNs = new AtomicLong(0);
        AtomicInteger inside = new AtomicInteger(0), peak = new AtomicInteger(0);
        AtomicBoolean violated = new AtomicBoolean(false);
        CountDownLatch latch = new CountDownLatch(nT);
        ExecutorService ex = Executors.newFixedThreadPool(nT);
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
                        maxWaitNs.accumulateAndGet(w, Math::max);
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
        System.out.printf("  FairMutex (ReentrantLock) maxWaitMs=%.3f peakInside=%d violated=%s%n",
            maxWaitNs.get()/1e6, peak.get(), violated.get());
    }
}
