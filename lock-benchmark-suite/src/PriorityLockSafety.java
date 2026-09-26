import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

/**
 * Correctness harness for AdaptivePriorityLock, run BEFORE any performance
 * claim -- same discipline that caught the real CLH bug earlier this
 * session. Checks:
 *   (1) Mutual exclusion: never more than 1 thread inside the critical
 *       section at once (same MEChecker-style occupancy counter).
 *   (2) No lost wakeups / no deadlock: every submitted request completes
 *       within a generous timeout.
 *   (3) Starvation bound: no single request waits more than
 *       MAX_WAIT_AGING_NS + one extra critical-section-time slack beyond
 *       the aging cutoff (i.e. the aging mechanism actually bounds worst
 *       case, it isn't just decorative).
 *   (4) Repeats the above under heavy oversubscription (nT=48 on this
 *       2-core sandbox) and mixed SHORT/LONG job classes, several trials.
 */
public class PriorityLockSafety {
    static final AtomicInteger inside = new AtomicInteger(0);
    static final AtomicInteger peak = new AtomicInteger(0);
    static final AtomicBoolean violated = new AtomicBoolean(false);
    static final AtomicLong maxObservedWaitNs = new AtomicLong(0);
    static final AtomicInteger completed = new AtomicInteger(0);

    public static void main(String[] args) throws Exception {
        int[] threadCounts = {8, 24, 48};
        int reqsPerThread = 150;
        int trials = 3;

        boolean anyFail = false;

        for (int trial = 0; trial < trials; trial++) {
            for (int nT : threadCounts) {
                inside.set(0); peak.set(0); violated.set(false);
                maxObservedWaitNs.set(0); completed.set(0);

                AdaptivePriorityLock lk = new AdaptivePriorityLock();
                CountDownLatch latch = new CountDownLatch(nT);
                ExecutorService ex = Executors.newFixedThreadPool(nT);
                int totalReqs = nT * reqsPerThread;

                for (int t = 0; t < nT; t++) {
                    final int tid = t;
                    ex.submit(() -> {
                        ThreadLocalRandom tlr = ThreadLocalRandom.current();
                        try {
                            for (int i = 0; i < reqsPerThread; i++) {
                                // 80% SHORT (~50us), 20% LONG (~2ms) -- deliberately
                                // skewed + heterogeneous to stress SJF+aging hardest.
                                AdaptivePriorityLock.JobClass jc =
                                    tlr.nextDouble() < 0.8
                                        ? AdaptivePriorityLock.JobClass.SHORT
                                        : AdaptivePriorityLock.JobClass.LONG;
                                long csUs = jc == AdaptivePriorityLock.JobClass.SHORT
                                    ? 50 : 2000;

                                long waitNs = lk.acquire(jc);
                                maxObservedWaitNs.accumulateAndGet(waitNs, Math::max);

                                int n = inside.incrementAndGet();
                                peak.updateAndGet(cur -> Math.max(cur, n));
                                if (n > 1) violated.set(true);
                                long t0 = System.nanoTime();
                                while (System.nanoTime() - t0 < csUs * 1000L) { /* busy wait CS work */ }
                                inside.decrementAndGet();

                                long actualDurationNs = System.nanoTime() - t0;
                                lk.release(jc, actualDurationNs);
                                completed.incrementAndGet();
                            }
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                        } finally {
                            latch.countDown();
                        }
                    });
                }

                boolean finished = latch.await(90, TimeUnit.SECONDS);
                ex.shutdownNow();
                ex.awaitTermination(5, TimeUnit.SECONDS);

                double maxWaitMs = maxObservedWaitNs.get() / 1e6;
                // Generous bound: aging cutoff + a few CS durations of slack
                // for scheduling/JVM jitter on a contended 2-core sandbox.
                double boundMs = (AdaptivePriorityLock.MAX_WAIT_AGING_NS / 1e6) + 50.0;
                boolean starvationOk = maxWaitMs <= boundMs;

                boolean pass = finished && !violated.get()
                    && completed.get() == totalReqs && starvationOk;
                if (!pass) anyFail = true;

                System.out.printf(
                    "trial=%d nT=%2d finished=%-5s completed=%d/%d peakInside=%d violated=%s maxWaitMs=%.3f boundMs=%.1f starvationOk=%s -> %s%n",
                    trial, nT, finished, completed.get(), totalReqs, peak.get(),
                    violated.get(), maxWaitMs, boundMs, starvationOk,
                    pass ? "PASS" : "FAIL");
            }
        }

        System.out.println();
        System.out.println(anyFail ? "OVERALL: FAIL" : "OVERALL: PASS");
    }
}
