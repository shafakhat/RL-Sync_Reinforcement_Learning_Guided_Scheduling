import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

/**
 * Correctness + liveness harness for CombiningLock, run BEFORE any
 * performance claim (same discipline as every prior mechanism in this
 * investigation).
 *
 * Checks:
 *   (1) Mutual exclusion: never more than 1 thread's work executing at
 *       once (occupancy counter around the actual work.run() body).
 *   (2) Liveness: every submitted request completes within a generous
 *       timeout (catches the park-forever bug class explicitly, since
 *       that class of bug was found and fixed during development).
 *   (3) No lost work: total completed == total submitted, and (for a
 *       shared-counter workload) the final counter value exactly equals
 *       the number of increments submitted -- this also transitively
 *       re-confirms mutual exclusion (a race on a naive increment would
 *       corrupt the final count).
 *   (4) Stress at heavy oversubscription (nT=64 on this 2-core sandbox)
 *       and repeated trials.
 */
public class CombiningLockSafety {
    static final AtomicInteger inside = new AtomicInteger(0);
    static final AtomicInteger peak = new AtomicInteger(0);
    static final AtomicBoolean violated = new AtomicBoolean(false);
    static final AtomicInteger completed = new AtomicInteger(0);

    // Deliberately NON-atomic shared counter, protected only by the lock
    // under test -- if mutual exclusion is broken, the final value will
    // not match the expected total (classic lost-update detector).
    static long[] sharedCounter = new long[1];

    public static void main(String[] args) throws Exception {
        int[] threadCounts = {8, 24, 64};
        int reqsPerThread = 300;
        int trials = 4;
        boolean anyFail = false;

        for (int trial = 0; trial < trials; trial++) {
            for (int nT : threadCounts) {
                inside.set(0); peak.set(0); violated.set(false); completed.set(0);
                sharedCounter[0] = 0;

                CombiningLock lk = new CombiningLock();
                CountDownLatch latch = new CountDownLatch(nT);
                ExecutorService ex = Executors.newFixedThreadPool(nT);
                int totalReqs = nT * reqsPerThread;

                for (int t = 0; t < nT; t++) {
                    ex.submit(() -> {
                        try {
                            for (int i = 0; i < reqsPerThread; i++) {
                                lk.execute(() -> {
                                    int n = inside.incrementAndGet();
                                    peak.updateAndGet(cur -> Math.max(cur, n));
                                    if (n > 1) violated.set(true);
                                    // Naive, non-atomic read-modify-write --
                                    // only safe if truly exclusive.
                                    long v = sharedCounter[0];
                                    // Tiny busy-work to widen any race window.
                                    long spin = 0;
                                    for (int k = 0; k < 50; k++) spin += k;
                                    sharedCounter[0] = v + 1 + (spin - spin);
                                    inside.decrementAndGet();
                                });
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

                boolean counterOk = sharedCounter[0] == totalReqs;
                boolean pass = finished && !violated.get()
                    && completed.get() == totalReqs && counterOk;
                if (!pass) anyFail = true;

                System.out.printf(
                    "trial=%d nT=%2d finished=%-5s completed=%d/%d peakInside=%d violated=%s counter=%d/%d counterOk=%s meanBatch=%.2f maxBatch=%d passes=%d -> %s%n",
                    trial, nT, finished, completed.get(), totalReqs, peak.get(), violated.get(),
                    sharedCounter[0], totalReqs, counterOk,
                    lk.meanBatchSize(), lk.maxBatchSizeSeen(), lk.totalCombinerPasses(),
                    pass ? "PASS" : "FAIL");
            }
        }

        System.out.println();
        System.out.println(anyFail ? "OVERALL: FAIL" : "OVERALL: PASS");
    }
}
