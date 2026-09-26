import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

/**
 * Correctness + liveness harness for AdaptiveCombiningLock, run BEFORE
 * any performance claim (same discipline as every prior mechanism in
 * this investigation -- CombiningLockSafety.java, PriorityLockV2Safety
 * .java, etc.).
 *
 * Checks:
 *   (1) Mutual exclusion via an occupancy counter around the actual
 *       work.run() body.
 *   (2) No lost work via an INDEPENDENT check: a naive, non-atomic
 *       shared counter that only produces the exact expected total if
 *       every increment was truly exclusive (a race silently corrupts
 *       it) -- this is a stronger correctness signal than the occupancy
 *       counter alone, and is the same technique that validated
 *       CombiningLock.
 *   (3) No hangs (liveness), at oversubscription up to nT=64 on this
 *       2-core sandbox, across repeated trials.
 *   (4) Reports the fast-path/combiner-path split and straggler count,
 *       to confirm the hybrid actually exercises BOTH code paths under
 *       these conditions (a safety check on the TEST, not just the
 *       lock: if one path is never hit, the corresponding correctness
 *       guarantee hasn't actually been exercised).
 */
public class AdaptiveCombiningLockSafety {
    static final AtomicInteger inside = new AtomicInteger(0);
    static final AtomicInteger peak = new AtomicInteger(0);
    static final AtomicBoolean violated = new AtomicBoolean(false);
    static final AtomicInteger completed = new AtomicInteger(0);
    static long[] sharedCounter = new long[1];

    public static void main(String[] args) throws Exception {
        int[] threadCounts = {1, 2, 8, 24, 64};
        int reqsPerThread = 300;
        int trials = 4;
        boolean anyFail = false;

        for (int trial = 0; trial < trials; trial++) {
            for (int nT : threadCounts) {
                inside.set(0); peak.set(0); violated.set(false); completed.set(0);
                sharedCounter[0] = 0;

                AdaptiveCombiningLock lk = new AdaptiveCombiningLock();
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
                                    long v = sharedCounter[0];
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
                    "trial=%d nT=%2d finished=%-5s completed=%d/%d peakInside=%d violated=%s counter=%d/%d counterOk=%s fastPath=%d combinerHits=%d meanBatch=%.2f -> %s%n",
                    trial, nT, finished, completed.get(), totalReqs, peak.get(), violated.get(),
                    sharedCounter[0], totalReqs, counterOk,
                    lk.fastPathHits(), lk.combinerHits(), lk.meanBatchSize(),
                    pass ? "PASS" : "FAIL");
            }
        }

        System.out.println();
        System.out.println(anyFail ? "OVERALL: FAIL" : "OVERALL: PASS");
    }
}
