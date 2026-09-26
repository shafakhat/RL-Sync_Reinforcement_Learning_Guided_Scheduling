import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

/** Same harness pattern as V1/V2 safety checks, applied to V3
 *  (CS-duration-aware spin sizing + spinner-count admission gate).
 *  Must pass before any V3 performance number is trusted. */
public class AdaptiveCombiningLockV3Safety {
    static final AtomicInteger inside = new AtomicInteger(0);
    static final AtomicInteger peak = new AtomicInteger(0);
    static final AtomicBoolean violated = new AtomicBoolean(false);
    static final AtomicInteger completed = new AtomicInteger(0);
    static long[] sharedCounter = new long[1];

    public static void main(String[] args) throws Exception {
        int[] threadCounts = {1, 2, 4, 8, 24, 64};
        int reqsPerThread = 300;
        int trials = 4;
        boolean anyFail = false;

        for (int trial = 0; trial < trials; trial++) {
            for (int nT : threadCounts) {
                inside.set(0); peak.set(0); violated.set(false); completed.set(0);
                sharedCounter[0] = 0;

                AdaptiveCombiningLockV3 lk = new AdaptiveCombiningLockV3();
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
                    "trial=%d nT=%2d finished=%-5s completed=%d/%d peakInside=%d violated=%s counter=%d/%d counterOk=%s fastPath=%d spinRetry=%d gateSkipped=%d combinerHits=%d csEst=%dns -> %s%n",
                    trial, nT, finished, completed.get(), totalReqs, peak.get(), violated.get(),
                    sharedCounter[0], totalReqs, counterOk,
                    lk.fastPathHits(), lk.spinRetryHits(), lk.spinGateSkipped(), lk.combinerHits(), lk.csEstimateNs(),
                    pass ? "PASS" : "FAIL");
            }
        }

        System.out.println();
        System.out.println(anyFail ? "OVERALL: FAIL" : "OVERALL: PASS");
    }
}
