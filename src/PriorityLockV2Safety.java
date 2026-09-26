import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

/** Same correctness discipline as PriorityLockSafety.java, applied to V2. */
public class PriorityLockV2Safety {
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

                AdaptivePriorityLockV2 lk = new AdaptivePriorityLockV2();
                CountDownLatch latch = new CountDownLatch(nT);
                ExecutorService ex = Executors.newFixedThreadPool(nT);
                int totalReqs = nT * reqsPerThread;

                for (int t = 0; t < nT; t++) {
                    ex.submit(() -> {
                        ThreadLocalRandom tlr = ThreadLocalRandom.current();
                        try {
                            for (int i = 0; i < reqsPerThread; i++) {
                                AdaptivePriorityLockV2.JobClass jc =
                                    tlr.nextDouble() < 0.8
                                        ? AdaptivePriorityLockV2.JobClass.SHORT
                                        : AdaptivePriorityLockV2.JobClass.LONG;
                                long csUs = jc == AdaptivePriorityLockV2.JobClass.SHORT ? 50 : 2000;

                                long waitNs = lk.acquire(jc);
                                maxObservedWaitNs.accumulateAndGet(waitNs, Math::max);

                                int n = inside.incrementAndGet();
                                peak.updateAndGet(cur -> Math.max(cur, n));
                                if (n > 1) violated.set(true);
                                long t0 = System.nanoTime();
                                while (System.nanoTime() - t0 < csUs * 1000L) {}
                                inside.decrementAndGet();

                                lk.release(jc, System.nanoTime() - t0);
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
                double boundMs = (AdaptivePriorityLockV2.MAX_WAIT_AGING_NS / 1e6) + 50.0;
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
