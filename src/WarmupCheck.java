import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

/** Checks if the one nT=8 starvation-bound miss in PriorityLockV2Safety
 * (trial=0, maxWaitMs=100.7 vs bound 55.0) is a JIT/JVM warmup artifact
 * by running nT=8 many times in a row within a single JVM. */
public class WarmupCheck {
    public static void main(String[] args) throws Exception {
        for (int run = 0; run < 8; run++) {
            AtomicLong maxWaitNs = new AtomicLong(0);
            AtomicInteger inside = new AtomicInteger(0), peak = new AtomicInteger(0);
            AtomicBoolean violated = new AtomicBoolean(false);
            int nT = 8, reqsPerThread = 150;
            AdaptivePriorityLockV2 lk = new AdaptivePriorityLockV2();
            CountDownLatch latch = new CountDownLatch(nT);
            ExecutorService ex = Executors.newFixedThreadPool(nT);
            for (int t = 0; t < nT; t++) {
                ex.submit(() -> {
                    ThreadLocalRandom tlr = ThreadLocalRandom.current();
                    try {
                        for (int i = 0; i < reqsPerThread; i++) {
                            AdaptivePriorityLockV2.JobClass jc = tlr.nextDouble() < 0.8
                                ? AdaptivePriorityLockV2.JobClass.SHORT : AdaptivePriorityLockV2.JobClass.LONG;
                            long csUs = jc == AdaptivePriorityLockV2.JobClass.SHORT ? 50 : 2000;
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
            latch.await(60, TimeUnit.SECONDS);
            ex.shutdownNow(); ex.awaitTermination(5, TimeUnit.SECONDS);
            System.out.printf("run=%d maxWaitMs=%.3f peakInside=%d violated=%s%n",
                run, maxWaitNs.get()/1e6, peak.get(), violated.get());
        }
    }
}
