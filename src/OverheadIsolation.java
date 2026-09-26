import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

/**
 * Isolates WHY AdaptivePriorityLock is slower than FairMutex overall
 * (PriorityLockDiagnostic.java showed 37% more wall time, and even SHORT
 * jobs -- the ones SJF should help -- were worse).
 *
 * Two candidate causes:
 *  (a) monitor wait()/notifyAll() Mesa-style handoff is inherently more
 *      expensive than java.util.concurrent's AQS-based ReentrantLock
 *      (a well-known, generic JVM-level cost, unrelated to scheduling
 *      policy).
 *  (b) the O(n) linear scan across all waiters on every release() (needed
 *      because "effective priority" is time-varying due to aging) adds
 *      cost that scales with queue depth.
 *
 * Tests a "PlainFifoMonitorLock" -- same monitor+wait/notify structure as
 * AdaptivePriorityLock, but FIFO ordering with an O(1) head-of-queue
 * dequeue (no scan, no priority, no aging) -- against FairMutex. If this
 * is ALSO much slower than FairMutex, cause (a) dominates and the
 * monitor pattern itself is the wrong tool. If it's close to FairMutex,
 * cause (b) -- the scan -- is the real culprit and is fixable.
 */
public class OverheadIsolation {
    static final class PlainFifoMonitorLock {
        private final Object monitor = new Object();
        private boolean held = false;
        private final ArrayDeque<Waiter> queue = new ArrayDeque<>();
        static final class Waiter { volatile boolean turn = false; }

        long acquire() throws InterruptedException {
            long t0 = System.nanoTime();
            synchronized (monitor) {
                if (!held && queue.isEmpty()) { held = true; return System.nanoTime() - t0; }
                Waiter w = new Waiter();
                queue.add(w);
                while (!w.turn) monitor.wait();
            }
            return System.nanoTime() - t0;
        }
        void release() {
            synchronized (monitor) {
                if (queue.isEmpty()) { held = false; return; }
                Waiter next = queue.poll();
                next.turn = true;
                monitor.notifyAll();
            }
        }
    }

    public static void main(String[] args) throws Exception {
        int nT = 24;
        int reqsPerThread = 150;

        System.out.println("=== PlainFifoMonitorLock (isolates monitor overhead vs AQS) ===");
        runFifoMonitor(nT, reqsPerThread);

        System.out.println("\n=== FairMutex (AQS ReentrantLock) ===");
        runFairMutex(nT, reqsPerThread);

        System.out.println("\n=== AdaptivePriorityLock (full mechanism, for reference) ===");
        runPriorityLock(nT, reqsPerThread);
    }

    static void runFifoMonitor(int nT, int reqsPerThread) throws Exception {
        PlainFifoMonitorLock lk = new PlainFifoMonitorLock();
        CountDownLatch latch = new CountDownLatch(nT);
        ExecutorService ex = Executors.newFixedThreadPool(nT);
        long wallT0 = System.nanoTime();
        for (int t = 0; t < nT; t++) {
            ex.submit(() -> {
                try {
                    for (int i = 0; i < reqsPerThread; i++) {
                        boolean isLong = ThreadLocalRandom.current().nextDouble() >= 0.8;
                        long csUs = isLong ? 2000 : 50;
                        lk.acquire();
                        long t0 = System.nanoTime();
                        while (System.nanoTime() - t0 < csUs * 1000L) {}
                        lk.release();
                    }
                } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
                finally { latch.countDown(); }
            });
        }
        latch.await(90, TimeUnit.SECONDS);
        long wallMs = (System.nanoTime() - wallT0) / 1_000_000;
        ex.shutdownNow(); ex.awaitTermination(5, TimeUnit.SECONDS);
        System.out.printf("  wallMs=%d%n", wallMs);
    }

    static void runFairMutex(int nT, int reqsPerThread) throws Exception {
        java.util.concurrent.locks.ReentrantLock lk = new java.util.concurrent.locks.ReentrantLock(true);
        CountDownLatch latch = new CountDownLatch(nT);
        ExecutorService ex = Executors.newFixedThreadPool(nT);
        long wallT0 = System.nanoTime();
        for (int t = 0; t < nT; t++) {
            ex.submit(() -> {
                try {
                    for (int i = 0; i < reqsPerThread; i++) {
                        boolean isLong = ThreadLocalRandom.current().nextDouble() >= 0.8;
                        long csUs = isLong ? 2000 : 50;
                        lk.lockInterruptibly();
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
        System.out.printf("  wallMs=%d%n", wallMs);
    }

    static void runPriorityLock(int nT, int reqsPerThread) throws Exception {
        AdaptivePriorityLock lk = new AdaptivePriorityLock();
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
                        lk.acquire(jc);
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
        System.out.printf("  wallMs=%d%n", wallMs);
    }
}
