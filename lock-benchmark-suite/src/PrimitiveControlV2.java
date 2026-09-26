import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.concurrent.locks.LockSupport;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Final falsification control. PriorityLockBenchmark showed
 * AdaptivePriorityLockV2 beating FairMutex even under HOMOGENEOUS load
 * (where SJF has no possible lever) -- a red flag that the win is
 * (again) primitive-implementation noise, not scheduling policy.
 *
 * This builds "PlainFifoV2Lock": the EXACT same low-level primitive as
 * AdaptivePriorityLockV2 (short CAS spinlock guarding a queue +
 * LockSupport.park/unpark for blocking/wakeup, direct one-thread
 * hand-off, no thundering herd) but with PLAIN FIFO ordering -- zero
 * priority logic, zero aging, zero per-class estimators. If this control
 * ALSO beats FairMutex by about the same margin as the full
 * AdaptivePriorityLockV2, the entire apparent "win" is explained by
 * ReentrantLock(true)'s fairness-mode bookkeeping being more expensive
 * than this simpler hand-rolled primitive -- NOT by shortest-job-first
 * scheduling doing anything useful. That must be reported honestly.
 */
public class PrimitiveControlV2 {
    static final class PlainFifoV2Lock {
        private static final class Waiter {
            final Thread thread; volatile boolean turn = false;
            Waiter(Thread t) { thread = t; }
        }
        private final AtomicBoolean bookkeepingLock = new AtomicBoolean(false);
        private boolean held = false;
        private final ArrayDeque<Waiter> queue = new ArrayDeque<>();

        private void lockBk() { while (!bookkeepingLock.compareAndSet(false, true)) Thread.onSpinWait(); }
        private void unlockBk() { bookkeepingLock.set(false); }

        long acquire() throws InterruptedException {
            long t0 = System.nanoTime();
            Waiter w = null;
            lockBk();
            try {
                if (!held && queue.isEmpty()) { held = true; return System.nanoTime() - t0; }
                w = new Waiter(Thread.currentThread());
                queue.add(w);
            } finally { unlockBk(); }
            while (!w.turn) {
                LockSupport.park();
                if (Thread.interrupted()) throw new InterruptedException();
            }
            return System.nanoTime() - t0;
        }
        void release() {
            Waiter next = null;
            lockBk();
            try {
                if (queue.isEmpty()) { held = false; return; }
                next = queue.poll();
            } finally { unlockBk(); }
            next.turn = true;
            LockSupport.unpark(next.thread);
        }
    }

    static final int[] THREADS = {8, 16, 24};
    static final int SEEDS = 12;
    static final int REQS_PER_THREAD = 120;

    public static void main(String[] args) throws Exception {
        for (int nT : THREADS) {
            System.out.println("=== HOMOGENEOUS nT=" + nT + " (control: plain FIFO on the V2 primitive) ===");
            // Warmup discard
            runFifo(nT, REQS_PER_THREAD);
            runFairMutex(nT, REQS_PER_THREAD);

            double[] fifoMeans = new double[SEEDS], mxMeans = new double[SEEDS];
            for (int s = 0; s < SEEDS; s++) {
                boolean fifoFirst = ThreadLocalRandom.current().nextBoolean();
                if (fifoFirst) {
                    fifoMeans[s] = runFifo(nT, REQS_PER_THREAD);
                    mxMeans[s] = runFairMutex(nT, REQS_PER_THREAD);
                } else {
                    mxMeans[s] = runFairMutex(nT, REQS_PER_THREAD);
                    fifoMeans[s] = runFifo(nT, REQS_PER_THREAD);
                }
                System.out.print(".");
            }
            System.out.println();
            double t = welchT(fifoMeans, mxMeans), d = cohensD(fifoMeans, mxMeans);
            System.out.printf("  PlainFifoV2Lock (same primitive, no SJF) mean=%.3fms +/-%.3f%n",
                mean(fifoMeans), ci95(fifoMeans));
            System.out.printf("  FairMutex                                mean=%.3fms +/-%.3f%n",
                mean(mxMeans), ci95(mxMeans));
            System.out.printf("  Welch t=%.3f  Cohen's d=%.3f  %s%n%n",
                t, d, Math.abs(t) > 2.2 ? "SIGNIFICANT" : "not significant");
        }
    }

    static double runFifo(int nT, int reqsPerThread) throws Exception {
        PlainFifoV2Lock lk = new PlainFifoV2Lock();
        List<Long> waits = Collections.synchronizedList(new ArrayList<>());
        CountDownLatch latch = new CountDownLatch(nT);
        ExecutorService ex = Executors.newFixedThreadPool(nT);
        for (int t = 0; t < nT; t++) {
            ex.submit(() -> {
                ThreadLocalRandom tlr = ThreadLocalRandom.current();
                try {
                    for (int i = 0; i < reqsPerThread; i++) {
                        sleepUs(80 + tlr.nextLong(60));
                        long w = lk.acquire();
                        waits.add(w);
                        long t0 = System.nanoTime();
                        while (System.nanoTime() - t0 < 150_000L) {}
                        lk.release();
                    }
                } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
                finally { latch.countDown(); }
            });
        }
        latch.await(90, TimeUnit.SECONDS);
        ex.shutdownNow(); ex.awaitTermination(5, TimeUnit.SECONDS);
        return meanMs(waits);
    }

    static double runFairMutex(int nT, int reqsPerThread) throws Exception {
        ReentrantLock lk = new ReentrantLock(true);
        List<Long> waits = Collections.synchronizedList(new ArrayList<>());
        CountDownLatch latch = new CountDownLatch(nT);
        ExecutorService ex = Executors.newFixedThreadPool(nT);
        for (int t = 0; t < nT; t++) {
            ex.submit(() -> {
                ThreadLocalRandom tlr = ThreadLocalRandom.current();
                try {
                    for (int i = 0; i < reqsPerThread; i++) {
                        sleepUs(80 + tlr.nextLong(60));
                        long t0w = System.nanoTime();
                        lk.lockInterruptibly();
                        waits.add(System.nanoTime() - t0w);
                        long t0 = System.nanoTime();
                        while (System.nanoTime() - t0 < 150_000L) {}
                        lk.unlock();
                    }
                } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
                finally { latch.countDown(); }
            });
        }
        latch.await(90, TimeUnit.SECONDS);
        ex.shutdownNow(); ex.awaitTermination(5, TimeUnit.SECONDS);
        return meanMs(waits);
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
