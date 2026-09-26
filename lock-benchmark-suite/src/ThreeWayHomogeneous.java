import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Resolves an inconsistency: PriorityLockBenchmark.java found
 * AdaptivePriorityLockV2 significantly beating FairMutex even under
 * HOMOGENEOUS load (nT=8 d=-1.1, nT=16 d=-0.95), but PrimitiveControlV2
 * found that a plain-FIFO lock built on the IDENTICAL low-level
 * primitive TIES FairMutex exactly (d~-0.02 to 0.3, all not
 * significant). Under homogeneous load, SJF+aging should be a complete
 * no-op (nothing to reorder by size), so these two findings should
 * agree. This runs all three lock types side by side, in ONE program,
 * randomized order, same conditions, to determine which result was
 * the fluke.
 */
public class ThreeWayHomogeneous {
    static final int[] THREADS = {8};
    static final int SEEDS = 40;
    static final int REQS_PER_THREAD = 120;

    public static void main(String[] args) throws Exception {
        for (int nT : THREADS) {
            System.out.println("=== HOMOGENEOUS nT=" + nT + " (3-way, randomized order) ===");
            // warmup discard
            runPriority(nT); runFifo(nT); runFairMutex(nT);

            double[] plMeans = new double[SEEDS], fifoMeans = new double[SEEDS], mxMeans = new double[SEEDS];
            for (int s = 0; s < SEEDS; s++) {
                int[] order = {0,1,2};
                for (int i = order.length-1;i>0;i--){int j=ThreadLocalRandom.current().nextInt(i+1);int tmp=order[i];order[i]=order[j];order[j]=tmp;}
                for (int slot : order) {
                    if (slot==0) plMeans[s] = runPriority(nT);
                    else if (slot==1) fifoMeans[s] = runFifo(nT);
                    else mxMeans[s] = runFairMutex(nT);
                }
                System.out.print(".");
            }
            System.out.println();
            System.out.printf("  AdaptivePriorityLockV2 mean=%.3fms +/-%.3f%n", mean(plMeans), ci95(plMeans));
            System.out.printf("  PlainFifoV2Lock        mean=%.3fms +/-%.3f%n", mean(fifoMeans), ci95(fifoMeans));
            System.out.printf("  FairMutex              mean=%.3fms +/-%.3f%n", mean(mxMeans), ci95(mxMeans));
            System.out.printf("  Priority vs Mutex: t=%.3f d=%.3f  %s%n",
                welchT(plMeans,mxMeans), cohensD(plMeans,mxMeans),
                Math.abs(welchT(plMeans,mxMeans))>2.2?"SIGNIFICANT":"not significant");
            System.out.printf("  Fifo vs Mutex:     t=%.3f d=%.3f  %s%n",
                welchT(fifoMeans,mxMeans), cohensD(fifoMeans,mxMeans),
                Math.abs(welchT(fifoMeans,mxMeans))>2.2?"SIGNIFICANT":"not significant");
            System.out.printf("  Priority vs Fifo:  t=%.3f d=%.3f  %s%n%n",
                welchT(plMeans,fifoMeans), cohensD(plMeans,fifoMeans),
                Math.abs(welchT(plMeans,fifoMeans))>2.2?"SIGNIFICANT":"not significant");
        }
    }

    static double runPriority(int nT) throws Exception {
        AdaptivePriorityLockV2 lk = new AdaptivePriorityLockV2();
        List<Long> waits = Collections.synchronizedList(new ArrayList<>());
        CountDownLatch latch = new CountDownLatch(nT);
        ExecutorService ex = Executors.newFixedThreadPool(nT);
        for (int t = 0; t < nT; t++) {
            ex.submit(() -> {
                ThreadLocalRandom tlr = ThreadLocalRandom.current();
                try {
                    for (int i = 0; i < REQS_PER_THREAD; i++) {
                        sleepUs(80 + tlr.nextLong(60));
                        long w = lk.acquire(AdaptivePriorityLockV2.JobClass.SHORT);
                        waits.add(w);
                        long t0 = System.nanoTime();
                        while (System.nanoTime() - t0 < 150_000L) {}
                        lk.release(AdaptivePriorityLockV2.JobClass.SHORT, System.nanoTime() - t0);
                    }
                } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
                finally { latch.countDown(); }
            });
        }
        latch.await(90, TimeUnit.SECONDS);
        ex.shutdownNow(); ex.awaitTermination(5, TimeUnit.SECONDS);
        return meanMs(waits);
    }

    static double runFifo(int nT) throws Exception {
        PrimitiveControlV2.PlainFifoV2Lock lk = new PrimitiveControlV2.PlainFifoV2Lock();
        List<Long> waits = Collections.synchronizedList(new ArrayList<>());
        CountDownLatch latch = new CountDownLatch(nT);
        ExecutorService ex = Executors.newFixedThreadPool(nT);
        for (int t = 0; t < nT; t++) {
            ex.submit(() -> {
                ThreadLocalRandom tlr = ThreadLocalRandom.current();
                try {
                    for (int i = 0; i < REQS_PER_THREAD; i++) {
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

    static double runFairMutex(int nT) throws Exception {
        ReentrantLock lk = new ReentrantLock(true);
        List<Long> waits = Collections.synchronizedList(new ArrayList<>());
        CountDownLatch latch = new CountDownLatch(nT);
        ExecutorService ex = Executors.newFixedThreadPool(nT);
        for (int t = 0; t < nT; t++) {
            ex.submit(() -> {
                ThreadLocalRandom tlr = ThreadLocalRandom.current();
                try {
                    for (int i = 0; i < REQS_PER_THREAD; i++) {
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
        double[] tc={0,12.706,4.303,3.182,2.776,2.571,2.447,2.365,2.306,2.262,2.228,2.201,2.179,2.160,2.145,2.131};
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
