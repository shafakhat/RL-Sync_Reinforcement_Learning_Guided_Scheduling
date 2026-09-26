import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.concurrent.locks.LockSupport;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Hunts down the mechanism behind ThreeWayHomogeneous's surprising,
 * high-powered (n=40 seeds) finding: AdaptivePriorityLockV2 significantly
 * beats BOTH a plain-FIFO lock on the identical primitive AND FairMutex,
 * even under homogeneous load where SJF reordering should be a no-op.
 *
 * The one live, structural difference between AdaptivePriorityLockV2 and
 * PlainFifoV2Lock is the per-class EMA duration estimator: even under
 * nominally "homogeneous" 150us critical sections, REAL observed
 * durations vary slightly (OS scheduling jitter on this shared 2-core
 * sandbox), so predictedDurationNs drifts between successive waiters
 * instead of being exactly tied -- meaning release() does NOT serve
 * strict FIFO order, it serves "most-recently-favorable-estimate first",
 * which is a subtly different (and apparently NOT neutral) discipline.
 *
 * This builds "FrozenEstimatorLock": identical to AdaptivePriorityLockV2
 * in every way EXCEPT the per-class estimate is a hardcoded constant
 * (never updated from observed durations), which forces exact ties on
 * every waiter and hence TRUE arrival-order FIFO (since the linear scan
 * picks the first-encountered minimum, and waiters are appended in
 * arrival order). If FrozenEstimatorLock's advantage over
 * PlainFifoV2Lock DISAPPEARS, the EMA-drift-driven reordering is the
 * real mechanism -- and needs to be understood on its own terms, not
 * silently kept as an unexplained "win".
 */
public class EstimatorFreezeTest {
    static final class FrozenEstimatorLock {
        private static final class Waiter {
            final long ticket; final Thread thread; volatile boolean turn=false;
            Waiter(long ticket, Thread t){this.ticket=ticket;thread=t;}
        }
        private final AtomicBoolean bk = new AtomicBoolean(false);
        private boolean held=false;
        private final ArrayList<Waiter> waiting = new ArrayList<>();
        private final AtomicLong seq = new AtomicLong(0);
        private void lock(){while(!bk.compareAndSet(false,true))Thread.onSpinWait();}
        private void unlock(){bk.set(false);}

        long acquire() throws InterruptedException {
            long t0=System.nanoTime(); Waiter w=null;
            lock();
            try {
                if (!held && waiting.isEmpty()) { held=true; return System.nanoTime()-t0; }
                w = new Waiter(seq.getAndIncrement(), Thread.currentThread());
                waiting.add(w);
            } finally { unlock(); }
            while(!w.turn){ LockSupport.park(); if(Thread.interrupted()) throw new InterruptedException(); }
            return System.nanoTime()-t0;
        }
        void release() {
            Waiter winner=null;
            lock();
            try {
                if (waiting.isEmpty()) { held=false; return; }
                // FROZEN priority: every waiter has the SAME constant
                // "predicted duration" (100_000.0 ns, arbitrary but fixed),
                // so the linear scan's strict "<" comparison always picks
                // the FIRST-encountered (= earliest-arrived, since we
                // append in arrival order) waiter -- i.e. true FIFO.
                double bestPr = Double.POSITIVE_INFINITY; int bestIdx=-1;
                for (int i=0;i<waiting.size();i++){
                    double pr = 100_000.0; // frozen constant, no EMA drift
                    if (pr < bestPr) { bestPr=pr; bestIdx=i; }
                }
                winner = waiting.remove(bestIdx);
            } finally { unlock(); }
            winner.turn = true;
            LockSupport.unpark(winner.thread);
        }
    }

    static final int nT = 8, SEEDS = 40, REQS_PER_THREAD = 120;

    public static void main(String[] args) throws Exception {
        System.out.println("=== HOMOGENEOUS nT=" + nT + " : FrozenEstimatorLock vs PlainFifoV2Lock vs FairMutex vs AdaptivePriorityLockV2 ===");
        runFrozen(nT); runFifo(nT); runFairMutex(nT); runAdaptive(nT); // warmup discard

        double[] frMeans=new double[SEEDS], fifoMeans=new double[SEEDS], mxMeans=new double[SEEDS], adMeans=new double[SEEDS];
        for (int s=0;s<SEEDS;s++){
            int[] order={0,1,2,3};
            for(int i=order.length-1;i>0;i--){int j=ThreadLocalRandom.current().nextInt(i+1);int t=order[i];order[i]=order[j];order[j]=t;}
            for (int slot: order){
                if(slot==0) frMeans[s]=runFrozen(nT);
                else if(slot==1) fifoMeans[s]=runFifo(nT);
                else if(slot==2) mxMeans[s]=runFairMutex(nT);
                else adMeans[s]=runAdaptive(nT);
            }
            System.out.print(".");
        }
        System.out.println();
        System.out.printf("  FrozenEstimatorLock (true FIFO, same scan) mean=%.3fms +/-%.3f%n", mean(frMeans), ci95(frMeans));
        System.out.printf("  PlainFifoV2Lock (ArrayDeque FIFO)          mean=%.3fms +/-%.3f%n", mean(fifoMeans), ci95(fifoMeans));
        System.out.printf("  FairMutex                                  mean=%.3fms +/-%.3f%n", mean(mxMeans), ci95(mxMeans));
        System.out.printf("  AdaptivePriorityLockV2 (live EMA)          mean=%.3fms +/-%.3f%n", mean(adMeans), ci95(adMeans));
        System.out.println();
        report("Frozen vs Fifo", frMeans, fifoMeans);
        report("Frozen vs Mutex", frMeans, mxMeans);
        report("Adaptive vs Frozen", adMeans, frMeans);
        report("Adaptive vs Mutex", adMeans, mxMeans);
    }

    static void report(String label, double[] a, double[] b) {
        double t=welchT(a,b), d=cohensD(a,b);
        System.out.printf("  %-20s t=%.3f d=%.3f %s%n", label, t, d, Math.abs(t)>2.2?"SIGNIFICANT":"not significant");
    }

    static double runFrozen(int nT) throws Exception {
        FrozenEstimatorLock lk = new FrozenEstimatorLock();
        List<Long> waits = Collections.synchronizedList(new ArrayList<>());
        CountDownLatch latch = new CountDownLatch(nT);
        ExecutorService ex = Executors.newFixedThreadPool(nT);
        for (int t=0;t<nT;t++) ex.submit(() -> {
            ThreadLocalRandom tlr = ThreadLocalRandom.current();
            try {
                for (int i=0;i<REQS_PER_THREAD;i++){
                    sleepUs(80 + tlr.nextLong(60));
                    long w = lk.acquire(); waits.add(w);
                    long t0=System.nanoTime(); while(System.nanoTime()-t0<150_000L){}
                    lk.release();
                }
            } catch (InterruptedException e){Thread.currentThread().interrupt();}
            finally { latch.countDown(); }
        });
        latch.await(90, TimeUnit.SECONDS); ex.shutdownNow(); ex.awaitTermination(5, TimeUnit.SECONDS);
        return meanMs(waits);
    }

    static double runFifo(int nT) throws Exception {
        PrimitiveControlV2.PlainFifoV2Lock lk = new PrimitiveControlV2.PlainFifoV2Lock();
        List<Long> waits = Collections.synchronizedList(new ArrayList<>());
        CountDownLatch latch = new CountDownLatch(nT);
        ExecutorService ex = Executors.newFixedThreadPool(nT);
        for (int t=0;t<nT;t++) ex.submit(() -> {
            ThreadLocalRandom tlr = ThreadLocalRandom.current();
            try {
                for (int i=0;i<REQS_PER_THREAD;i++){
                    sleepUs(80 + tlr.nextLong(60));
                    long w = lk.acquire(); waits.add(w);
                    long t0=System.nanoTime(); while(System.nanoTime()-t0<150_000L){}
                    lk.release();
                }
            } catch (InterruptedException e){Thread.currentThread().interrupt();}
            finally { latch.countDown(); }
        });
        latch.await(90, TimeUnit.SECONDS); ex.shutdownNow(); ex.awaitTermination(5, TimeUnit.SECONDS);
        return meanMs(waits);
    }

    static double runFairMutex(int nT) throws Exception {
        ReentrantLock lk = new ReentrantLock(true);
        List<Long> waits = Collections.synchronizedList(new ArrayList<>());
        CountDownLatch latch = new CountDownLatch(nT);
        ExecutorService ex = Executors.newFixedThreadPool(nT);
        for (int t=0;t<nT;t++) ex.submit(() -> {
            ThreadLocalRandom tlr = ThreadLocalRandom.current();
            try {
                for (int i=0;i<REQS_PER_THREAD;i++){
                    sleepUs(80 + tlr.nextLong(60));
                    long t0w=System.nanoTime(); lk.lockInterruptibly();
                    waits.add(System.nanoTime()-t0w);
                    long t0=System.nanoTime(); while(System.nanoTime()-t0<150_000L){}
                    lk.unlock();
                }
            } catch (InterruptedException e){Thread.currentThread().interrupt();}
            finally { latch.countDown(); }
        });
        latch.await(90, TimeUnit.SECONDS); ex.shutdownNow(); ex.awaitTermination(5, TimeUnit.SECONDS);
        return meanMs(waits);
    }

    static double runAdaptive(int nT) throws Exception {
        AdaptivePriorityLockV2 lk = new AdaptivePriorityLockV2();
        List<Long> waits = Collections.synchronizedList(new ArrayList<>());
        CountDownLatch latch = new CountDownLatch(nT);
        ExecutorService ex = Executors.newFixedThreadPool(nT);
        for (int t=0;t<nT;t++) ex.submit(() -> {
            ThreadLocalRandom tlr = ThreadLocalRandom.current();
            try {
                for (int i=0;i<REQS_PER_THREAD;i++){
                    sleepUs(80 + tlr.nextLong(60));
                    long w = lk.acquire(AdaptivePriorityLockV2.JobClass.SHORT); waits.add(w);
                    long t0=System.nanoTime(); while(System.nanoTime()-t0<150_000L){}
                    lk.release(AdaptivePriorityLockV2.JobClass.SHORT, System.nanoTime()-t0);
                }
            } catch (InterruptedException e){Thread.currentThread().interrupt();}
            finally { latch.countDown(); }
        });
        latch.await(90, TimeUnit.SECONDS); ex.shutdownNow(); ex.awaitTermination(5, TimeUnit.SECONDS);
        return meanMs(waits);
    }

    static void sleepUs(long us) throws InterruptedException {
        long deadline = System.nanoTime() + us*1000L;
        if (us>1500) Thread.sleep(us/1000);
        while (System.nanoTime()<deadline) Thread.onSpinWait();
    }
    static double meanMs(List<Long> w){ if(w.isEmpty())return 0; return w.stream().mapToLong(Long::longValue).average().orElse(0)/1e6; }
    static double mean(double[] a){double s=0;for(double v:a)s+=v;return s/a.length;}
    static double var(double[] a){double m=mean(a),s=0;for(double v:a)s+=(v-m)*(v-m);return s/Math.max(1,a.length-1);}
    static double std(double[] a){return Math.sqrt(var(a));}
    static double ci95(double[] a){
        double[] tc={0,12.706,4.303,3.182,2.776,2.571,2.447,2.365,2.306,2.262,2.228,2.201,2.179,2.160,2.145,2.131,2.120,2.110,2.101,2.093,2.086,2.080,2.074,2.069,2.064,2.060,2.056,2.052,2.048,2.045,2.042,2.040,2.037,2.035,2.032,2.030,2.028,2.026,2.024};
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
