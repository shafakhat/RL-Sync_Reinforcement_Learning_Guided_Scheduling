import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.concurrent.locks.LockSupport;

/**
 * Verifies the hypothesis for WHY AdaptivePriorityLockV2's SJF advantage
 * over FairMutex fades and reverses by nT=24 (priority_lock_benchmark.csv,
 * HETEROGENEOUS condition): as thread count grows, per-request queueing
 * delay approaches and then exceeds the 5ms aging cutoff for a growing
 * fraction of waiters, forcing them into FIFO order regardless of
 * predicted duration -- collapsing the SJF benefit by design (the aging
 * safety valve is doing exactly its job: preventing starvation at the
 * cost of giving up scheduling freedom under saturation).
 *
 * Instruments AdaptivePriorityLockV2's release() indirectly by measuring,
 * for the SAME heterogeneous workload used in the real benchmark, what
 * fraction of admitted waiters were "aged" (effective priority forced to
 * the -1e18 branch) vs "SJF-ranked" (effective priority = predicted
 * duration), at nT=8 vs nT=24.
 */
public class AgingTripRateCheck {
    // Local copy of AdaptivePriorityLockV2 with instrumentation counters
    // added, to avoid modifying the reported/benchmarked class.
    static final AtomicLong agedCount = new AtomicLong(0);
    static final AtomicLong sjfCount = new AtomicLong(0);

    static final class InstrumentedLock {
        enum JobClass { SHORT, LONG }
        static final long MAX_WAIT_AGING_NS = 5_000_000L;

        static final class Waiter {
            final long ticket; final long arrivalNs; final double predictedDurationNs;
            final Thread thread; volatile boolean turn = false;
            Waiter(long ticket, long arrivalNs, double predictedDurationNs, Thread thread) {
                this.ticket=ticket; this.arrivalNs=arrivalNs; this.predictedDurationNs=predictedDurationNs; this.thread=thread;
            }
            double effectivePriority(long now, boolean[] agedOut) {
                long waited = now - arrivalNs;
                if (waited > MAX_WAIT_AGING_NS) { agedOut[0] = true; return -1e18 + ticket; }
                agedOut[0] = false;
                return predictedDurationNs;
            }
        }
        static final class Estimator {
            volatile double emaNs; static final double ALPHA = 0.15;
            Estimator(double init) { emaNs = init; }
            void update(long o) { emaNs = ALPHA*o + (1-ALPHA)*emaNs; }
            double estimate() { return emaNs; }
        }

        final AtomicBoolean bk = new AtomicBoolean(false);
        boolean held = false;
        final ArrayList<Waiter> waiting = new ArrayList<>();
        final AtomicLong seq = new AtomicLong(0);
        final ConcurrentHashMap<JobClass, Estimator> estimators = new ConcurrentHashMap<>();
        InstrumentedLock() { for (JobClass jc: JobClass.values()) estimators.put(jc, new Estimator(100_000)); }
        void lock() { while(!bk.compareAndSet(false,true)) Thread.onSpinWait(); }
        void unlock() { bk.set(false); }

        long acquire(JobClass jc) throws InterruptedException {
            long t0 = System.nanoTime(); Waiter w = null;
            lock();
            try {
                if (!held && waiting.isEmpty()) { held = true; return System.nanoTime()-t0; }
                w = new Waiter(seq.getAndIncrement(), t0, estimators.get(jc).estimate(), Thread.currentThread());
                waiting.add(w);
            } finally { unlock(); }
            while (!w.turn) { LockSupport.park(); if (Thread.interrupted()) throw new InterruptedException(); }
            return System.nanoTime()-t0;
        }
        void release(JobClass jc, long durationNs) {
            Waiter winner = null;
            lock();
            try {
                estimators.get(jc).update(durationNs);
                if (waiting.isEmpty()) { held = false; return; }
                long now = System.nanoTime();
                double bestPr = Double.POSITIVE_INFINITY; int bestIdx = -1; boolean[] agedOut = new boolean[1]; boolean winnerAged=false;
                for (int i=0;i<waiting.size();i++) {
                    double pr = waiting.get(i).effectivePriority(now, agedOut);
                    if (pr < bestPr) { bestPr=pr; bestIdx=i; winnerAged=agedOut[0]; }
                }
                winner = waiting.remove(bestIdx);
                if (winnerAged) agedCount.incrementAndGet(); else sjfCount.incrementAndGet();
            } finally { unlock(); }
            winner.turn = true;
            LockSupport.unpark(winner.thread);
        }
    }

    public static void main(String[] args) throws Exception {
        for (int nT : new int[]{8, 16, 24, 32}) {
            agedCount.set(0); sjfCount.set(0);
            InstrumentedLock lk = new InstrumentedLock();
            int reqsPerThread = 120;
            CountDownLatch latch = new CountDownLatch(nT);
            ExecutorService ex = Executors.newFixedThreadPool(nT);
            for (int t = 0; t < nT; t++) {
                ex.submit(() -> {
                    ThreadLocalRandom tlr = ThreadLocalRandom.current();
                    try {
                        for (int i = 0; i < reqsPerThread; i++) {
                            sleepUs(80 + tlr.nextLong(60));
                            boolean isLong = tlr.nextDouble() < 0.2;
                            InstrumentedLock.JobClass jc = isLong ? InstrumentedLock.JobClass.LONG : InstrumentedLock.JobClass.SHORT;
                            long csUs = isLong ? 2000 : 150;
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
            ex.shutdownNow(); ex.awaitTermination(5, TimeUnit.SECONDS);
            long total = agedCount.get() + sjfCount.get();
            double agedPct = total > 0 ? 100.0 * agedCount.get() / total : 0;
            System.out.printf("nT=%2d  total_admissions=%d  aged=%d (%.1f%%)  sjf_ranked=%d (%.1f%%)%n",
                nT, total, agedCount.get(), agedPct, sjfCount.get(), 100-agedPct);
        }
    }

    static void sleepUs(long us) throws InterruptedException {
        long deadline = System.nanoTime() + us * 1000L;
        if (us > 1500) Thread.sleep(us / 1000);
        while (System.nanoTime() < deadline) Thread.onSpinWait();
    }
}
