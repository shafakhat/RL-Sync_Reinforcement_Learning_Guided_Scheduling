import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

/**
 * Diagnoses why CombiningBenchmark showed CombiningLock degrading
 * relative to FairMutex at nT=48/64 (19-33% worse mean, and a much
 * larger p99 blowup: 40ms vs FairMutex's 17ms at nT=64) DESPITE batch
 * sizes growing as predicted (38-50 requests/batch) -- the opposite of
 * what pure amortization should produce.
 *
 * Hypothesis: the liveness-bug fix in CombiningLock.execute() has every
 * non-combiner thread wake up and retry the combiner-CAS every 50us
 * "just in case" its targeted unpark was missed. With nT=64 threads on
 * a 2-core sandbox, that is up to ~1.28 million wake/CAS-attempt cycles
 * per second of pure scheduling overhead, competing for the same 2
 * cores the actual combiner needs to make progress on real work --
 * i.e. the fix for one bug (a liveness hang) introduced a new
 * performance bug (thrashing) that gets worse exactly as thread count
 * grows, which would explain both symptoms:
 *   (a) mean/p99 getting worse at high nT despite bigger batches
 *   (b) the disproportionate p99 blowup (thrashing is a tail-latency
 *       problem, hitting whichever unlucky thread's CAS attempts keep
 *       colliding with the busy combiner's need for CPU)
 *
 * This directly counts total combiner-CAS ATTEMPTS (won + lost) at
 * nT=16 vs nT=64 to see if the lost-attempt rate scales with nT as
 * predicted.
 */
public class PollingOverheadCheck {
    public static void main(String[] args) throws Exception {
        for (int nT : new int[]{8, 16, 32, 64}) {
            AtomicLong casAttempts = new AtomicLong(0);
            AtomicLong casWins = new AtomicLong(0);
            int reqsPerThread = 150;

            InstrumentedCombiningLock lk = new InstrumentedCombiningLock(casAttempts, casWins);
            CountDownLatch latch = new CountDownLatch(nT);
            ExecutorService ex = Executors.newFixedThreadPool(nT);
            long wallT0 = System.nanoTime();
            for (int t = 0; t < nT; t++) {
                ex.submit(() -> {
                    ThreadLocalRandom tlr = ThreadLocalRandom.current();
                    try {
                        for (int i = 0; i < reqsPerThread; i++) {
                            sleepUs(80 + tlr.nextLong(60));
                            lk.execute(() -> {
                                long cs0 = System.nanoTime();
                                while (System.nanoTime() - cs0 < 150_000L) {}
                            });
                        }
                    } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
                    finally { latch.countDown(); }
                });
            }
            latch.await(120, TimeUnit.SECONDS);
            long wallMs = (System.nanoTime() - wallT0) / 1_000_000;
            ex.shutdownNow(); ex.awaitTermination(5, TimeUnit.SECONDS);

            long attempts = casAttempts.get(), wins = casWins.get();
            double wastedPerWin = wins > 0 ? (double)(attempts - wins) / wins : 0;
            System.out.printf("nT=%2d wallMs=%6d casAttempts=%8d casWins=%6d wastedCASperWin=%.2f meanBatch=%.2f%n",
                nT, wallMs, attempts, wins, wastedPerWin, lk.meanBatchSize());
        }
    }

    static void sleepUs(long us) throws InterruptedException {
        long deadline = System.nanoTime() + us * 1000L;
        if (us > 1500) Thread.sleep(us / 1000);
        while (System.nanoTime() < deadline) Thread.onSpinWait();
    }

    // Instrumented copy of CombiningLock's core loop with CAS-attempt counters.
    static final class InstrumentedCombiningLock {
        interface Work { void run(); }
        static final class Request {
            final Work work; final Thread thread; volatile boolean done=false; volatile Request next;
            Request(Work w, Thread t){work=w;thread=t;}
        }
        private final AtomicReference<Request> stackTop = new AtomicReference<>();
        private final AtomicBoolean combining = new AtomicBoolean(false);
        private final AtomicLong totalCombined = new AtomicLong(0);
        private final AtomicLong totalPasses = new AtomicLong(0);
        private final AtomicLong casAttempts, casWins;

        InstrumentedCombiningLock(AtomicLong casAttempts, AtomicLong casWins) {
            this.casAttempts = casAttempts; this.casWins = casWins;
        }

        long execute(Work work) throws InterruptedException {
            long t0 = System.nanoTime();
            Request r = new Request(work, Thread.currentThread());
            Request oldTop;
            do { oldTop = stackTop.get(); r.next = oldTop; } while (!stackTop.compareAndSet(oldTop, r));
            long repollNs = CombiningLock.INITIAL_REPOLL_NS;
            while (!r.done) {
                if (Thread.interrupted()) throw new InterruptedException();
                casAttempts.incrementAndGet();
                if (combining.compareAndSet(false, true)) {
                    casWins.incrementAndGet();
                    try { combinePass(); } finally { combining.set(false); }
                } else {
                    java.util.concurrent.locks.LockSupport.parkNanos(repollNs);
                    repollNs = Math.min(repollNs * 2, CombiningLock.MAX_REPOLL_NS);
                }
            }
            return System.nanoTime() - t0;
        }
        private void combinePass() {
            Request batch = stackTop.getAndSet(null);
            if (batch == null) return;
            int n = 0; Request cur = batch;
            while (cur != null) {
                Request next = cur.next;
                cur.work.run(); cur.done = true; n++;
                if (cur.thread != Thread.currentThread()) java.util.concurrent.locks.LockSupport.unpark(cur.thread);
                cur = next;
            }
            totalCombined.addAndGet(n); totalPasses.incrementAndGet();
        }
        double meanBatchSize() { long p = totalPasses.get(); return p==0?0:(double)totalCombined.get()/p; }
    }
}
