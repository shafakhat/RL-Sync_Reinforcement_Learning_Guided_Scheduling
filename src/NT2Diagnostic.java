import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

/**
 * Root-cause diagnostic for the nT=2 anomaly found in
 * AdaptiveCombiningBenchmark: at nT=2, fastPathFrac=100.0% (i.e.
 * essentially every call succeeds on the single-CAS fast path, with
 * only ~1 straggler-drain event per 300-call run) yet the lock still
 * shows a SIGNIFICANT, reproducible loss vs FairMutex (ratio~1.27,
 * Cohen's d~5.8). Per the standing rule that surprising results must be
 * root-caused before being reported, this isolates candidate causes:
 *
 *  (a) Is the "always-run drainStragglers()" call on the fast path
 *      (an unconditional extra atomic getAndSet on stackTop, even when
 *      it will find null) adding real per-call cost once a SECOND core
 *      is touching the same fields (cache-line bounce), even though
 *      the *outcome* is "no straggler found"?
 *  (b) Is the cost instead concentrated in the rare actual-contention
 *      events (heap-allocate Request, CAS stack push, failed busy-CAS,
 *      park/unpark), i.e. a HEAVY tail that drags the mean up despite
 *      being rare?
 *
 * Method: instrument three variants at nT=2 with think-time matching
 * the real benchmark:
 *   1. AdaptiveCombiningLock as-is (baseline, reproduces the anomaly)
 *   2. A stripped variant that skips drainStragglers() entirely on the
 *      fast path (busy.CAS -> run -> busy.set(false), no stack check
 *      at all) -- UNSAFE (can strand stragglers) but fine for a
 *      microbenchmark isolating cost, not correctness.
 *   3. Single-thread control (nT=1) for both, as a zero-cross-core
 *      baseline.
 */
public class NT2Diagnostic {
    static final int REQS = 3000;
    static final int CS_WORK_US = 150;

    public static void main(String[] args) throws Exception {
        System.out.println("=== Variant 1: AdaptiveCombiningLock as-is, nT=2 ===");
        runReal(2);
        System.out.println("=== Variant 1: AdaptiveCombiningLock as-is, nT=1 ===");
        runReal(1);

        System.out.println("=== Variant 2: fast path WITHOUT drainStragglers(), nT=2 ===");
        runStripped(2);
        System.out.println("=== Variant 2: fast path WITHOUT drainStragglers(), nT=1 ===");
        runStripped(1);
    }

    static void runReal(int nT) throws Exception {
        AdaptiveCombiningLock lk = new AdaptiveCombiningLock();
        List<Long> waits = Collections.synchronizedList(new ArrayList<>());
        CountDownLatch latch = new CountDownLatch(nT);
        ExecutorService ex = Executors.newFixedThreadPool(nT);
        for (int t = 0; t < nT; t++) {
            ex.submit(() -> {
                ThreadLocalRandom tlr = ThreadLocalRandom.current();
                try {
                    for (int i = 0; i < REQS; i++) {
                        sleepUs(80 + tlr.nextLong(60));
                        long t0 = System.nanoTime();
                        lk.execute(() -> busyWait(CS_WORK_US));
                        waits.add(System.nanoTime() - t0);
                    }
                } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
                finally { latch.countDown(); }
            });
        }
        latch.await(120, TimeUnit.SECONDS);
        ex.shutdownNow(); ex.awaitTermination(5, TimeUnit.SECONDS);
        report(waits, lk.fastPathHits(), lk.combinerHits());
    }

    // Stripped copy: identical fast-path CAS but NO drainStragglers() call.
    static final AtomicBoolean strippedBusy = new AtomicBoolean(false);
    static final AtomicLong strippedFastHits = new AtomicLong(0);
    static final AtomicLong strippedContended = new AtomicLong(0);

    static void runStripped(int nT) throws Exception {
        strippedBusy.set(false);
        strippedFastHits.set(0);
        strippedContended.set(0);
        List<Long> waits = Collections.synchronizedList(new ArrayList<>());
        CountDownLatch latch = new CountDownLatch(nT);
        ExecutorService ex = Executors.newFixedThreadPool(nT);
        for (int t = 0; t < nT; t++) {
            ex.submit(() -> {
                ThreadLocalRandom tlr = ThreadLocalRandom.current();
                try {
                    for (int i = 0; i < REQS; i++) {
                        sleepUs(80 + tlr.nextLong(60));
                        long t0 = System.nanoTime();
                        if (strippedBusy.compareAndSet(false, true)) {
                            strippedFastHits.incrementAndGet();
                            try { busyWait(CS_WORK_US); }
                            finally { strippedBusy.set(false); }
                        } else {
                            strippedContended.incrementAndGet();
                            // crude spin-then-retry, no real queue -- fine
                            // for a cost-isolation microbenchmark only.
                            while (!strippedBusy.compareAndSet(false, true)) {
                                Thread.onSpinWait();
                            }
                            try { busyWait(CS_WORK_US); }
                            finally { strippedBusy.set(false); }
                        }
                        waits.add(System.nanoTime() - t0);
                    }
                } catch (Exception e) { throw new RuntimeException(e); }
                finally { latch.countDown(); }
            });
        }
        latch.await(120, TimeUnit.SECONDS);
        ex.shutdownNow(); ex.awaitTermination(5, TimeUnit.SECONDS);
        System.out.printf("  fastHits=%d contended=%d%n", strippedFastHits.get(), strippedContended.get());
        report(waits, strippedFastHits.get(), strippedContended.get());
    }

    static void busyWait(int us) {
        long deadline = System.nanoTime() + us * 1000L;
        while (System.nanoTime() < deadline) {}
    }
    static void sleepUs(long us) throws InterruptedException {
        long deadline = System.nanoTime() + us * 1000L;
        while (System.nanoTime() < deadline) Thread.onSpinWait();
    }
    static void report(List<Long> waits, long fp, long ch) {
        double mean = waits.stream().mapToLong(Long::longValue).average().orElse(0) / 1e6;
        List<Long> sorted = new ArrayList<>(waits);
        Collections.sort(sorted);
        double p50 = sorted.get(sorted.size()/2) / 1e6;
        double p99 = sorted.get((int)(sorted.size()*0.99)) / 1e6;
        System.out.printf("  n=%d mean=%.4fms p50=%.4fms p99=%.4fms fastHits=%d otherHits=%d%n%n",
            waits.size(), mean, p50, p99, fp, ch);
    }
}
