import java.util.concurrent.atomic.*;
import java.util.concurrent.locks.LockSupport;

/**
 * ADAPTIVE (HYBRID) COMBINING LOCK, V3: CS-DURATION-AWARE SPIN SIZING.
 *
 * BACKGROUND: V2 added a bounded spin-retry (fixed SPIN_LIMIT=40
 * iterations) before escalating to the heavyweight publish+combine
 * path, to fix a regression V1 had at nT=2. Verified result: V2's fix
 * only worked when the critical section was short (~1-30us); at
 * realistic CS durations of ~100-500us it did essentially nothing
 * (spinRetryHits ~0), because a FIXED iteration count is nowhere near
 * long enough to outlast a 100-500us critical section, so every call
 * still fell through to the expensive path regardless.
 *
 * THIS IS A NAMED, OPEN GAP IN THE PUBLISHED LITERATURE, not a novel
 * observation of this investigation's own -- the "Mutable Locks"
 * paper (De Sensi/Pellegrini et al., arXiv:1906.00490) states directly:
 * "one of the main limitations of our approach is not considering the
 * length of the critical section while sizing the spinning window."
 * That is precisely the mechanism gap V2 hit. V3 is a direct, disclosed
 * attempt to close that specific, citable gap -- on a DIFFERENT lock
 * family (adaptive flat-combining) than the spin/sleep hybrid the
 * Mutable Locks paper evaluated -- rather than a claim of a brand new
 * mechanism from scratch.
 *
 * MECHANISM: track a live, running estimate of ACTUAL critical-section
 * duration (exponential moving average, alpha=1/8, fed by every
 * completed work.run() regardless of which path served it -- fast
 * path, spin-retry, or combiner-drained batch member). On a fast-path
 * CAS failure, spin for up to that live estimate (clamped to
 * [MIN_SPIN_NS, MAX_SPIN_NS]) instead of a fixed iteration count. If
 * the lock frees up within that time, grab it directly -- no
 * allocation, no queue, no park/unpark. If not, fall back to the exact
 * same heavyweight protocol as V1/V2, unchanged.
 *
 * SECOND MECHANISM (also borrowed from established adaptive-mutex
 * practice, not invented here): gate spinning by a live spinnerCount
 * vs Runtime.getAvailableProcessors(). This mirrors the Solaris/
 * FreeBSD/macOS adaptive-mutex principle of "only spin if there's a
 * spare core to spin on" -- without it, spinning for up to hundreds of
 * microseconds at high thread counts (nT >> core count) could burn CPU
 * that real worker threads need, potentially making high-nT throughput
 * WORSE, not better. This gate is tested explicitly in the benchmark
 * (AdaptiveCombiningV3Benchmark.java) precisely to check for that
 * failure mode rather than assume the gate prevents it.
 *
 * CORRECTNESS: UNCHANGED from V1/V2. `busy` remains the sole exclusion
 * gate for both the fast/spin path and the combiner path; the spin
 * loop only ever retries the exact same CAS the immediate fast path
 * uses, and on giving up falls back to the identical, already-verified
 * heavyweight protocol. The duration estimate and spinner-count gate
 * are pure liveness/performance heuristics -- they affect WHEN a
 * thread gives up spinning, never WHAT holding the lock means. This
 * class is expected to pass the same safety harness pattern as V1/V2
 * (AdaptiveCombiningLockV3Safety.java) for exactly that reason.
 */
public class AdaptiveCombiningLockV3 {

    public interface Work { void run(); }

    private static final class Request {
        final Work work;
        final Thread thread;
        volatile boolean done = false;
        volatile Request next;
        Request(Work w, Thread t) { work = w; thread = t; }
    }

    static final long INITIAL_REPOLL_NS = 2_000_000L;
    static final long MAX_REPOLL_NS     = 20_000_000L;

    // Bounds on the duration-aware spin budget: a floor so the very
    // first (unseeded) attempts still get a token spin, and a cap so a
    // long-running estimate can't make a single failed attempt spin
    // for an unbounded amount of CPU time.
    static final long MIN_SPIN_NS = 500;        // 0.5us floor
    static final long MAX_SPIN_NS = 50_000;     // 50us cap
    static final long DEFAULT_ESTIMATE_NS = 2_000; // cold-start guess

    private final AtomicBoolean busy = new AtomicBoolean(false);
    private final AtomicReference<Request> stackTop = new AtomicReference<>();

    // Live-measured running estimate of actual critical-section
    // duration on THIS lock, fed by every execution regardless of path.
    private final AtomicLong csEstimateNs = new AtomicLong(DEFAULT_ESTIMATE_NS);

    // Spin-admission gate: don't add a spinner if there's no spare core.
    private final AtomicInteger spinnerCount = new AtomicInteger(0);
    private static final int AVAILABLE_CORES = Math.max(1, Runtime.getRuntime().availableProcessors());

    // Diagnostics only, not used for correctness.
    private final AtomicLong fastPathHits = new AtomicLong(0);
    private final AtomicLong spinRetryHits = new AtomicLong(0);
    private final AtomicLong spinGateSkipped = new AtomicLong(0);
    private final AtomicLong combinerHits = new AtomicLong(0);
    private final AtomicLong totalRequestsCombined = new AtomicLong(0);
    private final AtomicLong totalCombinerPasses = new AtomicLong(0);

    public long execute(Work work) throws InterruptedException {
        long t0 = System.nanoTime();

        if (busy.compareAndSet(false, true)) {
            fastPathHits.incrementAndGet();
            runFastPathAndRelease(work);
            return System.nanoTime() - t0;
        }

        long estimate = csEstimateNs.get();
        long spinBudgetNs = Math.max(MIN_SPIN_NS, Math.min(estimate, MAX_SPIN_NS));

        if (spinnerCount.incrementAndGet() <= AVAILABLE_CORES) {
            try {
                long spinDeadline = System.nanoTime() + spinBudgetNs;
                while (System.nanoTime() < spinDeadline) {
                    Thread.onSpinWait();
                    if (busy.compareAndSet(false, true)) {
                        spinRetryHits.incrementAndGet();
                        runFastPathAndRelease(work);
                        return System.nanoTime() - t0;
                    }
                }
            } finally {
                spinnerCount.decrementAndGet();
            }
        } else {
            spinnerCount.decrementAndGet();
            spinGateSkipped.incrementAndGet();
        }

        // ---- Heavyweight fallback: identical to V1/V2. ----
        Request r = new Request(work, Thread.currentThread());
        Request oldTop;
        do {
            oldTop = stackTop.get();
            r.next = oldTop;
        } while (!stackTop.compareAndSet(oldTop, r));

        long repollNs = INITIAL_REPOLL_NS;
        while (!r.done) {
            if (Thread.interrupted()) {
                throw new InterruptedException();
            }
            if (busy.compareAndSet(false, true)) {
                combinerHits.incrementAndGet();
                try {
                    combinePass();
                } finally {
                    busy.set(false);
                }
            } else {
                LockSupport.parkNanos(repollNs);
                repollNs = Math.min(repollNs * 2, MAX_REPOLL_NS);
            }
        }
        return System.nanoTime() - t0;
    }

    private void runFastPathAndRelease(Work work) {
        long cs0 = System.nanoTime();
        try {
            work.run();
        } finally {
            updateEstimate(System.nanoTime() - cs0);
            drainStragglers();
            busy.set(false);
        }
    }

    private void updateEstimate(long sampleNs) {
        long old, updated;
        do {
            old = csEstimateNs.get();
            // Clamp a single sample's pull on the estimate so one GC
            // pause / scheduling hiccup during a timed section can't
            // blow the running estimate up disproportionately.
            long clamped = Math.min(sampleNs, old * 8 + 10_000_000L);
            updated = old + (clamped - old) / 8; // EMA, alpha = 1/8
            if (updated < 1) updated = 1;
        } while (!csEstimateNs.compareAndSet(old, updated));
    }

    private void drainStragglers() {
        while (true) {
            Request batch = stackTop.getAndSet(null);
            if (batch == null) return;
            runBatch(batch);
        }
    }

    private void combinePass() {
        Request batch = stackTop.getAndSet(null);
        if (batch == null) return;
        runBatch(batch);
    }

    private void runBatch(Request batch) {
        int n = 0;
        Request cur = batch;
        while (cur != null) {
            Request next = cur.next;
            long cs0 = System.nanoTime();
            cur.work.run();
            updateEstimate(System.nanoTime() - cs0);
            cur.done = true;
            n++;
            if (cur.thread != Thread.currentThread()) {
                LockSupport.unpark(cur.thread);
            }
            cur = next;
        }
        totalRequestsCombined.addAndGet(n);
        totalCombinerPasses.incrementAndGet();
    }

    public long fastPathHits()    { return fastPathHits.get(); }
    public long spinRetryHits()   { return spinRetryHits.get(); }
    public long spinGateSkipped() { return spinGateSkipped.get(); }
    public long combinerHits()    { return combinerHits.get(); }
    public long csEstimateNs()    { return csEstimateNs.get(); }
    public double meanBatchSize() {
        long passes = totalCombinerPasses.get();
        return passes == 0 ? 0 : (double) totalRequestsCombined.get() / passes;
    }
}
