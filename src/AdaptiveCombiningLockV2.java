import java.util.concurrent.atomic.*;
import java.util.concurrent.locks.LockSupport;

/**
 * ADAPTIVE (HYBRID) COMBINING LOCK, V2.
 *
 * V1 (AdaptiveCombiningLock.java) added a single-CAS fast path in front
 * of CombiningLock's full combining protocol, and fixed the nT<=8
 * weakness for nT=4 upward -- but introduced a NEW, narrower regression
 * at exactly nT=2 (measured: ratio 1.274 vs FairMutex, SIGNIFICANT loss,
 * root-caused in NT2Diagnostic.java / ADAPTIVE_COMBINING_FINDINGS.md
 * Sec 6). That is a real design flaw, not a tolerable edge case: a
 * mechanism whose fast path "works" at nT=4 has no principled reason to
 * get WORSE at nT=2 (lower contention), so this needed a real fix, not
 * just a disclosure.
 *
 * ROOT CAUSE (confirmed via NT2Diagnostic.java): on ANY fast-path CAS
 * failure, V1 immediately escalates to the full heavyweight protocol --
 * allocate a Request, CAS-push it onto the stack, then park/unpark.
 * At nT=2, two threads alternate quickly enough that ~50% of calls hit
 * this failure, yet the resulting "batch" is essentially always size 1
 * (there's only one other thread, and it's usually already almost done).
 * The heavyweight machinery is paid in full with zero combining benefit
 * to offset it. A stripped variant that just spin-retries the CAS
 * instead was measurably FASTER at nT=2 in the diagnostic (0.192ms vs
 * 0.317ms mean).
 *
 * FIX: on fast-path CAS failure, do a bounded, cheap SPIN-RETRY of the
 * same raw CAS (no allocation, no stack push) before escalating to the
 * heavyweight path. This is the standard "spin-then-queue" pattern used
 * in real production locks (e.g. the JDK's own AbstractQueuedSynchronizer
 * spins briefly before parking; most futex-based mutexes spin on the
 * order of tens of iterations before falling back to a syscall). The
 * spin bound here (SPIN_LIMIT = 40 CAS attempts, each preceded by
 * Thread.onSpinWait()) is a fixed, small constant of the kind used
 * throughout that literature -- it was NOT reverse-engineered from this
 * benchmark's specific timings (CS_WORK_US=150us, think-time 80-140us).
 * To guard against exactly that overfitting risk, this version is
 * benchmarked in AdaptiveCombiningBenchmarkV2.java against BOTH the
 * original 150us critical section AND deliberately different critical-
 * section durations (30us and 500us) -- if the fix only works at the
 * one duration this investigation happened to pick, that will show up
 * as a regression at the other durations, and will be reported as such
 * rather than hidden.
 *
 * Everything else -- the busy-flag exclusion invariant, the straggler
 * drain, the exponential-backoff liveness net on the heavyweight path --
 * is UNCHANGED from V1 and inherits the same correctness argument
 * (see AdaptiveCombiningLock.java's javadoc). The spin-retry loop only
 * ever attempts the exact same CAS as the original immediate fast path;
 * if it succeeds, the thread proceeds exactly as the V1 fast path would
 * have (including the straggler drain before release). If it fails
 * SPIN_LIMIT times, behavior falls back to the existing, already-
 * verified V1 heavyweight path unchanged. No new state, no new gate --
 * this is a change in WHEN escalation happens, not a change in what
 * "having the lock" means.
 */
public class AdaptiveCombiningLockV2 {

    public interface Work { void run(); }

    private static final class Request {
        final Work work;
        final Thread thread;
        volatile boolean done = false;
        volatile Request next;
        Request(Work w, Thread t) { work = w; thread = t; }
    }

    static final long INITIAL_REPOLL_NS = 2_000_000L;   // 2ms
    static final long MAX_REPOLL_NS     = 20_000_000L;  // 20ms

    // Bounded spin-retry before escalating to the heavyweight path. A
    // small, fixed constant in the same order of magnitude as spin
    // bounds used in production spin-then-block locks (JDK AQS, futex-
    // based mutexes) -- not tuned to this benchmark's specific timings.
    static final int SPIN_LIMIT = 40;

    private final AtomicBoolean busy = new AtomicBoolean(false);
    private final AtomicReference<Request> stackTop = new AtomicReference<>();

    // Diagnostics only, not used for correctness.
    private final AtomicLong fastPathHits = new AtomicLong(0);
    private final AtomicLong spinRetryHits = new AtomicLong(0);
    private final AtomicLong combinerHits = new AtomicLong(0);
    private final AtomicLong totalRequestsCombined = new AtomicLong(0);
    private final AtomicLong totalCombinerPasses = new AtomicLong(0);
    private final AtomicLong maxBatchSizeSeen = new AtomicLong(0);

    public long execute(Work work) throws InterruptedException {
        long t0 = System.nanoTime();

        // ---- Fast path: immediate CAS. ----
        if (busy.compareAndSet(false, true)) {
            fastPathHits.incrementAndGet();
            runFastPathAndRelease(work);
            return System.nanoTime() - t0;
        }

        // ---- NEW: bounded spin-retry on the same cheap CAS before
        //      paying for the heavyweight protocol. ----
        for (int i = 0; i < SPIN_LIMIT; i++) {
            Thread.onSpinWait();
            if (busy.compareAndSet(false, true)) {
                spinRetryHits.incrementAndGet();
                runFastPathAndRelease(work);
                return System.nanoTime() - t0;
            }
        }

        // ---- Contended path: publish + combiner election, same
        //      protocol (and same liveness fix) as CombiningLock/V1. ----
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
        try {
            work.run();
        } finally {
            drainStragglers();
            busy.set(false);
        }
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
            cur.work.run();
            cur.done = true;
            n++;
            if (cur.thread != Thread.currentThread()) {
                LockSupport.unpark(cur.thread);
            }
            cur = next;
        }
        totalRequestsCombined.addAndGet(n);
        totalCombinerPasses.incrementAndGet();
        final int nFinal = n;
        maxBatchSizeSeen.updateAndGet(m -> Math.max(m, nFinal));
    }

    public long fastPathHits()   { return fastPathHits.get(); }
    public long spinRetryHits()  { return spinRetryHits.get(); }
    public long combinerHits()   { return combinerHits.get(); }
    public long totalRequestsCombined() { return totalRequestsCombined.get(); }
    public long totalCombinerPasses()   { return totalCombinerPasses.get(); }
    public long maxBatchSizeSeen()      { return maxBatchSizeSeen.get(); }
    public double meanBatchSize() {
        long passes = totalCombinerPasses.get();
        return passes == 0 ? 0 : (double) totalRequestsCombined.get() / passes;
    }
}
