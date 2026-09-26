import java.util.concurrent.atomic.*;
import java.util.concurrent.locks.LockSupport;

/**
 * ADAPTIVE (HYBRID) COMBINING LOCK.
 *
 * Motivation: CombiningLock.java is a real, verified, statistically
 * significant improvement over FairMutex from nT=24 through nT=96
 * (CombiningBenchmark.java), but LOSES to FairMutex at nT<=8 (up to 53%
 * worse at nT=4). Root cause: every single call -- even a completely
 * uncontended one with no other thread anywhere near the lock -- pays
 * THREE atomic read-modify-write operations (push request onto the
 * lock-free stack, CAS to become combiner, getAndSet to drain the
 * "batch" of exactly one item). A plain AQS-based lock's uncontended
 * fast path is roughly one CAS. That fixed 3x tax is invisible once
 * batches are large (nT>=24, tax amortized over dozens of requests) but
 * dominates when batches are usually size 1 (low thread count).
 *
 * FIX: add a genuine fast path. Before doing ANY stack/combiner
 * bookkeeping, try a single direct CAS on a `busy` flag. If it
 * succeeds, run the work directly -- no Request object, no stack push,
 * no combiner election. Only fall back to the full combining protocol
 * (push + elect + batch-drain) when the fast-path CAS fails, i.e. when
 * the lock is actually contended right now.
 *
 * CORRECTNESS ARGUMENT (why this doesn't reintroduce the mutual
 * exclusion or liveness problems already found and fixed elsewhere in
 * this investigation):
 *   - `busy` is the SOLE exclusion gate for BOTH the fast path and the
 *     combiner path: a thread only ever executes ITS OWN work (fast
 *     path) or SOMEONE ELSE'S published work (combiner path) while it
 *     holds `busy == true` via a winning CAS. Since CAS is atomic, at
 *     most one thread holds `busy` at a time, in either mode -- there is
 *     no third state where both a fast-path runner and a combiner could
 *     be inside a critical section simultaneously.
 *   - LATECOMER SAFETY: a fast-path runner might finish its own work
 *     just as another thread's request lands on the stack. Before
 *     releasing `busy`, the fast-path runner drains and executes ANY
 *     requests that arrived on the stack during its run (a "straggler
 *     drain", same mechanics as a normal combine pass) so nobody is
 *     stranded waiting for a combiner that already came and went.
 *   - LIVENESS: if a straggler arrives on the stack in the tiny window
 *     AFTER the runner's final drain check but BEFORE it sets
 *     busy=false, that straggler simply waits for the NEXT thread to
 *     win the busy-CAS (exactly the same eventual-service guarantee,
 *     backed by the same exponential-backoff safety net, as the
 *     original CombiningLock -- verified there via CombiningLockSafety
 *     and PollingOverheadCheck).
 */
public class AdaptiveCombiningLock {

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

    private final AtomicBoolean busy = new AtomicBoolean(false);
    private final AtomicReference<Request> stackTop = new AtomicReference<>();

    // Diagnostics only, not used for correctness.
    private final AtomicLong fastPathHits = new AtomicLong(0);
    private final AtomicLong combinerHits = new AtomicLong(0);
    private final AtomicLong totalRequestsCombined = new AtomicLong(0);
    private final AtomicLong totalCombinerPasses = new AtomicLong(0);
    private final AtomicLong maxBatchSizeSeen = new AtomicLong(0);
    private final AtomicLong stragglersHandledByFastPath = new AtomicLong(0);

    public long execute(Work work) throws InterruptedException {
        long t0 = System.nanoTime();

        // ---- Fast path: try direct ownership with a single CAS. ----
        if (busy.compareAndSet(false, true)) {
            fastPathHits.incrementAndGet();
            try {
                work.run();
            } finally {
                // Straggler safety: handle anything that queued while we
                // were running, before giving up `busy`.
                drainStragglers();
                busy.set(false);
            }
            return System.nanoTime() - t0;
        }

        // ---- Contended path: publish + combiner election, same
        //      protocol (and same liveness fix) as CombiningLock. ----
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

    /** Drains and executes any requests currently on the stack. Called
     *  while `busy == true` (either from the fast path's post-work
     *  straggler check, or from the contended path's combiner election).
     *  Loops until the stack is observed empty, since new stragglers can
     *  arrive mid-drain. */
    private void drainStragglers() {
        while (true) {
            Request batch = stackTop.getAndSet(null);
            if (batch == null) return;
            runBatch(batch);
            stragglersHandledByFastPath.incrementAndGet();
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
    public long combinerHits()   { return combinerHits.get(); }
    public long totalRequestsCombined() { return totalRequestsCombined.get(); }
    public long totalCombinerPasses()   { return totalCombinerPasses.get(); }
    public long maxBatchSizeSeen()      { return maxBatchSizeSeen.get(); }
    public double meanBatchSize() {
        long passes = totalCombinerPasses.get();
        return passes == 0 ? 0 : (double) totalRequestsCombined.get() / passes;
    }
}
