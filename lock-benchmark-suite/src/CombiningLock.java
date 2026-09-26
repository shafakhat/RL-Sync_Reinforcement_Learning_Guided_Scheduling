import java.util.concurrent.atomic.*;
import java.util.concurrent.locks.LockSupport;

/**
 * FLAT COMBINING LOCK.
 *
 * This is a DIFFERENT class of mechanism from everything tried so far in
 * this investigation (RL-Sync's backoff/admit choice, and the SJF+aging
 * priority lock). It is not "smarter admission ordering" -- it changes
 * WHO EXECUTES the protected work.
 *
 * MECHANISM (standard flat-combining pattern, Hendler/Incze/Shavit/
 * Tzafrir 2010 -- not invented for this benchmark):
 *   1. A thread wanting to run a critical section publishes a Request
 *      object (containing the work to run) onto a lock-free stack
 *      (single CAS push).
 *   2. It then tries to become the COMBINER via a single CAS on a
 *      boolean flag. Only one thread can hold this role at a time.
 *   3. If it becomes combiner: it atomically grabs the ENTIRE current
 *      stack of pending requests (one CAS, `getAndSet(null)`), and
 *      executes each one's work function itself, IN SEQUENCE, without
 *      ever releasing the combiner role in between. When done, it marks
 *      each request `done` and wakes (unparks) exactly that request's
 *      original thread, then releases the combiner role.
 *   4. If it does NOT become combiner: it parks, waiting for whichever
 *      thread IS the combiner to execute its request and unpark it.
 *
 * WHY THIS CAN HAVE A REAL, NON-HARDCODED LEVER (distinct from every
 * earlier mechanism tried):
 *   Every lock tested so far -- FairMutex, CLH, the SJF+aging lock, and
 *   RL-Sync itself -- pays ONE full acquire/handoff/wake cycle PER
 *   THREAD PER CRITICAL SECTION. Under contention with short critical
 *   sections, that per-thread scheduling overhead (not the protected
 *   work itself) dominates total wait time. Flat combining collapses N
 *   queued requests into ONE lock-acquisition-equivalent (the CAS to
 *   become combiner) plus N *sequential, handoff-free* executions --
 *   there is no re-acquire, no context switch, no wakeup between
 *   batched items. The busier the system, the bigger the batches, the
 *   more per-thread overhead gets amortized. This predicts the OPPOSITE
 *   scaling behavior from the SJF+aging lock (whose advantage shrank
 *   under saturation): flat combining's advantage should GROW with
 *   contention, not fade.
 *
 * MUTUAL EXCLUSION: trivially preserved by construction -- work.run()
 * is ONLY ever called by whichever single thread currently holds the
 * `combining` CAS flag, and that flag is held by exactly one thread at
 * a time. Non-combiner threads never execute anyone's work, including
 * their own; they only park and wait to be signaled. There is no
 * separate correctness argument needed beyond "only the CAS-flag holder
 * calls work.run(), and CAS is atomic."
 */
public class CombiningLock {

    public interface Work { void run(); }

    static final class Request {
        final Work work;
        final Thread thread;
        volatile boolean done = false;
        volatile Request next; // Treiber-stack linkage
        long waitStartNs;
        Request(Work w, Thread t) { work = w; thread = t; }
    }

    private final AtomicReference<Request> stackTop = new AtomicReference<>();
    private final AtomicBoolean combining = new AtomicBoolean(false);

    // Diagnostics (not used for correctness -- purely observational)
    private final AtomicLong totalRequestsCombined = new AtomicLong(0);
    private final AtomicLong totalCombinerPasses = new AtomicLong(0);
    private final AtomicLong maxBatchSizeSeen = new AtomicLong(0);

    // Bounds for the exponential-backoff re-poll interval used as a
    // liveness safety net (see execute() javadoc below). Starting high
    // and growing avoids the thrashing failure mode found in
    // PollingOverheadCheck.java: a flat, aggressively short retry
    // interval (e.g. 50us) caused wasted combiner-CAS attempts per
    // successful win to explode super-linearly with thread count (32 at
    // nT=8 -> 4221 at nT=64), because nearly every non-combiner thread
    // was waking up and burning CPU competing for the same cores the
    // actual combiner needed to make progress -- directly causing the
    // measured mean/p99 regression at nT>=48 in CombiningBenchmark.java.
    static final long INITIAL_REPOLL_NS = 2_000_000L;   // 2ms
    static final long MAX_REPOLL_NS     = 20_000_000L;  // 20ms

    /**
     * Executes `work` under mutual exclusion with every other call to
     * execute() on this lock, via the flat-combining protocol above.
     * Returns the queueing+combining wait in nanoseconds (time from
     * calling this method to the work having actually run).
     *
     * LIVENESS NOTE (bug found and fixed during development, before any
     * benchmark was run): a thread whose request is published AFTER the
     * current combiner has already snapshotted the stack (getAndSet)
     * will fail to become combiner (CAS loses) and must wait to be
     * served by some FUTURE combine pass. If it parks with no timeout
     * and only relies on a targeted unpark() from whoever eventually
     * processes its request, there is a genuine risk that no thread is
     * left attempting to become the next combiner at all -- a real
     * liveness bug (indefinite hang), not a mutual-exclusion violation.
     *
     * First fix attempt (flat 50us re-poll) eliminated the hang but
     * introduced a NEW, measured performance bug: wasted-CAS-per-win
     * grew super-linearly with thread count (PollingOverheadCheck.java),
     * causing net slowdowns at nT>=48 despite larger batch sizes. Fixed
     * properly here with an EXPONENTIAL-BACKOFF re-poll: a waiter's
     * fallback wake-and-retry interval starts at INITIAL_REPOLL_NS and
     * doubles (capped at MAX_REPOLL_NS) on every unsuccessful retry,
     * resetting to the initial value once it is finally served. This
     * preserves the liveness guarantee (every request is eventually
     * retried, so no permanent hang is possible) while making the
     * common case -- being served promptly via a targeted unpark from
     * the active combiner -- the ONLY thing that matters for latency;
     * the fallback poll essentially never fires under normal operation
     * and only protects against the rare missed-wakeup race.
     */
    public long execute(Work work) throws InterruptedException {
        long t0 = System.nanoTime();
        Request r = new Request(work, Thread.currentThread());
        r.waitStartNs = t0;

        // Publish our request (lock-free push).
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
            if (combining.compareAndSet(false, true)) {
                try {
                    combinePass();
                } finally {
                    combining.set(false);
                }
                // Our own request was necessarily included in this pass
                // (we published before attempting to combine), so r.done
                // is now true and the loop will exit.
            } else {
                // Someone else is combining. Wait to be served via a
                // targeted unpark() -- this is the fast, common-case
                // path (LockSupport.unpark() on an unparked/not-yet-
                // parked thread is remembered as a permit, so no race).
                // The bounded timeout here is ONLY a liveness safety
                // net for the rare missed-wakeup race, not a polling
                // loop: it backs off exponentially so it essentially
                // never fires under normal load.
                LockSupport.parkNanos(repollNs);
                repollNs = Math.min(repollNs * 2, MAX_REPOLL_NS);
            }
        }
        return System.nanoTime() - t0;
    }


    private void combinePass() {
        Request batch = stackTop.getAndSet(null);
        if (batch == null) return; // nothing to do (shouldn't happen: our own request is always present)

        int n = 0;
        Request cur = batch;
        while (cur != null) {
            Request next = cur.next; // save before running (work doesn't touch `next`, but be safe)
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

    public long totalRequestsCombined() { return totalRequestsCombined.get(); }
    public long totalCombinerPasses() { return totalCombinerPasses.get(); }
    public long maxBatchSizeSeen() { return maxBatchSizeSeen.get(); }
    public double meanBatchSize() {
        long passes = totalCombinerPasses.get();
        return passes == 0 ? 0 : (double) totalRequestsCombined.get() / passes;
    }
}
