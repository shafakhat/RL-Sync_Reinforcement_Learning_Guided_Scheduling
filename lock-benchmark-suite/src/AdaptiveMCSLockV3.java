import java.util.concurrent.atomic.*;

/**
 * ADAPTIVE MCS LOCK: same CS-duration-aware spin-sizing idea as
 * AdaptiveCombiningLockV3.java, applied to a STRUCTURALLY DIFFERENT
 * base lock family (MCS queue locking, not flat combining), to test
 * whether the idea's effect generalizes. Built at the user's explicit
 * request to test the technique "along with a different lock family."
 *
 * DESIGN NOTE / SELF-CAUGHT BUG (disclosed for transparency): an
 * earlier draft of this class used a SEPARATE `owner` AtomicBoolean
 * gate in front of the MCS queue, with the queue-head thread
 * re-acquiring `owner` via CAS after being unblocked, guarded only by
 * an `assert` that the CAS "must" succeed. That reasoning was WRONG: a
 * brand-new fast-path thread's CAS on `owner` could race in and win
 * between the predecessor releasing `owner` and the real queue head
 * re-acquiring it (these are two separate memory writes in the
 * predecessor's unlock sequence, not one atomic step) -- and since
 * Java assertions are disabled by default, that failure would have
 * been silently ignored, allowing TWO threads to run their critical
 * sections concurrently. This would have been a genuine mutual-
 * exclusion violation, caught during design review before ANY
 * benchmark or safety harness was run against it (i.e. before the
 * bug could contaminate a single reported number).
 *
 * FIX (this version): there is only ONE exclusion mechanism, `tail`,
 * exactly as in plain MCSLock.java -- no second flag. The "fast path"
 * is simply MCS's own inherent uncontended case (tail transitions
 * null -> node with no predecessor) attempted repeatedly via a bounded,
 * duration-aware CAS-retry loop BEFORE falling back to the
 * unconditional queue-join (tail.getAndSet) that plain MCS always does.
 * If the retry loop never observes tail==null, the thread falls
 * through to the ordinary, unmodified MCS queue-join and spin-wait --
 * bit-for-bit the same code path as MCSLock.java. There is no new
 * state, no new gate, and no separate ownership flag to race:
 * correctness is inherited directly from plain MCS's own proof
 * (single CAS on a single tail reference is the sole arbiter of both
 * "am I the head" and "who is the head"), verified with the same
 * safety-harness pattern used throughout this investigation.
 */
public class AdaptiveMCSLockV3 {
    static final class QNode {
        volatile boolean locked = false;
        volatile QNode next = null;
    }

    public interface Work { void run(); }

    static final long MIN_SPIN_NS = 500;
    static final long MAX_SPIN_NS = 50_000;
    static final long DEFAULT_ESTIMATE_NS = 2_000;

    private final AtomicReference<QNode> tail = new AtomicReference<>(null);
    private final ThreadLocal<QNode> myNode = ThreadLocal.withInitial(QNode::new);

    private final AtomicLong csEstimateNs = new AtomicLong(DEFAULT_ESTIMATE_NS);
    private final AtomicInteger spinnerCount = new AtomicInteger(0);
    private static final int AVAILABLE_CORES = Math.max(1, Runtime.getRuntime().availableProcessors());

    private final AtomicLong fastPathHits = new AtomicLong(0);
    private final AtomicLong spinRetryHits = new AtomicLong(0);
    private final AtomicLong spinGateSkipped = new AtomicLong(0);
    private final AtomicLong queueHits = new AtomicLong(0);

    public long execute(Work work) throws InterruptedException {
        long t0 = System.nanoTime();
        QNode node = myNode.get();
        node.locked = false; // will be set true only if we actually queue
        node.next = null;

        // ---- Immediate attempt: the same "tail was null" case plain
        //      MCS already treats as its uncontended fast path. ----
        if (tail.compareAndSet(null, node)) {
            fastPathHits.incrementAndGet();
            runAndRelease(work, node);
            return System.nanoTime() - t0;
        }

        // ---- Bounded, duration-aware spin-retry of the SAME CAS,
        //      before paying for an unconditional queue-join. This is
        //      the only change relative to plain MCSLock: repeatedly
        //      re-check "is tail null right now" for a budget derived
        //      from the live-measured CS duration, instead of joining
        //      the queue on the very first failure. ----
        long estimate = csEstimateNs.get();
        long spinBudgetNs = Math.max(MIN_SPIN_NS, Math.min(estimate, MAX_SPIN_NS));

        if (spinnerCount.incrementAndGet() <= AVAILABLE_CORES) {
            try {
                long spinDeadline = System.nanoTime() + spinBudgetNs;
                while (System.nanoTime() < spinDeadline) {
                    Thread.onSpinWait();
                    if (tail.compareAndSet(null, node)) {
                        spinRetryHits.incrementAndGet();
                        runAndRelease(work, node);
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

        // ---- Fall back to plain, unmodified MCS queueing. ----
        queueHits.incrementAndGet();
        node.locked = true;
        node.next = null;
        QNode pred = tail.getAndSet(node);
        if (pred != null) {
            pred.next = node;
            // Same disclosed fix as MCSLock.java: periodic yield during
            // the local spin-wait to avoid near-livelock under heavy
            // core oversubscription. Applied identically here so the
            // V1(plain MCS)-vs-V3(adaptive MCS) comparison isn't
            // confounded by one arm having the fix and the other not.
            int spins = 0;
            while (node.locked) {
                if (Thread.interrupted()) throw new InterruptedException();
                if (++spins % 64 == 0) {
                    Thread.yield();
                } else {
                    Thread.onSpinWait();
                }
            }
        }
        runAndRelease(work, node);
        return System.nanoTime() - t0;
    }

    private void runAndRelease(Work work, QNode node) {
        long cs0 = System.nanoTime();
        try {
            work.run();
        } finally {
            updateEstimate(System.nanoTime() - cs0);
            if (node.next == null) {
                if (tail.compareAndSet(node, null)) {
                    return;
                }
                while (node.next == null) {
                    Thread.onSpinWait();
                }
            }
            node.next.locked = false;
        }
    }

    private void updateEstimate(long sampleNs) {
        long old, updated;
        do {
            old = csEstimateNs.get();
            long clamped = Math.min(sampleNs, old * 8 + 10_000_000L);
            updated = old + (clamped - old) / 8;
            if (updated < 1) updated = 1;
        } while (!csEstimateNs.compareAndSet(old, updated));
    }

    public long fastPathHits()    { return fastPathHits.get(); }
    public long spinRetryHits()   { return spinRetryHits.get(); }
    public long spinGateSkipped() { return spinGateSkipped.get(); }
    public long queueHits()       { return queueHits.get(); }
    public long csEstimateNs()    { return csEstimateNs.get(); }
}
