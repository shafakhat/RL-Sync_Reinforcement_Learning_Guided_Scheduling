import java.util.concurrent.atomic.*;

/**
 * Baseline MCS queue lock (Mellor-Crummey & Scott, 1991) -- a
 * structurally DIFFERENT lock family from flat-combining: instead of
 * one thread executing a batch of others' work, each thread executes
 * its OWN critical section, but waits by SPINNING ON A LOCAL (per-
 * thread) memory location rather than a shared one, which is the
 * classic fix for the cache-line-bouncing pathology of a naive spin-
 * on-shared-flag lock.
 *
 * This is a plain, unmodified baseline (no fast path, no duration-
 * awareness) used as: (a) a safety-verified reference, and (b) the
 * substrate onto which AdaptiveMCSLockV3 later adds the SAME CS-
 * duration-aware spin-sizing idea tested on the combining-lock family,
 * to check whether that idea's benefit (or lack thereof) is specific
 * to flat-combining or generalizes across lock families.
 */
public class MCSLock {
    static final class QNode {
        volatile boolean locked = false;
        volatile QNode next = null;
    }

    private final AtomicReference<QNode> tail = new AtomicReference<>(null);
    private final ThreadLocal<QNode> myNode = ThreadLocal.withInitial(QNode::new);

    public interface Work { void run(); }

    public long execute(Work work) throws InterruptedException {
        long t0 = System.nanoTime();
        QNode node = myNode.get();
        node.locked = true;
        node.next = null;

        QNode pred = tail.getAndSet(node);
        if (pred != null) {
            pred.next = node;
            // DISCLOSED FIX (found during this investigation, applied
            // identically to AdaptiveMCSLockV3 for a fair comparison):
            // pure onSpinWait()-only spinning caused severe, non-
            // deadlocked but near-livelocked slowdowns on this 2-core
            // sandbox at high oversubscription (e.g. nT=64), because
            // 62 busy-spinning threads compete for 2 cores with no
            // hint to the OS scheduler to prioritize the actual lock
            // holder. A periodic Thread.yield() during the spin -- a
            // standard, well-known MCS refinement, not novel to this
            // investigation -- fixes this by giving the scheduler a
            // chance to run the thread that can actually make progress.
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

        try {
            work.run();
        } finally {
            if (node.next == null) {
                if (tail.compareAndSet(node, null)) {
                    return System.nanoTime() - t0;
                }
                while (node.next == null) {
                    Thread.onSpinWait();
                }
            }
            node.next.locked = false;
        }
        return System.nanoTime() - t0;
    }
}
