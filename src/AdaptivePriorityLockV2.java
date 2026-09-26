import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.concurrent.locks.LockSupport;

/**
 * V2 of the adaptive SJF+aging priority lock. Same admission POLICY as
 * AdaptivePriorityLock (V1): Shortest-Predicted-Job-First with an aging
 * cutoff to bound starvation, and an online per-class EMA duration
 * estimator. What changed is purely the LOW-LEVEL PRIMITIVE:
 *
 *   V1: java "synchronized" monitor + wait()/notifyAll()
 *       -> diagnosed (OverheadIsolation.java) as the actual bottleneck:
 *          notifyAll() wakes every waiter (thundering herd), and they all
 *          re-contend on the monitor just to recheck their turn flag and
 *          go back to sleep -- a JVM-monitor-specific tax, unrelated to
 *          the scheduling policy itself. A bare FIFO monitor lock paid
 *          almost the same tax as the full priority lock (2605ms vs
 *          2403ms), while FairMutex (AQS-based) ran in 1729ms.
 *
 *   V2: a short spinlock (CAS on an AtomicBoolean, held only for the
 *       brief O(log n)/O(n) bookkeeping operations) guards the waiter
 *       set, and actual blocking/wakeup uses LockSupport.park()/
 *       unpark(specific Thread) -- the exact primitive AQS itself uses
 *       internally. Only the ONE thread selected to go next is ever
 *       unparked; no thundering herd.
 *
 * This isolates whether SJF+aging has genuine leverage once the
 * primitive-choice confound (the actual lesson learned from the FairMutex
 * vs RL-Sync semaphore/lock investigation) is removed.
 */
public class AdaptivePriorityLockV2 {

    public enum JobClass { SHORT, LONG }

    static final long MAX_WAIT_AGING_NS = 5_000_000L; // 5 ms

    private static final class Waiter {
        final long ticket;
        final long arrivalNs;
        final double predictedDurationNs;
        final Thread thread;
        volatile boolean turn = false;

        Waiter(long ticket, long arrivalNs, double predictedDurationNs, Thread thread) {
            this.ticket = ticket; this.arrivalNs = arrivalNs;
            this.predictedDurationNs = predictedDurationNs; this.thread = thread;
        }

        double effectivePriority(long now) {
            long waited = now - arrivalNs;
            if (waited > MAX_WAIT_AGING_NS) return -1e18 + ticket;
            return predictedDurationNs;
        }
    }

    static final class Estimator {
        volatile double emaNs;
        static final double ALPHA = 0.15;
        Estimator(double initNs) { emaNs = initNs; }
        void update(long observedNs) { emaNs = ALPHA * observedNs + (1 - ALPHA) * emaNs; }
        double estimate() { return emaNs; }
    }

    // Short-held spinlock protecting `held` and `waiting` ONLY -- no
    // thread ever blocks while holding this, so critical sections here
    // are microseconds, same design constraint AQS's own internal queue
    // manipulation follows.
    private final AtomicBoolean bookkeepingLock = new AtomicBoolean(false);
    private boolean held = false;
    private final ArrayList<Waiter> waiting = new ArrayList<>();
    private final AtomicLong seqGen = new AtomicLong(0);

    private final ConcurrentHashMap<JobClass, Estimator> estimators = new ConcurrentHashMap<>();

    public AdaptivePriorityLockV2() {
        for (JobClass jc : JobClass.values()) estimators.put(jc, new Estimator(100_000));
    }

    private void lockBookkeeping() {
        while (!bookkeepingLock.compareAndSet(false, true)) Thread.onSpinWait();
    }
    private void unlockBookkeeping() { bookkeepingLock.set(false); }

    public long acquire(JobClass jobClass) throws InterruptedException {
        long t0 = System.nanoTime();
        Waiter w = null;
        lockBookkeeping();
        try {
            if (!held && waiting.isEmpty()) {
                held = true;
                return System.nanoTime() - t0;
            }
            double predicted = estimators.get(jobClass).estimate();
            w = new Waiter(seqGen.getAndIncrement(), t0, predicted, Thread.currentThread());
            waiting.add(w);
        } finally {
            unlockBookkeeping();
        }
        // Park until our specific turn flag is set by some release().
        while (!w.turn) {
            LockSupport.park();
            if (Thread.interrupted()) {
                // Best-effort cleanup: remove ourselves if still queued.
                lockBookkeeping();
                try { waiting.remove(w); } finally { unlockBookkeeping(); }
                throw new InterruptedException();
            }
        }
        return System.nanoTime() - t0;
    }

    public void release(JobClass jobClass, long durationNs) {
        Waiter winner = null;
        lockBookkeeping();
        try {
            estimators.get(jobClass).update(durationNs);
            if (waiting.isEmpty()) {
                held = false;
                return;
            }
            long now = System.nanoTime();
            double bestPr = Double.POSITIVE_INFINITY;
            int bestIdx = -1;
            for (int i = 0; i < waiting.size(); i++) {
                double pr = waiting.get(i).effectivePriority(now);
                if (pr < bestPr) { bestPr = pr; bestIdx = i; }
            }
            winner = waiting.remove(bestIdx);
            // held stays true -- ownership transferred directly to winner.
        } finally {
            unlockBookkeeping();
        }
        winner.turn = true;
        LockSupport.unpark(winner.thread); // wake exactly one thread
    }

    public int queueLength() {
        lockBookkeeping();
        try { return waiting.size(); } finally { unlockBookkeeping(); }
    }
}
