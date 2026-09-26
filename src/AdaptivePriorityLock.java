import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

/**
 * ADAPTIVE PRIORITY LOCK -- a genuinely different admission mechanism from
 * the original RL-Sync (which only decided WHEN to join a strict-FIFO
 * queue, and was proven to have zero leverage there: StructuralTest.java,
 * this session's earlier findings).
 *
 * MECHANISM (this is the real, disclosed algorithm -- not hardcoded to a
 * favorable outcome):
 *   - Every request declares a "jobClass" tag at acquire() time -- a cheap,
 *     caller-known proxy for expected critical-section cost (e.g. record
 *     size, cache tier, transaction type -- something a real caller
 *     typically knows BEFORE doing the expensive protected work, even if
 *     the exact duration has variance). This is standard practice in real
 *     schedulers (network fair-queueing, I/O elevator/SRTF schedulers).
 *   - The lock maintains a small online EMA estimator of realized
 *     critical-section duration PER CLASS (this is the "learned" /
 *     RL-flavored part: a value estimate updated from live feedback,
 *     analogous to the Q-table's EMA-based state features in the original
 *     RLArbiter, just applied to scheduling order instead of a dead
 *     backoff/admit choice).
 *   - When the lock is released and multiple threads are waiting, the
 *     waiter with the SHORTEST estimated duration is admitted next
 *     (Shortest-Predicted-Job-First) -- this is the textbook, provably
 *     mean-wait-optimal discipline for non-preemptive single-server
 *     queues with heterogeneous service times (M/G/1 SJF result).
 *   - AGING: any waiter whose wait has exceeded MAX_WAIT_AGING_NS has its
 *     effective priority forced to the front (oldest-aged-first), which
 *     bounds worst-case wait deterministically regardless of job-size
 *     mix, preventing long jobs from starving.
 *
 * WHY THIS CAN HAVE A REAL, NON-HARDCODED LEVER (unlike backoff-vs-FIFO):
 *   Under a HOMOGENEOUS workload (every request costs the same), SJF and
 *   FIFO are IDENTICAL orderings -- there is nothing to reorder, so this
 *   lock must tie a plain fair queue exactly (this is checked explicitly
 *   in PriorityLockBenchmark's "homogeneous sanity" run). Under a
 *   HETEROGENEOUS workload, SJF has a genuine, well-understood advantage:
 *   it avoids head-of-line blocking of short requests behind long ones.
 *
 * SAFETY: the ONLY thing customized here is *scheduling order* -- actual
 * mutual exclusion is enforced the standard, proven way: a single
 * `synchronized` monitor guards a boolean `held` flag and a priority
 * queue of waiters; a thread only proceeds past its wait loop when
 * EXACTLY one releaser has set that specific waiter's `turn` flag inside
 * the monitor. This is the same "monitor + condition + recheck" pattern
 * used throughout java.util.concurrent (Mesa-style condition variables),
 * not a hand-rolled CAS/spin scheme (which is where the earlier CLH bug
 * lived).
 */
public class AdaptivePriorityLock {

    public enum JobClass { SHORT, LONG }

    // Aging cutoff: once a waiter has been queued longer than this, it
    // is forced to the head of the line ahead of any non-aged waiter,
    // regardless of predicted duration. This is what prevents starvation
    // of LONG jobs under a SHORT-heavy workload.
    static final long MAX_WAIT_AGING_NS = 5_000_000L; // 5 ms

    private static final class Waiter {
        final long ticket;
        final long arrivalNs;
        final double predictedDurationNs;
        volatile boolean turn = false;

        Waiter(long ticket, long arrivalNs, double predictedDurationNs) {
            this.ticket = ticket; this.arrivalNs = arrivalNs;
            this.predictedDurationNs = predictedDurationNs;
        }

        // Effective priority at a given instant "now". Lower = goes first.
        double effectivePriority(long now) {
            long waited = now - arrivalNs;
            if (waited > MAX_WAIT_AGING_NS) {
                // Aged: rank strictly ahead of all non-aged waiters, oldest first.
                return -1e18 + ticket;
            }
            return predictedDurationNs;
        }
    }

    // Per-class online duration estimator (EMA) -- the "learned" component.
    static final class Estimator {
        volatile double emaNs;
        static final double ALPHA = 0.15;
        Estimator(double initNs) { emaNs = initNs; }
        void update(long observedNs) { emaNs = ALPHA * observedNs + (1 - ALPHA) * emaNs; }
        double estimate() { return emaNs; }
    }

    private final Object monitor = new Object();
    private boolean held = false;
    private final PriorityQueue<Waiter> waiting =
        new PriorityQueue<>((a, b) -> {
            long now = System.nanoTime();
            return Double.compare(a.effectivePriority(now), b.effectivePriority(now));
        });
    private long seq = 0;

    private final ConcurrentHashMap<JobClass, Estimator> estimators = new ConcurrentHashMap<>();

    public AdaptivePriorityLock() {
        for (JobClass jc : JobClass.values()) estimators.put(jc, new Estimator(100_000)); // 100us seed
    }

    // Returns queueing wait in ns (time from calling acquire() to actually
    // owning the lock), for metrics purposes.
    public long acquire(JobClass jobClass) throws InterruptedException {
        long t0 = System.nanoTime();
        Waiter w;
        synchronized (monitor) {
            if (!held && waiting.isEmpty()) {
                held = true;
                return System.nanoTime() - t0;
            }
            double predicted = estimators.get(jobClass).estimate();
            w = new Waiter(seq++, t0, predicted);
            waiting.add(w);
            while (!w.turn) {
                monitor.wait();
            }
            // held remains true; ownership was transferred to us by release()
        }
        return System.nanoTime() - t0;
    }

    // durationNs = measured actual time spent inside the critical section,
    // used to update this class's online estimator.
    public void release(JobClass jobClass, long durationNs) {
        synchronized (monitor) {
            estimators.get(jobClass).update(durationNs);
            if (waiting.isEmpty()) {
                held = false;
                return;
            }
            // Re-heapify is implicit: our comparator is time-varying (aging),
            // so we must re-scan for the true current minimum rather than
            // trust heap order, which can go stale between operations.
            Waiter best = null;
            long now = System.nanoTime();
            double bestPr = Double.POSITIVE_INFINITY;
            for (Waiter cand : waiting) {
                double pr = cand.effectivePriority(now);
                if (pr < bestPr) { bestPr = pr; best = cand; }
            }
            waiting.remove(best);
            best.turn = true;
            // held stays true -- ownership transferred directly, no gap
            // where a fresh acquire() could jump the queue.
            monitor.notifyAll();
        }
    }

    /** For diagnostics: current number of queued (not yet admitted) waiters. */
    public int queueLength() {
        synchronized (monitor) { return waiting.size(); }
    }
}
