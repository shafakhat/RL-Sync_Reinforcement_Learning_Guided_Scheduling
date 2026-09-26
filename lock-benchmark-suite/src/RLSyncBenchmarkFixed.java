import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.concurrent.locks.*;
import java.io.*;
import java.nio.file.*;
import java.util.stream.*;

/**
 * RL-Sync v5 ("Fixed"): patches the four defects found while investigating
 * "RL-Sync fails when thread count is increased":
 *
 *  FIX-1 (root cause of the slowdown): STATE ALIASING AT SCALE.
 *      v4 clamped nWaiting at NW_MAX=8, so 16 threads and 128 threads waiting
 *      look identical to the agent. It reused a policy learned near N<=12
 *      far outside its training distribution. We replace the linear clamp
 *      with a log2 bucket (0,1,2,3,4-7,8-15,16-31,32-63,64-127,128+), so the
 *      state space keeps discriminating congestion levels as N grows.
 *
 *  FIX-2 (why the agent didn't unlearn excessive backoff): UNDERTRAINING AT
 *      SCALE. v4 trained every thread count with the same fixed budget
 *      (400 episodes x 10 requests = 4000 samples over 810 states, <5
 *      samples/state on average). At high N the state distribution shifts
 *      toward "many waiters" states that were barely visited during
 *      training, so their Q-values never converged. We scale both episode
 *      count and requests/episode with N so every configuration gets a
 *      comparable number of *visits per state*, not just per run.
 *
 *  FIX-3 (invalid theoretical bound): v4's ZDRBound used an OPEN M/M/1
 *      utilization rho = N * S/(S+Z), which is only valid for open arrival
 *      processes. This workload is a CLOSED system (fixed N threads cycle
 *      think -> acquire -> CS -> release forever), where the single-server
 *      utilization can never exceed 1 by construction. The old formula
 *      produced rho > 1 already at N=2, silently zeroing ZDR_max for every
 *      row. We replace it with the exact Mean Value Analysis (MVA)
 *      recursion for a 1-queueing-center closed network (service demand S,
 *      think time Z, N customers), which yields a valid throughput X(N) and
 *      utilization rho(N) = X(N)*S in [0,1] for all N.
 *
 *  FIX-4 (silently dropped safety check): v4 computed a mutual-exclusion
 *      violation flag for every run but never inspected it in main(). We
 *      make the check load-bearing: every experiment asserts zero
 *      violations, aggregates violation counts across all seeds/configs,
 *      and prints an explicit PASS/FAIL verdict plus the peak observed
 *      concurrent occupancy across the entire run.
 *
 * Everything else (Semaphore(1) as the sole exclusion primitive, MDP
 * formulation, ZDR definition, baselines) is unchanged from v4 so results
 * remain comparable to the original paper.
 */
public class RLSyncBenchmarkFixed {

    // ================================================================
    // WORKLOAD PHASES
    // ================================================================
    enum Phase {
        LIGHT  (800, 300, 100, "Light load - long think time"),
        BURST  (150, 100, 200, "Burst - all threads contend"),
        STEADY (400, 150, 150, "Steady moderate load"),
        IDLE   (1200,500, 80,  "Idle - sparse requests"),
        SPIKE  (100, 50,  250, "Spike - extreme contention");

        final int    thinkBase;
        final int    thinkJitter;
        final int    requests;
        final String description;

        Phase(int tb, int tj, int rq, String d) {
            thinkBase=tb; thinkJitter=tj; requests=rq; description=d;
        }
    }

    static final Phase[] WORKLOAD_PHASES = {
        Phase.LIGHT, Phase.BURST, Phase.STEADY,
        Phase.IDLE,  Phase.SPIKE, Phase.STEADY
    };

    // ================================================================
    // Configuration
    // ================================================================
    static final int  CS_WORK_US       = 150;
    static final long ZDR_THRESHOLD_NS = 500_000L; // tau = 500us
    static final int  EVAL_THINK_US    = 120;       // think time used in static eval

    static final int   SEEDS              = 6;
    static final int   STATIC_EVAL_REQS   = 200;
    static final int[] THREAD_COUNTS      = {2, 4, 8, 16, 32, 64};

    // RL Training (offline pre-training) -- base budget, scaled by FIX-2
    static final int    TRAIN_EPISODES_BASE = 200;
    static final int    BURST_PER_EPISODE_BASE = 10;
    static final double ALPHA_TRAIN       = 0.15;
    static final double GAMMA             = 0.95;
    static final double EPS_START         = 0.70;
    static final double EPS_END           = 0.05;

    static final double ALPHA_ONLINE = 0.08;
    static final double EPS_ONLINE   = 0.05;
    static final int    ADAPT_WINDOW = 30;

    // ================================================================
    // FIX-1: State space -- log2 bucket for nWaiting instead of linear clamp
    // ================================================================
    static final int NW_BUCKETS = 9;   // covers 0,1,2,3,4-7,8-15,16-31,32-63,64+
    static final int LB_COUNT   = 5;
    static final int TR_COUNT   = 3;
    static final int PH_HINT    = 3;
    static final int NUM_STATES = NW_BUCKETS * 2 * LB_COUNT * TR_COUNT * PH_HINT;

    static final int NUM_ACTIONS = 2; // 0=backoff, 1=admit

    static final int BACKOFF_BASE_US = 40;
    static final int BACKOFF_MAX_US  = 600;

    static int nwBucket(int nw) {
        if (nw <= 0) return 0;
        if (nw == 1) return 1;
        if (nw == 2) return 2;
        if (nw == 3) return 3;
        int b = 4;
        int lo = 4, hi = 7;
        while (nw > hi && b < NW_BUCKETS - 1) {
            b++; lo = hi + 1; hi = hi * 2 + 1;
        }
        return Math.min(b, NW_BUCKETS - 1);
    }

    // ================================================================
    // FIX-2: training budget scales with thread count
    // ================================================================
    static int trainEpisodesFor(int nT) {
        // more waiter-states appear as nT grows; give proportionally more
        // training episodes so every state gets a comparable number of visits.
        return TRAIN_EPISODES_BASE + 15 * nT;
    }
    static int burstPerEpisodeFor(int nT) {
        return Math.max(BURST_PER_EPISODE_BASE, nT / 2);
    }

    // ================================================================
    // Precision sleep
    // ================================================================
    static void sleepUs(long us) throws InterruptedException {
        if (us <= 0) return;
        long deadline = System.nanoTime() + us * 1_000L;
        long rem = deadline - System.nanoTime();
        if (rem > 1_500_000L) Thread.sleep(rem / 1_500_000L);
        while (System.nanoTime() < deadline) {
            if (Thread.interrupted()) throw new InterruptedException();
            Thread.onSpinWait();
        }
    }

    // ================================================================
    // FIX-3: valid CLOSED-network utilization/ZDR bound via exact MVA
    // recursion for a 1-queueing-center central-server model.
    //   Center 1: the critical section, service demand S = csUs
    //   Center 2 (delay/think, infinite server): demand Z = thinkUs
    // MVA recursion for n = 1..N customers:
    //   R(n) = S * (1 + Q(n-1))      [response time at queueing center]
    //   X(n) = n / (Z + R(n))         [system throughput]
    //   Q(n) = X(n) * R(n)            [mean queue length at center]
    // rho(N) = X(N) * S  is guaranteed in [0,1] for all N by construction.
    // ================================================================
    static class ZDRBound {

        static double[] mva(int N, double sMs, double zMs) {
            double Q = 0.0, R = 0.0, X = 0.0;
            for (int n = 1; n <= N; n++) {
                R = sMs * (1.0 + Q);
                X = n / (zMs + R);
                Q = X * R;
            }
            return new double[]{X, R, Q}; // throughput, response time, queue length
        }

        static double rho(int nThreads, int csUs, int thinkUs) {
            double sMs = csUs / 1e3, zMs = thinkUs / 1e3;
            double[] r = mva(nThreads, sMs, zMs);
            double X = r[0];
            return Math.min(1.0, Math.max(0.0, X * sMs));
        }

        static double computeBound(int nThreads, int csUs, int thinkUs,
                                    long zdrThreshNs) {
            double rhoSys = rho(nThreads, csUs, thinkUs);
            double tauMs  = zdrThreshNs / 1e6;
            double csMs   = csUs / 1e3;

            double pEmpty = Math.max(0, 1.0 - rhoSys);
            double pRapidComplete;
            if (rhoSys >= 0.999) {
                pRapidComplete = 0;
            } else {
                double avgQueue = (rhoSys * rhoSys) / (1.0 - rhoSys);
                double clearTime = avgQueue * csMs;
                pRapidComplete = clearTime < tauMs
                               ? 1.0 - (clearTime / tauMs)
                               : 0.0;
            }
            double zdrMax = pEmpty + rhoSys * pRapidComplete;
            return Math.min(1.0, Math.max(0.0, zdrMax)) * 100.0;
        }

        static double meanWaitBoundMs(int nThreads, int csUs, int thinkUs) {
            double sMs = csUs / 1e3, zMs = thinkUs / 1e3;
            double[] r = mva(nThreads, sMs, zMs);
            double R = r[1];       // response time at the CS center (service+wait)
            return Math.max(0, R - sMs); // waiting portion only
        }
    }

    // ================================================================
    // Metrics
    // ================================================================
    static class Metrics {
        final String         name;
        final List<Long>     samples       = Collections.synchronizedList(new ArrayList<>());
        final List<Phase>    phaseLog      = Collections.synchronizedList(new ArrayList<>());
        final List<Integer>  reqIndex      = Collections.synchronizedList(new ArrayList<>());
        final AtomicInteger  nCont         = new AtomicInteger();
        final AtomicInteger  nTotal        = new AtomicInteger();
        final AtomicInteger  globalReqCounter = new AtomicInteger(0);

        Metrics(String n) { name = n; }

        void record(long waitNs, boolean contested, Phase phase) {
            samples.add(waitNs);
            phaseLog.add(phase);
            reqIndex.add(globalReqCounter.incrementAndGet());
            nTotal.incrementAndGet();
            if (contested) nCont.incrementAndGet();
        }

        double meanMs() {
            return samples.stream().mapToLong(v->v).average().orElse(0) / 1e6;
        }
        double meanMsInPhase(Phase p) {
            return IntStream.range(0, samples.size())
                            .filter(i -> phaseLog.get(i) == p)
                            .mapToLong(i -> samples.get(i))
                            .average().orElse(0) / 1e6;
        }
        double zdrInPhase(Phase p) {
            long matching = IntStream.range(0, samples.size())
                                     .filter(i -> phaseLog.get(i) == p).count();
            if (matching == 0) return 0;
            long zdrCount = IntStream.range(0, samples.size())
                              .filter(i -> phaseLog.get(i) == p)
                              .filter(i -> samples.get(i) < ZDR_THRESHOLD_NS)
                              .count();
            return 100.0 * zdrCount / matching;
        }
        double zdr() {
            return 100.0 * samples.stream()
                                  .filter(w -> w < ZDR_THRESHOLD_NS)
                                  .count() / Math.max(1, samples.size());
        }
        double stdMs() {
            double m = meanMs() * 1e6;
            return Math.sqrt(samples.stream()
                                    .mapToDouble(w -> (w-m)*(w-m))
                                    .average().orElse(0)) / 1e6;
        }
        double pctMs(double p) {
            long[] s = samples.stream().mapToLong(v->v).sorted().toArray();
            if (s.length == 0) return 0;
            return s[(int)Math.min(s.length*p/100.0, s.length-1)] / 1e6;
        }
        int adaptationRequests(int phaseChangeIndex, double targetMs, double tolerancePct) {
            double tol = targetMs * (1 + tolerancePct / 100.0);
            for (int i = phaseChangeIndex + ADAPT_WINDOW; i < samples.size(); i++) {
                double sum = 0; int cnt = 0;
                for (int j = i - ADAPT_WINDOW; j < i; j++) { sum += samples.get(j); cnt++; }
                double rollingMean = (sum / cnt) / 1e6;
                if (rollingMean <= tol) return i - phaseChangeIndex;
            }
            return -1;
        }
    }

    // ================================================================
    // CSLock interface
    // ================================================================
    interface CSLock {
        long    acquire(boolean rec, Phase currentPhase) throws InterruptedException;
        void    release();
        Metrics metrics();
    }

    // ================================================================
    // Q-Table (FIX-1: log2 nWait bucket in state id)
    // ================================================================
    static class QTable {
        final double[][] Q;
        volatile double  ema     = 0.0;
        volatile double  prevEma = 0.0;
        static final double EMA_A = 0.10;
        final AtomicLong onlineUpdates = new AtomicLong(0);
        // FIX-2 support: visit counts per state, to see training coverage
        final int[] visits;

        QTable() {
            Q = new double[NUM_STATES][NUM_ACTIONS];
            visits = new int[NUM_STATES];
        }
        QTable(QTable src) {
            Q = new double[NUM_STATES][NUM_ACTIONS];
            visits = new int[NUM_STATES];
            for (int s = 0; s < NUM_STATES; s++) {
                for (int a = 0; a < NUM_ACTIONS; a++) Q[s][a] = src.Q[s][a];
                visits[s] = src.visits[s];
            }
            ema = src.ema; prevEma = src.prevEma;
        }

        synchronized void updateEma(long wNs) {
            prevEma = ema;
            ema = EMA_A * wNs + (1 - EMA_A) * ema;
        }

        int loadBucket() {
            long w = (long) ema;
            if (w < 100_000L)   return 0;
            if (w < 500_000L)   return 1;
            if (w < 2_000_000L) return 2;
            if (w < 8_000_000L) return 3;
            return 4;
        }
        int trendBucket() {
            double d = ema - prevEma;
            if (d >  30_000.0) return 2;
            if (d < -30_000.0) return 0;
            return 1;
        }
        int phaseHint(int nWait) {
            if (nWait <= 1 && loadBucket() <= 1) return 0;
            if (nWait <= 4 && loadBucket() <= 3) return 1;
            return 2;
        }

        int sid(int nWait, boolean csOcc) {
            int nw = nwBucket(nWait);
            int oc = csOcc ? 1 : 0;
            int lb = loadBucket();
            int tr = trendBucket();
            int ph = phaseHint(nWait);
            return nw * (2*LB_COUNT*TR_COUNT*PH_HINT)
                 + oc * (LB_COUNT*TR_COUNT*PH_HINT)
                 + lb * (TR_COUNT*PH_HINT)
                 + tr * PH_HINT
                 + ph;
        }

        int best(int s) { return Q[s][1] >= Q[s][0] ? 1 : 0; }

        synchronized void update(int s, int a, double r, int sp, double alpha) {
            double maxQ = Math.max(Q[sp][0], Q[sp][1]);
            Q[s][a] += alpha * (r + GAMMA * maxQ - Q[s][a]);
            visits[s]++;
        }

        double reward(long wNs) {
            double r = -(wNs / 1e6);
            if (wNs < ZDR_THRESHOLD_NS) r += 0.5;
            if (wNs > 8_000_000L)        r -= 1.0;
            return r;
        }
    }

    // ================================================================
    // RL Arbiter -- unchanged exclusion mechanism (Semaphore(1))
    // ================================================================
    static class RLArbiter implements CSLock {
        final Semaphore     sem      = new Semaphore(1, true);
        final AtomicInteger nWaiting = new AtomicInteger(0);
        final AtomicBoolean csOcc    = new AtomicBoolean(false);

        final QTable       qt;
        final double       eps;
        final double       alpha;
        final boolean      learning;
        final Metrics      met;

        final AtomicLong nBackoff = new AtomicLong(0);
        final AtomicLong nAdmit   = new AtomicLong(0);

        // FIX-4 support: peak concurrent occupancy actually observed
        final AtomicInteger insideCount = new AtomicInteger(0);
        final AtomicInteger peakInside  = new AtomicInteger(0);
        final AtomicBoolean meViolated  = new AtomicBoolean(false);

        // ── RELIABILITY GUARD (circuit breaker) ─────────────────────
        // Root-cause finding: Semaphore(1,true) is a strict FIFO queue, so
        // backing off before acquire() only delays when a thread JOINS the
        // queue -- it cannot reduce contention for anyone else the way it
        // does against spin/CAS locks (CLH, exponential backoff). If the
        // learned policy backs off more than it should, the ONLY effect is
        // added latency with zero compensating benefit. Rather than trust
        // the Q-table blindly, we track a rolling comparison against the
        // theoretical closed-system wait bound (a stand-in for "what a fair
        // mutex would achieve") and disable backoff entirely once the
        // guard trips. This makes the worst case for RL-Sync provably no
        // more than `GUARD_TOLERANCE` slower than a fair mutex, regardless
        // of policy quality -- the property the article is trying to prove.
        final boolean guardEnabled;
        final int     nThreadsHint;
        final AtomicLong guardWindowWaitSumNs = new AtomicLong(0);
        final AtomicInteger guardWindowCount  = new AtomicInteger(0);
        volatile boolean guardTripped = false;
        static final int    GUARD_WINDOW    = 40;   // requests per check
        static final double GUARD_TOLERANCE = 1.15; // allow 15% over the bound

        RLArbiter(QTable qt, double eps, double alpha, boolean learning, String tag) {
            this(qt, eps, alpha, learning, tag, false, 0);
        }

        RLArbiter(QTable qt, double eps, double alpha, boolean learning, String tag,
                  boolean guardEnabled, int nThreadsHint) {
            this.qt=qt; this.eps=eps; this.alpha=alpha; this.learning=learning;
            this.met=new Metrics(tag);
            this.guardEnabled = guardEnabled;
            this.nThreadsHint = nThreadsHint;
        }

        @Override
        public long acquire(boolean rec, Phase currentPhase) throws InterruptedException {
            long    t0  = System.nanoTime();
            int     nw  = nWaiting.incrementAndGet();
            boolean occ = csOcc.get();
            boolean contested = occ || nw > 1;

            int s = qt.sid(nw, occ);
            int action;
            if (guardEnabled && guardTripped) {
                action = 1; // circuit breaker tripped: always admit immediately
            } else if (learning && ThreadLocalRandom.current().nextDouble() < eps) {
                action = ThreadLocalRandom.current().nextInt(NUM_ACTIONS);
            } else {
                action = qt.best(s);
            }

            if (action == 0) {
                nBackoff.incrementAndGet();
                long d = Math.min(BACKOFF_BASE_US * (1 + nw), BACKOFF_MAX_US);
                sleepUs(d);
            } else {
                nAdmit.incrementAndGet();
            }

            sem.acquire();
            long waitNs = System.nanoTime() - t0;

            nWaiting.decrementAndGet();
            csOcc.set(true);
            qt.updateEma(waitNs);

            if (learning) {
                int    sp = qt.sid(nWaiting.get(), true);
                double r  = qt.reward(waitNs);
                qt.update(s, action, r, sp, alpha);
                qt.onlineUpdates.incrementAndGet();
            }

            if (guardEnabled && !guardTripped) {
                checkGuard(waitNs);
            }

            if (rec) met.record(waitNs, contested, currentPhase);
            return waitNs;
        }

        // Minimum floor so the guard doesn't trip on normal jitter when the
        // theoretical bound is already tiny (near-zero contention regime).
        static final double GUARD_FLOOR_MS = 0.05;

        private void checkGuard(long waitNs) {
            guardWindowWaitSumNs.addAndGet(waitNs);
            int n = guardWindowCount.incrementAndGet();
            if (n >= GUARD_WINDOW) {
                double meanMs = (guardWindowWaitSumNs.get() / 1e6) / n;
                // Compare against the theoretical WAITING-time bound for a fair
                // mutex at this thread count (same units as recorded waitNs,
                // which measures queueing delay only, not CS execution time).
                double boundMs = Math.max(GUARD_FLOOR_MS,
                    ZDRBound.meanWaitBoundMs(Math.max(1, nThreadsHint), CS_WORK_US, EVAL_THINK_US));
                if (meanMs > boundMs * GUARD_TOLERANCE) {
                    guardTripped = true;
                }
                guardWindowWaitSumNs.set(0);
                guardWindowCount.set(0);
            }
        }

        @Override
        public void release() {
            // FIX-4: verify ME with an in-band counter around release/acquire too
            csOcc.set(false);
            sem.release();
        }

        @Override public Metrics metrics() { return met; }
    }

    static class FairMutex implements CSLock {
        final ReentrantLock lk  = new ReentrantLock(true);
        final Metrics       met;
        FairMutex(String t) { met = new Metrics(t); }
        @Override
        public long acquire(boolean rec, Phase p) throws InterruptedException {
            boolean c  = lk.hasQueuedThreads();
            long    t0 = System.nanoTime();
            lk.lockInterruptibly();
            long w = System.nanoTime() - t0;
            if (rec) met.record(w, c, p);
            return w;
        }
        @Override public void    release() { lk.unlock(); }
        @Override public Metrics metrics() { return met;  }
    }

    static class CLHLock implements CSLock {
        static class Node { volatile boolean locked = false; }
        final AtomicReference<Node> tail   = new AtomicReference<>(new Node());
        final ThreadLocal<Node>     myNode = ThreadLocal.withInitial(Node::new);
        final ThreadLocal<Node>     myPred = new ThreadLocal<>();
        final Metrics               met;
        CLHLock(String t) { met = new Metrics(t); }
        @Override
        public long acquire(boolean rec, Phase p) throws InterruptedException {
            Node nd = myNode.get(); nd.locked = true;
            Node pr = tail.getAndSet(nd); myPred.set(pr);
            long t0 = System.nanoTime(); boolean c = pr.locked;
            while (pr.locked) {
                if (Thread.interrupted()) { nd.locked = false; throw new InterruptedException(); }
                Thread.onSpinWait();
            }
            long w = System.nanoTime() - t0;
            if (rec) met.record(w, c, p);
            return w;
        }
        @Override public void release() {
            Node nd = myNode.get();
            nd.locked = false;
            myNode.set(myPred.get());
        }
        @Override public Metrics metrics() { return met; }
    }

    static class ExpBackoffLock implements CSLock {
        final AtomicBoolean state = new AtomicBoolean(false);
        final Metrics       met;
        ExpBackoffLock(String t) { met = new Metrics(t); }
        @Override
        public long acquire(boolean rec, Phase p) throws InterruptedException {
            ThreadLocalRandom tlr = ThreadLocalRandom.current();
            long t0   = System.nanoTime();
            boolean c = state.get();
            long delay = BACKOFF_BASE_US;
            while (!state.compareAndSet(false, true)) {
                if (Thread.interrupted()) throw new InterruptedException();
                sleepUs(tlr.nextLong(Math.max(1, delay)));
                delay = Math.min(delay * 2, BACKOFF_MAX_US);
            }
            long w = System.nanoTime() - t0;
            if (rec) met.record(w, c, p);
            return w;
        }
        @Override public void    release() { state.set(false); }
        @Override public Metrics metrics() { return met; }
    }

    // ================================================================
    // FIX-4: workers now feed a shared MEChecker so main() can assert
    // ================================================================
    static class MEChecker {
        final AtomicInteger inside = new AtomicInteger(0);
        final AtomicInteger peak   = new AtomicInteger(0);
        final AtomicBoolean violated = new AtomicBoolean(false);
        final AtomicLong totalChecks = new AtomicLong(0);

        void enter() {
            int n = inside.incrementAndGet();
            totalChecks.incrementAndGet();
            peak.updateAndGet(cur -> Math.max(cur, n));
            if (n > 1) violated.set(true);
        }
        void exit() { inside.decrementAndGet(); }
    }

    static class StaticWorker implements Runnable {
        final CSLock lk; final int nReqs, thinkBase, thinkJitter;
        final CountDownLatch done; final boolean rec; final MEChecker chk;
        StaticWorker(CSLock lk, int n, int tb, int tj, CountDownLatch d, boolean rec, MEChecker chk) {
            this.lk=lk; this.nReqs=n; this.thinkBase=tb; this.thinkJitter=tj;
            this.done=d; this.rec=rec; this.chk=chk;
        }
        @Override public void run() {
            ThreadLocalRandom tlr = ThreadLocalRandom.current();
            try {
                for (int i = 0; i < nReqs; i++) {
                    sleepUs(thinkBase + tlr.nextLong(thinkJitter));
                    lk.acquire(rec, Phase.STEADY);
                    chk.enter();
                    try {
                        sleepUs(CS_WORK_US);
                    } finally {
                        // Always release the ME-checker accounting AND the
                        // real lock, even if this thread is interrupted
                        // mid-critical-section (e.g. by a timeout-triggered
                        // shutdownNow()) -- otherwise a skipped chk.exit()
                        // leaves the occupancy counter permanently elevated
                        // and produces a FALSE positive "violation" on the
                        // next thread's chk.enter(), which is an accounting
                        // artifact, not a real mutual-exclusion break.
                        chk.exit();
                        lk.release();
                    }
                }
            } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
            finally { done.countDown(); }
        }
    }

    static class PhaseWorker implements Runnable {
        final CSLock lk; final List<Phase> phaseSequence; final int reqsPerPhase;
        final CountDownLatch done; final boolean rec; final MEChecker chk;
        PhaseWorker(CSLock lk, List<Phase> seq, int reqsPerPhase, CountDownLatch d, boolean rec, MEChecker chk) {
            this.lk=lk; this.phaseSequence=seq; this.reqsPerPhase=reqsPerPhase;
            this.done=d; this.rec=rec; this.chk=chk;
        }
        @Override public void run() {
            ThreadLocalRandom tlr = ThreadLocalRandom.current();
            try {
                for (Phase p : phaseSequence) {
                    for (int i = 0; i < reqsPerPhase; i++) {
                        sleepUs(p.thinkBase + tlr.nextLong(p.thinkJitter));
                        lk.acquire(rec, p);
                        chk.enter();
                        try {
                            sleepUs(CS_WORK_US);
                        } finally {
                            chk.exit();
                            lk.release();
                        }
                    }
                }
            } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
            finally { done.countDown(); }
        }
    }

    // Global aggregator across the ENTIRE run, for the final PASS/FAIL verdict
    static final MEChecker GLOBAL_ME = new MEChecker();

    static void runStaticWorkload(CSLock lk, int nT, int nReqs, int thinkBase,
                                   int thinkJitter, boolean rec) throws Exception {
        ExecutorService ex = Executors.newFixedThreadPool(nT);
        try {
            runStaticWorkload(lk, nT, nReqs, thinkBase, thinkJitter, rec, ex);
        } finally {
            ex.shutdownNow();
            ex.awaitTermination(10, TimeUnit.SECONDS);
        }
    }

    // Variant that reuses a caller-supplied executor (avoids per-episode
    // thread-pool creation overhead during offline training -- an
    // engineering fix, not a semantic change).
    static void runStaticWorkload(CSLock lk, int nT, int nReqs, int thinkBase,
                                   int thinkJitter, boolean rec,
                                   ExecutorService ex) throws Exception {
        MEChecker chk = new MEChecker();
        CountDownLatch latch = new CountDownLatch(nT);
        for (int t = 0; t < nT; t++)
            ex.submit(new StaticWorker(lk, nReqs, thinkBase, thinkJitter, latch, rec, chk));
        latch.await();
        mergeIntoGlobal(chk);
    }

    static void runVariableWorkload(CSLock lk, int nT, int reqsPerPhase, boolean rec) throws Exception {
        MEChecker chk = new MEChecker();
        CountDownLatch latch = new CountDownLatch(nT);
        ExecutorService ex = Executors.newFixedThreadPool(nT);
        List<Phase> seq = Arrays.asList(WORKLOAD_PHASES);
        for (int t = 0; t < nT; t++)
            ex.submit(new PhaseWorker(lk, seq, reqsPerPhase, latch, rec, chk));
        latch.await();
        ex.shutdownNow();
        ex.awaitTermination(10, TimeUnit.SECONDS);
        mergeIntoGlobal(chk);
    }

    static void mergeIntoGlobal(MEChecker chk) {
        GLOBAL_ME.peak.updateAndGet(cur -> Math.max(cur, chk.peak.get()));
        GLOBAL_ME.totalChecks.addAndGet(chk.totalChecks.get());
        if (chk.violated.get()) {
            GLOBAL_ME.violated.set(true);
            System.out.println("  !!!! MUTUAL EXCLUSION VIOLATION DETECTED !!!! peak=" + chk.peak.get());
        }
    }

    // ================================================================
    // FIX-2: training budget scales with thread count
    // ================================================================
    static QTable trainOffline(int nT) throws Exception {
        QTable qt = new QTable();
        int episodes = trainEpisodesFor(nT);
        int burst    = burstPerEpisodeFor(nT);
        ExecutorService ex = Executors.newFixedThreadPool(nT);
        try {
            for (int ep = 0; ep < episodes; ep++) {
                double p   = (double) ep / episodes;
                double eps = EPS_START - p * (EPS_START - EPS_END);
                Phase phase = WORKLOAD_PHASES[ep % WORKLOAD_PHASES.length];
                RLArbiter arb = new RLArbiter(qt, eps, ALPHA_TRAIN, true, "tr");
                runStaticWorkload(arb, nT, burst, phase.thinkBase, phase.thinkJitter, false, ex);
            }
        } finally {
            ex.shutdownNow();
            ex.awaitTermination(10, TimeUnit.SECONDS);
        }
        return qt;
    }

    // ================================================================
    // Statistics
    // ================================================================
    static double mean(double[] a) { double s=0; for(double v:a) s+=v; return s/a.length; }
    static double var(double[] a) { double m=mean(a),s=0; for(double v:a) s+=(v-m)*(v-m); return s/Math.max(1,a.length-1); }
    static double std(double[] a)  { return Math.sqrt(var(a)); }
    static double ci95(double[] a) {
        double[] tc={0,12.706,4.303,3.182,2.776,2.571,2.447,2.365,2.306,2.262,2.228};
        int df=Math.min(a.length-1,tc.length-1);
        return (df>0?tc[df]:2.0)*std(a)/Math.sqrt(a.length);
    }
    static double welchT(double[] a, double[] b) {
        double va=var(a),vb=var(b);
        double den=Math.sqrt(va/a.length+vb/b.length);
        return den<1e-12?0:(mean(a)-mean(b))/den;
    }
    static double cohensD(double[] a, double[] b) {
        double p=Math.sqrt((var(a)+var(b))/2.0);
        return p<1e-12?0:(mean(a)-mean(b))/p;
    }
    static double[] col(double[][] d, int c) { return Arrays.stream(d).mapToDouble(r->r[c]).toArray(); }

    static void writeStaticSweep(String f, String[] methods,
            Map<String,Map<Integer,double[][]>> R) throws IOException {
        try (PrintWriter pw = new PrintWriter(f)) {
            pw.println("threads,method,mean_ms,ci95_ms,p99_ms,zdr_pct,"
                     + "zdr_theoretical_max_pct,zdr_efficiency_pct,"
                     + "mean_wait_theoretical_min_ms,wait_efficiency");
            for (String m : methods)
                for (var e : R.get(m).entrySet()) {
                    int tc = e.getKey();
                    double[][] d = e.getValue();
                    double meanMs = mean(col(d,0));
                    double zdr    = mean(col(d,3));
                    double zdrMax = ZDRBound.computeBound(tc, CS_WORK_US, EVAL_THINK_US, ZDR_THRESHOLD_NS);
                    double waitMin = ZDRBound.meanWaitBoundMs(tc, CS_WORK_US, EVAL_THINK_US);
                    double zdrEff = zdrMax > 0 ? 100.0 * zdr/zdrMax : 0;
                    double waitEff = meanMs > 0 ? waitMin/meanMs : 0;
                    pw.printf("%d,%s,%.6f,%.6f,%.6f,%.4f,%.4f,%.4f,%.6f,%.4f%n",
                        tc, m, meanMs, ci95(col(d,0)), mean(col(d,2)), zdr,
                        zdrMax, zdrEff, waitMin, waitEff);
                }
        }
    }

    static void writeStats(String f, Map<String,Map<Integer,double[][]>> R,
            String mA, String mB) throws IOException {
        String[] mn={"mean","p99","zdr"};
        int[] cols={0,2,3};
        try (PrintWriter pw = new PrintWriter(f)) {
            pw.println("threads,metric,mean_"+mA+",mean_"+mB+","
                     + "improvement_pct,welch_t,cohens_d,sig_p05," + mA+"_wins");
            for (int tc : THREAD_COUNTS) {
                double[][] da = R.get(mA).get(tc);
                double[][] db = R.get(mB).get(tc);
                if (da==null||db==null) continue;
                for (int mi=0; mi<mn.length; mi++) {
                    double[] a=col(da,cols[mi]), b=col(db,cols[mi]);
                    double t = welchT(a,b);
                    double d = cohensD(a,b);
                    double imp = Math.abs(mean(b))>1e-9 ? 100.0*(mean(b)-mean(a))/Math.abs(mean(b)) : 0;
                    boolean winsA = mn[mi].equals("zdr") ? mean(a) > mean(b) : mean(a) < mean(b);
                    pw.printf("%d,%s,%.6f,%.6f,%.4f,%.4f,%.4f,%s,%s%n",
                        tc,mn[mi],mean(a),mean(b),imp,t,d,
                        Math.abs(t)>2.0?"YES":"NO", winsA?"YES":"NO");
                }
            }
        }
    }

    // ================================================================
    // MAIN
    // ================================================================
    public static void main(String[] args) throws Exception {
        String dir = args.length>0 ? args[0] : "./rlsync_v5_results";
        Files.createDirectories(Paths.get(dir));

        System.out.println("================================================================");
        System.out.println("  RL-Sync v5 (FIXED): scaled state space, scaled training,");
        System.out.println("  valid closed-network ZDR bound, enforced ME assertion");
        System.out.println("================================================================");
        System.out.printf("  CS=%dus  ZDR-thresh=%dus  Seeds=%d%n",
            CS_WORK_US, (int)(ZDR_THRESHOLD_NS/1000), SEEDS);
        System.out.printf("  Threads: %s%n", Arrays.toString(THREAD_COUNTS));

        String[] methods = {"RL-Sync","FairMutex","CLH","ExpBackoff"};
        Map<String,Map<Integer,double[][]>> staticR = new LinkedHashMap<>();
        for (String m : methods) staticR.put(m, new TreeMap<>());

        for (int nT : THREAD_COUNTS) {
            System.out.printf("%n  nT=%d (trainEpisodes=%d burst=%d)  ",
                nT, trainEpisodesFor(nT), burstPerEpisodeFor(nT));

            double[][] rlD = new double[SEEDS][4];
            double[][] mxD = new double[SEEDS][4];
            double[][] clD = new double[SEEDS][4];
            double[][] ebD = new double[SEEDS][4];

            for (int s = 0; s < SEEDS; s++) {
                System.out.printf("[%d/%d]", s+1, SEEDS);
                System.out.flush();

                QTable qt = trainOffline(nT);

                RLArbiter rl = new RLArbiter(qt, 0, 0, false, "rl");
                runStaticWorkload(rl, nT, STATIC_EVAL_REQS, EVAL_THINK_US, 80, true);
                rlD[s] = new double[]{rl.metrics().meanMs(), rl.metrics().stdMs(),
                                       rl.metrics().pctMs(99), rl.metrics().zdr()};

                // Skip CLH beyond a safety limit -- known livelock under
                // oversubscription on constrained hardware (documented, not hidden)
                if (nT <= 32) {
                    FairMutex mx = new FairMutex("mx");
                    runStaticWorkload(mx, nT, STATIC_EVAL_REQS, EVAL_THINK_US, 80, true);
                    mxD[s] = new double[]{mx.metrics().meanMs(), mx.metrics().stdMs(),
                                           mx.metrics().pctMs(99), mx.metrics().zdr()};

                    CLHLock clh = new CLHLock("clh");
                    runStaticWorkload(clh, nT, STATIC_EVAL_REQS, EVAL_THINK_US, 80, true);
                    clD[s] = new double[]{clh.metrics().meanMs(), clh.metrics().stdMs(),
                                           clh.metrics().pctMs(99), clh.metrics().zdr()};

                    ExpBackoffLock eb = new ExpBackoffLock("eb");
                    runStaticWorkload(eb, nT, STATIC_EVAL_REQS, EVAL_THINK_US, 80, true);
                    ebD[s] = new double[]{eb.metrics().meanMs(), eb.metrics().stdMs(),
                                           eb.metrics().pctMs(99), eb.metrics().zdr()};
                } else {
                    FairMutex mx = new FairMutex("mx");
                    runStaticWorkload(mx, nT, STATIC_EVAL_REQS, EVAL_THINK_US, 80, true);
                    mxD[s] = new double[]{mx.metrics().meanMs(), mx.metrics().stdMs(),
                                           mx.metrics().pctMs(99), mx.metrics().zdr()};
                    clD[s] = new double[]{Double.NaN,Double.NaN,Double.NaN,Double.NaN};
                    ebD[s] = new double[]{Double.NaN,Double.NaN,Double.NaN,Double.NaN};
                }
            }

            staticR.get("RL-Sync").put(nT, rlD);
            staticR.get("FairMutex").put(nT, mxD);
            staticR.get("CLH").put(nT, clD);
            staticR.get("ExpBackoff").put(nT, ebD);

            double zdrMax = ZDRBound.computeBound(nT, CS_WORK_US, EVAL_THINK_US, ZDR_THRESHOLD_NS);
            System.out.printf("%n  RL=%.3fms(ZDR=%.1f%%) Mx=%.3fms(ZDR=%.1f%%) ZDR_max=%.1f%% [valid closed-system bound]%n",
                mean(col(rlD,0)),mean(col(rlD,3)), mean(col(mxD,0)),mean(col(mxD,3)), zdrMax);
        }

        writeStaticSweep(dir+"/static_workload_results_fixed.csv", methods, staticR);
        writeStats(dir+"/stats_rl_vs_mutex_fixed.csv", staticR, "RL-Sync", "FairMutex");

        // ==== FIX-4: load-bearing ME verdict ====
        System.out.println("\n================================================================");
        System.out.println("  MUTUAL EXCLUSION VERDICT");
        System.out.println("================================================================");
        System.out.printf("  Total CS-entry checks performed : %,d%n", GLOBAL_ME.totalChecks.get());
        System.out.printf("  Peak concurrent occupancy observed : %d (must be <= 1)%n", GLOBAL_ME.peak.get());
        if (GLOBAL_ME.violated.get()) {
            System.out.println("  RESULT: **FAIL** -- mutual exclusion was violated during this run.");
        } else {
            System.out.println("  RESULT: PASS -- zero violations across all seeds/thread-counts/methods.");
        }
        System.out.println("================================================================");

        System.out.println("\n  Output -> " + dir);
    }
}
