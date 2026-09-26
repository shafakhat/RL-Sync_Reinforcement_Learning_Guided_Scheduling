import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.concurrent.locks.*;
import java.io.*;
import java.nio.file.*;
import java.util.stream.*;

/**
 * RL-Sync v4: Variable Workload + Online Adaptation + ZDR Bound
 * ===============================================================
 *
 * THREE NEW CONTRIBUTIONS BEYOND BASELINE PAPER:
 *
 * 1. VARIABLE WORKLOAD (Section: Adaptive Evaluation)
 *    Workload phases: LIGHT -> BURST -> STEADY -> IDLE -> SPIKE -> STEADY
 *    Each phase = N requests with different (think, jitter) params
 *    Static locks (FairMutex, CLH) cannot adapt their policy
 *    RL with online learning CAN adapt -> genuine measurable advantage
 *
 * 2. ONLINE ADAPTATION (Section: Lifelong RL)
 *    During evaluation, RL continues Q-updates with eta_online
 *    eps_online > 0 for continued exploration
 *    Measure: adaptation time = #requests to recover within 10% of
 *             pre-shift performance after a workload phase change
 *
 * 3. THEORETICAL ZDR BOUND (Section: Optimality Analysis)
 *    For M/G/1 queue with utilisation rho and ZDR threshold tau:
 *      ZDR_max(rho, tau) = 1 - rho + rho * P(S < tau - W_q)
 *    where W_q is Pollaczek-Khinchine queue wait
 *    Report ZDR_efficiency = measured_ZDR / ZDR_max
 *    Approach to 1.0 indicates near-optimal policy
 *
 * MUTUAL EXCLUSION: Still guaranteed by Semaphore(1) (Theorem 1)
 * Q-table policy cannot violate ME (proved in original paper)
 */
public class RLSyncBenchmark {

    // ================================================================
    // WORKLOAD PHASES (Novelty 1)
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

    // The variable workload sequence (paper Figure 6)
    static final Phase[] WORKLOAD_PHASES = {
        Phase.LIGHT, Phase.BURST, Phase.STEADY,
        Phase.IDLE,  Phase.SPIKE, Phase.STEADY
    };

    // ================================================================
    // Configuration
    // ================================================================
    static final int  CS_WORK_US       = 150;
    static final long ZDR_THRESHOLD_NS = 500_000L; // tau = 500us

    // Experiment
    static final int   SEEDS              = 10;
    static final int   STATIC_EVAL_REQS   = 300;
    static final int[] THREAD_COUNTS      = {2, 4, 6, 8, 10, 12};

    // RL Training (offline pre-training)
    static final int    TRAIN_EPISODES    = 400;
    static final int    BURST_PER_EPISODE = 10;
    static final double ALPHA_TRAIN       = 0.15;
    static final double GAMMA             = 0.95;
    static final double EPS_START         = 0.70;
    static final double EPS_END           = 0.05;

    // ================================================================
    // ONLINE ADAPTATION PARAMETERS (Novelty 2)
    // ================================================================
    static final double ALPHA_ONLINE = 0.08;  // smaller LR for stability
    static final double EPS_ONLINE   = 0.05;  // small exploration during eval
    static final int    ADAPT_WINDOW = 30;    // rolling window for adaptation

    // ================================================================
    // State Space
    // ================================================================
    static final int NW_MAX     = 8;
    static final int LB_COUNT   = 5;
    static final int TR_COUNT   = 3;
    static final int PH_HINT    = 3;  // phase hint: LOW, MED, HIGH load
    static final int NUM_STATES = (NW_MAX+1) * 2 * LB_COUNT * TR_COUNT * PH_HINT;

    static final int NUM_ACTIONS = 2; // 0=backoff, 1=admit

    // Backoff (paper Section 4.2)
    static final int BACKOFF_BASE_US = 40;
    static final int BACKOFF_MAX_US  = 600;

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
    // THEORETICAL ZDR BOUND (Novelty 3)
    // ================================================================
    static class ZDRBound {

        static double computeBound(int nThreads, int csUs, int thinkUs,
                                    long zdrThreshNs) {
            double rhoI    = (double) csUs / (csUs + thinkUs);
            double rhoSys  = nThreads * rhoI;
            double tauMs   = zdrThreshNs / 1e6;
            double csMs    = csUs / 1e3;

            double pEmpty = Math.max(0, 1.0 - rhoSys);

            double pRapidComplete;
            if (rhoSys >= 1.0) {
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
            double rhoI   = (double) csUs / (csUs + thinkUs);
            double rhoSys = nThreads * rhoI;
            double csMs   = csUs / 1e3;
            if (rhoSys >= 1.0) {
                return (nThreads - 1) * csMs;
            }
            return (rhoSys * csMs) / (2.0 * (1.0 - rhoSys));
        }
    }

    // ================================================================
    // Metrics with PHASE-AWARE tracking (Novelty 1)
    // ================================================================
    static class Metrics {
        final String         name;
        final List<Long>     samples       = Collections.synchronizedList(
                                                 new ArrayList<>());
        final List<Phase>    phaseLog      = Collections.synchronizedList(
                                                 new ArrayList<>());
        final List<Integer>  reqIndex      = Collections.synchronizedList(
                                                 new ArrayList<>());
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
            return samples.stream().mapToLong(v->v)
                          .average().orElse(0) / 1e6;
        }

        double meanMsInPhase(Phase p) {
            return IntStream.range(0, samples.size())
                            .filter(i -> phaseLog.get(i) == p)
                            .mapToLong(i -> samples.get(i))
                            .average().orElse(0) / 1e6;
        }

        double zdrInPhase(Phase p) {
            long matching = IntStream.range(0, samples.size())
                                     .filter(i -> phaseLog.get(i) == p)
                                     .count();
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

        double contRate() {
            return 100.0 * nCont.get() / Math.max(1, nTotal.get());
        }

        int adaptationRequests(int phaseChangeIndex, double targetMs,
                                double tolerancePct) {
            double tol = targetMs * (1 + tolerancePct / 100.0);
            for (int i = phaseChangeIndex + ADAPT_WINDOW;
                 i < samples.size(); i++) {
                double sum = 0;
                int cnt = 0;
                for (int j = i - ADAPT_WINDOW; j < i; j++) {
                    sum += samples.get(j);
                    cnt++;
                }
                double rollingMean = (sum / cnt) / 1e6;
                if (rollingMean <= tol) {
                    return i - phaseChangeIndex;
                }
            }
            return -1;
        }

        int totalCount() { return samples.size(); }
    }

    // ================================================================
    // CSLock interface (now phase-aware)
    // ================================================================
    interface CSLock {
        long    acquire(boolean rec, Phase currentPhase)
                    throws InterruptedException;
        void    release();
        Metrics metrics();
    }

    // ================================================================
    // Q-Table with PHASE HINT in state (Novelty 1 + 2)
    // ================================================================
    static class QTable {
        final double[][] Q;
        volatile double  ema     = 0.0;
        volatile double  prevEma = 0.0;
        static final double EMA_A = 0.10;

        final AtomicLong onlineUpdates = new AtomicLong(0);

        QTable() {
            Q = new double[NUM_STATES][NUM_ACTIONS];
        }

        QTable(QTable src) {
            Q = new double[NUM_STATES][NUM_ACTIONS];
            for (int s = 0; s < NUM_STATES; s++)
                for (int a = 0; a < NUM_ACTIONS; a++)
                    Q[s][a] = src.Q[s][a];
            ema = src.ema;
            prevEma = src.prevEma;
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
            if (nWait <= 1 && loadBucket() <= 1) return 0; // LOW
            if (nWait <= 4 && loadBucket() <= 3) return 1; // MED
            return 2;                                       // HIGH
        }

        int sid(int nWait, boolean csOcc) {
            int nw = Math.min(nWait, NW_MAX);
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

        int best(int s) {
            return Q[s][1] >= Q[s][0] ? 1 : 0;
        }

        void update(int s, int a, double r, int sp, double alpha) {
            double maxQ = Math.max(Q[sp][0], Q[sp][1]);
            Q[s][a] += alpha * (r + GAMMA * maxQ - Q[s][a]);
        }

        double reward(long wNs) {
            double r = -(wNs / 1e6);
            if (wNs < ZDR_THRESHOLD_NS) r += 0.5; // ZDR bonus
            if (wNs > 8_000_000L)        r -= 1.0; // starvation penalty
            return r;
        }

        void exportCSV(String path) throws IOException {
            try (PrintWriter pw = new PrintWriter(path)) {
                pw.println("state,nWait,csOcc,loadBucket,trend,phaseHint,"
                         + "Q_backoff,Q_admit,policy");
                for (int nw=0; nw<=NW_MAX; nw++)
                    for (int oc=0; oc<=1; oc++)
                        for (int lb=0; lb<LB_COUNT; lb++)
                            for (int tr=0; tr<TR_COUNT; tr++)
                                for (int ph=0; ph<PH_HINT; ph++) {
                                    int s = nw*(2*LB_COUNT*TR_COUNT*PH_HINT)
                                          + oc*(LB_COUNT*TR_COUNT*PH_HINT)
                                          + lb*(TR_COUNT*PH_HINT)
                                          + tr*PH_HINT + ph;
                                    pw.printf(
                                        "%d,%d,%d,%d,%d,%d,%.6f,%.6f,%s%n",
                                        s,nw,oc,lb,tr,ph,
                                        Q[s][0],Q[s][1],
                                        Q[s][1]>=Q[s][0]?"ADMIT":"BACKOFF");
                                }
            }
        }
    }

    // ================================================================
    // RL Arbiter with ONLINE ADAPTATION (Novelty 2)
    // ================================================================
    static class RLArbiter implements CSLock {

        // Structural ME - Semaphore(1)
        final Semaphore     sem      = new Semaphore(1, true);
        final AtomicInteger nWaiting = new AtomicInteger(0);
        final AtomicBoolean csOcc    = new AtomicBoolean(false);

        final QTable       qt;
        final double       eps;
        final double       alpha;
        final boolean      learning;   // online adaptation enabled
        final Metrics      met;
        final List<Double> convLog;

        volatile long epWaitSum = 0;
        volatile int  epCount   = 0;
        final Object  epLock    = new Object();

        final AtomicLong nBackoff = new AtomicLong(0);
        final AtomicLong nAdmit   = new AtomicLong(0);

        RLArbiter(QTable qt, double eps, double alpha, boolean learning,
                  String tag, List<Double> convLog) {
            this.qt=qt; this.eps=eps; this.alpha=alpha;
            this.learning=learning;
            this.met=new Metrics(tag); this.convLog=convLog;
        }

        @Override
        public long acquire(boolean rec, Phase currentPhase)
                throws InterruptedException {
            long    t0  = System.nanoTime();
            int     nw  = nWaiting.incrementAndGet();
            boolean occ = csOcc.get();
            boolean contested = occ || nw > 1;

            int s = qt.sid(nw, occ);

            int action;
            if (learning && ThreadLocalRandom.current().nextDouble() < eps)
                action = ThreadLocalRandom.current().nextInt(NUM_ACTIONS);
            else
                action = qt.best(s);

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

            // -- ONLINE Q-UPDATE (continues during evaluation) --------
            if (learning) {
                int    sp = qt.sid(nWaiting.get(), true);
                double r  = qt.reward(waitNs);
                qt.update(s, action, r, sp, alpha);
                qt.onlineUpdates.incrementAndGet();
                synchronized (epLock) {
                    epWaitSum += waitNs;
                    epCount++;
                }
            }

            if (rec) met.record(waitNs, contested, currentPhase);
            return waitNs;
        }

        @Override
        public void release() {
            csOcc.set(false);
            sem.release();
        }

        @Override
        public Metrics metrics() { return met; }

        void flushEpisode() {
            synchronized (epLock) {
                if (epCount > 0 && convLog != null) {
                    convLog.add(epWaitSum / 1e6 / epCount);
                    epWaitSum = 0;
                    epCount   = 0;
                }
            }
        }
    }

    // ================================================================
    // Baselines (unchanged but now phase-aware)
    // ================================================================
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
                if (Thread.interrupted()) {
                    nd.locked = false;
                    throw new InterruptedException();
                }
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
    // PHASE-AWARE WORKER (Novelty 1)
    // ================================================================
    static class PhaseWorker implements Runnable {
        final CSLock         lk;
        final List<Phase>    phaseSequence;  // shared phase plan
        final int            reqsPerPhase;
        final AtomicInteger  globalPhaseIdx;
        final CountDownLatch done;
        final boolean        rec;
        final AtomicInteger  inside;
        final AtomicBoolean  viol;

        PhaseWorker(CSLock lk, List<Phase> seq, int reqsPerPhase,
                    AtomicInteger phaseIdx, CountDownLatch d,
                    boolean rec, AtomicInteger inside, AtomicBoolean viol) {
            this.lk=lk; this.phaseSequence=seq;
            this.reqsPerPhase=reqsPerPhase;
            this.globalPhaseIdx=phaseIdx;
            this.done=d; this.rec=rec;
            this.inside=inside; this.viol=viol;
        }

        @Override
        public void run() {
            ThreadLocalRandom tlr = ThreadLocalRandom.current();
            try {
                for (int phaseIdx = 0;
                     phaseIdx < phaseSequence.size(); phaseIdx++) {
                    Phase p = phaseSequence.get(phaseIdx);
                    globalPhaseIdx.set(phaseIdx);

                    for (int i = 0; i < reqsPerPhase; i++) {
                        sleepUs(p.thinkBase + tlr.nextLong(p.thinkJitter));
                        lk.acquire(rec, p);
                        int n = inside.incrementAndGet();
                        if (n > 1) viol.set(true);
                        sleepUs(CS_WORK_US);
                        inside.decrementAndGet();
                        lk.release();
                    }
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            } finally {
                done.countDown();
            }
        }
    }

    // ================================================================
    // Static workload worker (for baseline comparison)
    // ================================================================
    static class StaticWorker implements Runnable {
        final CSLock         lk;
        final int            nReqs;
        final int            thinkBase;
        final int            thinkJitter;
        final CountDownLatch done;
        final boolean        rec;
        final AtomicInteger  inside;
        final AtomicBoolean  viol;

        StaticWorker(CSLock lk, int n, int tb, int tj, CountDownLatch d,
                     boolean rec, AtomicInteger inside, AtomicBoolean viol) {
            this.lk=lk; this.nReqs=n;
            this.thinkBase=tb; this.thinkJitter=tj;
            this.done=d; this.rec=rec;
            this.inside=inside; this.viol=viol;
        }

        @Override
        public void run() {
            ThreadLocalRandom tlr = ThreadLocalRandom.current();
            try {
                for (int i = 0; i < nReqs; i++) {
                    sleepUs(thinkBase + tlr.nextLong(thinkJitter));
                    lk.acquire(rec, Phase.STEADY);
                    int n = inside.incrementAndGet();
                    if (n > 1) viol.set(true);
                    sleepUs(CS_WORK_US);
                    inside.decrementAndGet();
                    lk.release();
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            } finally {
                done.countDown();
            }
        }
    }

    // ================================================================
    // Run experiments
    // ================================================================
    static boolean runVariableWorkload(CSLock lk, int nT, int reqsPerPhase,
                                         boolean rec) throws Exception {
        AtomicInteger inside = new AtomicInteger(0);
        AtomicBoolean viol   = new AtomicBoolean(false);
        AtomicInteger pIdx   = new AtomicInteger(0);
        CountDownLatch latch = new CountDownLatch(nT);
        ExecutorService ex   = Executors.newFixedThreadPool(nT);

        List<Phase> seq = Arrays.asList(WORKLOAD_PHASES);
        for (int t = 0; t < nT; t++)
            ex.submit(new PhaseWorker(lk, seq, reqsPerPhase, pIdx,
                                       latch, rec, inside, viol));
        latch.await();
        ex.shutdownNow();
        ex.awaitTermination(10, TimeUnit.SECONDS);
        return !viol.get();
    }

    static boolean runStaticWorkload(CSLock lk, int nT, int nReqs,
                                       int thinkBase, int thinkJitter,
                                       boolean rec) throws Exception {
        AtomicInteger inside = new AtomicInteger(0);
        AtomicBoolean viol   = new AtomicBoolean(false);
        CountDownLatch latch = new CountDownLatch(nT);
        ExecutorService ex   = Executors.newFixedThreadPool(nT);
        for (int t = 0; t < nT; t++)
            ex.submit(new StaticWorker(lk, nReqs, thinkBase, thinkJitter,
                                       latch, rec, inside, viol));
        latch.await();
        ex.shutdownNow();
        ex.awaitTermination(10, TimeUnit.SECONDS);
        return !viol.get();
    }

    // ================================================================
    // OFFLINE TRAINING (pre-trains across all phases)
    // ================================================================
    static QTable trainOffline(int nT, List<Double> convLog) throws Exception {
        QTable qt = new QTable();
        for (int ep = 0; ep < TRAIN_EPISODES; ep++) {
            double p   = (double) ep / TRAIN_EPISODES;
            double eps = EPS_START - p * (EPS_START - EPS_END);
            Phase phase = WORKLOAD_PHASES[ep % WORKLOAD_PHASES.length];
            RLArbiter arb = new RLArbiter(qt, eps, ALPHA_TRAIN, true,
                                           "tr", convLog);
            runStaticWorkload(arb, nT, BURST_PER_EPISODE,
                              phase.thinkBase, phase.thinkJitter, false);
            arb.flushEpisode();
        }
        return qt;
    }

    // ================================================================
    // Statistics
    // ================================================================
    static double mean(double[] a) {
        double s=0; for(double v:a) s+=v; return s/a.length;
    }
    static double var(double[] a) {
        double m=mean(a),s=0;
        for(double v:a) s+=(v-m)*(v-m);
        return s/Math.max(1,a.length-1);
    }
    static double std(double[] a)  { return Math.sqrt(var(a)); }
    static double ci95(double[] a) {
        double[] tc={0,12.706,4.303,3.182,2.776,2.571,2.447,
                       2.365,2.306,2.262,2.228};
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
    static double[] col(double[][] d, int c) {
        return Arrays.stream(d).mapToDouble(r->r[c]).toArray();
    }

    // ================================================================
    // CSV writers
    // ================================================================
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
                    double zdrMax = ZDRBound.computeBound(
                        tc, CS_WORK_US, 120, ZDR_THRESHOLD_NS);
                    double waitMin = ZDRBound.meanWaitBoundMs(
                        tc, CS_WORK_US, 120);
                    double zdrEff = zdrMax > 0 ? 100.0 * zdr/zdrMax : 0;
                    double waitEff = waitMin > 0 ? waitMin/meanMs : 0;
                    pw.printf("%d,%s,%.6f,%.6f,%.6f,%.4f,"
                            + "%.4f,%.4f,%.6f,%.4f%n",
                        tc, m, meanMs, ci95(col(d,0)),
                        mean(col(d,2)), zdr,
                        zdrMax, zdrEff, waitMin, waitEff);
                }
        }
    }

    static void writePhaseResults(String f, String[] methods,
            Map<String,Map<Integer,Map<Phase,double[]>>> R)
            throws IOException {
        try (PrintWriter pw = new PrintWriter(f)) {
            pw.println("threads,method,phase,phase_desc,"
                     + "mean_ms,zdr_pct,phase_think_base,"
                     + "phase_think_jitter");
            for (String m : methods)
                for (var tcE : R.get(m).entrySet()) {
                    int tc = tcE.getKey();
                    for (var pE : tcE.getValue().entrySet()) {
                        Phase p = pE.getKey();
                        double[] vals = pE.getValue();
                        pw.printf("%d,%s,%s,\"%s\",%.6f,%.4f,%d,%d%n",
                            tc,m,p.name(),p.description,
                            vals[0],vals[1],p.thinkBase,p.thinkJitter);
                    }
                }
        }
    }

    static void writeAdaptation(String f,
            Map<Integer,Map<String,int[]>> adapt) throws IOException {
        try (PrintWriter pw = new PrintWriter(f)) {
            pw.println("threads,method,phase_transition,"
                     + "requests_to_adapt,adapted_within_window");
            for (var tcE : adapt.entrySet()) {
                int tc = tcE.getKey();
                for (var mE : tcE.getValue().entrySet()) {
                    String m = mE.getKey();
                    int[] times = mE.getValue();
                    for (int i = 0; i < times.length; i++) {
                        String trans = WORKLOAD_PHASES[i].name()
                                     + "->"
                                     + (i+1 < WORKLOAD_PHASES.length
                                        ? WORKLOAD_PHASES[i+1].name()
                                        : "END");
                        pw.printf("%d,%s,%s,%d,%s%n",
                            tc,m,trans,times[i],
                            times[i]>0?"YES":"NO");
                    }
                }
            }
        }
    }

    static void writeZDRBounds(String f) throws IOException {
        try (PrintWriter pw = new PrintWriter(f)) {
            pw.println("threads,rho_per_thread,rho_system,"
                     + "zdr_theoretical_max_pct,"
                     + "mean_wait_theoretical_min_ms,"
                     + "explanation");
            for (Phase p : Phase.values()) {
                pw.printf("# Phase: %s (think=%dus)%n",
                    p.name(), p.thinkBase);
                for (int nT : THREAD_COUNTS) {
                    double rhoI = (double)CS_WORK_US
                                 /(CS_WORK_US+p.thinkBase);
                    double rhoS = nT * rhoI;
                    double zdr = ZDRBound.computeBound(
                        nT, CS_WORK_US, p.thinkBase, ZDR_THRESHOLD_NS);
                    double wait = ZDRBound.meanWaitBoundMs(
                        nT, CS_WORK_US, p.thinkBase);
                    String expl = rhoS<0.5  ? "LOW load"
                                : rhoS<0.9  ? "MODERATE load"
                                : rhoS<1.0  ? "HIGH load"
                                : "SATURATED";
                    pw.printf("%d,%.4f,%.4f,%.4f,%.6f,%s%n",
                        nT,rhoI,rhoS,zdr,wait,expl);
                }
                pw.println();
            }
        }
    }

    static void writeStats(String f,
            Map<String,Map<Integer,double[][]>> R,
            String mA, String mB) throws IOException {
        String[] mn={"mean","p99","zdr"};
        int[] cols={0,2,3};
        try (PrintWriter pw = new PrintWriter(f)) {
            pw.println("threads,metric,mean_"+mA+",mean_"+mB+","
                     + "improvement_pct,welch_t,cohens_d,sig_p05,"
                     + mA+"_wins");
            for (int tc : THREAD_COUNTS) {
                double[][] da = R.get(mA).get(tc);
                double[][] db = R.get(mB).get(tc);
                if (da==null||db==null) continue;
                for (int mi=0; mi<mn.length; mi++) {
                    double[] a=col(da,cols[mi]), b=col(db,cols[mi]);
                    double t = welchT(a,b);
                    double d = cohensD(a,b);
                    double imp = Math.abs(mean(b))>1e-9
                               ? 100.0*(mean(b)-mean(a))/Math.abs(mean(b))
                               : 0;
                    boolean winsA;
                    if (mn[mi].equals("zdr"))
                        winsA = mean(a) > mean(b);
                    else
                        winsA = mean(a) < mean(b);
                    pw.printf("%d,%s,%.6f,%.6f,%.4f,%.4f,%.4f,%s,%s%n",
                        tc,mn[mi],mean(a),mean(b),imp,t,d,
                        Math.abs(t)>2.0?"YES":"NO",
                        winsA?"YES":"NO");
                }
            }
        }
    }

    // ================================================================
    // MAIN
    // ================================================================
    public static void main(String[] args) throws Exception {

        String dir = args.length>0 ? args[0] : "./rlsync_v4_results";
        Files.createDirectories(Paths.get(dir));

        System.out.println("================================================================");
        System.out.println("  RL-Sync v4: Variable Workload + Online Adaptation + ZDR Bound");
        System.out.println("================================================================");
        System.out.printf(
            "  CS=%dus  ZDR-thresh=%dus  Seeds=%d  Episodes=%d%n",
            CS_WORK_US, (int)(ZDR_THRESHOLD_NS/1000),
            SEEDS, TRAIN_EPISODES);
        System.out.printf("  Threads: %s%n",
            Arrays.toString(THREAD_COUNTS));
        System.out.printf("  Workload phases: %s%n",
            Arrays.toString(WORKLOAD_PHASES));
        System.out.println("================================================================");

        String[] methods = {"RL-Sync","RL-Sync-Online",
                            "FairMutex","CLH","ExpBackoff"};

        // === PHASE 1: Static workload evaluation ===
        System.out.println("\n+- EXPERIMENT 1: Static Workload (paper baseline) --------------+");

        Map<String,Map<Integer,double[][]>> staticR = new LinkedHashMap<>();
        for (String m : methods) staticR.put(m, new TreeMap<>());

        for (int nT : THREAD_COUNTS) {
            System.out.printf("%n  nT=%d  ", nT);

            double[][] rlD   = new double[SEEDS][4]; // mean,std,p99,zdr
            double[][] rlOnD = new double[SEEDS][4];
            double[][] mxD   = new double[SEEDS][4];
            double[][] clD   = new double[SEEDS][4];
            double[][] ebD   = new double[SEEDS][4];

            for (int s = 0; s < SEEDS; s++) {
                System.out.printf("[%d/%d] ", s+1, SEEDS);
                System.out.flush();

                QTable qt = trainOffline(nT, null);

                RLArbiter rl = new RLArbiter(qt, 0, 0, false, "rl", null);
                runStaticWorkload(rl, nT, STATIC_EVAL_REQS, 120, 80, true);
                rlD[s] = new double[]{
                    rl.metrics().meanMs(), rl.metrics().stdMs(),
                    rl.metrics().pctMs(99), rl.metrics().zdr()};

                QTable qtO = new QTable(qt);
                RLArbiter rlO = new RLArbiter(qtO, EPS_ONLINE, ALPHA_ONLINE,
                                               true, "rlOn", null);
                runStaticWorkload(rlO, nT, STATIC_EVAL_REQS, 120, 80, true);
                rlOnD[s] = new double[]{
                    rlO.metrics().meanMs(), rlO.metrics().stdMs(),
                    rlO.metrics().pctMs(99), rlO.metrics().zdr()};

                FairMutex mx = new FairMutex("mx");
                runStaticWorkload(mx, nT, STATIC_EVAL_REQS, 120, 80, true);
                mxD[s] = new double[]{
                    mx.metrics().meanMs(), mx.metrics().stdMs(),
                    mx.metrics().pctMs(99), mx.metrics().zdr()};

                CLHLock clh = new CLHLock("clh");
                runStaticWorkload(clh, nT, STATIC_EVAL_REQS, 120, 80, true);
                clD[s] = new double[]{
                    clh.metrics().meanMs(), clh.metrics().stdMs(),
                    clh.metrics().pctMs(99), clh.metrics().zdr()};

                ExpBackoffLock eb = new ExpBackoffLock("eb");
                runStaticWorkload(eb, nT, STATIC_EVAL_REQS, 120, 80, true);
                ebD[s] = new double[]{
                    eb.metrics().meanMs(), eb.metrics().stdMs(),
                    eb.metrics().pctMs(99), eb.metrics().zdr()};
            }

            staticR.get("RL-Sync")       .put(nT, rlD);
            staticR.get("RL-Sync-Online").put(nT, rlOnD);
            staticR.get("FairMutex")     .put(nT, mxD);
            staticR.get("CLH")           .put(nT, clD);
            staticR.get("ExpBackoff")    .put(nT, ebD);

            double zdrMax = ZDRBound.computeBound(
                nT, CS_WORK_US, 120, ZDR_THRESHOLD_NS);
            System.out.printf(
                "%n  RL=%.3fms(ZDR=%.0f%%) RL-On=%.3fms(ZDR=%.0f%%) "
              + "Mx=%.3fms(ZDR=%.0f%%) ZDR_max=%.0f%%%n",
                mean(col(rlD,0)),mean(col(rlD,3)),
                mean(col(rlOnD,0)),mean(col(rlOnD,3)),
                mean(col(mxD,0)),mean(col(mxD,3)),zdrMax);
        }

        // === PHASE 2: Variable workload evaluation ===
        System.out.println(
            "\n+- EXPERIMENT 2: Variable Workload (Novelty 1+2) -----------+");

        Map<String,Map<Integer,Map<Phase,double[]>>> phaseR
            = new LinkedHashMap<>();
        for (String m : methods) phaseR.put(m, new TreeMap<>());

        Map<Integer,Map<String,int[]>> adaptR = new TreeMap<>();

        for (int nT : THREAD_COUNTS) {
            System.out.printf("%n  nT=%d  Phases: ", nT);

            Map<Phase,double[][]> rlPhase   = new EnumMap<>(Phase.class);
            Map<Phase,double[][]> rlOnPhase = new EnumMap<>(Phase.class);
            Map<Phase,double[][]> mxPhase   = new EnumMap<>(Phase.class);
            for (Phase p : WORKLOAD_PHASES) {
                rlPhase.put  (p, new double[SEEDS][2]);
                rlOnPhase.put(p, new double[SEEDS][2]);
                mxPhase.put  (p, new double[SEEDS][2]);
            }

            int[] rlAdapt   = new int[WORKLOAD_PHASES.length];
            int[] rlOnAdapt = new int[WORKLOAD_PHASES.length];
            int[] mxAdapt   = new int[WORKLOAD_PHASES.length];

            for (int s = 0; s < SEEDS; s++) {
                System.out.printf("[%d/%d]", s+1, SEEDS);
                System.out.flush();

                QTable qt = trainOffline(nT, null);

                RLArbiter rl = new RLArbiter(qt, 0, 0, false, "rl_var", null);
                runVariableWorkload(rl, nT, 50, true);
                for (Phase p : WORKLOAD_PHASES) {
                    rlPhase.get(p)[s][0] = rl.metrics().meanMsInPhase(p);
                    rlPhase.get(p)[s][1] = rl.metrics().zdrInPhase(p);
                }

                QTable qtO = new QTable(qt);
                RLArbiter rlO = new RLArbiter(qtO, EPS_ONLINE, ALPHA_ONLINE,
                                               true, "rlOn_var", null);
                runVariableWorkload(rlO, nT, 50, true);
                for (Phase p : WORKLOAD_PHASES) {
                    rlOnPhase.get(p)[s][0] = rlO.metrics().meanMsInPhase(p);
                    rlOnPhase.get(p)[s][1] = rlO.metrics().zdrInPhase(p);
                }

                FairMutex mx = new FairMutex("mx_var");
                runVariableWorkload(mx, nT, 50, true);
                for (Phase p : WORKLOAD_PHASES) {
                    mxPhase.get(p)[s][0] = mx.metrics().meanMsInPhase(p);
                    mxPhase.get(p)[s][1] = mx.metrics().zdrInPhase(p);
                }

                if (s == 0) {
                    int reqsPerPhase = 50 * nT;
                    for (int i = 0; i < WORKLOAD_PHASES.length-1; i++) {
                        int changeIdx = (i+1) * reqsPerPhase;
                        double rlOnPhaseMean = mean(col(
                            rlOnPhase.get(WORKLOAD_PHASES[i+1]), 0));
                        double mxPhaseMean = mean(col(
                            mxPhase.get(WORKLOAD_PHASES[i+1]), 0));
                        rlOnAdapt[i] = rlO.metrics().adaptationRequests(
                            changeIdx, rlOnPhaseMean, 10);
                        mxAdapt[i]   = mx.metrics().adaptationRequests(
                            changeIdx, mxPhaseMean, 10);
                        rlAdapt[i]   = rl.metrics().adaptationRequests(
                            changeIdx, mean(col(
                                rlPhase.get(WORKLOAD_PHASES[i+1]),0)), 10);
                    }
                }
            }

            Map<Phase,double[]> rlAgg   = new EnumMap<>(Phase.class);
            Map<Phase,double[]> rlOnAgg = new EnumMap<>(Phase.class);
            Map<Phase,double[]> mxAgg   = new EnumMap<>(Phase.class);

            for (Phase p : WORKLOAD_PHASES) {
                rlAgg.put  (p, new double[]{
                    mean(col(rlPhase.get(p),0)),
                    mean(col(rlPhase.get(p),1))});
                rlOnAgg.put(p, new double[]{
                    mean(col(rlOnPhase.get(p),0)),
                    mean(col(rlOnPhase.get(p),1))});
                mxAgg.put  (p, new double[]{
                    mean(col(mxPhase.get(p),0)),
                    mean(col(mxPhase.get(p),1))});
            }
            phaseR.get("RL-Sync")       .put(nT, rlAgg);
            phaseR.get("RL-Sync-Online").put(nT, rlOnAgg);
            phaseR.get("FairMutex")     .put(nT, mxAgg);

            Map<String,int[]> ad = new LinkedHashMap<>();
            ad.put("RL-Sync", rlAdapt);
            ad.put("RL-Sync-Online", rlOnAdapt);
            ad.put("FairMutex", mxAdapt);
            adaptR.put(nT, ad);

            System.out.printf("%n  Phase results:%n");
            for (Phase p : WORKLOAD_PHASES) {
                System.out.printf(
                    "    %s: RL=%.3fms RL-On=%.3fms Mx=%.3fms%n",
                    p.name(),
                    rlAgg.get(p)[0], rlOnAgg.get(p)[0], mxAgg.get(p)[0]);
            }
        }

        // === Write all outputs ===
        System.out.println("\n-- Writing all results -------------------------");

        writeStaticSweep(dir+"/static_workload_results.csv",
                          methods, staticR);
        System.out.println("  [OK] static_workload_results.csv");

        writePhaseResults(dir+"/variable_workload_per_phase.csv",
                           methods, phaseR);
        System.out.println("  [OK] variable_workload_per_phase.csv");

        writeAdaptation(dir+"/adaptation_times.csv", adaptR);
        System.out.println("  [OK] adaptation_times.csv");

        writeZDRBounds(dir+"/theoretical_bounds.csv");
        System.out.println("  [OK] theoretical_bounds.csv");

        writeStats(dir+"/stats_rl_vs_mutex_static.csv",
                   staticR, "RL-Sync", "FairMutex");
        writeStats(dir+"/stats_rlonline_vs_mutex_static.csv",
                   staticR, "RL-Sync-Online", "FairMutex");
        writeStats(dir+"/stats_rlonline_vs_rl_static.csv",
                   staticR, "RL-Sync-Online", "RL-Sync");
        System.out.println("  [OK] stats_*.csv");

        // === Honest summary ===
        System.out.println("\n==== NOVELTY VERIFICATION ====================");

        System.out.println("\nNovelty 1 (Variable Workload):");
        System.out.println("  Did RL-Online beat FairMutex in BURST/SPIKE phases?");
        for (int nT : THREAD_COUNTS) {
            double rlOnBurst = phaseR.get("RL-Sync-Online")
                                     .get(nT).get(Phase.BURST)[1]; // ZDR
            double mxBurst   = phaseR.get("FairMutex")
                                     .get(nT).get(Phase.BURST)[1];
            System.out.printf(
                "  nT=%d BURST ZDR: RL-On=%.1f%% Mx=%.1f%% [%s]%n",
                nT, rlOnBurst, mxBurst,
                rlOnBurst>mxBurst?"RL WINS":"RL loses");
        }

        System.out.println("\nNovelty 2 (Online Adaptation):");
        System.out.println("  Adaptation requests after each phase change (lower=better):");
        for (var tcE : adaptR.entrySet()) {
            System.out.printf("  nT=%d:%n", tcE.getKey());
            for (var mE : tcE.getValue().entrySet()) {
                System.out.printf("    %s: %s%n",
                    mE.getKey(), Arrays.toString(mE.getValue()));
            }
        }

        System.out.println("\nNovelty 3 (Theoretical ZDR Bound):");
        System.out.println("  ZDR efficiency (measured/theoretical, %):");
        for (int nT : THREAD_COUNTS) {
            double rlZdr = mean(col(staticR.get("RL-Sync").get(nT), 3));
            double mxZdr = mean(col(staticR.get("FairMutex").get(nT), 3));
            double zdrMax = ZDRBound.computeBound(
                nT, CS_WORK_US, 120, ZDR_THRESHOLD_NS);
            double rlEff = zdrMax>0 ? 100*rlZdr/zdrMax : 0;
            double mxEff = zdrMax>0 ? 100*mxZdr/zdrMax : 0;
            System.out.printf(
                "  nT=%d: ZDR_max=%.1f%% RL_eff=%.1f%% Mx_eff=%.1f%%%n",
                nT, zdrMax, rlEff, mxEff);
        }

        System.out.println("\n  Output -> " + dir);

        System.out.println(
            "================================================================");
    }
}
