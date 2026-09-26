import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

/**
 * Diagnoses WHY RL-Sync's mean wait / wall-time grows faster than FairMutex's
 * as thread count increases (root cause of "fails when thread number is
 * increased").
 *
 * Hypothesis: the learned policy over-uses the BACKOFF action at high
 * nWaiting because:
 *   (a) state nWait is capped at NW_MAX=8, so it can't tell 9 waiters from
 *       128 waiters,
 *   (b) offline training happens at the SAME thread count as eval, with only
 *       BURST_PER_EPISODE=10 requests/episode -- far too few samples to
 *       converge a stable policy when nT is large,
 *   (c) backoff delay = min(40*(1+nw), 600) us is added *in addition* to the
 *       fair Semaphore(1) queueing delay, so every backoff decision is pure
 *       extra latency once the system is already saturated (no throughput
 *       benefit from delaying admission when the queue is fair FIFO).
 */
public class PolicyDiagnostic {
    public static void main(String[] args) throws Exception {
        int[] threadCounts = {2, 8, 16, 32, 64, 128};
        int reqsPerThread = 300;

        System.out.println("threads,nBackoff,nAdmit,backoffFrac,meanWaitMs,fairMutexMeanMs,overheadMs");

        for (int nT : threadCounts) {
            RLSyncBenchmark.QTable qt = RLSyncBenchmark.trainOffline(nT, null);

            RLSyncBenchmark.RLArbiter rl = new RLSyncBenchmark.RLArbiter(
                qt, 0, 0, false, "rl", null);

            AtomicInteger inside = new AtomicInteger(0);
            AtomicBoolean viol = new AtomicBoolean(false);
            CountDownLatch latch = new CountDownLatch(nT);
            ExecutorService ex = Executors.newFixedThreadPool(nT);
            for (int i = 0; i < nT; i++) {
                ex.submit(() -> {
                    ThreadLocalRandom tlr = ThreadLocalRandom.current();
                    try {
                        for (int r = 0; r < reqsPerThread; r++) {
                            RLSyncBenchmark.sleepUs(50 + tlr.nextLong(50));
                            rl.acquire(true, RLSyncBenchmark.Phase.STEADY);
                            int n = inside.incrementAndGet();
                            if (n > 1) viol.set(true);
                            RLSyncBenchmark.sleepUs(RLSyncBenchmark.CS_WORK_US);
                            inside.decrementAndGet();
                            rl.release();
                        }
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    } finally {
                        latch.countDown();
                    }
                });
            }
            latch.await(120, TimeUnit.SECONDS);
            ex.shutdownNow();

            long nb = rl.nBackoff.get();
            long na = rl.nAdmit.get();
            double backoffFrac = 100.0 * nb / Math.max(1, nb + na);
            double meanWaitMs = rl.metrics().meanMs();

            // Reference: FairMutex under identical workload
            RLSyncBenchmark.FairMutex mx = new RLSyncBenchmark.FairMutex("mx");
            AtomicInteger inside2 = new AtomicInteger(0);
            AtomicBoolean viol2 = new AtomicBoolean(false);
            CountDownLatch latch2 = new CountDownLatch(nT);
            ExecutorService ex2 = Executors.newFixedThreadPool(nT);
            for (int i = 0; i < nT; i++) {
                ex2.submit(() -> {
                    ThreadLocalRandom tlr = ThreadLocalRandom.current();
                    try {
                        for (int r = 0; r < reqsPerThread; r++) {
                            RLSyncBenchmark.sleepUs(50 + tlr.nextLong(50));
                            mx.acquire(true, RLSyncBenchmark.Phase.STEADY);
                            int n = inside2.incrementAndGet();
                            if (n > 1) viol2.set(true);
                            RLSyncBenchmark.sleepUs(RLSyncBenchmark.CS_WORK_US);
                            inside2.decrementAndGet();
                            mx.release();
                        }
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    } finally {
                        latch2.countDown();
                    }
                });
            }
            latch2.await(120, TimeUnit.SECONDS);
            ex2.shutdownNow();
            double mxMeanMs = mx.metrics().meanMs();

            System.out.printf("%d,%d,%d,%.1f,%.4f,%.4f,%.4f%n",
                nT, nb, na, backoffFrac, meanWaitMs, mxMeanMs, meanWaitMs - mxMeanMs);

            // Dump learned Q-table policy distribution for this nT
            int admitCount = 0, backoffCount = 0;
            for (int nw = 0; nw <= RLSyncBenchmark.NW_MAX; nw++) {
                for (int oc = 0; oc <= 1; oc++) {
                    for (int lb = 0; lb < RLSyncBenchmark.LB_COUNT; lb++) {
                        for (int tr = 0; tr < RLSyncBenchmark.TR_COUNT; tr++) {
                            for (int ph = 0; ph < RLSyncBenchmark.PH_HINT; ph++) {
                                int s = nw*(2*RLSyncBenchmark.LB_COUNT*RLSyncBenchmark.TR_COUNT*RLSyncBenchmark.PH_HINT)
                                      + oc*(RLSyncBenchmark.LB_COUNT*RLSyncBenchmark.TR_COUNT*RLSyncBenchmark.PH_HINT)
                                      + lb*(RLSyncBenchmark.TR_COUNT*RLSyncBenchmark.PH_HINT)
                                      + tr*RLSyncBenchmark.PH_HINT + ph;
                                if (qt.best(s) == 1) admitCount++; else backoffCount++;
                            }
                        }
                    }
                }
            }
            System.out.printf("  -> learned policy: %d/%d states prefer ADMIT, %d/%d prefer BACKOFF%n",
                admitCount, admitCount+backoffCount, backoffCount, admitCount+backoffCount);
        }
    }
}
