import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

/**
 * Focused diagnostic: does RLSyncBenchmark's RLArbiter (Semaphore(1)-backed)
 * ever violate mutual exclusion, and does the harness survive as thread
 * count grows well beyond the paper's 2-12 range (16, 32, 64, 128)?
 *
 * This reuses the exact same classes as RLSyncBenchmark by re-implementing
 * a minimal harness that DOES check+report the violation flag that the
 * original main() silently discards.
 */
public class MeStressTest {

    public static void main(String[] args) throws Exception {
        int[] threadCounts = {2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128};
        int reqsPerThread = 400;

        System.out.println("threads,method,violation,wall_ms,mean_wait_ms,p99_ms,zdr_pct,throughput_ops_s");

        for (int nT : threadCounts) {
            // Train a policy sized for a moderate thread count once, reuse.
            RLSyncBenchmark.QTable qt = RLSyncBenchmark.trainOffline(Math.min(nT, 12), null);

            runOne("RL-Sync-Frozen", () -> new RLSyncBenchmark.RLArbiter(qt, 0, 0, false, "rl", null),
                   nT, reqsPerThread);

            RLSyncBenchmark.QTable qtO = new RLSyncBenchmark.QTable(qt);
            runOne("RL-Sync-Online", () -> new RLSyncBenchmark.RLArbiter(qtO, RLSyncBenchmark.EPS_ONLINE,
                       RLSyncBenchmark.ALPHA_ONLINE, true, "rlOn", null),
                   nT, reqsPerThread);

            runOne("FairMutex", () -> new RLSyncBenchmark.FairMutex("mx"), nT, reqsPerThread);
            runOne("CLH", () -> new RLSyncBenchmark.CLHLock("clh"), nT, reqsPerThread);
            runOne("ExpBackoff", () -> new RLSyncBenchmark.ExpBackoffLock("eb"), nT, reqsPerThread);
        }
    }

    interface LockFactory { RLSyncBenchmark.CSLock make(); }

    static void runOne(String label, LockFactory factory, int nT, int nReqs) throws Exception {
        RLSyncBenchmark.CSLock lk = factory.make();
        AtomicInteger inside = new AtomicInteger(0);
        AtomicBoolean viol = new AtomicBoolean(false);
        AtomicLong maxObservedInside = new AtomicLong(0);
        CountDownLatch latch = new CountDownLatch(nT);

        long t0 = System.nanoTime();
        ExecutorService ex;
        try {
            ex = Executors.newFixedThreadPool(nT);
        } catch (Throwable t) {
            System.out.printf("%d,%s,EXCEPTION_THREADPOOL,-1,-1,-1,-1,-1%n", nT, label);
            System.out.printf("  -> %s%n", t);
            return;
        }

        try {
            for (int i = 0; i < nT; i++) {
                ex.submit(() -> {
                    ThreadLocalRandom tlr = ThreadLocalRandom.current();
                    try {
                        for (int r = 0; r < nReqs; r++) {
                            RLSyncBenchmark.sleepUs(50 + tlr.nextLong(50));
                            lk.acquire(true, RLSyncBenchmark.Phase.STEADY);
                            int n = inside.incrementAndGet();
                            if (n > 1) viol.set(true);
                            maxObservedInside.updateAndGet(cur -> Math.max(cur, n));
                            RLSyncBenchmark.sleepUs(RLSyncBenchmark.CS_WORK_US);
                            inside.decrementAndGet();
                            lk.release();
                        }
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    } catch (Throwable t) {
                        System.out.println("  WORKER EXCEPTION: " + t);
                    } finally {
                        latch.countDown();
                    }
                });
            }

            boolean finished = latch.await(120, TimeUnit.SECONDS);
            long wallMs = (System.nanoTime() - t0) / 1_000_000;

            if (!finished) {
                System.out.printf("%d,%s,TIMEOUT_HANG,%d,-1,-1,-1,-1%n", nT, label, wallMs);
                ex.shutdownNow();
                return;
            }

            RLSyncBenchmark.Metrics m = lk.metrics();
            double meanMs = m.meanMs();
            double p99 = m.pctMs(99);
            double zdr = m.zdr();
            double throughput = (nT * nReqs) / (wallMs / 1000.0);

            System.out.printf("%d,%s,%s,%d,%.4f,%.4f,%.2f,%.1f%n",
                nT, label, viol.get() ? "VIOLATION(maxInside="+maxObservedInside.get()+")" : "OK",
                wallMs, meanMs, p99, zdr, throughput);

        } finally {
            ex.shutdownNow();
            ex.awaitTermination(5, TimeUnit.SECONDS);
        }
    }
}
