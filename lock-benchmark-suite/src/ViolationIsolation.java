import java.util.*;
import java.util.concurrent.*;

/**
 * Isolates exactly which lock implementation (if any) shows a genuine
 * mutual-exclusion violation at nT=32, after fixing the
 * chk.exit()/lk.release() ordering bug (try/finally) that could produce a
 * false positive if a worker thread was interrupted mid-critical-section
 * by the harness's 120s timeout / shutdownNow().
 *
 * Runs each lock type in isolation, single seed, verbose peak-occupancy
 * reporting, repeated 5x to rule out a rare race.
 */
public class ViolationIsolation {
    public static void main(String[] args) throws Exception {
        int nT = 32;
        int reqs = 250;

        for (int trial = 0; trial < 5; trial++) {
            System.out.println("=== TRIAL " + trial + " ===");

            RLSyncBenchmarkFixed.QTable qtBase = RLSyncBenchmarkFixed.trainOffline(12);

            test("RL-Sync-Unguarded", new RLSyncBenchmarkFixed.RLArbiter(
                new RLSyncBenchmarkFixed.QTable(qtBase), 0, 0, false, "rlUn"), nT, reqs);

            test("RL-Sync-Guarded", new RLSyncBenchmarkFixed.RLArbiter(
                new RLSyncBenchmarkFixed.QTable(qtBase), 0, 0, false, "rlGd", true, nT), nT, reqs);

            test("FairMutex", new RLSyncBenchmarkFixed.FairMutex("mx"), nT, reqs);
        }
    }

    static void test(String label, RLSyncBenchmarkFixed.CSLock lk, int nT, int reqs) throws Exception {
        RLSyncBenchmarkFixed.MEChecker chk = new RLSyncBenchmarkFixed.MEChecker();
        CountDownLatch latch = new CountDownLatch(nT);
        ExecutorService ex = Executors.newFixedThreadPool(nT);
        long t0 = System.nanoTime();
        for (int i = 0; i < nT; i++)
            ex.submit(new RLSyncBenchmarkFixed.StaticWorker(lk, reqs, 120, 80, latch, true, chk));
        boolean finished = latch.await(60, TimeUnit.SECONDS);
        long wallMs = (System.nanoTime() - t0) / 1_000_000;
        ex.shutdownNow();
        ex.awaitTermination(5, TimeUnit.SECONDS);

        System.out.printf("  %-20s finished=%s wallMs=%d peakInside=%d violated=%s totalChecks=%d%n",
            label, finished, wallMs, chk.peak.get(), chk.violated.get(), chk.totalChecks.get());
    }
}
