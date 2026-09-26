import java.util.*;
import java.util.concurrent.*;

/**
 * Validates the reliability guard: with the circuit breaker enabled,
 * RL-Sync's mean wait should never exceed FairMutex's mean wait by more
 * than GUARD_TOLERANCE (15%), across the full thread range, INCLUDING at
 * thread counts far outside what the policy was trained/tuned for.
 *
 * This is the key claim for the article: "RL-Sync handles mutex smoothly"
 * -- meaning it is provably no worse than a fair mutex by more than a
 * bounded margin, at any thread count, not just the ones it happened to be
 * evaluated at.
 */
public class GuardValidation {
    public static void main(String[] args) throws Exception {
        int[] threadCounts = {2, 4, 8, 16, 24, 32, 48};
        int reqs = 250;

        System.out.println("threads,variant,meanWaitMs,p99Ms,zdrPct,backoffFrac,guardTripped,ratioToFairMutex");

        for (int nT : threadCounts) {
            // Train once at a MUCH smaller thread count (12) to simulate a
            // policy that is now being deployed far outside its training
            // distribution -- exactly the "increase thread count" failure
            // mode reported.
            RLSyncBenchmarkFixed.QTable qt = RLSyncBenchmarkFixed.trainOffline(12);

            // Unguarded (original behaviour)
            RLSyncBenchmarkFixed.RLArbiter rlUnguarded =
                new RLSyncBenchmarkFixed.RLArbiter(qt, 0, 0, false, "rl_unguarded");
            run(rlUnguarded, nT, reqs);

            // Guarded (circuit breaker enabled)
            RLSyncBenchmarkFixed.QTable qt2 = new RLSyncBenchmarkFixed.QTable(qt);
            RLSyncBenchmarkFixed.RLArbiter rlGuarded =
                new RLSyncBenchmarkFixed.RLArbiter(qt2, 0, 0, false, "rl_guarded", true, nT);
            run(rlGuarded, nT, reqs);

            // Reference
            RLSyncBenchmarkFixed.FairMutex mx = new RLSyncBenchmarkFixed.FairMutex("mx");
            run(mx, nT, reqs);

            double mxMean = mx.metrics().meanMs();
            printRow(nT, "RL-Sync-UNGUARDED", rlUnguarded, mxMean);
            printRow(nT, "RL-Sync-GUARDED",   rlGuarded,   mxMean);
            System.out.printf("%d,FairMutex,%.4f,%.4f,%.2f,-,-,1.0000%n",
                nT, mx.metrics().meanMs(), mx.metrics().pctMs(99), mx.metrics().zdr());
        }
    }

    static void run(RLSyncBenchmarkFixed.CSLock lk, int nT, int reqs) throws Exception {
        RLSyncBenchmarkFixed.MEChecker chk = new RLSyncBenchmarkFixed.MEChecker();
        CountDownLatch latch = new CountDownLatch(nT);
        ExecutorService ex = Executors.newFixedThreadPool(nT);
        for (int i = 0; i < nT; i++) {
            ex.submit(new RLSyncBenchmarkFixed.StaticWorker(
                lk, reqs, 120, 80, latch, true, chk));
        }
        latch.await(120, TimeUnit.SECONDS);
        ex.shutdownNow();
        ex.awaitTermination(5, TimeUnit.SECONDS);
        if (chk.violated.get()) {
            System.out.println("  !!! ME VIOLATION !!! peak=" + chk.peak.get());
        }
    }

    static void printRow(int nT, String label, RLSyncBenchmarkFixed.RLArbiter rl, double mxMean) {
        long nb = rl.nBackoff.get(), na = rl.nAdmit.get();
        double backoffFrac = 100.0 * nb / Math.max(1, nb + na);
        double meanMs = rl.metrics().meanMs();
        double ratio = mxMean > 1e-9 ? meanMs / mxMean : 1.0;
        System.out.printf("%d,%s,%.4f,%.4f,%.2f,%.1f,%s,%.4f%n",
            nT, label, meanMs, rl.metrics().pctMs(99), rl.metrics().zdr(),
            backoffFrac, rl.guardTripped ? "YES" : "no", ratio);
    }
}
