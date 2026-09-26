import java.util.*;
import java.util.concurrent.*;

/**
 * Targeted stress test to characterize the single CLH violation found in
 * FinalSweep v2 (nT=24, seed=6). Runs the CLH baseline lock ONLY, at
 * nT=24, many independent trials, to determine reproduction rate and
 * whether it's a genuine race in the CLH implementation under heavy
 * (12x) thread oversubscription on this 2-core sandbox, vs a one-off
 * fluke.
 */
public class CLHStress {
    public static void main(String[] args) throws Exception {
        int nT = 24;
        int reqs = 250;
        int trials = 30;
        int violations = 0;

        for (int trial = 0; trial < trials; trial++) {
            RLSyncBenchmarkFixed.CLHLock clh = new RLSyncBenchmarkFixed.CLHLock("clh");
            RLSyncBenchmarkFixed.MEChecker chk = new RLSyncBenchmarkFixed.MEChecker();
            CountDownLatch latch = new CountDownLatch(nT);
            ExecutorService ex = Executors.newFixedThreadPool(nT);
            long t0 = System.nanoTime();
            for (int i = 0; i < nT; i++)
                ex.submit(new RLSyncBenchmarkFixed.StaticWorker(clh, reqs, 120, 80, latch, true, chk));
            boolean finished = latch.await(120, TimeUnit.SECONDS);
            long wallMs = (System.nanoTime() - t0) / 1_000_000;
            ex.shutdownNow();
            ex.awaitTermination(5, TimeUnit.SECONDS);

            boolean v = chk.violated.get();
            if (v) violations++;
            System.out.printf("trial=%2d finished=%s wallMs=%6d peakInside=%d violated=%s totalChecks=%d%n",
                trial, finished, wallMs, chk.peak.get(), v, chk.totalChecks.get());
        }
        System.out.println();
        System.out.printf("Violations: %d / %d trials (%.1f%%)%n", violations, trials, 100.0*violations/trials);
    }
}
