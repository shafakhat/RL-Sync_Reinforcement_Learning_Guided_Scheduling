import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

/**
 * Tests the STRUCTURAL hypothesis behind why RL-Sync can't beat FairMutex
 * no matter how well-trained the policy is:
 *
 *   Semaphore(1, true) is a strict FIFO/AQS queue. A thread's position in
 *   that queue is fixed at the moment it calls sem.acquire(). Backing off
 *   BEFORE calling acquire() only delays when you join the queue -- it does
 *   not, and structurally cannot, reduce contention for anyone else, because
 *   there is no retry-storm / CAS-thrash cost in a park-based fair queue
 *   (unlike spin locks / CAS locks, where reducing concurrent retries
 *   genuinely helps).
 *
 * Prediction: an RLArbiter with the backoff action forcibly disabled
 * (always admits immediately) should perform STATISTICALLY IDENTICALLY to
 * FairMutex, because at that point RL-Sync literally reduces to "call
 * Semaphore.acquire() immediately", which is functionally the same
 * admission discipline as ReentrantLock(fair=true).
 */
public class StructuralTest {
    public static void main(String[] args) throws Exception {
        int[] threadCounts = {2, 8, 16, 32};
        int reqs = 300;

        System.out.println("threads,variant,meanWaitMs,p99Ms,zdrPct");

        for (int nT : threadCounts) {
            // Variant A: normal trained RL-Sync (backoff enabled)
            RLSyncBenchmarkFixed.QTable qt = RLSyncBenchmarkFixed.trainOffline(nT);
            RLSyncBenchmarkFixed.RLArbiter rlNormal =
                new RLSyncBenchmarkFixed.RLArbiter(qt, 0, 0, false, "rl_normal");
            run(rlNormal, nT, reqs);
            printRow(nT, "RL-Sync(trained-policy)", rlNormal.metrics());

            // Variant B: RL-Sync with backoff action forcibly disabled
            // (force action=1/admit always, by using an eps=0 QTable whose
            //  Q[*][1] is always >= Q[*][0])
            RLSyncBenchmarkFixed.QTable qtForced = new RLSyncBenchmarkFixed.QTable();
            // leave all Q at 0,0 -> best() returns 1 (admit) since Q[s][1]>=Q[s][0] ties to admit
            RLSyncBenchmarkFixed.RLArbiter rlForced =
                new RLSyncBenchmarkFixed.RLArbiter(qtForced, 0, 0, false, "rl_forced_admit");
            run(rlForced, nT, reqs);
            printRow(nT, "RL-Sync(backoff-disabled)", rlForced.metrics());

            // Variant C: FairMutex baseline
            RLSyncBenchmarkFixed.FairMutex mx = new RLSyncBenchmarkFixed.FairMutex("mx");
            run(mx, nT, reqs);
            printRow(nT, "FairMutex", mx.metrics());

            System.out.println();
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

    static void printRow(int nT, String label, RLSyncBenchmarkFixed.Metrics m) {
        System.out.printf("%d,%s,%.4f,%.4f,%.2f%n",
            nT, label, m.meanMs(), m.pctMs(99), m.zdr());
    }
}
