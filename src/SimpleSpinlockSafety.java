import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

/** Same safety harness pattern as every other lock in this
 *  investigation, applied to TASLock, TATASLock, and TicketLock. */
public class SimpleSpinlockSafety {
    static final AtomicInteger inside = new AtomicInteger(0);
    static final AtomicInteger peak = new AtomicInteger(0);
    static final AtomicBoolean violated = new AtomicBoolean(false);
    static final AtomicInteger completed = new AtomicInteger(0);
    static long[] sharedCounter = new long[1];

    static final int[] THREAD_COUNTS = {1, 2, 4, 8, 24, 64};
    static final int REQS_PER_THREAD = 300;
    static final int TRIALS = 4;
    static boolean anyFail = false;

    public static void main(String[] args) throws Exception {
        System.out.println("=== TASLock ===");
        runAll(nT -> new TASLockRunner(nT));
        System.out.println("=== TATASLock ===");
        runAll(nT -> new TATASLockRunner(nT));
        System.out.println("=== TicketLock ===");
        runAll(nT -> new TicketLockRunner(nT));

        System.out.println();
        System.out.println(anyFail ? "OVERALL: FAIL" : "OVERALL: PASS");
    }

    interface RunnerFactory { Runner make(int nT); }
    interface Runner {
        boolean run() throws Exception; // returns true on pass
    }

    static void runAll(RunnerFactory factory) throws Exception {
        for (int trial = 0; trial < TRIALS; trial++) {
            for (int nT : THREAD_COUNTS) {
                Runner r = factory.make(nT);
                boolean pass = r.run();
                if (!pass) anyFail = true;
            }
        }
    }

    static class TASLockRunner implements Runner {
        final int nT;
        TASLockRunner(int nT) { this.nT = nT; }
        public boolean run() throws Exception {
            inside.set(0); peak.set(0); violated.set(false); completed.set(0);
            sharedCounter[0] = 0;
            TASLock lk = new TASLock();
            CountDownLatch latch = new CountDownLatch(nT);
            ExecutorService ex = Executors.newFixedThreadPool(nT);
            int totalReqs = nT * REQS_PER_THREAD;
            for (int t = 0; t < nT; t++) {
                ex.submit(() -> {
                    try {
                        for (int i = 0; i < REQS_PER_THREAD; i++) {
                            lk.execute(SimpleSpinlockSafety::criticalSection);
                            completed.incrementAndGet();
                        }
                    } catch (InterruptedException ie) { Thread.currentThread().interrupt(); }
                    finally { latch.countDown(); }
                });
            }
            boolean finished = latch.await(60, TimeUnit.SECONDS);
            ex.shutdownNow(); ex.awaitTermination(5, TimeUnit.SECONDS);
            return report("TASLock", nT, finished, totalReqs);
        }
    }

    static class TATASLockRunner implements Runner {
        final int nT;
        TATASLockRunner(int nT) { this.nT = nT; }
        public boolean run() throws Exception {
            inside.set(0); peak.set(0); violated.set(false); completed.set(0);
            sharedCounter[0] = 0;
            TATASLock lk = new TATASLock();
            CountDownLatch latch = new CountDownLatch(nT);
            ExecutorService ex = Executors.newFixedThreadPool(nT);
            int totalReqs = nT * REQS_PER_THREAD;
            for (int t = 0; t < nT; t++) {
                ex.submit(() -> {
                    try {
                        for (int i = 0; i < REQS_PER_THREAD; i++) {
                            lk.execute(SimpleSpinlockSafety::criticalSection);
                            completed.incrementAndGet();
                        }
                    } catch (InterruptedException ie) { Thread.currentThread().interrupt(); }
                    finally { latch.countDown(); }
                });
            }
            boolean finished = latch.await(60, TimeUnit.SECONDS);
            ex.shutdownNow(); ex.awaitTermination(5, TimeUnit.SECONDS);
            return report("TATASLock", nT, finished, totalReqs);
        }
    }

    static class TicketLockRunner implements Runner {
        final int nT;
        TicketLockRunner(int nT) { this.nT = nT; }
        public boolean run() throws Exception {
            inside.set(0); peak.set(0); violated.set(false); completed.set(0);
            sharedCounter[0] = 0;
            TicketLock lk = new TicketLock();
            CountDownLatch latch = new CountDownLatch(nT);
            ExecutorService ex = Executors.newFixedThreadPool(nT);
            int totalReqs = nT * REQS_PER_THREAD;
            for (int t = 0; t < nT; t++) {
                ex.submit(() -> {
                    try {
                        for (int i = 0; i < REQS_PER_THREAD; i++) {
                            lk.execute(SimpleSpinlockSafety::criticalSection);
                            completed.incrementAndGet();
                        }
                    } catch (InterruptedException ie) { Thread.currentThread().interrupt(); }
                    finally { latch.countDown(); }
                });
            }
            boolean finished = latch.await(60, TimeUnit.SECONDS);
            ex.shutdownNow(); ex.awaitTermination(5, TimeUnit.SECONDS);
            return report("TicketLock", nT, finished, totalReqs);
        }
    }

    static void criticalSection() {
        int n = inside.incrementAndGet();
        peak.updateAndGet(cur -> Math.max(cur, n));
        if (n > 1) violated.set(true);
        long v = sharedCounter[0];
        long spin = 0;
        for (int k = 0; k < 50; k++) spin += k;
        sharedCounter[0] = v + 1 + (spin - spin);
        inside.decrementAndGet();
    }

    static boolean report(String name, int nT, boolean finished, int totalReqs) {
        boolean counterOk = sharedCounter[0] == totalReqs;
        boolean pass = finished && !violated.get() && completed.get() == totalReqs && counterOk;
        System.out.printf(
            "  %-11s nT=%2d finished=%-5s completed=%d/%d peakInside=%d violated=%s counter=%d/%d counterOk=%s -> %s%n",
            name, nT, finished, completed.get(), totalReqs, peak.get(), violated.get(),
            sharedCounter[0], totalReqs, counterOk, pass ? "PASS" : "FAIL");
        return pass;
    }
}
