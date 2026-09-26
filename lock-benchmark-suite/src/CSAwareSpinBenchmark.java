import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.concurrent.locks.ReentrantLock;
import java.io.*;

/**
 * Tests whether CS-DURATION-AWARE spin sizing (V3, live EMA-based
 * budget) actually fixes the nT=2 regression that FIXED-count spin
 * sizing (V2) could not fix at realistic (~100-500us) critical
 * sections -- and whether the effect holds on a second, structurally
 * different lock family (MCS queue locking), per explicit user request
 * to test "along with a different lock family."
 *
 * Compares, at each (CS duration, nT):
 *   - FairMutex            (ReentrantLock(true), the standing baseline)
 *   - CombiningLock family: V1 (immediate escalation), V2 (fixed spin),
 *     V3 (duration-aware spin)
 *   - MCS family:           plain MCSLock, AdaptiveMCSLockV3 (duration-
 *     aware spin)
 *
 * Reports whether V3 actually improves on V2 specifically at nT=2 for
 * the two longer, realistic CS durations (100us, 500us) where V2 was
 * previously shown to fail -- the exact, falsifiable question asked.
 */
public class CSAwareSpinBenchmark {
    static final int[] THREADS = {1, 2, 4, 8, 16, 24, 32, 48, 64, 96};
    static final long[] CS_DURATIONS_NS = {10_000, 100_000, 500_000};
    static final int SEEDS = 8;
    static final int REQS_PER_THREAD = 120;

    public static void main(String[] args) throws Exception {
        List<String[]> csvRows = new ArrayList<>();
        csvRows.add(new String[]{"cs_ns","threads","method","mean_ms","ratio_to_fairmutex"});

        for (long csNs : CS_DURATIONS_NS) {
            for (int nT : THREADS) {
                System.out.println("\n=== CS=" + csNs + "ns nT=" + nT + " ===");
                // warmup discard
                runFair(nT, csNs); runCombV1(nT, csNs); runCombV2(nT, csNs); runCombV3(nT, csNs);
                runMCS(nT, csNs); runAdaptiveMCS(nT, csNs);
                runTAS(nT, csNs); runTATAS(nT, csNs); runTicket(nT, csNs);

                double[] fairM = new double[SEEDS];
                double[] cv1M = new double[SEEDS], cv2M = new double[SEEDS], cv3M = new double[SEEDS];
                double[] mcsM = new double[SEEDS], amcsM = new double[SEEDS];
                double[] tasM = new double[SEEDS], tatasM = new double[SEEDS], ticketM = new double[SEEDS];
                boolean anyViolation = false;

                for (int s = 0; s < SEEDS; s++) {
                    List<Integer> perm = new ArrayList<>(List.of(0,1,2,3,4,5,6,7,8));
                    Collections.shuffle(perm);
                    Object[] rf=null, r1=null, r2=null, r3=null, rm=null, ram=null, rtas=null, rtatas=null, rticket=null;
                    for (int idx : perm) {
                        switch (idx) {
                            case 0: rf = runFair(nT, csNs); break;
                            case 1: r1 = runCombV1(nT, csNs); break;
                            case 2: r2 = runCombV2(nT, csNs); break;
                            case 3: r3 = runCombV3(nT, csNs); break;
                            case 4: rm = runMCS(nT, csNs); break;
                            case 5: ram = runAdaptiveMCS(nT, csNs); break;
                            case 6: rtas = runTAS(nT, csNs); break;
                            case 7: rtatas = runTATAS(nT, csNs); break;
                            case 8: rticket = runTicket(nT, csNs); break;
                        }
                    }
                    fairM[s] = (double) rf[0]; if ((boolean) rf[1]) anyViolation = true;
                    cv1M[s] = (double) r1[0]; if ((boolean) r1[1]) anyViolation = true;
                    cv2M[s] = (double) r2[0]; if ((boolean) r2[1]) anyViolation = true;
                    cv3M[s] = (double) r3[0]; if ((boolean) r3[1]) anyViolation = true;
                    mcsM[s] = (double) rm[0]; if ((boolean) rm[1]) anyViolation = true;
                    amcsM[s] = (double) ram[0]; if ((boolean) ram[1]) anyViolation = true;
                    tasM[s] = (double) rtas[0]; if ((boolean) rtas[1]) anyViolation = true;
                    tatasM[s] = (double) rtatas[0]; if ((boolean) rtatas[1]) anyViolation = true;
                    ticketM[s] = (double) rticket[0]; if ((boolean) rticket[1]) anyViolation = true;
                    System.out.print(".");
                }
                System.out.println();
                if (anyViolation) System.out.println("  !!!! ME VIOLATION DETECTED !!!!");

                double fFair = mean(fairM);
                System.out.printf("  FairMutex        mean=%.4fms  ratio=1.000%n", fFair);
                System.out.printf("  CombV1(no spin)  mean=%.4fms  ratio=%.3f%n", mean(cv1M), mean(cv1M)/fFair);
                System.out.printf("  CombV2(fixedspin)mean=%.4fms  ratio=%.3f%n", mean(cv2M), mean(cv2M)/fFair);
                System.out.printf("  CombV3(CS-aware) mean=%.4fms  ratio=%.3f%n", mean(cv3M), mean(cv3M)/fFair);
                System.out.printf("  MCS(plain)       mean=%.4fms  ratio=%.3f%n", mean(mcsM), mean(mcsM)/fFair);
                System.out.printf("  AdaptiveMCS(CS-aware) mean=%.4fms  ratio=%.3f%n", mean(amcsM), mean(amcsM)/fFair);
                System.out.printf("  TASLock          mean=%.4fms  ratio=%.3f%n", mean(tasM), mean(tasM)/fFair);
                System.out.printf("  TATASLock        mean=%.4fms  ratio=%.3f%n", mean(tatasM), mean(tatasM)/fFair);
                System.out.printf("  TicketLock       mean=%.4fms  ratio=%.3f%n", mean(ticketM), mean(ticketM)/fFair);

                csvRows.add(row(csNs, nT, "FairMutex", fFair, 1.0));
                csvRows.add(row(csNs, nT, "CombV1", mean(cv1M), mean(cv1M)/fFair));
                csvRows.add(row(csNs, nT, "CombV2", mean(cv2M), mean(cv2M)/fFair));
                csvRows.add(row(csNs, nT, "CombV3", mean(cv3M), mean(cv3M)/fFair));
                csvRows.add(row(csNs, nT, "MCS", mean(mcsM), mean(mcsM)/fFair));
                csvRows.add(row(csNs, nT, "AdaptiveMCS", mean(amcsM), mean(amcsM)/fFair));
                csvRows.add(row(csNs, nT, "TASLock", mean(tasM), mean(tasM)/fFair));
                csvRows.add(row(csNs, nT, "TATASLock", mean(tatasM), mean(tatasM)/fFair));
                csvRows.add(row(csNs, nT, "TicketLock", mean(ticketM), mean(ticketM)/fFair));
            }
        }

        String outPath = args.length > 0 ? args[0] : "cs_aware_spin_benchmark.csv";
        try (PrintWriter pw = new PrintWriter(new FileWriter(outPath))) {
            for (String[] r : csvRows) pw.println(String.join(",", r));
        }
        System.out.println("\nWritten to " + outPath);
    }

    static String[] row(long csNs, int nT, String method, double meanMs, double ratio) {
        return new String[]{String.valueOf(csNs), String.valueOf(nT), method, fmt(meanMs), fmt(ratio)};
    }

    static Object[] runFair(int nT, long csNs) throws Exception {
        ReentrantLock lk = new ReentrantLock(true);
        List<Long> waits = Collections.synchronizedList(new ArrayList<>());
        AtomicInteger inside = new AtomicInteger(0);
        AtomicBoolean violated = new AtomicBoolean(false);
        CountDownLatch latch = new CountDownLatch(nT);
        ExecutorService ex = Executors.newFixedThreadPool(nT);
        long thinkNs = Math.max(500, (long)(csNs * 0.7));
        for (int t = 0; t < nT; t++) {
            ex.submit(() -> {
                ThreadLocalRandom tlr = ThreadLocalRandom.current();
                try {
                    for (int i = 0; i < REQS_PER_THREAD; i++) {
                        sleepNs(thinkNs + tlr.nextLong(Math.max(1,thinkNs/2)));
                        long t0 = System.nanoTime();
                        lk.lockInterruptibly();
                        int n = inside.incrementAndGet();
                        if (n > 1) violated.set(true);
                        long cs0 = System.nanoTime();
                        while (System.nanoTime() - cs0 < csNs) {}
                        inside.decrementAndGet();
                        lk.unlock();
                        waits.add(System.nanoTime() - t0);
                    }
                } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
                finally { latch.countDown(); }
            });
        }
        latch.await(150, TimeUnit.SECONDS);
        ex.shutdownNow(); ex.awaitTermination(5, TimeUnit.SECONDS);
        return new Object[]{meanMs(waits), violated.get()};
    }

    static Object[] runCombV1(int nT, long csNs) throws Exception {
        AdaptiveCombiningLock lk = new AdaptiveCombiningLock();
        return runWithBody(nT, csNs, null, lk);
    }
    static Object[] runCombV2(int nT, long csNs) throws Exception {
        AdaptiveCombiningLockV2 lk = new AdaptiveCombiningLockV2();
        return runV2(nT, csNs, lk);
    }
    static Object[] runCombV3(int nT, long csNs) throws Exception {
        AdaptiveCombiningLockV3 lk = new AdaptiveCombiningLockV3();
        return runV3(nT, csNs, lk);
    }
    static Object[] runMCS(int nT, long csNs) throws Exception {
        MCSLock lk = new MCSLock();
        return runMcsPlain(nT, csNs, lk);
    }
    static Object[] runAdaptiveMCS(int nT, long csNs) throws Exception {
        AdaptiveMCSLockV3 lk = new AdaptiveMCSLockV3();
        return runMcsAdaptive(nT, csNs, lk);
    }

    static Object[] runWithBody(int nT, long csNs, Object unused, AdaptiveCombiningLock lk) throws Exception {
        List<Long> waits = Collections.synchronizedList(new ArrayList<>());
        AtomicInteger inside = new AtomicInteger(0);
        AtomicBoolean violated = new AtomicBoolean(false);
        CountDownLatch latch = new CountDownLatch(nT);
        ExecutorService ex = Executors.newFixedThreadPool(nT);
        long thinkNs = Math.max(500, (long)(csNs * 0.7));
        for (int t = 0; t < nT; t++) {
            ex.submit(() -> {
                ThreadLocalRandom tlr = ThreadLocalRandom.current();
                try {
                    for (int i = 0; i < REQS_PER_THREAD; i++) {
                        sleepNs(thinkNs + tlr.nextLong(Math.max(1,thinkNs/2)));
                        long t0 = System.nanoTime();
                        lk.execute(() -> {
                            int n = inside.incrementAndGet();
                            if (n > 1) violated.set(true);
                            long cs0 = System.nanoTime();
                            while (System.nanoTime() - cs0 < csNs) {}
                            inside.decrementAndGet();
                        });
                        waits.add(System.nanoTime() - t0);
                    }
                } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
                finally { latch.countDown(); }
            });
        }
        latch.await(150, TimeUnit.SECONDS);
        ex.shutdownNow(); ex.awaitTermination(5, TimeUnit.SECONDS);
        return new Object[]{meanMs(waits), violated.get()};
    }

    static Object[] runV2(int nT, long csNs, AdaptiveCombiningLockV2 lk) throws Exception {
        List<Long> waits = Collections.synchronizedList(new ArrayList<>());
        AtomicInteger inside = new AtomicInteger(0);
        AtomicBoolean violated = new AtomicBoolean(false);
        CountDownLatch latch = new CountDownLatch(nT);
        ExecutorService ex = Executors.newFixedThreadPool(nT);
        long thinkNs = Math.max(500, (long)(csNs * 0.7));
        for (int t = 0; t < nT; t++) {
            ex.submit(() -> {
                ThreadLocalRandom tlr = ThreadLocalRandom.current();
                try {
                    for (int i = 0; i < REQS_PER_THREAD; i++) {
                        sleepNs(thinkNs + tlr.nextLong(Math.max(1,thinkNs/2)));
                        long t0 = System.nanoTime();
                        lk.execute(() -> {
                            int n = inside.incrementAndGet();
                            if (n > 1) violated.set(true);
                            long cs0 = System.nanoTime();
                            while (System.nanoTime() - cs0 < csNs) {}
                            inside.decrementAndGet();
                        });
                        waits.add(System.nanoTime() - t0);
                    }
                } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
                finally { latch.countDown(); }
            });
        }
        latch.await(150, TimeUnit.SECONDS);
        ex.shutdownNow(); ex.awaitTermination(5, TimeUnit.SECONDS);
        return new Object[]{meanMs(waits), violated.get()};
    }

    static Object[] runV3(int nT, long csNs, AdaptiveCombiningLockV3 lk) throws Exception {
        List<Long> waits = Collections.synchronizedList(new ArrayList<>());
        AtomicInteger inside = new AtomicInteger(0);
        AtomicBoolean violated = new AtomicBoolean(false);
        CountDownLatch latch = new CountDownLatch(nT);
        ExecutorService ex = Executors.newFixedThreadPool(nT);
        long thinkNs = Math.max(500, (long)(csNs * 0.7));
        for (int t = 0; t < nT; t++) {
            ex.submit(() -> {
                ThreadLocalRandom tlr = ThreadLocalRandom.current();
                try {
                    for (int i = 0; i < REQS_PER_THREAD; i++) {
                        sleepNs(thinkNs + tlr.nextLong(Math.max(1,thinkNs/2)));
                        long t0 = System.nanoTime();
                        lk.execute(() -> {
                            int n = inside.incrementAndGet();
                            if (n > 1) violated.set(true);
                            long cs0 = System.nanoTime();
                            while (System.nanoTime() - cs0 < csNs) {}
                            inside.decrementAndGet();
                        });
                        waits.add(System.nanoTime() - t0);
                    }
                } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
                finally { latch.countDown(); }
            });
        }
        latch.await(150, TimeUnit.SECONDS);
        ex.shutdownNow(); ex.awaitTermination(5, TimeUnit.SECONDS);
        return new Object[]{meanMs(waits), violated.get()};
    }

    static Object[] runMcsPlain(int nT, long csNs, MCSLock lk) throws Exception {
        List<Long> waits = Collections.synchronizedList(new ArrayList<>());
        AtomicInteger inside = new AtomicInteger(0);
        AtomicBoolean violated = new AtomicBoolean(false);
        CountDownLatch latch = new CountDownLatch(nT);
        ExecutorService ex = Executors.newFixedThreadPool(nT);
        long thinkNs = Math.max(500, (long)(csNs * 0.7));
        for (int t = 0; t < nT; t++) {
            ex.submit(() -> {
                ThreadLocalRandom tlr = ThreadLocalRandom.current();
                try {
                    for (int i = 0; i < REQS_PER_THREAD; i++) {
                        sleepNs(thinkNs + tlr.nextLong(Math.max(1,thinkNs/2)));
                        long t0 = System.nanoTime();
                        lk.execute(() -> {
                            int n = inside.incrementAndGet();
                            if (n > 1) violated.set(true);
                            long cs0 = System.nanoTime();
                            while (System.nanoTime() - cs0 < csNs) {}
                            inside.decrementAndGet();
                        });
                        waits.add(System.nanoTime() - t0);
                    }
                } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
                finally { latch.countDown(); }
            });
        }
        latch.await(150, TimeUnit.SECONDS);
        ex.shutdownNow(); ex.awaitTermination(5, TimeUnit.SECONDS);
        return new Object[]{meanMs(waits), violated.get()};
    }

    static Object[] runMcsAdaptive(int nT, long csNs, AdaptiveMCSLockV3 lk) throws Exception {
        List<Long> waits = Collections.synchronizedList(new ArrayList<>());
        AtomicInteger inside = new AtomicInteger(0);
        AtomicBoolean violated = new AtomicBoolean(false);
        CountDownLatch latch = new CountDownLatch(nT);
        ExecutorService ex = Executors.newFixedThreadPool(nT);
        long thinkNs = Math.max(500, (long)(csNs * 0.7));
        for (int t = 0; t < nT; t++) {
            ex.submit(() -> {
                ThreadLocalRandom tlr = ThreadLocalRandom.current();
                try {
                    for (int i = 0; i < REQS_PER_THREAD; i++) {
                        sleepNs(thinkNs + tlr.nextLong(Math.max(1,thinkNs/2)));
                        long t0 = System.nanoTime();
                        lk.execute(() -> {
                            int n = inside.incrementAndGet();
                            if (n > 1) violated.set(true);
                            long cs0 = System.nanoTime();
                            while (System.nanoTime() - cs0 < csNs) {}
                            inside.decrementAndGet();
                        });
                        waits.add(System.nanoTime() - t0);
                    }
                } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
                finally { latch.countDown(); }
            });
        }
        latch.await(150, TimeUnit.SECONDS);
        ex.shutdownNow(); ex.awaitTermination(5, TimeUnit.SECONDS);
        return new Object[]{meanMs(waits), violated.get()};
    }

    static Object[] runTAS(int nT, long csNs) throws Exception {
        TASLock lk = new TASLock();
        List<Long> waits = Collections.synchronizedList(new ArrayList<>());
        AtomicInteger inside = new AtomicInteger(0);
        AtomicBoolean violated = new AtomicBoolean(false);
        CountDownLatch latch = new CountDownLatch(nT);
        ExecutorService ex = Executors.newFixedThreadPool(nT);
        long thinkNs = Math.max(500, (long)(csNs * 0.7));
        for (int t = 0; t < nT; t++) {
            ex.submit(() -> {
                ThreadLocalRandom tlr = ThreadLocalRandom.current();
                try {
                    for (int i = 0; i < REQS_PER_THREAD; i++) {
                        sleepNs(thinkNs + tlr.nextLong(Math.max(1,thinkNs/2)));
                        long t0 = System.nanoTime();
                        lk.execute(() -> {
                            int n = inside.incrementAndGet();
                            if (n > 1) violated.set(true);
                            long cs0 = System.nanoTime();
                            while (System.nanoTime() - cs0 < csNs) {}
                            inside.decrementAndGet();
                        });
                        waits.add(System.nanoTime() - t0);
                    }
                } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
                finally { latch.countDown(); }
            });
        }
        latch.await(150, TimeUnit.SECONDS);
        ex.shutdownNow(); ex.awaitTermination(5, TimeUnit.SECONDS);
        return new Object[]{meanMs(waits), violated.get()};
    }

    static Object[] runTATAS(int nT, long csNs) throws Exception {
        TATASLock lk = new TATASLock();
        List<Long> waits = Collections.synchronizedList(new ArrayList<>());
        AtomicInteger inside = new AtomicInteger(0);
        AtomicBoolean violated = new AtomicBoolean(false);
        CountDownLatch latch = new CountDownLatch(nT);
        ExecutorService ex = Executors.newFixedThreadPool(nT);
        long thinkNs = Math.max(500, (long)(csNs * 0.7));
        for (int t = 0; t < nT; t++) {
            ex.submit(() -> {
                ThreadLocalRandom tlr = ThreadLocalRandom.current();
                try {
                    for (int i = 0; i < REQS_PER_THREAD; i++) {
                        sleepNs(thinkNs + tlr.nextLong(Math.max(1,thinkNs/2)));
                        long t0 = System.nanoTime();
                        lk.execute(() -> {
                            int n = inside.incrementAndGet();
                            if (n > 1) violated.set(true);
                            long cs0 = System.nanoTime();
                            while (System.nanoTime() - cs0 < csNs) {}
                            inside.decrementAndGet();
                        });
                        waits.add(System.nanoTime() - t0);
                    }
                } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
                finally { latch.countDown(); }
            });
        }
        latch.await(150, TimeUnit.SECONDS);
        ex.shutdownNow(); ex.awaitTermination(5, TimeUnit.SECONDS);
        return new Object[]{meanMs(waits), violated.get()};
    }

    static Object[] runTicket(int nT, long csNs) throws Exception {
        TicketLock lk = new TicketLock();
        List<Long> waits = Collections.synchronizedList(new ArrayList<>());
        AtomicInteger inside = new AtomicInteger(0);
        AtomicBoolean violated = new AtomicBoolean(false);
        CountDownLatch latch = new CountDownLatch(nT);
        ExecutorService ex = Executors.newFixedThreadPool(nT);
        long thinkNs = Math.max(500, (long)(csNs * 0.7));
        for (int t = 0; t < nT; t++) {
            ex.submit(() -> {
                ThreadLocalRandom tlr = ThreadLocalRandom.current();
                try {
                    for (int i = 0; i < REQS_PER_THREAD; i++) {
                        sleepNs(thinkNs + tlr.nextLong(Math.max(1,thinkNs/2)));
                        long t0 = System.nanoTime();
                        lk.execute(() -> {
                            int n = inside.incrementAndGet();
                            if (n > 1) violated.set(true);
                            long cs0 = System.nanoTime();
                            while (System.nanoTime() - cs0 < csNs) {}
                            inside.decrementAndGet();
                        });
                        waits.add(System.nanoTime() - t0);
                    }
                } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
                finally { latch.countDown(); }
            });
        }
        latch.await(150, TimeUnit.SECONDS);
        ex.shutdownNow(); ex.awaitTermination(5, TimeUnit.SECONDS);
        return new Object[]{meanMs(waits), violated.get()};
    }

    static void sleepNs(long ns) throws InterruptedException {
        long deadline = System.nanoTime() + ns;
        if (ns > 2_000_000) Thread.sleep(ns / 1_000_000);
        while (System.nanoTime() < deadline) Thread.onSpinWait();
    }
    static double meanMs(List<Long> waits) {
        if (waits.isEmpty()) return 0;
        return waits.stream().mapToLong(Long::longValue).average().orElse(0) / 1e6;
    }
    static String fmt(double v) { return String.format("%.6f", v); }
    static double mean(double[] a){double s=0;for(double v:a)s+=v;return s/a.length;}
}
