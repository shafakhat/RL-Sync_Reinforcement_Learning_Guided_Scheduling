import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

/**
 * Traces exactly WHY the live single-class EMA estimator in
 * AdaptivePriorityLockV2 beats a frozen/plain FIFO tie-break, even under
 * homogeneous load with only ONE job class (so there's no "shortest job"
 * to identify -- only one EMA value exists at any instant, shared by
 * whichever waiters are queued).
 *
 * Hypothesis: the estimator is seeded at 100us (below the true 150us
 * workload), so early in a run -- and after any idle gap where the queue
 * drains to empty and the EMA has decayed toward whatever it last was --
 * successive predictedDurationNs snapshots taken at DIFFERENT wall-clock
 * moments differ slightly as the EMA tracks noisy observed durations.
 * Because release() always picks the waiter with the CURRENT lowest
 * snapshot instead of strict arrival order, this can systematically
 * reorder waiters by "which EMA reading they happened to capture," which
 * is only neutral in EXPECTATION, not in every finite run -- and CI
 * ordering combined with queueing dynamics could make it correlate with
 * something real (e.g. favoring threads that arrived when the system
 * was momentarily under-loaded, which are exactly the threads that
 * would otherwise wait less anyway... or MORE, depending on direction).
 *
 * This directly logs: for a single representative run, the sequence of
 * (ticket, arrivalNs relative offset, predictedDurationNs snapshot,
 * selection order vs arrival order) to see the actual reordering pattern
 * empirically instead of guessing.
 */
public class MechanismTrace {
    public static void main(String[] args) throws Exception {
        int nT = 8, reqsPerThread = 40;
        AdaptivePriorityLockV2 lk = new AdaptivePriorityLockV2();

        List<long[]> log = Collections.synchronizedList(new ArrayList<>()); // {ticketOrder, arrivalNs, predictedNs, admitOrder}
        AtomicLong admitCounter = new AtomicLong(0);
        AtomicLong arrivalCounter = new AtomicLong(0);

        CountDownLatch latch = new CountDownLatch(nT);
        ExecutorService ex = Executors.newFixedThreadPool(nT);
        long startNs = System.nanoTime();
        for (int t = 0; t < nT; t++) {
            ex.submit(() -> {
                ThreadLocalRandom tlr = ThreadLocalRandom.current();
                try {
                    for (int i = 0; i < reqsPerThread; i++) {
                        sleepUs(80 + tlr.nextLong(60));
                        long arrivalOrder = arrivalCounter.getAndIncrement();
                        long t0 = System.nanoTime();
                        long w = lk.acquire(AdaptivePriorityLockV2.JobClass.SHORT);
                        long admitOrder = admitCounter.getAndIncrement();
                        log.add(new long[]{arrivalOrder, (t0-startNs), w, admitOrder});
                        long tcs = System.nanoTime();
                        while (System.nanoTime() - tcs < 150_000L) {}
                        lk.release(AdaptivePriorityLockV2.JobClass.SHORT, System.nanoTime() - tcs);
                    }
                } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
                finally { latch.countDown(); }
            });
        }
        latch.await(60, TimeUnit.SECONDS);
        ex.shutdownNow(); ex.awaitTermination(5, TimeUnit.SECONDS);

        // Compute: fraction of requests admitted OUT of arrival order,
        // and correlation between arrival order and admit order.
        long[][] rows = log.toArray(new long[0][]);
        Arrays.sort(rows, Comparator.comparingLong(r -> r[0])); // sort by arrivalOrder

        int outOfOrder = 0;
        long lastAdmit = -1;
        for (long[] r : rows) {
            if (r[3] < lastAdmit) outOfOrder++;
            lastAdmit = Math.max(lastAdmit, r[3]);
        }
        System.out.println("Total requests: " + rows.length);
        System.out.println("Out-of-arrival-order admissions (admitOrder < running max so far): " + outOfOrder);

        // Show first 30 rows: arrivalOrder, arrivalTimeMs, waitMs, admitOrder
        System.out.println("\narrivalOrder  arrivalTimeMs  waitMs  admitOrder  delta(admit-arrival)");
        for (int i = 0; i < Math.min(30, rows.length); i++) {
            long[] r = rows[i];
            System.out.printf("%12d  %13.3f  %6.3f  %10d  %+d%n",
                r[0], r[1]/1e6, r[2]/1e6, r[3], r[3]-r[0]);
        }

        // Mean wait vs "delta rank" bucket: do requests that get promoted
        // ahead (negative delta admit-arrival) actually wait less?
        double meanWaitMs = Arrays.stream(rows).mapToDouble(r -> r[2]/1e6).average().orElse(0);
        System.out.printf("%nMean wait across all requests: %.3fms%n", meanWaitMs);
    }

    static void sleepUs(long us) throws InterruptedException {
        long deadline = System.nanoTime() + us * 1000L;
        if (us > 1500) Thread.sleep(us / 1000);
        while (System.nanoTime() < deadline) Thread.onSpinWait();
    }
}
