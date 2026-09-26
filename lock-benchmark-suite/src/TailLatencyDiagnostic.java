import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

/**
 * Checks WHY CombiningLock's p99 is worse than FairMutex's at every
 * thread count tested (e.g. 37ms vs 26ms at nT=96), despite a
 * consistently better mean. Two very different explanations:
 *
 *  (a) EXPECTED/ACCEPTABLE: the combiner itself pays a long wait
 *      because it has to sequentially execute the whole batch (N
 *      requests' worth of critical-section time) before the pass
 *      finishes -- this is an inherent, well-understood cost of flat
 *      combining (you trade "N threads take turns waiting a little
 *      each" for "the batch takes one thread a while to plow through",
 *      but net queueing time still goes down because there's no
 *      repeated re-acquisition tax). This is a fair, disclosed
 *      trade-off, not a bug.
 *
 *  (b) A REAL DEFECT: an unlucky NON-combiner waiter occasionally gets
 *      stuck for a long time because of the exponential backoff
 *      fallback (up to 20ms) combined with a missed targeted unpark,
 *      i.e. the tail is dominated by the liveness safety-net path
 *      firing, not by legitimate combiner batch-processing time. This
 *      would indicate the backoff bounds need tightening.
 *
 * This tags every observed wait with whether the observing thread WAS
 * the combiner for that pass or not, and reports p99 separately for
 * each population.
 */
public class TailLatencyDiagnostic {
    public static void main(String[] args) throws Exception {
        for (int nT : new int[]{64, 96}) {
            System.out.println("=== nT=" + nT + " ===");
            runInstrumented(nT);
            System.out.println();
        }
    }

    static void runInstrumented(int nT) throws Exception {
        int reqsPerThread = 150;
        InstrumentedLock lk = new InstrumentedLock();
        List<Long> combinerWaits = Collections.synchronizedList(new ArrayList<>());
        List<Long> nonCombinerWaits = Collections.synchronizedList(new ArrayList<>());
        CountDownLatch latch = new CountDownLatch(nT);
        ExecutorService ex = Executors.newFixedThreadPool(nT);
        for (int t = 0; t < nT; t++) {
            ex.submit(() -> {
                ThreadLocalRandom tlr = ThreadLocalRandom.current();
                try {
                    for (int i = 0; i < reqsPerThread; i++) {
                        sleepUs(80 + tlr.nextLong(60));
                        boolean[] wasCombiner = new boolean[1];
                        long w = lk.execute(() -> {
                            long cs0 = System.nanoTime();
                            while (System.nanoTime() - cs0 < 150_000L) {}
                        }, wasCombiner);
                        (wasCombiner[0] ? combinerWaits : nonCombinerWaits).add(w);
                    }
                } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
                finally { latch.countDown(); }
            });
        }
        latch.await(120, TimeUnit.SECONDS);
        ex.shutdownNow(); ex.awaitTermination(5, TimeUnit.SECONDS);

        System.out.printf("  Combiner-role waits:     n=%d meanMs=%.3f p99Ms=%.3f maxMs=%.3f%n",
            combinerWaits.size(), meanMs(combinerWaits), p99Ms(combinerWaits), maxMs(combinerWaits));
        System.out.printf("  Non-combiner-role waits: n=%d meanMs=%.3f p99Ms=%.3f maxMs=%.3f%n",
            nonCombinerWaits.size(), meanMs(nonCombinerWaits), p99Ms(nonCombinerWaits), maxMs(nonCombinerWaits));

        List<Long> all = new ArrayList<>(combinerWaits);
        all.addAll(nonCombinerWaits);
        System.out.printf("  Overall: n=%d meanMs=%.3f p99Ms=%.3f%n", all.size(), meanMs(all), p99Ms(all));
    }

    // Local copy of CombiningLock's real logic (same constants), with a
    // wasCombiner[0] output flag added for diagnosis.
    static final class InstrumentedLock {
        interface Work { void run(); }
        static final class Request {
            final Work work; final Thread thread; volatile boolean done=false; volatile Request next;
            Request(Work w, Thread t){work=w;thread=t;}
        }
        private final AtomicReference<Request> stackTop = new AtomicReference<>();
        private final AtomicBoolean combining = new AtomicBoolean(false);

        long execute(Work work, boolean[] wasCombinerOut) throws InterruptedException {
            long t0 = System.nanoTime();
            Request r = new Request(work, Thread.currentThread());
            Request oldTop;
            do { oldTop = stackTop.get(); r.next = oldTop; } while (!stackTop.compareAndSet(oldTop, r));
            long repollNs = CombiningLock.INITIAL_REPOLL_NS;
            boolean everWasCombiner = false;
            while (!r.done) {
                if (Thread.interrupted()) throw new InterruptedException();
                if (combining.compareAndSet(false, true)) {
                    everWasCombiner = true;
                    try { combinePass(); } finally { combining.set(false); }
                } else {
                    java.util.concurrent.locks.LockSupport.parkNanos(repollNs);
                    repollNs = Math.min(repollNs * 2, CombiningLock.MAX_REPOLL_NS);
                }
            }
            wasCombinerOut[0] = everWasCombiner;
            return System.nanoTime() - t0;
        }
        private void combinePass() {
            Request batch = stackTop.getAndSet(null);
            if (batch == null) return;
            Request cur = batch;
            while (cur != null) {
                Request next = cur.next;
                cur.work.run(); cur.done = true;
                if (cur.thread != Thread.currentThread()) java.util.concurrent.locks.LockSupport.unpark(cur.thread);
                cur = next;
            }
        }
    }

    static void sleepUs(long us) throws InterruptedException {
        long deadline = System.nanoTime() + us * 1000L;
        if (us > 1500) Thread.sleep(us / 1000);
        while (System.nanoTime() < deadline) Thread.onSpinWait();
    }
    static double meanMs(List<Long> w){ if(w.isEmpty())return 0; return w.stream().mapToLong(Long::longValue).average().orElse(0)/1e6; }
    static double p99Ms(List<Long> w){
        if (w.isEmpty()) return 0;
        List<Long> s = new ArrayList<>(w); Collections.sort(s);
        return s.get((int)(s.size()*0.99))/1e6;
    }
    static double maxMs(List<Long> w){ if(w.isEmpty())return 0; return w.stream().mapToLong(Long::longValue).max().orElse(0)/1e6; }
}
