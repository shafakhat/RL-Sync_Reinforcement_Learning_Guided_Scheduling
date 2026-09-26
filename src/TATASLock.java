import java.util.concurrent.atomic.*;

/**
 * Test-And-Test-And-Set (TATAS) spinlock -- the standard fix for
 * TASLock's cache-coherence-storm problem: spin on a plain READ of the
 * flag first (which every core can do from its own cached copy without
 * generating coherence traffic), and only attempt the actual
 * getAndSet(true) write once the read suggests the lock might be free.
 * This is textbook, decades-old technique (Rudolph & Segall 1984),
 * included here as the standard "fixed" baseline above plain TAS.
 *
 * Still has no fairness and no local spinning -- multiple threads can
 * still race on the same shared flag when it frees up (only one wins,
 * the rest re-enter the read-spin), and there's no bound on how long a
 * given thread might wait, unlike Ticket or MCS. Included purely as
 * the standard next-step-up reference point.
 */
public class TATASLock {
    private final AtomicBoolean state = new AtomicBoolean(false);

    public interface Work { void run(); }

    public long execute(Work work) throws InterruptedException {
        long t0 = System.nanoTime();
        while (true) {
            while (state.get()) {
                if (Thread.interrupted()) throw new InterruptedException();
                Thread.onSpinWait();
            }
            if (!state.getAndSet(true)) {
                break;
            }
        }
        try {
            work.run();
        } finally {
            state.set(false);
        }
        return System.nanoTime() - t0;
    }
}
