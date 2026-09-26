import java.util.concurrent.atomic.*;

/**
 * Test-And-Set (TAS) spinlock -- the simplest possible mutual-exclusion
 * primitive: a single AtomicBoolean, acquired via getAndSet(true) in a
 * tight loop until it returns false.
 *
 * KNOWN, TEXTBOOK WEAKNESS (not discovered here, included for
 * completeness of the comparison): every failed attempt is itself a
 * write (getAndSet always writes, even when it "fails"), so under
 * contention every spinning thread continuously invalidates the cache
 * line for every other spinning thread and the holder. This causes
 * severe cache-coherence traffic that gets worse, not better, as
 * thread count rises -- the classic reason TATASLock (test-THEN-test-
 * and-set) was introduced. TASLock is included here purely as the
 * bottom-of-the-barrel reference point every lock survey uses.
 *
 * No fairness, no local spinning, no backoff -- deliberately the most
 * naive possible baseline.
 */
public class TASLock {
    private final AtomicBoolean state = new AtomicBoolean(false);

    public interface Work { void run(); }

    public long execute(Work work) throws InterruptedException {
        long t0 = System.nanoTime();
        while (state.getAndSet(true)) {
            if (Thread.interrupted()) throw new InterruptedException();
            Thread.onSpinWait();
        }
        try {
            work.run();
        } finally {
            state.set(false);
        }
        return System.nanoTime() - t0;
    }
}
