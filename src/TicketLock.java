import java.util.concurrent.atomic.*;

/**
 * Ticket lock -- classic FIFO spinlock (Mellor-Crummey & Scott 1991
 * also describe this, alongside MCS itself). Each arriving thread
 * atomically draws a ticket number (nextTicket.getAndIncrement()) and
 * spins reading a shared "now serving" counter until it matches its
 * own ticket. Provides strict FIFO fairness (unlike TAS/TATAS/plain
 * MCS-without-modification, which are all technically unfair), at the
 * cost of every waiter spinning on the SAME shared cache line (unlike
 * MCS, where each thread spins on its own local location) -- so it
 * still suffers from some coherence traffic under contention, though
 * less than TAS/TATAS since it's a read-spin, not a write-spin.
 *
 * DISCLOSED, PRE-EMPTIVELY APPLIED FIX: this investigation already
 * found (with plain MCSLock.java, see ADAPTIVE_COMBINING_FINDINGS.md)
 * that pure onSpinWait()-only waiting causes near-livelock on this
 * 2-core sandbox at high thread-count oversubscription (e.g. nT=64 on
 * 2 cores). Since Ticket lock has the same "spin with no yield" risk
 * profile, the same standard, well-known fix (periodic Thread.yield()
 * during the wait) is applied here from the start, disclosed up front
 * rather than discovered by a hang during testing.
 */
public class TicketLock {
    private final AtomicLong nextTicket = new AtomicLong(0);
    private final AtomicLong nowServing = new AtomicLong(0);

    public interface Work { void run(); }

    public long execute(Work work) throws InterruptedException {
        long t0 = System.nanoTime();
        long myTicket = nextTicket.getAndIncrement();
        int spins = 0;
        while (nowServing.get() != myTicket) {
            if (Thread.interrupted()) throw new InterruptedException();
            if (++spins % 64 == 0) {
                Thread.yield();
            } else {
                Thread.onSpinWait();
            }
        }
        try {
            work.run();
        } finally {
            nowServing.incrementAndGet();
        }
        return System.nanoTime() - t0;
    }
}
