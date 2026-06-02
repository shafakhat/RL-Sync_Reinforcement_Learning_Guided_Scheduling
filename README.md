# RLSyncBenchmark

A Java benchmark that applies **Q-Learning** to adaptively control critical-section (CS) access, comparing the RL-based arbiter against three classical synchronisation mechanisms across varying thread counts and random seeds.

---

## Overview

`RLSyncBenchmark` trains a tabular Q-Learning agent — called **RL-Sync** — to decide whether each competing thread should *admit* immediately or *back off* before entering a critical section. The trained policy is then evaluated head-to-head against:

| Method | Description |
|---|---|
| **RL-Sync** | Q-Learning arbiter (trained online, greedy at eval) |
| **FairMutex** | Java `ReentrantLock` in fair mode |
| **CLH** | Craig–Landin–Hagersten queue lock (spin-based) |
| **ExpBackoff** | Exponential back-off spin lock |

---

## Key Features

- **Q-Learning with extended state space** — state encodes queue depth, CS occupancy, and an EMA-based load bucket
- **ε-greedy exploration** with linear annealing from 0.40 → 0.04 over 220 training episodes
- **Mutual exclusion verification** — runtime check ensures no two threads are ever inside the CS simultaneously
- **Sub-millisecond timing** via busy-wait (`Thread.onSpinWait`) for microsecond-precision measurements
- **Statistical reporting** — mean, std, P99, Zero-Delay Ratio (ZDR), contention rate, Welch t-test, and Cohen's d across 6 seeds

---

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `CS_WORK_US` | 150 µs | Critical section work duration |
| `THINK_US_BASE` | 120 µs | Base think time between CS entries |
| `THINK_US_JITTER` | 80 µs | Random jitter added to think time |
| `TRAIN_EPISODES` | 220 | Q-Learning training episodes |
| `EVAL_REQUESTS` | 100 | Requests per thread during evaluation |
| `SEEDS` | 6 | Independent repetitions per configuration |
| `THREAD_COUNTS` | 2, 4, 6, 8 | Thread counts swept during benchmark |
| `ALPHA` | 0.12 | Q-Learning rate |
| `GAMMA` | 0.92 | Discount factor |

---

## Requirements

- Java 11 or later (uses `Thread.onSpinWait()`)
- No external dependencies — pure Java standard library

---

## Build & Run

```bash
# Compile
javac RLSyncBenchmark.java

# Run with default output directory (./mlsync_experiment_data/data)
java RLSyncBenchmark

# Run with custom output directory
java RLSyncBenchmark /path/to/output/dir
```

---

## Output Files

All CSV files are written to the specified data directory:

| File | Contents |
|---|---|
| `sweep_results.csv` | Mean wait, CI95, std, P99, ZDR, contention for all methods × thread counts |
| `convergence.csv` | Per-episode mean wait for RL-Sync vs FairMutex baseline (at 6 threads, seed 0) |
| `qtable.csv` | Final Q-values and derived policy (ADMIT / BACKOFF) for all states |
| `stats_rl_vs_mutex.csv` | Welch t-test and Cohen's d: RL-Sync vs FairMutex |
| `stats_rl_vs_clh.csv` | Welch t-test and Cohen's d: RL-Sync vs CLH |
| `stats_rl_vs_eb.csv` | Welch t-test and Cohen's d: RL-Sync vs ExpBackoff |

---

## Project Structure

```
RLSyncBenchmark.java
│
├── QTable                  # Tabular Q-Learning state/action model
├── RLArbiter               # RL-based CS lock (CSLock implementation)
├── FairMutex               # ReentrantLock(fair=true) wrapper
├── CLHLock                 # CLH queue lock implementation
├── ExpBackoff              # Exponential back-off lock
├── Worker                  # Runnable worker simulating think + CS access
├── Metrics                 # Per-lock latency and contention tracking
└── main()                  # Training loop, evaluation sweep, CSV export
```

---

## Metrics Explained

| Metric | Description |
|---|---|
| **Mean Wait (ms)** | Average time from lock request to acquisition |
| **P99 (ms)** | 99th percentile wait latency |
| **ZDR (%)** | Zero-Delay Ratio — fraction of acquisitions under 500 µs |
| **Contention (%)** | Fraction of acquisitions where CS was occupied or queue was non-empty |
| **Welch t** | Two-sample t-statistic (unequal variance) |
| **Cohen's d** | Standardised effect size between two methods |

---

## License

MIT — free to use, modify, and distribute.
