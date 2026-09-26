# Lock Benchmark Suite

A collection of Java lock implementations and micro-benchmark/diagnostic
programs for measuring concurrent-lock throughput, latency, and correctness
under configurable thread counts and workloads.

## Requirements

- JDK 11 or later (`java`, `javac` on your `PATH`)
- No external libraries or build tool required — everything is plain Java
  in the default package with no third-party dependencies.

Check your setup:

```bash
java -version
javac -version
```

## Clone

```bash
git clone <this-repository-url>
cd lock-benchmark-suite
```

## Build

All sources live in `src/` and compile together with a single `javac`
invocation:

```bash
mkdir -p out
javac -encoding UTF-8 -d out src/*.java
```

This compiles all 50 `.java` files into `out/`.

## Run

Every program is a standalone class with a `public static void main`. Run
any of them with:

```bash
java -cp out <ClassName>
```

Most programs print CSV-formatted results to standard output; redirect to a
file if you want to keep the output:

```bash
java -cp out <ClassName> > results.csv
```

Some programs accept no arguments and use fixed internal configurations;
a few accept optional command-line arguments (thread count, duration, etc.)
— run the class with `--help` or check the top of its `main` method if a
particular run needs tuning.

### Runnable classes (contain `main`)

| Class | What it runs |
|---|---|
| `AdaptiveCombiningBenchmark` | Benchmark harness for the adaptive combining lock (V1) |
| `AdaptiveCombiningBenchmarkV2` | Benchmark harness for the adaptive combining lock (V2) |
| `AdaptiveCombiningLockSafety` | Correctness/safety check for the adaptive combining lock (V1) |
| `AdaptiveCombiningLockV2Safety` | Correctness/safety check for the adaptive combining lock (V2) |
| `AdaptiveCombiningLockV3Safety` | Correctness/safety check for the adaptive combining lock (V3) |
| `AdaptiveCombiningRealWorldBenchmark` | Benchmark harness using a real-world-style workload profile |
| `AdaptiveMCSLockV3Safety` | Correctness/safety check for the adaptive MCS lock (V3) |
| `AgingTripRateCheck` | Diagnostic for lock-aging/priority trip-rate behavior |
| `CLHStress` | Stress test for the CLH queue lock |
| `CSAwareSpinBenchmark` | Benchmark harness for critical-section-duration-aware spin sizing |
| `CombiningBenchmark` | Benchmark harness for the base flat-combining lock |
| `CombiningLockSafety` | Correctness/safety check for the base combining lock |
| `EstimatorFreezeTest` | Diagnostic for the online duration-estimator behavior |
| `FinalSweep` | Full parameter sweep across lock variants/thread counts/durations |
| `GuardValidation` | Validation check for the reliability guard mechanism |
| `JitterProbe` | Diagnostic for measuring host timing jitter |
| `MCSLockSafety` | Correctness/safety check for the plain MCS lock |
| `MeStressTest` | Stress-test harness |
| `MechanismTrace` | Diagnostic that traces internal lock mechanism transitions |
| `NT2Diagnostic` | Diagnostic focused on the 2-thread configuration |
| `OverheadIsolation` | Micro-benchmark isolating fixed per-operation overhead |
| `PhaseSweep` | Sweep across workload phases |
| `PolicyDiagnostic` | Diagnostic for lock scheduling/backoff policy behavior |
| `PollingOverheadCheck` | Micro-benchmark of polling-loop overhead |
| `PrimitiveControlV2` | Control benchmark using primitive synchronization only |
| `PrimitiveProbe` | Diagnostic probing primitive synchronization costs |
| `PriorityLockBenchmark` | Benchmark harness for the priority lock |
| `PriorityLockDiagnostic` | Diagnostic for priority-lock internal state |
| `PriorityLockSafety` | Correctness/safety check for the priority lock (V1) |
| `PriorityLockV2Safety` | Correctness/safety check for the priority lock (V2) |
| `RLSyncBenchmark` | Main benchmark harness for the RL-Sync lock variants |
| `RLSyncBenchmarkFixed` | Fixed-configuration variant of the RL-Sync benchmark |
| `SimpleSpinlockSafety` | Correctness/safety check for a basic spinlock |
| `StarvationCompare` | Benchmark comparing starvation behavior across locks |
| `StructuralTest` | Structural/sanity test across lock variants |
| `TailLatencyDiagnostic` | Diagnostic for tail-latency (p99, etc.) measurement |
| `ThreeWayHomogeneous` | Benchmark comparing three lock variants under identical load |
| `ViolationIsolation` | Diagnostic isolating mutual-exclusion violations |
| `WarmupCheck` | Diagnostic for JIT/host warm-up effects on measurements |

### Supporting classes (no `main`, used by the programs above)

`AdaptiveCombiningLock`, `AdaptiveCombiningLockV2`, `AdaptiveCombiningLockV3`,
`AdaptiveMCSLockV3`, `AdaptivePriorityLock`, `AdaptivePriorityLockV2`,
`CombiningLock`, `MCSLock`, `TASLock`, `TATASLock`, `TicketLock` — lock
implementations imported by the benchmark/diagnostic/safety classes above.

## Example

```bash
mkdir -p out
javac -encoding UTF-8 -d out src/*.java
java -cp out StructuralTest > structural_test_output.csv
```

## Project layout

```
lock-benchmark-suite/
├── README.md
├── .gitignore
└── src/            # all Java sources (default package, no build tool needed)
```
