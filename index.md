---
layout: home
title: Daniel's Blog
---

Welcome to my math notes. Heavily WIP, feel free to tell me suggestions and things I can improve.

If there is something you want to discuss feel free to reach out, I will add comment functionality later on.

# Deterministic replay and simulation for market data pipelines

Research notes on deterministic replay, event linearisation, fault injection and race reproduction, mapped onto a single-hot-thread market data / book building pipeline running inside a simulated environment.

---

## 0. System and constraints

### System as it stands

- One hot thread. The decoder spins until input shows up, then the templated pipeline runs end to end in one go. FPGA decoding sits upstream and is not modelled.
- Inputs:
  - one or two UDP multicast streams (A/B), replayed from pcaps
  - one bidirectional TCP session per venue, backed by a TCP mock that knows the supported message types plus an exchange simulation that reacts to outbound messages
  - a reference data lookup
  - filesystem effects ignored
- Output: the publisher has no observable side effects, so a sink captures it. Apart from errors, TCP out is the only external observable.
- One authoritative simulated clock in shared memory. Modes:
  - realtime
  - linear (time scaled by a constant factor, i.e. a faster or slower machine)
  - event-based (infinitely fast machine, never stalls from the sim's point of view)
  - transport and processing latency models can be layered on top
- Separate logging and error recovery threads.
- Introspection by serialising memory regions through an attached debugger that can freeze threads (e.g. reading the book out of the matching engine into a dashboard during replay).
- Binaries are swappable (pre-fix vs post-fix).

### Constraints

- Around 50 market data ingestion implementations exist, mostly sharing one matching engine. They are 15 years of legacy and will not be rewritten to make them testable. Newer implementations may be cleaner but that is not guaranteed.
- Goals: full determinism, precise control over reproduction scenarios, replayable end-to-end fixtures, and cheap onboarding of new venues into the sim.
- Benchmarking is out of scope.

### Modelling assumption

Event processing is atomic. One UDP packet goes through the pipeline end to end before any other input is observed. There is no modelling of a TCP message landing between decoder and handler vs inside the handler.

---

## 1. Lineage: deterministic simulation testing (DST)

The pattern is DST as practised by FoundationDB and TigerBeetle, narrowed to one node. Both of those were designed for simulation from day one, which does not apply to legacy code (section 3 covers that route). What carries over is the testing discipline.

- **FoundationDB.** Built the simulator before the database. Network, process, disk and request-level faults are simulated inside a single process by swapping physical interfaces for shims and the epoll run loop for a simulated-time loop. Also uses `BUGGIFY`, in-code fault points that fire randomly in simulation (unusable on untouched legacy code, usable in new implementations).
- **TigerBeetle VOPR.** Seed plus git commit reproduces a run exactly, simulated time runs arbitrarily fast, assertions stay on in production.
  - Liveness mode: after the fault phase, heal a core quorum, restart downed core replicas, make faults on the rest permanent, and require progress.
  - Feed analogue: stop injecting at time T, then assert the book reconverges to the reference within a bounded sim-time budget. Recovery livelocks (gap -> snapshot -> gap during snapshot -> ...) only show up under a convergence check like this.
- **Antithesis.** Deterministic hypervisor that runs the whole system on one core, instruction-level deterministic for arbitrary x86 binaries, multicast supported. Explicitly aimed at systems that were never designed for DST. dhyve is an open-source bhyve-based project along the same lines.
- **Hermit (Meta).** User-space determinisation layer that intercepts syscalls and preempts threads after a fixed number of conditional branches counted by the PMU, so it can generate schedules reproducibly instead of only recording them. `hermit analyze` searches for a failing and a passing schedule and reports the racing operations whose reordering flips the outcome. Maintenance mode, x86_64 Linux only.

The search problem here is much smaller than in a replicated database: one node, atomic steps, no consensus protocol. The hard parts move to retrofitting determinism onto code that can't change (section 3) and to mock fidelity across venues (section 9).

---

## 2. The atomic-event assumption

Run-to-completion handlers are the standard step granularity in message-passing model checking. DEMi models a system as independent single-threaded processes exchanging messages, and the P language uses the same semantics. The assumption is sound exactly when no handler can observe another input while it runs.

### What it buys

- The ordering space collapses to sequences of whole events across sources. Within-event interleavings drop out. What remains is the order of ready events plus arrival-time perturbations.
- Every debugger stop between events lands on a quiescent, consistent state. Memory dumps become well-defined snapshots, and checkpointing / branching is clean.

### Where it breaks

- **Shared-memory inputs.** If refdata is read from memory another process writes, a handler can see a half-applied update mid-event. Kernel/NIC-queued inputs only become visible at the next poll; shared memory has no such boundary. Modelling refdata as versioned snapshots that only change between events keeps the assumption intact, at the cost of never reproducing a torn read that production can produce.
- **Mid-event polling.** Any handler that polls or drains another source inside an event, e.g. checking the TCP socket in the UDP path, or pulling snapshot packets synchronously during recovery.
- **Other threads.** Logging or recovery threads writing state the hot path reads.

### What remains: the poll policy

"Arrived during the handler" vs "arrived just before the decoder picked up the packet" only differ in which ready input the loop takes next. That makes the hot loop's poll policy part of the model.

- Fixed UDP priority, round-robin, and draining a burst of N packets before checking TCP each order identical arrival times differently.
- If the sim uses timestamp order with a generic tie-break while production drains bursts, TCP responses land at different points in the book sequence than they would live.

---

## 3. Determinism without touching legacy code

Control has to come from seams that already exist outside the source (Feathers, *Working Effectively with Legacy Code*). With one hot thread and atomic events, determinism reduces to making every input pass through a seam the sim controls. That turns it into an audit problem: find every input that bypasses a seam, then trap it or prove it doesn't matter.

### Seams

| Seam | Mechanism | Catches | Bypassed by / cost |
|---|---|---|---|
| Link | `-Wl,--wrap=sym`, mock libraries | out-of-line symbols | inlined templates, intrinsics; needs a relink, so "pre-fix binary" means pre-fix objects relinked against the harness |
| Preload | `LD_PRELOAD` | anything resolved through the dynamic linker: libc I/O, `clock_gettime`, vendor NIC libraries | static linking, direct vDSO use, inline syscalls. Upside: shipped binary runs unmodified, closest to prod, binary swap is trivial |
| Syscall / process | Hermit | syscalls, thread scheduling | x86_64 Linux only, maintenance mode |
| Machine | Antithesis | everything | commercial, single core |

### Leak checklist and how to trap each one without code changes

- **`rdtsc` / `rdtscp`.** `prctl(PR_SET_TSC, PR_TSC_SIGSEGV)` makes the instruction fault. A SIGSEGV handler installed from a preload constructor can return sim time. rr relies on the same mechanism.
- **Wall-clock reads** (`clock_gettime`, `gettimeofday`, `std::chrono`). Interposable via preload when they resolve through the dynamic linker. Perturbation check: run the same fixture under different `libfaketime` offsets and with injected wall-clock stalls. Any output change means something reads real time behind the sim clock.
- **Uninitialised memory.** Run with `MALLOC_PERTURB_` set to two different values and diff the sink. Valgrind memcheck works on the unmodified binary, just slowly.
- **Address-dependent behaviour** (pointer-keyed hashing, iteration order following allocator addresses). Diff runs with ASLR off (`setarch -R`) and on.
- **Spin-count-as-timeout logic.** Spinning is free in event mode, so anything counting spin iterations as a timeout proxy behaves differently from realtime.
- **Floating point.** `-ffp-contract` differences between pre-fix and post-fix builds change results if any price math uses doubles.
- **Logging / recovery threads.** Can't be made deterministic without code changes short of Hermit or Antithesis. Check output invariance instead: freeze them at random points via the debugger (a cheap chaos mode) and assert TCP out and book state are unchanged.
- **Hot path <-> logging/recovery handoff** (atomics, SPSC queues). GenMC model-checks C/C++ atomics under RC11, exploring each consistent execution exactly once, with automatic spinloop bounding. Extract the queue protocol into a harness; the harness is new code, so the legacy constraint doesn't apply.
- **Realtime mode.** Emit the realised event order in fixture format, so any realtime-mode failure becomes an event-mode fixture (record at the boundary, the same idea rr is built on).

### Detection over prevention

- Hash the serialised state at every event boundary. Run each fixture twice under different perturbations. The first event whose hash diverges localises the leak; bisect inside that event with the debugger.
- This is the per-tick state checksum technique lockstep games use to catch desync (Bettner and Terrano, "1500 Archers on a 28.8", Age of Empires, 2001), which was also retrofitted onto code nobody had written with determinism in mind.

---

## 4. Ordering space (linearisation)

### Simultaneous events

- Standard tie-breaking fixes one order for same-timestamp events and implicitly treats the others as irrelevant. MOOSE instead branches over the possible orders of simultaneous events.
- Make the tie-break policy and the poll policy (section 2) explicit fixture parameters.
- Add a mode that permutes ties and near-ties within an epsilon window, since realtime jitter turns "near" into "tied". In an event-queue design this is the cheapest race search available.

### Partial-order reduction

- DPOR (Flanagan and Godefroid, POPL 2005) explores one linearisation per equivalence class of orderings that only swap independent events.
- Constrained DPOR (CAV 2018) adds conditional independence: two events commute only when declared constraints hold.
- Opaque templated code gives no read/write sets, so declare independence at the domain level:
  - different instruments / partitions
  - A and B copies of the same sequence number once one has been applied
  - refdata responses for unrelated symbols
- This is the approach of SAMC (semantic-aware model checking, OSDI 2014).

### Bug-depth randomised schedulers

- **PCT** (Burckhardt et al., ASPLOS 2010): finds a depth-d bug with probability >= 1/(n * k^(d-1)) per run, for n threads and k steps.
- **PCTCP** (Ozkan et al., OOPSLA 2018): adapts PCT to message-passing systems, whose structure breaks PCT's assumptions.
- **taPCTCP** (OOPSLA 2019): combines partial-order reduction with bug depth.
- **POS** (Yuan, Yang, Gu, CAV 2018): samples partial orders rather than total orders, and stays effective where PCT degrades under many ordering events. The Fray paper (OOPSLA 2025) reports POS needing fewer iterations than PCT on real races in Kafka, Lucene and Guava, with a resample probability of 0.5 as a reasonable default.
- **Mapping.** Each event source (UDP-A, UDP-B, TCP, refdata, timers, recovery) plays the role of a thread. Where production's poll policy is deterministic, the sim's freedom lies in arrival times, and PCT/POS priorities decide which source gets delayed past which.

### Bounded deviation

Delay-bounded scheduling (Emmi, Qadeer, Rakamaric, POPL 2011) enumerates every schedule within k deviations from a canonical one. For small k this is a systematic alternative to sampling: every single swap of adjacent cross-source events, then pairs, restricted to windows around interesting points (gaps, recovery, session events).

---

## 5. Fault targeting

### Importance sampling vs search

- Importance sampling earns its keep when the goal is a failure probability. O'Kelly et al. (NeurIPS 2018) fit the sampling distribution iteratively with the cross-entropy method and compute in log space to keep high-dimensional weights numerically stable.
- That needs a calibrated base distribution. For bug finding, every system surveyed here uses biased search with no reweighting.

### Loss models

- I.i.d. drops underweight microburst loss. The standard is Gilbert-Elliott burst loss (two-state Markov; Gilbert 1960, Elliott 1963), fitted to production gap statistics.
- Correlated A+B loss on the same sequence range is the case that forces recovery. It is close to impossible under independent per-line drops, so it needs its own fault type.

### Lineage-driven fault injection (LDFI)

- Alvaro, Rosen, Hellerstein, SIGMOD 2015. Inject only faults capable of changing an outcome already observed to be correct.
- Their tool, Molly, needed about an order of magnitude fewer runs than random injection, and when it finds nothing it certifies the configuration bug-free up to a bounded depth.
- This is the formal version of building the TCP-out messages into a DAG and targeting faults at it.
- Lineage per output has to come from outside: sequence numbers visible at the sink, or delta runs (drop input i, check whether output j changes).

### Time-of-fault bugs

- FCatch (ASPLOS 2018) targets bugs that need a crash or dropped message to land at one specific moment, predicted from fault-free runs.
- Anchor injections to phase transitions rather than spreading them uniformly in time: gap detected, snapshot cycle start/end, TCP logon / resend, refdata reload.

### Swarm testing

- Groce, Zhang, Eide, Chen, Regehr, ISSTA 2012. Each run omits a random subset of features, because some features suppress bug-exposing behaviour and features crowd each other out within one test. Found 42% more distinct compiler crashes in a week than a hand-tuned default.
- Per seed, enable a random subset of {A drop, B drop, A+B correlated drop, reorder, duplicate, TCP disconnect, refdata delay, latency jitter}. An all-faults-on profile drowns rare compound sequences.

---

## 6. Feedback-guided search

- **Mallory** (CCS 2023). Abstracts Lamport timelines into happens-before summaries as fuzzing feedback and learns a fault-injection policy with Q-learning. Found 22 zero-days in systems already tested with Jepsen.
- **ModelFuzz** (Gulcan, Ozkan, Majumdar). Uses state coverage of an abstract formal model as the feedback signal, and beats random, code-coverage-guided and trace-coverage-guided fuzzing.
  - Works on untouched binaries, since coverage is defined over the model as observed at the sink.
  - Candidate abstract state: per-channel feed state (in-sync, gap, recovering, snapshot-applied, stale) x TCP session state x last fault kind.
- **Code coverage feedback** (AFL-style) needs `-fsanitize-coverage` or binary instrumentation. Only available if build flags can change.
- **RFF** (ASPLOS 2024). Uses the reads-from relation as feedback for schedule fuzzing. Relevant only at the hot path / recovery thread boundary.
- **Snapshot branching.** Freeze state before a fault window and fork many continuations instead of replaying the pcap prefix each time. Antithesis frames its search as exploring a multiverse of execution histories.

---

## 7. Minimisation and fixture format

### Internal reduction (Hypothesis)

- MacIver and Donaldson, ECOOP 2020. Shrink the generator's recorded random decisions instead of the generated test case, so every shrunk candidate is regenerated and therefore valid.
- Every sim decision (drop, delay, tie order, latency draw, poll choice) is drawn from one choice stream, with 0 meaning canonical.
- Fixture = (pcap hash, binary hash, sim config, choice stream).
- A shrunk stream reads as a short list of deviations from canonical replay, and sequence-number consistency can't break because the generator re-derives everything.

### Execution minimisation (DEMi)

- Scott et al., NSDI 2016. For failures captured from production, with no choice stream behind them.
- Minimise external input events first, then the internal message schedule, then payloads. Landed within 1x to 4.6x of optimal on Raft and Spark.
- On a sequenced feed, deleting a packet doesn't remove it; it creates a gap the feed logic will react to. Deletion has to be expressed as gaps or whole sequence ranges.

### Schedule bisection

`hermit analyze` bisects between a passing and a failing schedule to find the racing pair. The same idea works in-sim: given a passing and a failing choice stream, bisect over the deviations.

---

## 8. Oracles

- **Trace validation.** CCF (Microsoft, NSDI 2025) binds a TLA+ spec to its C++ implementation through trace validation running in CI, and caught six bugs. Trace validation is non-exhaustive, but about a dozen executions surfaced two serious data-loss bugs. Mapping: spec the feed recovery machine and the TCP session, then validate sink and log traces against it.
- **Imandra.** Formal venue models, with region decomposition enumerating every edge-case region of matching logic (SIX Swiss fill pricing as a worked example). Its protocol modelling generates both test packs and venue response simulators from a FIX spec. Same object as the per-venue TCP mock plus exchange sim; region decomposition gives a coverage target for the mock.
- **Metamorphic relations** (no reference implementation needed):
  - A-only, B-only and A+B yield identical books at quiescent points.
  - Drop-then-recover converges to the no-drop book.
  - Linear-mode time scaling leaves event-mode outputs unchanged, except where behaviour is latency-dependent by design.
  - Permuting events declared independent (section 4) leaves outputs unchanged.
- **Differential testing.** Pre-fix vs post-fix binary on the TCP-out stream, with normalisers for allowed divergences (timestamps, IDs).
- **Reference book.** One generic book builder over normalised events, compared against each implementation's book via debugger dumps at quiescent points.
- **Convergence / liveness.** See section 1.

---

## 9. Scaling to 50 venues

### Mock fidelity is the bottleneck

50 exchange sims each have to match real venue behaviour, including recovery quirks the specs describe ambiguously.

### Common-mode error

- When the same reading of a venue spec produces both the handler and the mock, both encode the same misreading and the replay passes.
- Knight and Leveson (IEEE TSE, 1986) found that even independently written versions fail together more often than independence predicts. One person writing both from one reading gives no independence at all.
- Mitigations:
  - Conformance-check each mock against captured production TCP sessions and pcaps: replay real exchange responses and diff them against the mock's responses.
  - Have different people write the handler and the mock where possible.

### Keeping venue logic constrained

Decompose each venue into reusable archetypes plus data:

- **codec:** wire format -> normalised events
- **book type:** MBO, MBP, depth-N price levels
- **recovery archetype:** snapshot multicast channel, retransmission request (MoldUDP64-style), TCP snapshot / replay, refresh-cycle only
- **session state machine** for the TCP side

The generic layer is written once and every venue inherits it: reference book, metamorphic relations, convergence check, and fault profiles keyed by recovery archetype. A new venue then costs a codec, a config, and occasionally a new archetype. Imandra's protocol language is a commercial instance of the same split, with the protocol spec as data that generates test packs and simulators.

---

## 10. Debugging and introspection

- **Re-execution time travel.** Deterministic replay plus periodic checkpoints lets you seek to any (fixture, event index) by restoring the nearest checkpoint and replaying forward. rr does the general-purpose version of this (O'Callahan et al., USENIX ATC 2017). Debugger dumps become queries at (fixture, event index).
- **rr chaos mode**, for the real binary outside the sim. It inserts frequent short starvation windows, because many races need a thread starved only briefly inside a narrow window of vulnerability. MongoDB's experiment with intermittent test failures showed chaos mode raising reproduction rates substantially.

---

## References

### DST and deterministic execution

- FoundationDB, SIGMOD 2021: https://doi.org/10.1145/3448016.3457559
- FoundationDB, CACM version: https://cacmb4.acm.org/magazines/2023/6/273229-foundationdb-a-distributed-key-value-store
- Will Wilson, "Testing Distributed Systems w/ Deterministic Simulation", Strange Loop 2014: https://thestrangeloop.com/2014/testing-distributed-systems-w-slash-deterministic-simulation.html
- TigerBeetle VOPR: https://docs.tigerbeetle.com/about/vopr
- TigerBeetle, protocol-aware DST: https://tigerbeetle.com/blog/2026-08-20-protocol-aware-dst
- DST overview incl. TigerBeetle liveness mode: https://mohakchugh.is-a.dev/blog/deterministic-simulation-testing
- Antithesis on DST: https://antithesis.com/resources/deterministic_simulation_testing/
- Antithesis deterministic hypervisor: https://antithesis.com/blog/deterministic_hypervisor/
- Antithesis product (incl. dhyve pointer): https://antithesis.com/product/
- Hermit blog: https://developers.facebook.com/blog/post/2022/11/22/hermit-deterministic-linux-testing/
- Hermit repo: https://github.com/facebookexperimental/hermit

### Ordering and scheduling

- MOOSE, simultaneous events: https://eudl.eu/doi/10.4108/nstools.2007.2019
- Deterministic tie-breaking in PDES: https://arxiv.org/pdf/2105.00069
- Flanagan and Godefroid, "Dynamic Partial-Order Reduction for Model Checking Software", POPL 2005
- Constrained DPOR, CAV 2018: https://docta.ucm.es/entities/publication/d33f594c-85f0-42db-9c62-36d1cfacfd35/full
- Leesatapornwongsa et al., "SAMC: Semantic-Aware Model Checking for Fast Discovery of Deep Bugs in Cloud Systems", OSDI 2014
- Burckhardt et al., "A Randomized Scheduler with Probabilistic Guarantees of Finding Bugs", ASPLOS 2010
- PCTCP, OOPSLA 2018: https://2018.splashcon.org/event/splash-2018-oopsla-randomized-testing-of-distributed-systems-with-probabilistic-guarantees
- taPCTCP, OOPSLA 2019: https://2019.splashcon.org/details/splash-2019-oopsla/61/Trace-Aware-Random-Testing-for-Distributed-Systems
- POS, CAV 2018: https://www.cs.columbia.edu/~junfeng/papers/pos-cav18.pdf
- Fray, OOPSLA 2025: https://arxiv.org/abs/2501.12618
- Emmi, Qadeer, Rakamaric, "Delay-Bounded Scheduling", POPL 2011

### Fault targeting and search

- O'Kelly et al., rare-event simulation, NeurIPS 2018: https://arxiv.org/abs/1811.00145
- LDFI, SIGMOD 2015: https://doi.org/10.1145/2723372.2723711
- LDFI summary: https://blog.acolyer.org/2015/03/26/lineage-driven-fault-injection/
- FCatch, ASPLOS 2018: https://www.aminer.org/pub/6218c4b15aee126c0fa1aa64/fcatch
- Haopeng Liu dissertation (DCatch / FCatch): https://knowledge.uchicago.edu/record/1996/files/Liu_uchicago_0330D_14954.pdf
- Swarm testing, ISSTA 2012: https://www.cefns.nau.edu/~adg326/issta12.pdf
- Regehr on swarm testing: https://blog.regehr.org/archives/591
- Gilbert, "Capacity of a Burst-Noise Channel", BSTJ 1960
- Elliott, "Estimates of Error Rates for Codes on Burst-Noise Channels", BSTJ 1963
- Mallory, CCS 2023: https://arxiv.org/abs/2305.02601
- ModelFuzz: https://arxiv.org/abs/2410.02307
- RFF, ASPLOS 2024: https://www.comp.nus.edu.sg/~umathur/publications/2024-asplos-wolff-rff-greybox-concurrency-fuzzing/
- RFF repo: https://github.com/dylanjwolff/RFF

### Minimisation

- Hypothesis internal reduction, ECOOP 2020: https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ECOOP.2020.13
- DEMi, NSDI 2016: https://www.usenix.org/conference/nsdi16/technical-sessions/presentation/scott

### Oracles and specification

- CCF smart casual verification, NSDI 2025: https://www.usenix.org/conference/nsdi25/presentation/howard
- TLA+ trace validation notes: https://docs.tlapl.us/using:tlc:trace_validation
- Imandra region decomposition, SIX example: https://docs.imandra.ai/imandra-docs/notebooks/six-swiss-exchange-pricing
- Imandra Connectivity: https://imandra.ai/connectivity
- Knight and Leveson, "An Experimental Evaluation of the Assumption of Independence in Multiversion Programming", IEEE TSE 1986

### Legacy code, concurrency checking, debugging

- Feathers, *Working Effectively with Legacy Code*, 2004
- GenMC: https://plv.mpi-sws.org/genmc/
- Bettner and Terrano, "1500 Archers on a 28.8: Network Programming in Age of Empires and Beyond", 2001
- O'Callahan et al., "Engineering Record and Replay for Deployability", USENIX ATC 2017
- rr chaos mode: https://robert.ocallahan.org/2016/02/introducing-rr-chaos-mode.html
- rr, deeper into chaos: https://robert.ocallahan.org/2016/02/deeper-into-chaos.html
- rr chaos mode, MongoDB data: https://robert.ocallahan.org/2019/01/experimental-data-on-reproducing.html
