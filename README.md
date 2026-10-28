This is the README.m
# Tail latency of the shm telemetry writer on the market data hot path

## Setup and symptom

- Single writer on the market data ingestion hot path, publishing telemetry into a shared-memory broadcast ring that a separate exporter process reads.
- Ring: huge pages, prefaulted, every record padded to whole cachelines, always overwrite, no backpressure.
- Writes are memcpy-style 16-byte SSE stores (AVX2/AVX-512 not allowed), with prefetchw on the destination.
- 128-256 bytes per publish, so 8-16 stores into 2-4 cachelines. No tracing on the fast abort path.
- CPU: Skylake.
- Result: median and p99 unchanged, p99.99 +8%, p99.999 +17%.

## Mechanism

- Skylake's store buffer has 56 entries. One publish uses 8-16 of them, so the telemetry stores alone can't fill it.
- Stores commit to L1 in program order (x86 TSO). If one of the 2-4 destination lines isn't owned when its store reaches the head of the buffer, every younger store waits behind it.
- After the publish the hot path keeps issuing its own stores. Once ~40 more are queued, the core stalls until the RFO completes, which is a few hundred cycles for a cross-core or L3-miss RFO.
- Fences and locked instructions turn any in-flight RFO into a stall right away, since they wait for the store buffer to drain.
- p99 stays clean because prefetchw usually lands in time. The tail is the cases where it doesn't:
  - back-to-back publishes during a burst leave one iteration of lead time for "prefetch the next record", and bursts are where iterations are shortest
  - bursts also fill the ~10-12 L1 fill buffers with demand misses, so the prefetchw waits behind them
  - the exporter reads a line between the prefetchw and the store and pulls it back to shared

## Fixes

### 1. Take the RFO off the hot path with an L1-resident staging buffer

- Small private ring, e.g. 8-16 slots x 256B = 2-4KB, reused cyclically and never touched by another core. Its lines stay modified in L1/L2, so every publish is 8-16 store hits.
- Copy staging -> shm in the idle branch of the poll loop (rx ring empty). Do one record per idle spin so a packet arriving mid-flush waits for at most one copy.
- Size it to the burst depth you want to absorb. On overflow either drop and count, or write through to shm.
- Cost: 2-4KB of the 32KB L1D that the hot path loses.
- Same pattern as NanoLog's per-thread staging buffers.

### 2. Nothing that drains the store buffer

- x86 TSO already orders store->store, so payload stores followed by a release store need no fence. The release store should compile to a plain `mov`; check the asm.
- These all wait for the store buffer to drain: `mfence`, `std::atomic_thread_fence(seq_cst)`, seq_cst stores (`xchg`), `fetch_add` (`lock xadd`). With a single writer use load + release store, never an RMW.
- `sfence` is only needed with NT stores.

### 3. Prefetch

- GCC/Clang emit `prefetchw` for `__builtin_prefetch(p, 1, 3)` / `_MM_HINT_ET0` only when PRFCHW is enabled (`-mprfchw` or a Broadwell+ `-march`). A conservative baseline build silently emits `prefetcht0`, which brings the line in shared and leaves the RFO for the store. Enable it for the Skylake path and check the disassembly.
- Prefetch 1-2 records (4-8 lines) ahead and re-issue in the idle branch, so lead time doesn't depend on publish spacing.

### 4. Watermark publication instead of a shared counter per publish

Writer:

- Each record header holds a seq word. Store an in-progress value, write the payload, store the final seq (a seqlock per record). The header shares the first padded line with the payload, so this costs no extra line.
- Don't store a global index on every publish. Publish a watermark from the idle branch, and only when it changed.
- Forced publish every N records (N a fraction of capacity, e.g. capacity/4). Without it, a burst that never reaches the idle branch freezes the watermark until the burst's own records get overwritten, which loses exactly the burst telemetry. Prefetchw the watermark line a few records before each forced publish.

Exporter:

- Reads strictly below the watermark, so it never touches lines in the writer's next few records.
- Polls the watermark on a coarse timer. That keeps the line modified in the writer's L1 most of the time, so idle-branch publishes don't need an RFO. A pending RFO from the idle store would otherwise sit at the head of the store buffer if a packet arrives right after.
- Validates each record: load seq, copy, reload seq, drop on mismatch.
- If it's more than `capacity - margin` behind the watermark, jump forward and count the gap. `margin` covers how far the writer can advance during one exporter copy batch.

### 5. Exporter placement

- Different physical core from the writer (not the HT sibling), same socket, shm bound to that NUMA node.
- With the sibling hyperthread active the store buffer is statically partitioned, so the writer gets roughly half of it.
- Ring large enough that the exporter is long done with a slot before the writer wraps around to it.

### 6. Cache pollution (measure first)

- Streaming through a large ring evicts the decoder/book working set from L1/L2, which shows up later as misses in code that looks unrelated. Compare hot-path L1D/L2 misses with telemetry on vs off.
- Options: fewer bytes per record, a ring small enough to stay L2-resident, or SSE2 NT stores (`_mm_stream_si128`, full 64B lines only, one `sfence` per batch before publishing).
- NT stores occupy fill buffers that the hot path's loads need too, so only use them if pollution is the measured cause.

## Diagnosis

- Detach the exporter. Tails gone -> coherence (4, 5). Still there -> store buffer / fences (2) or pollution (6).
- Skylake counters: `RESOURCE_STALLS.SB`, `L2_RQSTS.RFO_MISS`, `OFFCORE_REQUESTS.DEMAND_RFO`.
- rdtsc-bracket the publish call and check whether hot-path tail samples line up with writer tail samples.

## Literature

### Broadcast-buffer implementations

None of these prefetch on the writer side. The differences are protocol and layout.

- Agrona/Aeron `BroadcastTransmitter`: tail-intent, tail and latest counters at offsets 0/8/16 of a two-cacheline trailer, so all on one line. The transmitter does an ordered store of tail intent plus a release fence before writing the record. Receivers compare cursor + capacity against tail intent to detect being lapped. Every message writes the line all receivers poll.
- bcast (Rust): one producer position on its own padded line, release-stored on every commit, frames only 8-byte aligned. Readers re-read the position after copying to detect overrun.
- MengRao SPMC_Queue: the writer never waits and doesn't know readers exist; a reader more than one queue length behind drops messages.
- hft-channel (Rust): per-slot dirty bit + 63-bit seq. The sender marks the next slot dirty, writes the payload, clears dirty; a changed seq tells a receiver the slot was overwritten.
- Agrona and bcast store a global counter on every message. The per-slot seq designs don't, and that's the part relevant to the tail here.

### Papers and posts

- FastForward (Giacomoni et al., PPoPP 2008): cache-aware timing so the hardware prefetcher removes most compulsory misses. Temporal slipping keeps producer and consumer on different cache lines (3.7x over Lamport's buffer); the consumer is held back at least until the producer has filled a line.
- MCRingBuffer: producer-local, consumer-local and shared control variables on separate lines; shared indices updated once per batch; up to 5x throughput.
- Rigtorp's ring buffer: the producer caches the consumer's index and only re-reads it when the cache says the ring might be full. The writer keeps its own index on the same line as that cache, so it only ever writes its own line.
- NanoLog (Yang, Park, Ousterhout, USENIX ATC 2018): per-thread SPSC staging buffers drained by an emission thread, formatting deferred offline, ~1MB per thread plus one core, median just over 7ns per log call.
- Store-Prefetch Bursts (Cebrian, Kaxiras, Ros, MICRO 2020): store latency only shows when the store buffer is full; hardware already issues an ownership prefetch when a store commits; most store-buffer stalls come from a few stores; SMT statically partitions the store buffer; Skylake has 56 entries.
- CLDEMOTE: moves a line from the private cache to the LLC so another core's access skips the snoop. Intel's patent claims more than 2x lower core-to-core latency. Not available on Skylake (Tremont / Sapphire Rapids onward).

### Links

- [aeron-go broadcast receiver](https://github.com/lirm/aeron-go/blob/f6902ab3164a/aeron/broadcast/receiver.go)
- [aeron-rs BroadcastTransmitter](https://docs.rs/aeron-rs/0.1.1/src/aeron_rs/concurrent/broadcast/broadcast_transmitter.rs.html)
- [bcast source](https://docs.rs/crate/bcast/latest/source/src/lib.rs)
- [SPMC_Queue](https://github.com/Elite-stay/SPMC_Queue)
- [hft-channel](https://docs.rs/hft-channel)
- [FastForward (Giacomoni et al.)](https://scholar.colorado.edu/downloads/v405sb189)
- [SPSC queues on shared-cache multicores (Torquati)](https://arxiv.org/pdf/1012.1824)
- [MCRingBuffer (survey section)](https://arxiv.org/pdf/1801.10556)
- [Rigtorp ring buffer discussion, DPDK list](https://mails.dpdk.org/archives/dev/2023-June/271923.html) (original post: https://rigtorp.se/ringbuffer/)
- [NanoLog, USENIX ATC 18](https://www.usenix.org/conference/atc18/presentation/yang-stephen)
- [Store-Prefetch Bursts, MICRO 53](https://www.microarch.org/micro53/papers/738300a568.pdf)
- [CLDEMOTE analysis](https://arxiv.org/pdf/2503.10074)
- [Intel CLDEMOTE patent](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10564972)
