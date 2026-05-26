//! Bounded-channel async MCAP writer (W7 PR4).
//!
//! [`AsyncRecorder`] wraps a synchronous [`AsyncSink`] implementor (in the
//! recorder hot path this is the same `mcap::write::Writer<BufWriter<File>>`
//! used by the sync path) behind a bounded
//! [`crossbeam_channel`] queue. The Bevy `PostUpdate` schedule pushes
//! frames into the queue via [`AsyncRecorder::try_send_frame`]; a single
//! background worker thread drains the queue and routes every
//! [`RecorderMessage::Frame`] through the sink.
//!
//! # Rationale
//!
//! Per WS7-plan § 8 risk 4 the sync recorder is a step-rate bottleneck
//! whenever the disk hiccups (a 50 ms `fsync` stalls the entire physics
//! tick). Decoupling the writer onto a background thread keeps the
//! simulation step at its target Hz at the cost of bounded memory
//! pressure when the producer outruns the consumer.
//!
//! # Backpressure model
//!
//! The channel is **bounded** (configurable; default 256). When the queue
//! is full, [`AsyncRecorder::try_send_frame`] returns immediately and
//! atomically increments a [`DroppedFrames`] counter — it **never blocks
//! the producer**. This is the explicit trade-off documented in the W7
//! plan: a slow disk costs you log fidelity, not simulation throughput.
//!
//! ```text
//! [Bevy PostUpdate] --try_send_frame--> [bounded(256)] --worker--> [AsyncSink]
//!                          ↓ Full
//!                  fetch_add(DroppedFrames)
//! ```
//!
//! # Determinism
//!
//! The worker drains messages in FIFO order. Across runs with the same
//! producer cadence the sink sees identical bytes (the channel preserves
//! send order). Drops are observable only when the producer outruns the
//! consumer; the test suite covers the zero-drop sufficient-buffer
//! invariant.
//!
//! # Shutdown
//!
//! [`AsyncRecorder::close`] sends a [`RecorderMessage::Close`] sentinel
//! and joins the worker. `Drop for AsyncRecorder` also closes — bounded
//! in time only by the worker draining whatever frames are still in the
//! queue. Callers that need a hard upper bound should call
//! [`AsyncRecorder::close`] explicitly with their own timeout.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::{self, JoinHandle};

use bevy::prelude::Resource;
use crossbeam_channel::{Sender, TrySendError, bounded};

// ---------------------------------------------------------------------------
// AsyncSink
// ---------------------------------------------------------------------------

/// Backend abstraction consumed by the async worker thread.
///
/// In the production recorder hot path this is implemented by an adapter
/// that wraps the [`mcap::write::Writer`] held by [`crate::recorder::Recorder`].
/// Tests in `crates/clankers-record/tests/async_backpressure.rs`
/// implement a `SleepingSink` to simulate a slow disk.
pub trait AsyncSink {
    /// Write a single MCAP message to the underlying writer.
    ///
    /// # Errors
    ///
    /// Returns an [`std::io::Error`] if the underlying writer reports a
    /// failure. The async worker logs the error and continues processing
    /// remaining messages — a single failed write does not tear down
    /// the worker.
    fn write_message(
        &mut self,
        channel_id: u16,
        log_time_ns: u64,
        payload: &[u8],
    ) -> std::io::Result<()>;

    /// Flush any buffered bytes to the underlying writer.
    ///
    /// # Errors
    ///
    /// Returns an [`std::io::Error`] if the underlying writer cannot
    /// flush.
    fn flush(&mut self) -> std::io::Result<()>;
}

// ---------------------------------------------------------------------------
// RecorderMessage
// ---------------------------------------------------------------------------

/// One message sent to the async writer worker thread.
///
/// `Frame` carries a full owned payload (`Vec<u8>`) so the producer can
/// drop its borrow as soon as `try_send_frame` returns. `Flush` is a
/// best-effort sync point; `Close` is the worker shutdown sentinel.
#[derive(Debug)]
pub enum RecorderMessage {
    /// Append a single MCAP message on `channel_id` at `log_time_ns`
    /// with the given owned `payload`.
    Frame {
        /// MCAP channel id (matches the value returned by
        /// `mcap::Writer::add_channel`).
        channel_id: u16,
        /// MCAP log timestamp in nanoseconds (sim time).
        log_time_ns: u64,
        /// Owned message payload bytes.
        payload: Vec<u8>,
    },
    /// Best-effort `flush` on the sink. Not used by the production
    /// recorder; available for explicit sync points in tests.
    Flush,
    /// Worker shutdown sentinel. Sent by [`AsyncRecorder::close`] and
    /// `Drop for AsyncRecorder`.
    Close,
}

// ---------------------------------------------------------------------------
// DroppedFrames
// ---------------------------------------------------------------------------

/// Bevy resource exposing the running count of frames dropped on the
/// recorder hot path because the bounded queue was full.
///
/// Cloned from the [`AsyncRecorder`]'s internal `Arc<AtomicU64>` at
/// construction so reading the resource and reading
/// [`AsyncRecorder::dropped_frames`] always agree.
///
/// # Default
///
/// `DroppedFrames::default()` constructs a fresh `Arc<AtomicU64::new(0)>`.
/// When the recorder runs in sync mode (the default), this counter stays
/// at 0 — no drops are possible on the sync path because every
/// [`crate::recorder::Recorder::write_json`] call blocks until the
/// underlying writer accepts the bytes.
#[derive(Resource, Clone, Debug, Default)]
pub struct DroppedFrames(pub Arc<AtomicU64>);

impl DroppedFrames {
    /// Load the current dropped-frame count (relaxed ordering — the
    /// counter is a monotonic monitoring metric, not a synchronisation
    /// primitive).
    #[must_use]
    pub fn get(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// AsyncRecorder
// ---------------------------------------------------------------------------

/// Producer-side handle for the bounded-channel async writer.
///
/// Construct via [`AsyncRecorder::new`] with a capacity, a sink, and an
/// optional shared `Arc<AtomicU64>`. The shared counter lets the Bevy
/// [`DroppedFrames`] resource and the recorder agree on the running drop
/// count without cross-thread coordination.
pub struct AsyncRecorder {
    tx: Sender<RecorderMessage>,
    dropped: Arc<AtomicU64>,
    worker: Option<JoinHandle<()>>,
}

impl AsyncRecorder {
    /// Spawn the worker thread and return a producer handle.
    ///
    /// `capacity` bounds the queue depth; once full,
    /// [`Self::try_send_frame`] increments the dropped-frame counter and
    /// drops the payload. `dropped` is an optional shared counter — if
    /// `None`, a fresh `Arc<AtomicU64>` is allocated and can be read back
    /// via [`Self::dropped_frames_handle`].
    pub fn new<S>(capacity: usize, sink: S, dropped: Option<Arc<AtomicU64>>) -> Self
    where
        S: AsyncSink + Send + 'static,
    {
        let (tx, rx) = bounded::<RecorderMessage>(capacity);
        let dropped = dropped.unwrap_or_else(|| Arc::new(AtomicU64::new(0)));
        let mut sink = sink;
        let worker = thread::Builder::new()
            .name("clankers-record-async".into())
            .spawn(move || {
                while let Ok(msg) = rx.recv() {
                    match msg {
                        RecorderMessage::Frame {
                            channel_id,
                            log_time_ns,
                            payload,
                        } => {
                            if let Err(e) = sink.write_message(channel_id, log_time_ns, &payload) {
                                // We deliberately log via eprintln rather
                                // than bevy::log because the worker thread
                                // outlives the Bevy app's logger setup.
                                eprintln!("clankers-record async worker: write failed: {e}");
                            }
                        }
                        RecorderMessage::Flush => {
                            if let Err(e) = sink.flush() {
                                eprintln!("clankers-record async worker: flush failed: {e}");
                            }
                        }
                        RecorderMessage::Close => break,
                    }
                }
                // Best-effort final flush on shutdown.
                let _ = sink.flush();
            })
            .expect("failed to spawn clankers-record async worker");
        Self {
            tx,
            dropped,
            worker: Some(worker),
        }
    }

    /// Attempt to enqueue a single frame.
    ///
    /// Never blocks: on a full queue, increments the dropped-frame
    /// counter and discards the payload. On worker disconnection
    /// (i.e. the worker thread panicked), the counter is still
    /// incremented so callers see the loss.
    pub fn try_send_frame(&self, channel_id: u16, log_time_ns: u64, payload: Vec<u8>) {
        match self.tx.try_send(RecorderMessage::Frame {
            channel_id,
            log_time_ns,
            payload,
        }) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
            }
            Err(TrySendError::Disconnected(_)) => {
                // Worker died — count and continue. The sync recorder
                // fallback will eventually be observed by the producer
                // through Drop semantics on the Recorder itself.
                self.dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Current dropped-frame count (relaxed load).
    #[must_use]
    pub fn dropped_frames(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// Cloneable handle to the shared dropped-frame counter.
    ///
    /// Insert the corresponding [`DroppedFrames`] resource into the Bevy
    /// world by wrapping this handle: `DroppedFrames(handle.clone())`.
    #[must_use]
    pub fn dropped_frames_handle(&self) -> Arc<AtomicU64> {
        self.dropped.clone()
    }

    /// Send the [`RecorderMessage::Close`] sentinel and join the worker.
    ///
    /// Blocks until the worker drains the queue and exits. The wait time
    /// is bounded by (queue depth × per-frame sink latency); callers
    /// that need a hard upper bound should run `close` from a spawned
    /// thread with a join timeout.
    pub fn close(mut self) {
        // Best-effort: if the worker already exited the send fails; we
        // still try to join.
        let _ = self.tx.send(RecorderMessage::Close);
        if let Some(j) = self.worker.take() {
            let _ = j.join();
        }
    }
}

impl Drop for AsyncRecorder {
    fn drop(&mut self) {
        // Best-effort shutdown — see `close` docstring.
        let _ = self.tx.send(RecorderMessage::Close);
        if let Some(j) = self.worker.take() {
            let _ = j.join();
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    type Writes = Arc<Mutex<Vec<(u16, u64, Vec<u8>)>>>;

    /// Test sink that records every write.
    struct VecSink {
        writes: Writes,
    }

    impl AsyncSink for VecSink {
        fn write_message(
            &mut self,
            channel_id: u16,
            log_time_ns: u64,
            payload: &[u8],
        ) -> std::io::Result<()> {
            self.writes.lock().expect("VecSink mutex poisoned").push((
                channel_id,
                log_time_ns,
                payload.to_vec(),
            ));
            Ok(())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn dropped_frames_default_is_zero() {
        let d = DroppedFrames::default();
        assert_eq!(d.get(), 0);
    }

    #[test]
    fn fifo_order_preserved_under_sufficient_buffer() {
        let writes = Arc::new(Mutex::new(Vec::new()));
        let sink = VecSink {
            writes: writes.clone(),
        };
        let rec = AsyncRecorder::new(1024, sink, None);
        for i in 0..16_u64 {
            rec.try_send_frame(0, i, vec![u8::try_from(i & 0xFF).unwrap_or(0)]);
        }
        rec.close();
        let writes = writes.lock().unwrap();
        assert_eq!(writes.len(), 16);
        for (i, w) in writes.iter().enumerate() {
            assert_eq!(w.1, i as u64, "log_time_ns mismatch");
        }
    }
}
