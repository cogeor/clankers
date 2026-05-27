//! Integration tests for the W7 PR4 async recorder backpressure model.
//!
//! Three cases per WS7-plan § 6:
//!
//! 1. `async_recorder_drops_frames_under_backpressure` — overflow case.
//! 2. `async_recorder_zero_drops_when_buffer_sufficient` — happy path.
//! 3. `async_recorder_close_flushes_pending_frames` — clean shutdown.
//!
//! Each test wraps the assertion in a join-timeout guard so a deadlock
//! fails the test (and CI) within 5 seconds rather than hanging
//! indefinitely.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use clankers_record::prelude::{AsyncRecorder, AsyncSink};

// ---------------------------------------------------------------------------
// SleepingSink — simulates a slow disk
// ---------------------------------------------------------------------------

/// Recorded `(channel_id, log_time_ns, payload)` triples.
type Writes = Arc<Mutex<Vec<(u16, u64, Vec<u8>)>>>;

/// `AsyncSink` adapter that records every write and sleeps `delay`
/// before returning. Used to force the bounded queue to fill so the
/// `try_send_frame` backpressure behaviour is observable.
struct SleepingSink {
    writes: Writes,
    delay: Duration,
}

impl AsyncSink for SleepingSink {
    fn write_message(
        &mut self,
        channel_id: u16,
        log_time_ns: u64,
        payload: &[u8],
    ) -> std::io::Result<()> {
        if !self.delay.is_zero() {
            thread::sleep(self.delay);
        }
        self.writes
            .lock()
            .expect("SleepingSink mutex poisoned")
            .push((channel_id, log_time_ns, payload.to_vec()));
        Ok(())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Test harness
// ---------------------------------------------------------------------------

/// Run `body` on a worker thread; fail the test if it does not return
/// within `timeout`. Returns the body's return value on success.
fn run_with_timeout<F, T>(timeout: Duration, body: F) -> T
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    let (tx, rx) = mpsc::channel();
    let handle = thread::spawn(move || {
        let result = body();
        let _ = tx.send(result);
    });
    let Ok(result) = rx.recv_timeout(timeout) else {
        panic!(
            "test exceeded {} ms timeout — likely a deadlock in the async recorder",
            timeout.as_millis()
        );
    };
    // body returned; join to clean up
    let _ = handle.join();
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn async_recorder_drops_frames_under_backpressure() {
    let dropped = run_with_timeout(Duration::from_secs(15), || {
        let writes: Writes = Arc::new(Mutex::new(Vec::new()));
        let sink = SleepingSink {
            writes,
            delay: Duration::from_millis(50),
        };
        // Tight capacity + slow sink = guaranteed overflow at 100 frames
        // pushed at producer speed. The worst-case worker drain time is
        // ~5 s (100 × 50 ms) — well under the 15 s test budget.
        let rec = AsyncRecorder::new(4, sink, None);
        for i in 0..100_u64 {
            rec.try_send_frame(0, i, vec![u8::try_from(i & 0xFF).unwrap_or(0)]);
        }
        let dropped_before_close = rec.dropped_frames();
        rec.close();
        dropped_before_close
    });
    assert!(
        dropped > 0,
        "expected at least one dropped frame under backpressure, got 0"
    );
}

#[test]
fn async_recorder_zero_drops_when_buffer_sufficient() {
    let (dropped, written) = run_with_timeout(Duration::from_secs(5), || {
        let writes = Arc::new(Mutex::new(Vec::new()));
        let sink = SleepingSink {
            writes: writes.clone(),
            delay: Duration::ZERO,
        };
        // Capacity well above the 100 producer pushes — no drops
        // possible because the bounded channel never fills before the
        // worker drains.
        let rec = AsyncRecorder::new(1024, sink, None);
        for i in 0..100_u64 {
            rec.try_send_frame(0, i, vec![u8::try_from(i & 0xFF).unwrap_or(0)]);
        }
        rec.close();
        let written_count = writes.lock().unwrap().len();
        let dropped_count = 0_u64; // queried below from a closure-local clone
        // We rely on the shared Arc: the close() consumed `rec` so we
        // can't query dropped_frames() after — but the SleepingSink
        // recorded every successful write. Drops are inferred as
        // `100 - written` for the assertion below.
        (
            (100_u64.saturating_sub(written_count as u64)).max(dropped_count),
            written_count,
        )
    });
    assert_eq!(
        dropped, 0,
        "expected zero drops with 1024 capacity vs 100 frames"
    );
    assert_eq!(written, 100, "expected all 100 frames flushed to the sink");
}

#[test]
fn async_recorder_close_flushes_pending_frames() {
    let written = run_with_timeout(Duration::from_secs(5), || {
        let writes = Arc::new(Mutex::new(Vec::new()));
        let sink = SleepingSink {
            writes: writes.clone(),
            delay: Duration::ZERO,
        };
        let rec = AsyncRecorder::new(256, sink, None);
        for i in 0..10_u64 {
            rec.try_send_frame(0, i, vec![u8::try_from(i & 0xFF).unwrap_or(0)]);
        }
        // Close immediately — the worker MUST drain the queue before
        // joining (the W7 PR4 shutdown contract).
        rec.close();
        let writes = writes.lock().unwrap();
        writes.len()
    });
    assert_eq!(
        written, 10,
        "close() must flush all queued frames before joining"
    );
}
