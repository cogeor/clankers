//! Proves `ObservationBuffer::view()` returns a byte-equal borrow with no
//! allocations across repeated calls.
//!
//! The pointer-stability assertion is the proxy for "zero allocations" that
//! does not require a custom allocator harness — the contract is satisfied
//! because the borrow IS `&self.data` directly.

use clankers_env::buffer::ObservationBuffer;

#[test]
fn view_matches_as_observation_clone() {
    let mut buf = ObservationBuffer::new();
    let a = buf.register("pos", 3);
    let b = buf.register("vel", 2);
    buf.write(a, &[1.0, 2.0, 3.0]);
    buf.write(b, &[4.0, 5.0]);

    let owned = buf.as_observation();
    let view = buf.view();
    assert_eq!(view.as_f32(), owned.as_slice());
    assert_eq!(view.shape(), &[5][..]);
}

#[test]
fn view_is_zero_alloc_across_calls() {
    let mut buf = ObservationBuffer::new();
    buf.register("pos", 3);
    buf.write(0, &[1.0, 2.0, 3.0]);

    // 1000 view calls must reuse the same pointer.
    let first_ptr = buf.view().as_f32().as_ptr();
    for _ in 0..1000 {
        assert_eq!(buf.view().as_f32().as_ptr(), first_ptr);
    }
}
