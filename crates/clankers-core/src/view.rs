//! Zero-copy borrow type for observation buffers.
//!
//! [`ObservationView<'a>`] is a borrow of a backing `&'a [f32]` (typically
//! `ObservationBuffer::data` or a row of `VecObsBuffer::data`) plus its dtype
//! and shape metadata. It exists to replace the cloning
//! [`ObservationBuffer::as_observation`](../../clankers_env/buffer/struct.ObservationBuffer.html#method.as_observation)
//! read on hot paths (sensor write → read, vec-runner row copy, MPC read-back)
//! without changing the serialisation-boundary `Observation` type used by the
//! gym protocol.
//!
//! Workstream 3 PR1 ships the additive type; W3 PR2 migrates internal
//! callers (e.g. `VecObsBuffer::set` consuming `view().as_f32()`).
//!
//! ## Naming deviation from the WS3 plan
//!
//! The plan uses the bare name `Dtype` for the dtype field. The workspace
//! has no `Dtype` type — W1 shipped it as
//! [`SchemaDtype`]. `ObservationView` reuses `SchemaDtype` directly. A
//! future workstream may add a prelude alias.

use crate::schema::SchemaDtype;

/// Zero-copy borrow of a contiguous f32 observation slice plus its dtype and
/// shape metadata.
///
/// All fields are public-by-value so callers can pattern-match the view in
/// hot loops without allocating intermediate `Vec<usize>` shape buffers.
/// The accessor methods (`as_f32`, `shape`, `dtype`) are provided for the
/// canonical read pattern.
///
/// # Example
///
/// ```
/// use clankers_core::schema::SchemaDtype;
/// use clankers_core::view::ObservationView;
///
/// let data = [1.0_f32, 2.0, 3.0];
/// let shape = [3_usize];
/// let view = ObservationView::new(&data, SchemaDtype::F32, &shape);
/// assert_eq!(view.as_f32(), &[1.0, 2.0, 3.0]);
/// assert_eq!(view.shape(), &[3]);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ObservationView<'a> {
    /// Borrowed f32 data slice.
    pub data: &'a [f32],
    /// Element dtype. Always `SchemaDtype::F32` for buffers backed by
    /// `Vec<f32>` storage; preserved for symmetry with future typed buffers.
    pub dtype: SchemaDtype,
    /// Tensor shape (row-major). For flat `ObservationBuffer` views this is
    /// `&[total_dim]`; for `VecObsBuffer::row` views this is `&[obs_dim]`.
    pub shape: &'a [usize],
}

impl<'a> ObservationView<'a> {
    /// Construct a view from a borrowed slice + dtype + shape.
    ///
    /// No allocations; the returned `ObservationView` borrows `data` and
    /// `shape` for `'a`.
    #[must_use]
    pub const fn new(data: &'a [f32], dtype: SchemaDtype, shape: &'a [usize]) -> Self {
        Self { data, dtype, shape }
    }

    /// Return the borrowed f32 slice.
    #[must_use]
    pub const fn as_f32(&self) -> &[f32] {
        self.data
    }

    /// Return the borrowed shape slice.
    #[must_use]
    pub const fn shape(&self) -> &[usize] {
        self.shape
    }

    /// Return the dtype.
    #[must_use]
    pub const fn dtype(&self) -> SchemaDtype {
        self.dtype
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_and_accessors() {
        let data = [10.0_f32, 20.0, 30.0, 40.0];
        let shape = [2_usize, 2];
        let view = ObservationView::new(&data, SchemaDtype::F32, &shape);
        assert_eq!(view.as_f32(), &[10.0, 20.0, 30.0, 40.0]);
        assert_eq!(view.shape(), &[2, 2]);
        assert_eq!(view.dtype(), SchemaDtype::F32);
    }

    #[test]
    fn is_zero_copy_borrow() {
        let data = vec![1.0_f32, 2.0, 3.0];
        let shape = vec![3_usize];
        let expected_ptr = data.as_ptr();
        let view = ObservationView::new(&data, SchemaDtype::F32, &shape);
        // The view's data slice must point at the original buffer.
        assert_eq!(view.as_f32().as_ptr(), expected_ptr);
        // Cloning the view also reuses the same backing pointer (Copy semantics).
        let view2 = view;
        assert_eq!(view2.as_f32().as_ptr(), expected_ptr);
    }
}
