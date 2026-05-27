//! Bin-specific viz overlays — egui control panels, camera glue, and
//! recording hooks lifted out of `examples/examples/arm_*.rs`.
//!
//! # Status (W8 PR1)
//!
//! Loop 7 lands the **module skeleton** so the new
//! [`clankers_sim::scenarios`](clankers_sim::scenarios) registry +
//! xtask LOC checker can ship without blocking on the much larger
//! per-bin lift. The three planned overlay plugins
//! (`ArmIkOverlayPlugin`, `ArmPolicyOverlayPlugin`,
//! `ArmPickReplayOverlayPlugin`) are declared here as no-op stubs and
//! deferred to a follow-up loop (see
//! `.delegate/work/20260526-045330-w6-w7-w8-impl/07/IMPLEMENTATION.md`
//! under "Deferred work"). Loop 8 (or a dedicated follow-up) will
//! populate the plugin bodies and migrate the bins.
//!
//! # Design (W8 PR1 PLAN Design C)
//!
//! Each overlay is a Bevy `Plugin` (not a free function), matching
//! `ClankersVizPlugin`. Bin call-sites become:
//!
//! ```ignore
//! app.add_plugins((
//!     DefaultPlugins,
//!     clankers_sim::ClankersSimPlugin,
//!     ClankersVizPlugin::default(),
//!     ArmIkOverlayPlugin,
//! ));
//! ```
//!
//! Per-bin inputs (model paths, trace paths, recording knobs) ride as
//! Bevy resources inserted by the bin before `add_plugins`. The
//! overlay reads them in `Plugin::build`.

pub mod arm_ik_overlay;
pub mod arm_pick_replay_overlay;
pub mod arm_policy_overlay;

/// Re-exports for ergonomic `use clankers_viz::overlays::prelude::*;`.
pub mod prelude {
    pub use super::arm_ik_overlay::ArmIkOverlayPlugin;
    pub use super::arm_pick_replay_overlay::ArmPickReplayOverlayPlugin;
    pub use super::arm_policy_overlay::ArmPolicyOverlayPlugin;
}
