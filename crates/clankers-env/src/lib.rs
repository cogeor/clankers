//! Environment state management, sensors, and episode lifecycle for Clankers.
//!
//! This crate provides the ECS integration layer for environment simulation:
//! episode lifecycle, observation collection, and sensor management.

pub mod buffer;
pub mod episode;
pub mod sensors;
