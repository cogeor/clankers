//! Error types for URDF parsing and robot spawning.

use std::path::PathBuf;

/// Errors that can occur during URDF processing.
#[derive(Debug, thiserror::Error)]
pub enum UrdfError {
    /// Failed to read the URDF file.
    #[error("IO error reading {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },

    /// Failed to parse URDF XML content.
    #[error("URDF parse error: {0}")]
    Parse(String),

    /// A referenced link was not found in the model.
    #[error("missing link: {0}")]
    MissingLink(String),

    /// A referenced joint was not found in the model.
    #[error("missing joint: {0}")]
    MissingJoint(String),

    /// Invalid or unsupported joint type.
    #[error("unsupported joint type: {0}")]
    UnsupportedJointType(String),

    /// Invalid geometry specification.
    #[error("invalid geometry: {0}")]
    InvalidGeometry(String),

    /// The URDF has no root link (no link that is never a child).
    #[error("no root link found")]
    NoRootLink,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_messages() {
        let e = UrdfError::Parse("bad xml".into());
        assert_eq!(e.to_string(), "URDF parse error: bad xml");

        let e = UrdfError::MissingLink("base_link".into());
        assert_eq!(e.to_string(), "missing link: base_link");

        let e = UrdfError::MissingJoint("joint1".into());
        assert_eq!(e.to_string(), "missing joint: joint1");

        let e = UrdfError::UnsupportedJointType("ball".into());
        assert_eq!(e.to_string(), "unsupported joint type: ball");

        let e = UrdfError::NoRootLink;
        assert_eq!(e.to_string(), "no root link found");
    }

    #[test]
    fn io_error_includes_path() {
        let e = UrdfError::Io {
            path: PathBuf::from("/tmp/robot.urdf"),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "not found"),
        };
        let msg = e.to_string();
        assert!(msg.contains("/tmp/robot.urdf"));
        assert!(msg.contains("not found"));
    }

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn error_is_send_sync() {
        assert_send_sync::<UrdfError>();
    }
}
