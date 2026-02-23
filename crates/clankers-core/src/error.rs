use thiserror::Error;

/// Top-level error type for clankers-core.
#[derive(Debug, Error)]
pub enum ClankersError {
    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),

    #[error("Simulation error: {0}")]
    Simulation(#[from] SimError),

    #[error("Space error: {0}")]
    Space(#[from] SpaceError),

    #[error("Validation error: {0}")]
    Validation(#[from] ValidationError),
}

/// Configuration errors.
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("TOML parse error: {0}")]
    Toml(#[from] toml::de::Error),

    #[error("Invalid physics_dt: {0} (must be > 0)")]
    InvalidPhysicsDt(f64),

    #[error("control_dt must be >= physics_dt")]
    ControlDtLessThanPhysicsDt,

    #[error("Missing required field: {0}")]
    MissingField(String),

    #[error("Invalid value for {field}: {message}")]
    InvalidValue { field: String, message: String },

    #[error("Incompatible configuration: {0}")]
    Incompatible(String),
}

/// Simulation runtime errors.
#[derive(Debug, Error)]
pub enum SimError {
    #[error("Physics diverged: NaN detected in state")]
    PhysicsDiverged,

    #[error("Reset failed: {0}")]
    ResetFailed(String),

    #[error("Step failed: {0}")]
    StepFailed(String),

    #[error("Entity not found: {0}")]
    EntityNotFound(String),
}

/// Space definition errors.
#[derive(Debug, Error)]
pub enum SpaceError {
    #[error("Mismatched low/high dimensions: low={low}, high={high}")]
    DimensionMismatch { low: usize, high: usize },

    #[error("Space/data type mismatch: expected {expected}, got {got}")]
    TypeMismatch { expected: String, got: String },

    #[error("Invalid space definition: {0}")]
    InvalidDefinition(String),
}

/// Action/observation validation errors.
///
/// Copy + static messages for cheap propagation in hot paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum ValidationError {
    #[error("Action dimension mismatch: expected {expected}, got {got}")]
    ActionDimMismatch { expected: usize, got: usize },

    #[error("Action contains NaN")]
    ActionContainsNan,

    #[error("Action contains Inf")]
    ActionContainsInf,

    #[error("Action out of bounds at dimension {dim}")]
    ActionOutOfBounds { dim: usize },

    #[error("Discrete action out of range: {value} >= {max}")]
    DiscreteOutOfRange { value: u64, max: usize },

    #[error("Observation dimension mismatch: expected {expected}, got {got}")]
    ObservationDimMismatch { expected: usize, got: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clankers_error_from_config_error() {
        let err = ConfigError::InvalidPhysicsDt(-1.0);
        let clankers_err: ClankersError = err.into();
        assert!(matches!(clankers_err, ClankersError::Config(_)));
        assert!(clankers_err.to_string().contains("-1"));
    }

    #[test]
    fn clankers_error_from_sim_error() {
        let err = SimError::PhysicsDiverged;
        let clankers_err: ClankersError = err.into();
        assert!(matches!(clankers_err, ClankersError::Simulation(_)));
        assert!(clankers_err.to_string().contains("NaN"));
    }

    #[test]
    fn clankers_error_from_space_error() {
        let err = SpaceError::DimensionMismatch { low: 3, high: 5 };
        let clankers_err: ClankersError = err.into();
        assert!(matches!(clankers_err, ClankersError::Space(_)));
        assert!(clankers_err.to_string().contains("low=3"));
    }

    #[test]
    fn clankers_error_from_validation_error() {
        let err = ValidationError::ActionContainsNan;
        let clankers_err: ClankersError = err.into();
        assert!(matches!(clankers_err, ClankersError::Validation(_)));
    }

    #[test]
    fn config_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let config_err: ConfigError = io_err.into();
        assert!(matches!(config_err, ConfigError::Io(_)));
    }

    #[test]
    fn validation_error_is_copy() {
        let err = ValidationError::ActionContainsNan;
        let err2 = err; // Copy
        assert_eq!(err, err2);
    }

    #[test]
    fn validation_error_display_messages() {
        assert_eq!(
            ValidationError::ActionDimMismatch {
                expected: 6,
                got: 3
            }
            .to_string(),
            "Action dimension mismatch: expected 6, got 3"
        );
        assert_eq!(
            ValidationError::ActionContainsNan.to_string(),
            "Action contains NaN"
        );
        assert_eq!(
            ValidationError::ActionContainsInf.to_string(),
            "Action contains Inf"
        );
        assert_eq!(
            ValidationError::ActionOutOfBounds { dim: 2 }.to_string(),
            "Action out of bounds at dimension 2"
        );
        assert_eq!(
            ValidationError::DiscreteOutOfRange { value: 5, max: 3 }.to_string(),
            "Discrete action out of range: 5 >= 3"
        );
        assert_eq!(
            ValidationError::ObservationDimMismatch {
                expected: 10,
                got: 8
            }
            .to_string(),
            "Observation dimension mismatch: expected 10, got 8"
        );
    }

    #[test]
    fn config_error_display_messages() {
        assert_eq!(
            ConfigError::InvalidPhysicsDt(0.0).to_string(),
            "Invalid physics_dt: 0 (must be > 0)"
        );
        assert_eq!(
            ConfigError::ControlDtLessThanPhysicsDt.to_string(),
            "control_dt must be >= physics_dt"
        );
        assert_eq!(
            ConfigError::MissingField("name".into()).to_string(),
            "Missing required field: name"
        );
        assert_eq!(
            ConfigError::InvalidValue {
                field: "seed".into(),
                message: "must be non-negative".into()
            }
            .to_string(),
            "Invalid value for seed: must be non-negative"
        );
        assert_eq!(
            ConfigError::Incompatible("headless mode requires no window".into()).to_string(),
            "Incompatible configuration: headless mode requires no window"
        );
    }

    #[test]
    fn sim_error_display_messages() {
        assert_eq!(
            SimError::PhysicsDiverged.to_string(),
            "Physics diverged: NaN detected in state"
        );
        assert_eq!(
            SimError::ResetFailed("timeout".into()).to_string(),
            "Reset failed: timeout"
        );
        assert_eq!(
            SimError::StepFailed("invalid action".into()).to_string(),
            "Step failed: invalid action"
        );
        assert_eq!(
            SimError::EntityNotFound("robot_arm".into()).to_string(),
            "Entity not found: robot_arm"
        );
    }

    #[test]
    fn space_error_display_messages() {
        assert_eq!(
            SpaceError::DimensionMismatch { low: 3, high: 5 }.to_string(),
            "Mismatched low/high dimensions: low=3, high=5"
        );
        assert_eq!(
            SpaceError::TypeMismatch {
                expected: "Box".into(),
                got: "Discrete".into()
            }
            .to_string(),
            "Space/data type mismatch: expected Box, got Discrete"
        );
        assert_eq!(
            SpaceError::InvalidDefinition("empty shape".into()).to_string(),
            "Invalid space definition: empty shape"
        );
    }
}
