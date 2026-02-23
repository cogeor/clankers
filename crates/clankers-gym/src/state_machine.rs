//! Protocol state machine enforcing valid message transitions.
//!
//! [`ProtocolStateMachine`] tracks the current [`ProtocolState`] and validates
//! that incoming messages are legal given the current state. See
//! `PROTOCOL_SPEC.md` Section 3 for the full transition table.

use crate::protocol::{ProtocolError, ProtocolState, Request, Response};

/// Tracks protocol state and enforces valid transitions.
///
/// # Example
///
/// ```
/// use clankers_gym::state_machine::ProtocolStateMachine;
/// use clankers_gym::protocol::ProtocolState;
///
/// let mut sm = ProtocolStateMachine::new();
/// assert_eq!(sm.state(), ProtocolState::Connected);
/// ```
#[derive(Debug)]
pub struct ProtocolStateMachine {
    state: ProtocolState,
}

impl ProtocolStateMachine {
    /// Create a new state machine in the [`Connected`](ProtocolState::Connected) state.
    ///
    /// This assumes a TCP connection has already been established.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state: ProtocolState::Connected,
        }
    }

    /// Current protocol state.
    #[must_use]
    pub const fn state(&self) -> ProtocolState {
        self.state
    }

    /// Validate and apply a transition for an incoming request.
    ///
    /// Returns `Ok(())` if the request is valid in the current state,
    /// advancing the state machine. Returns a [`ProtocolError::UnexpectedMessage`]
    /// if the request is not allowed.
    pub fn on_request(&mut self, request: &Request) -> Result<(), ProtocolError> {
        let msg_type = request_type_name(request);
        let allowed = self.allowed_request_types();

        if !allowed.contains(&msg_type) {
            return Err(ProtocolError::UnexpectedMessage {
                current_state: self.state,
                expected: allowed.iter().map(|s| (*s).to_string()).collect(),
                got: msg_type.to_string(),
            });
        }

        self.state = match (&self.state, request) {
            (ProtocolState::Connected, Request::Init { .. }) => ProtocolState::Handshaking,
            (_, Request::Close) => ProtocolState::Closing,
            // Ping, Reset, Spaces, Step — stay in current state
            (_, Request::Ping { .. })
            | (ProtocolState::Ready, Request::Reset { .. } | Request::Spaces)
            | (ProtocolState::EpisodeRunning, Request::Reset { .. } | Request::Step { .. } | Request::Spaces) => {
                self.state
            }
            _ => {
                return Err(ProtocolError::UnexpectedMessage {
                    current_state: self.state,
                    expected: allowed.iter().map(|s| (*s).to_string()).collect(),
                    got: msg_type.to_string(),
                });
            }
        };

        Ok(())
    }

    /// Update state after sending a response.
    ///
    /// Some responses cause state transitions (e.g. a successful
    /// `InitResponse` moves from `Handshaking` to `Ready`).
    pub const fn on_response(&mut self, response: &Response) {
        match (&self.state, response) {
            (ProtocolState::Handshaking, Response::InitResponse { .. }) => {
                self.state = ProtocolState::Ready;
            }
            (ProtocolState::Handshaking, Response::Error { .. }) => {
                self.state = ProtocolState::Error;
            }
            (ProtocolState::Ready, Response::Reset { .. }) => {
                self.state = ProtocolState::EpisodeRunning;
            }
            (ProtocolState::EpisodeRunning, Response::Step { terminated, truncated, .. })
                if *terminated || *truncated =>
            {
                self.state = ProtocolState::Ready;
            }
            (ProtocolState::Closing, Response::Close) => {
                self.state = ProtocolState::Disconnected;
            }
            _ => {}
        }
    }

    /// Transition to the error state.
    pub const fn enter_error(&mut self) {
        self.state = ProtocolState::Error;
    }

    /// Message types allowed in the current state.
    const fn allowed_request_types(&self) -> &'static [&'static str] {
        match self.state {
            ProtocolState::Connected => &["init"],
            ProtocolState::Ready => &["reset", "spaces", "close", "ping"],
            ProtocolState::EpisodeRunning => &["step", "reset", "spaces", "close", "ping"],
            ProtocolState::Handshaking
            | ProtocolState::Closing
            | ProtocolState::Disconnected
            | ProtocolState::Error => &[],
        }
    }
}

impl Default for ProtocolStateMachine {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract the message type name from a request (matches serde tag).
const fn request_type_name(request: &Request) -> &'static str {
    match request {
        Request::Init { .. } => "init",
        Request::Spaces => "spaces",
        Request::Reset { .. } => "reset",
        Request::Step { .. } => "step",
        Request::Close => "close",
        Request::Ping { .. } => "ping",
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use clankers_core::types::Action;
    use std::collections::HashMap;

    fn init_request() -> Request {
        Request::Init {
            protocol_version: "1.0.0".into(),
            client_name: "test".into(),
            client_version: "0.1.0".into(),
            capabilities: HashMap::new(),
            seed: None,
        }
    }

    #[test]
    fn new_starts_connected() {
        let sm = ProtocolStateMachine::new();
        assert_eq!(sm.state(), ProtocolState::Connected);
    }

    #[test]
    fn default_starts_connected() {
        let sm = ProtocolStateMachine::default();
        assert_eq!(sm.state(), ProtocolState::Connected);
    }

    #[test]
    fn connected_accepts_init() {
        let mut sm = ProtocolStateMachine::new();
        sm.on_request(&init_request()).unwrap();
        assert_eq!(sm.state(), ProtocolState::Handshaking);
    }

    #[test]
    fn connected_rejects_step() {
        let mut sm = ProtocolStateMachine::new();
        let err = sm
            .on_request(&Request::Step {
                action: Action::Discrete(0),
            })
            .unwrap_err();
        assert!(matches!(err, ProtocolError::UnexpectedMessage { .. }));
    }

    #[test]
    fn handshake_to_ready() {
        let mut sm = ProtocolStateMachine::new();
        sm.on_request(&init_request()).unwrap();

        sm.on_response(&Response::InitResponse {
            protocol_version: "1.0.0".into(),
            env_name: "test".into(),
            env_version: "0.1.0".into(),
            env_info: crate::protocol::EnvInfo {
                n_agents: 1,
                observation_space: clankers_core::types::ObservationSpace::Box {
                    low: vec![0.0],
                    high: vec![1.0],
                },
                action_space: clankers_core::types::ActionSpace::Discrete { n: 2 },
                reward_range: None,
            },
            capabilities: HashMap::new(),
            seed_accepted: false,
        });
        assert_eq!(sm.state(), ProtocolState::Ready);
    }

    #[test]
    fn ready_accepts_reset() {
        let mut sm = ProtocolStateMachine::new();
        sm.on_request(&init_request()).unwrap();
        sm.on_response(&make_init_response());
        assert_eq!(sm.state(), ProtocolState::Ready);

        sm.on_request(&Request::Reset { seed: None }).unwrap();
        assert_eq!(sm.state(), ProtocolState::Ready);
    }

    #[test]
    fn reset_response_transitions_to_episode() {
        let mut sm = ProtocolStateMachine::new();
        sm.on_request(&init_request()).unwrap();
        sm.on_response(&make_init_response());

        sm.on_request(&Request::Reset { seed: None }).unwrap();
        sm.on_response(&Response::Reset {
            observation: clankers_core::types::Observation::zeros(1),
            info: clankers_core::types::ResetInfo::default(),
        });
        assert_eq!(sm.state(), ProtocolState::EpisodeRunning);
    }

    #[test]
    fn episode_accepts_step() {
        let mut sm = ready_episode_sm();
        sm.on_request(&Request::Step {
            action: Action::Discrete(0),
        })
        .unwrap();
        assert_eq!(sm.state(), ProtocolState::EpisodeRunning);
    }

    #[test]
    fn terminated_step_returns_to_ready() {
        let mut sm = ready_episode_sm();
        sm.on_request(&Request::Step {
            action: Action::Discrete(0),
        })
        .unwrap();

        sm.on_response(&Response::Step {
            observation: clankers_core::types::Observation::zeros(1),
            reward: 0.0,
            terminated: true,
            truncated: false,
            info: clankers_core::types::StepInfo::default(),
        });
        assert_eq!(sm.state(), ProtocolState::Ready);
    }

    #[test]
    fn truncated_step_returns_to_ready() {
        let mut sm = ready_episode_sm();
        sm.on_request(&Request::Step {
            action: Action::Discrete(0),
        })
        .unwrap();

        sm.on_response(&Response::Step {
            observation: clankers_core::types::Observation::zeros(1),
            reward: 0.0,
            terminated: false,
            truncated: true,
            info: clankers_core::types::StepInfo::default(),
        });
        assert_eq!(sm.state(), ProtocolState::Ready);
    }

    #[test]
    fn close_from_ready() {
        let mut sm = ProtocolStateMachine::new();
        sm.on_request(&init_request()).unwrap();
        sm.on_response(&make_init_response());

        sm.on_request(&Request::Close).unwrap();
        assert_eq!(sm.state(), ProtocolState::Closing);

        sm.on_response(&Response::Close);
        assert_eq!(sm.state(), ProtocolState::Disconnected);
    }

    #[test]
    fn close_from_episode() {
        let mut sm = ready_episode_sm();
        sm.on_request(&Request::Close).unwrap();
        assert_eq!(sm.state(), ProtocolState::Closing);
    }

    #[test]
    fn ping_in_ready() {
        let mut sm = ProtocolStateMachine::new();
        sm.on_request(&init_request()).unwrap();
        sm.on_response(&make_init_response());

        sm.on_request(&Request::Ping { timestamp: 123 })
            .unwrap();
        assert_eq!(sm.state(), ProtocolState::Ready);
    }

    #[test]
    fn ping_in_episode() {
        let mut sm = ready_episode_sm();
        sm.on_request(&Request::Ping { timestamp: 123 })
            .unwrap();
        assert_eq!(sm.state(), ProtocolState::EpisodeRunning);
    }

    #[test]
    fn ready_rejects_step() {
        let mut sm = ProtocolStateMachine::new();
        sm.on_request(&init_request()).unwrap();
        sm.on_response(&make_init_response());

        let err = sm
            .on_request(&Request::Step {
                action: Action::Discrete(0),
            })
            .unwrap_err();
        assert!(matches!(err, ProtocolError::UnexpectedMessage { .. }));
    }

    #[test]
    fn error_state_rejects_all() {
        let mut sm = ProtocolStateMachine::new();
        sm.enter_error();
        assert_eq!(sm.state(), ProtocolState::Error);

        let err = sm.on_request(&Request::Reset { seed: None }).unwrap_err();
        assert!(matches!(err, ProtocolError::UnexpectedMessage { .. }));
    }

    #[test]
    fn spaces_allowed_in_ready() {
        let mut sm = ProtocolStateMachine::new();
        sm.on_request(&init_request()).unwrap();
        sm.on_response(&make_init_response());

        sm.on_request(&Request::Spaces).unwrap();
        assert_eq!(sm.state(), ProtocolState::Ready);
    }

    #[test]
    fn handshake_error_transitions_to_error() {
        let mut sm = ProtocolStateMachine::new();
        sm.on_request(&init_request()).unwrap();
        assert_eq!(sm.state(), ProtocolState::Handshaking);

        sm.on_response(&Response::Error {
            message: "version mismatch".into(),
        });
        assert_eq!(sm.state(), ProtocolState::Error);
    }

    // --- Helpers ---

    fn make_init_response() -> Response {
        Response::InitResponse {
            protocol_version: "1.0.0".into(),
            env_name: "test".into(),
            env_version: "0.1.0".into(),
            env_info: crate::protocol::EnvInfo {
                n_agents: 1,
                observation_space: clankers_core::types::ObservationSpace::Box {
                    low: vec![0.0],
                    high: vec![1.0],
                },
                action_space: clankers_core::types::ActionSpace::Discrete { n: 2 },
                reward_range: None,
            },
            capabilities: HashMap::new(),
            seed_accepted: false,
        }
    }

    fn ready_episode_sm() -> ProtocolStateMachine {
        let mut sm = ProtocolStateMachine::new();
        sm.on_request(&init_request()).unwrap();
        sm.on_response(&make_init_response());
        sm.on_request(&Request::Reset { seed: None }).unwrap();
        sm.on_response(&Response::Reset {
            observation: clankers_core::types::Observation::zeros(1),
            info: clankers_core::types::ResetInfo::default(),
        });
        assert_eq!(sm.state(), ProtocolState::EpisodeRunning);
        sm
    }
}
