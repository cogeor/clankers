//! TCP server for remote training communication.
//!
//! [`GymServer`] listens on a TCP port and handles one connection at a time,
//! dispatching [`Request`] messages to a [`GymEnv`] using length-prefixed
//! JSON framing per `PROTOCOL_SPEC.md`.

use std::collections::HashMap;
use std::net::{TcpListener, TcpStream};

use clankers_core::types::ObservationSpace;

use crate::env::GymEnv;
use crate::framing::{read_message, write_binary_frame, write_message};
use crate::protocol::{
    EnvInfo, ObsEncoding, PROTOCOL_VERSION, ProtocolError, ProtocolState, Request, Response,
    negotiate_version,
};
use crate::state_machine::ProtocolStateMachine;
use crate::vec_env::GymVecEnv;

// ---------------------------------------------------------------------------
// SessionState
// ---------------------------------------------------------------------------

/// Per-connection session state tracking negotiated capabilities.
#[derive(Debug, Default)]
struct SessionState {
    /// Whether the client negotiated the `binary_obs` capability.
    binary_obs: bool,
}

// ---------------------------------------------------------------------------
// ServerConfig
// ---------------------------------------------------------------------------

/// Configuration for the gym server's handshake response.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Environment name reported during handshake.
    pub env_name: String,
    /// Environment version reported during handshake.
    pub env_version: String,
    /// Server-side capabilities.
    pub capabilities: HashMap<String, bool>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        let mut capabilities = HashMap::new();
        capabilities.insert("binary_obs".into(), true);
        Self {
            env_name: "clankers".into(),
            env_version: "0.1.0".into(),
            capabilities,
        }
    }
}

// ---------------------------------------------------------------------------
// GymServer
// ---------------------------------------------------------------------------

/// TCP server that exposes a [`GymEnv`] over the network.
///
/// Binds to a local address and handles one client connection at a time.
/// Uses length-prefixed JSON framing and validates message ordering via
/// [`ProtocolStateMachine`].
pub struct GymServer {
    listener: TcpListener,
    config: ServerConfig,
}

impl GymServer {
    /// Bind to the given address (e.g. `"127.0.0.1:9876"`).
    ///
    /// # Errors
    ///
    /// Returns an IO error if the address cannot be bound.
    pub fn bind(addr: &str) -> std::io::Result<Self> {
        let listener = TcpListener::bind(addr)?;
        Ok(Self {
            listener,
            config: ServerConfig::default(),
        })
    }

    /// Bind with a custom server configuration.
    ///
    /// # Errors
    ///
    /// Returns an IO error if the address cannot be bound.
    pub fn bind_with_config(addr: &str, config: ServerConfig) -> std::io::Result<Self> {
        let listener = TcpListener::bind(addr)?;
        Ok(Self { listener, config })
    }

    /// The local address the server is bound to.
    pub fn local_addr(&self) -> std::io::Result<std::net::SocketAddr> {
        self.listener.local_addr()
    }

    /// Accept one client connection and run the request-response loop.
    ///
    /// Blocks until a client connects. The first message must be an `Init`
    /// handshake. Processes messages until the client sends
    /// [`Close`](Request::Close) or the connection drops.
    ///
    /// # Errors
    ///
    /// Returns a [`ProtocolError`] if communication or protocol validation fails.
    pub fn serve_one(&self, env: &mut GymEnv) -> Result<(), ProtocolError> {
        let (stream, _addr) = self.listener.accept()?;
        handle_connection(stream, env, &self.config)
    }
}

// ---------------------------------------------------------------------------
// VecGymServer
// ---------------------------------------------------------------------------

/// TCP server that exposes a [`GymVecEnv`] over the network.
///
/// Like [`GymServer`] but supports batched operations (`batch_step`,
/// `batch_reset`) in addition to single-env commands. The `batch_step`
/// capability is automatically advertised.
pub struct VecGymServer {
    listener: TcpListener,
    config: ServerConfig,
}

impl VecGymServer {
    /// Bind to the given address.
    ///
    /// Automatically enables the `batch_step` capability.
    ///
    /// # Errors
    ///
    /// Returns an IO error if the address cannot be bound.
    pub fn bind(addr: &str) -> std::io::Result<Self> {
        Self::bind_with_config(addr, ServerConfig::default())
    }

    /// Bind with a custom server configuration.
    ///
    /// The `batch_step` capability is always enabled.
    ///
    /// # Errors
    ///
    /// Returns an IO error if the address cannot be bound.
    pub fn bind_with_config(addr: &str, mut config: ServerConfig) -> std::io::Result<Self> {
        config.capabilities.insert("batch_step".into(), true);
        let listener = TcpListener::bind(addr)?;
        Ok(Self { listener, config })
    }

    /// The local address the server is bound to.
    pub fn local_addr(&self) -> std::io::Result<std::net::SocketAddr> {
        self.listener.local_addr()
    }

    /// Accept one client connection and run the request-response loop.
    ///
    /// # Errors
    ///
    /// Returns a [`ProtocolError`] if communication or protocol validation fails.
    pub fn serve_one(&self, vec_env: &mut GymVecEnv) -> Result<(), ProtocolError> {
        let (stream, _addr) = self.listener.accept()?;
        handle_vec_connection(stream, vec_env, &self.config)
    }
}

// ---------------------------------------------------------------------------
// Connection handlers
// ---------------------------------------------------------------------------

fn handle_connection(
    stream: TcpStream,
    env: &mut GymEnv,
    config: &ServerConfig,
) -> Result<(), ProtocolError> {
    let mut reader = stream.try_clone().map_err(ProtocolError::Io)?;
    let mut writer = stream;
    let mut sm = ProtocolStateMachine::new();
    let mut session = SessionState::default();

    loop {
        let request: Option<Request> = read_message(&mut reader)?;
        let Some(request) = request else {
            break; // Client disconnected
        };

        // Validate state transition
        let (response, binary_payload) = match sm.on_request(&request) {
            Ok(()) => dispatch(env, &request, config, &sm, &mut session),
            Err(e) => (e.into_response(), None),
        };

        // Update state machine after response
        sm.on_response(&response);
        write_message(&mut writer, &response)?;

        // Send binary observation frame immediately after the JSON response
        if let Some(payload) = binary_payload {
            write_binary_frame(&mut writer, &payload)?;
        }

        if sm.state() == ProtocolState::Disconnected {
            break;
        }
    }

    Ok(())
}

/// Returns the response and an optional binary payload to send after the JSON frame.
fn dispatch(
    env: &mut GymEnv,
    request: &Request,
    config: &ServerConfig,
    _sm: &ProtocolStateMachine,
    session: &mut SessionState,
) -> (Response, Option<Vec<u8>>) {
    match request {
        Request::Init {
            protocol_version,
            capabilities,
            seed,
            ..
        } => {
            // Negotiate protocol version
            let negotiated_version = match negotiate_version(protocol_version, PROTOCOL_VERSION) {
                Ok(v) => v,
                Err(e) => return (e.into_response(), None),
            };

            // Negotiate capabilities (logical AND)
            let negotiated: HashMap<String, bool> = capabilities
                .iter()
                .map(|(k, v)| {
                    let server_has = config.capabilities.get(k).copied().unwrap_or(false);
                    (k.clone(), *v && server_has)
                })
                .collect();

            // Update session state from negotiated capabilities
            session.binary_obs = negotiated.get("binary_obs").copied().unwrap_or(false);

            (
                Response::InitResponse {
                    protocol_version: negotiated_version,
                    env_name: config.env_name.clone(),
                    env_version: config.env_version.clone(),
                    env_info: EnvInfo {
                        n_agents: 1,
                        observation_space: env.observation_space().clone(),
                        action_space: env.action_space().clone(),
                    },
                    capabilities: negotiated,
                    seed_accepted: seed.is_some(),
                },
                None,
            )
        }
        Request::Spaces => (
            Response::Spaces {
                observation_space: env.observation_space().clone(),
                action_space: env.action_space().clone(),
            },
            None,
        ),
        Request::Reset { seed } => (Response::from_reset(env.reset(*seed)), None),
        Request::Step { action } => {
            let result = env.step(action);
            if session.binary_obs {
                if let ObservationSpace::Image {
                    width,
                    height,
                    channels,
                } = env.observation_space()
                {
                    // Convert f32 observation data to u8 pixels (multiply by 255)
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let pixel_bytes: Vec<u8> = result
                        .observation
                        .as_slice()
                        .iter()
                        .map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8)
                        .collect();

                    let encoding = ObsEncoding::RawU8 {
                        width: *width,
                        height: *height,
                        #[allow(clippy::cast_possible_truncation)]
                        channels: *channels as u8,
                    };
                    return (Response::from_step_binary(result, encoding), Some(pixel_bytes));
                }
            }
            (Response::from_step(result), None)
        }
        Request::Close => (Response::Close, None),
        Request::BatchReset { .. } | Request::BatchStep { .. } => (
            // Batch operations require a VecEnv server (not single-env GymServer)
            Response::error("batch operations not supported on single-env server"),
            None,
        ),
        Request::Ping { timestamp } => (
            Response::Pong {
                timestamp: *timestamp,
                server_time: 0,
            },
            None,
        ),
    }
}

// ---------------------------------------------------------------------------
// Vec connection handler
// ---------------------------------------------------------------------------

fn handle_vec_connection(
    stream: TcpStream,
    vec_env: &mut GymVecEnv,
    config: &ServerConfig,
) -> Result<(), ProtocolError> {
    let mut reader = stream.try_clone().map_err(ProtocolError::Io)?;
    let mut writer = stream;
    let mut sm = ProtocolStateMachine::new();

    loop {
        let request: Option<Request> = read_message(&mut reader)?;
        let Some(request) = request else {
            break;
        };

        let response = match sm.on_request(&request) {
            Ok(()) => dispatch_vec(vec_env, &request, config),
            Err(e) => e.into_response(),
        };

        sm.on_response(&response);
        write_message(&mut writer, &response)?;

        if sm.state() == ProtocolState::Disconnected {
            break;
        }
    }

    Ok(())
}

fn dispatch_vec(vec_env: &mut GymVecEnv, request: &Request, config: &ServerConfig) -> Response {
    match request {
        Request::Init {
            protocol_version,
            capabilities,
            seed,
            ..
        } => {
            let negotiated_version = match negotiate_version(protocol_version, PROTOCOL_VERSION) {
                Ok(v) => v,
                Err(e) => return e.into_response(),
            };

            let negotiated: HashMap<String, bool> = capabilities
                .iter()
                .map(|(k, v)| {
                    let server_has = config.capabilities.get(k).copied().unwrap_or(false);
                    (k.clone(), *v && server_has)
                })
                .collect();

            Response::InitResponse {
                protocol_version: negotiated_version,
                env_name: config.env_name.clone(),
                env_version: config.env_version.clone(),
                env_info: vec_env.env_info(),
                capabilities: negotiated,
                seed_accepted: seed.is_some(),
            }
        }
        Request::Spaces => Response::Spaces {
            observation_space: vec_env.observation_space().clone(),
            action_space: vec_env.action_space().clone(),
        },
        Request::Reset { seed } => {
            let result = vec_env.reset_all(*seed);
            Response::from_batch_reset(result)
        }
        Request::Step { action } => {
            // Single-step: apply to all envs (broadcast same action)
            let actions: Vec<_> = (0..vec_env.num_envs()).map(|_| action.clone()).collect();
            let result = vec_env.step_all(&actions);
            Response::from_batch_step(result)
        }
        Request::BatchReset { env_ids, seeds } => {
            let result = vec_env.reset_envs(env_ids, seeds.as_deref());
            Response::from_batch_reset(result)
        }
        Request::BatchStep { actions } => {
            let result = vec_env.step_all(actions);
            Response::from_batch_step(result)
        }
        Request::Close => Response::Close,
        Request::Ping { timestamp } => Response::Pong {
            timestamp: *timestamp,
            server_time: 0,
        },
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::GymEnv;
    use crate::framing;
    use clankers_actuator::components::{Actuator, JointCommand, JointState, JointTorque};
    use clankers_core::traits::ActionApplicator;
    use clankers_core::types::{Action, ActionSpace, ObservationSpace};
    use clankers_env::prelude::*;
    use std::net::TcpStream;

    struct NoopApplicator;

    impl ActionApplicator for NoopApplicator {
        fn apply(&self, _world: &mut bevy::prelude::World, _action: &Action) {}

        #[allow(clippy::unnecessary_literal_bound)]
        fn name(&self) -> &str {
            "NoopApplicator"
        }
    }

    fn build_test_env() -> GymEnv {
        let mut app = bevy::prelude::App::new();
        app.add_plugins(clankers_core::ClankersCorePlugin);
        app.add_plugins(ClankersEnvPlugin);
        app.world_mut().spawn((
            Actuator::default(),
            JointCommand::default(),
            JointState::default(),
            JointTorque::default(),
        ));

        let obs_space = ObservationSpace::Box {
            low: vec![-1.0],
            high: vec![1.0],
        };
        let act_space = ActionSpace::Discrete { n: 2 };

        GymEnv::new(app, obs_space, act_space, Box::new(NoopApplicator))
    }

    fn send_recv(stream: &mut TcpStream, req: &Request) -> Response {
        framing::write_message(stream, req).unwrap();
        framing::read_message::<Response>(stream).unwrap().unwrap()
    }

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
    fn server_handshake_and_spaces() {
        let server = GymServer::bind("127.0.0.1:0").unwrap();
        let addr = server.local_addr().unwrap();

        let handle = std::thread::spawn(move || {
            let mut env = build_test_env();
            server.serve_one(&mut env).unwrap();
        });

        let mut stream = TcpStream::connect(addr).unwrap();

        // Handshake
        let resp = send_recv(&mut stream, &init_request());
        assert!(matches!(resp, Response::InitResponse { .. }));

        // Spaces
        let resp = send_recv(&mut stream, &Request::Spaces);
        assert!(matches!(resp, Response::Spaces { .. }));

        // Close
        let resp = send_recv(&mut stream, &Request::Close);
        assert!(matches!(resp, Response::Close));

        drop(stream);
        handle.join().unwrap();
    }

    #[test]
    fn server_handshake_reset_step() {
        let server = GymServer::bind("127.0.0.1:0").unwrap();
        let addr = server.local_addr().unwrap();

        let handle = std::thread::spawn(move || {
            let mut env = build_test_env();
            server.serve_one(&mut env).unwrap();
        });

        let mut stream = TcpStream::connect(addr).unwrap();

        // Handshake
        send_recv(&mut stream, &init_request());

        // Reset
        let resp = send_recv(&mut stream, &Request::Reset { seed: Some(42) });
        assert!(matches!(resp, Response::Reset { .. }));

        // Step
        let resp = send_recv(
            &mut stream,
            &Request::Step {
                action: Action::Discrete(0),
            },
        );
        assert!(matches!(resp, Response::Step { .. }));

        // Close
        send_recv(&mut stream, &Request::Close);
        drop(stream);
        handle.join().unwrap();
    }

    #[test]
    fn server_rejects_step_before_handshake() {
        let server = GymServer::bind("127.0.0.1:0").unwrap();
        let addr = server.local_addr().unwrap();

        let handle = std::thread::spawn(move || {
            let mut env = build_test_env();
            // Will encounter error response but should complete
            let _ = server.serve_one(&mut env);
        });

        let mut stream = TcpStream::connect(addr).unwrap();

        // Send step without init — should get error
        let resp = send_recv(
            &mut stream,
            &Request::Step {
                action: Action::Discrete(0),
            },
        );
        assert!(matches!(resp, Response::Error { .. }));

        drop(stream);
        handle.join().unwrap();
    }

    #[test]
    fn server_ping_pong() {
        let server = GymServer::bind("127.0.0.1:0").unwrap();
        let addr = server.local_addr().unwrap();

        let handle = std::thread::spawn(move || {
            let mut env = build_test_env();
            server.serve_one(&mut env).unwrap();
        });

        let mut stream = TcpStream::connect(addr).unwrap();

        // Handshake first
        send_recv(&mut stream, &init_request());

        // Ping
        let resp = send_recv(&mut stream, &Request::Ping { timestamp: 12345 });
        if let Response::Pong { timestamp, .. } = resp {
            assert_eq!(timestamp, 12345);
        } else {
            panic!("expected Pong");
        }

        // Close
        send_recv(&mut stream, &Request::Close);
        drop(stream);
        handle.join().unwrap();
    }

    #[test]
    fn server_handles_client_disconnect() {
        let server = GymServer::bind("127.0.0.1:0").unwrap();
        let addr = server.local_addr().unwrap();

        let handle = std::thread::spawn(move || {
            let mut env = build_test_env();
            let _ = server.serve_one(&mut env);
        });

        let stream = TcpStream::connect(addr).unwrap();
        // Immediately drop — server should handle gracefully
        drop(stream);
        handle.join().unwrap();
    }

    #[test]
    fn server_capability_negotiation() {
        let mut server_caps = HashMap::new();
        server_caps.insert("batch_step".into(), true);
        server_caps.insert("shared_memory".into(), false);

        let config = ServerConfig {
            env_name: "test_env".into(),
            env_version: "1.0.0".into(),
            capabilities: server_caps,
        };

        let server = GymServer::bind_with_config("127.0.0.1:0", config).unwrap();
        let addr = server.local_addr().unwrap();

        let handle = std::thread::spawn(move || {
            let mut env = build_test_env();
            server.serve_one(&mut env).unwrap();
        });

        let mut stream = TcpStream::connect(addr).unwrap();

        let mut client_caps = HashMap::new();
        client_caps.insert("batch_step".into(), true);
        client_caps.insert("shared_memory".into(), true);

        let resp = send_recv(
            &mut stream,
            &Request::Init {
                protocol_version: "1.0.0".into(),
                client_name: "test".into(),
                client_version: "0.1.0".into(),
                capabilities: client_caps,
                seed: None,
            },
        );

        if let Response::InitResponse {
            env_name,
            capabilities,
            ..
        } = resp
        {
            assert_eq!(env_name, "test_env");
            // batch_step: both true → true
            assert_eq!(capabilities.get("batch_step"), Some(&true));
            // shared_memory: server false → false
            assert_eq!(capabilities.get("shared_memory"), Some(&false));
        } else {
            panic!("expected InitResponse");
        }

        send_recv(&mut stream, &Request::Close);
        drop(stream);
        handle.join().unwrap();
    }

    // ---- VecGymServer integration tests ----

    use clankers_core::types::{Observation, ResetInfo, ResetResult, StepInfo, StepResult};
    use clankers_env::vec_runner::VecEnvInstance;

    struct ConstVecEnvInstance {
        obs_dim: usize,
        step_count: u32,
    }

    impl ConstVecEnvInstance {
        fn new(obs_dim: usize) -> Self {
            Self {
                obs_dim,
                step_count: 0,
            }
        }
    }

    impl VecEnvInstance for ConstVecEnvInstance {
        fn reset(&mut self, _seed: Option<u64>) -> ResetResult {
            self.step_count = 0;
            ResetResult {
                observation: Observation::zeros(self.obs_dim),
                info: ResetInfo::default(),
            }
        }

        fn step(&mut self, _action: &Action) -> StepResult {
            self.step_count += 1;
            StepResult {
                #[allow(clippy::cast_precision_loss)]
                observation: Observation::new(vec![self.step_count as f32; self.obs_dim]),
                terminated: false,
                truncated: false,
                info: StepInfo::default(),
            }
        }

        fn obs_dim(&self) -> usize {
            self.obs_dim
        }
    }

    fn build_test_vec_env(num_envs: usize, obs_dim: usize) -> GymVecEnv {
        let envs: Vec<Box<dyn VecEnvInstance>> = (0..num_envs)
            .map(|_| Box::new(ConstVecEnvInstance::new(obs_dim)) as Box<dyn VecEnvInstance>)
            .collect();
        let config = clankers_env::vec_env::VecEnvConfig::new(
            u16::try_from(num_envs).expect("too many envs"),
        );
        let obs_space = ObservationSpace::Box {
            low: vec![-10.0; obs_dim],
            high: vec![10.0; obs_dim],
        };
        let act_space = ActionSpace::Box {
            low: vec![-1.0; obs_dim],
            high: vec![1.0; obs_dim],
        };
        GymVecEnv::new(envs, config, obs_space, act_space)
    }

    fn init_with_batch() -> Request {
        let mut caps = HashMap::new();
        caps.insert("batch_step".into(), true);
        Request::Init {
            protocol_version: "1.0.0".into(),
            client_name: "test".into(),
            client_version: "0.1.0".into(),
            capabilities: caps,
            seed: None,
        }
    }

    #[test]
    fn vec_server_handshake_and_batch_reset() {
        let server = VecGymServer::bind("127.0.0.1:0").unwrap();
        let addr = server.local_addr().unwrap();

        let handle = std::thread::spawn(move || {
            let mut env = build_test_vec_env(3, 2);
            server.serve_one(&mut env).unwrap();
        });

        let mut stream = TcpStream::connect(addr).unwrap();

        // Handshake
        let resp = send_recv(&mut stream, &init_with_batch());
        if let Response::InitResponse {
            env_info,
            capabilities,
            ..
        } = &resp
        {
            assert_eq!(env_info.n_agents, 3);
            assert_eq!(capabilities.get("batch_step"), Some(&true));
        } else {
            panic!("expected InitResponse");
        }

        // Batch reset all envs
        let resp = send_recv(
            &mut stream,
            &Request::BatchReset {
                env_ids: vec![0, 1, 2],
                seeds: None,
            },
        );
        if let Response::BatchReset { observations, .. } = &resp {
            assert_eq!(observations.len(), 3);
            assert_eq!(observations[0].len(), 2);
        } else {
            panic!("expected BatchReset response, got {resp:?}");
        }

        // Close
        send_recv(&mut stream, &Request::Close);
        drop(stream);
        handle.join().unwrap();
    }

    #[test]
    fn vec_server_batch_step() {
        let server = VecGymServer::bind("127.0.0.1:0").unwrap();
        let addr = server.local_addr().unwrap();

        let handle = std::thread::spawn(move || {
            let mut env = build_test_vec_env(2, 2);
            server.serve_one(&mut env).unwrap();
        });

        let mut stream = TcpStream::connect(addr).unwrap();

        // Handshake
        send_recv(&mut stream, &init_with_batch());

        // Reset
        send_recv(
            &mut stream,
            &Request::BatchReset {
                env_ids: vec![0, 1],
                seeds: None,
            },
        );

        // Batch step
        let resp = send_recv(
            &mut stream,
            &Request::BatchStep {
                actions: vec![Action::zeros(2), Action::zeros(2)],
            },
        );
        if let Response::BatchStep {
            observations,
            terminated,
            ..
        } = &resp
        {
            assert_eq!(observations.len(), 2);
            assert_eq!(observations[0].as_slice(), &[1.0, 1.0]);
            assert!(!terminated[0]);
        } else {
            panic!("expected BatchStep response, got {resp:?}");
        }

        // Close
        send_recv(&mut stream, &Request::Close);
        drop(stream);
        handle.join().unwrap();
    }

    #[test]
    fn vec_server_selective_reset() {
        let server = VecGymServer::bind("127.0.0.1:0").unwrap();
        let addr = server.local_addr().unwrap();

        let handle = std::thread::spawn(move || {
            let mut env = build_test_vec_env(3, 2);
            server.serve_one(&mut env).unwrap();
        });

        let mut stream = TcpStream::connect(addr).unwrap();
        send_recv(&mut stream, &init_with_batch());

        // Reset all
        send_recv(
            &mut stream,
            &Request::BatchReset {
                env_ids: vec![0, 1, 2],
                seeds: None,
            },
        );

        // Step all
        send_recv(
            &mut stream,
            &Request::BatchStep {
                actions: vec![Action::zeros(2), Action::zeros(2), Action::zeros(2)],
            },
        );

        // Reset only env 1
        let resp = send_recv(
            &mut stream,
            &Request::BatchReset {
                env_ids: vec![1],
                seeds: Some(vec![Some(42)]),
            },
        );
        if let Response::BatchReset { observations, .. } = &resp {
            assert_eq!(observations.len(), 1);
            // Env 1 was reset, so obs should be zeros
            assert_eq!(observations[0].as_slice(), &[0.0, 0.0]);
        } else {
            panic!("expected BatchReset response");
        }

        // Close
        send_recv(&mut stream, &Request::Close);
        drop(stream);
        handle.join().unwrap();
    }

    #[test]
    fn vec_server_batch_step_capability_negotiated() {
        let server = VecGymServer::bind("127.0.0.1:0").unwrap();
        let addr = server.local_addr().unwrap();

        let handle = std::thread::spawn(move || {
            let mut env = build_test_vec_env(1, 2);
            server.serve_one(&mut env).unwrap();
        });

        let mut stream = TcpStream::connect(addr).unwrap();

        // Client requests batch_step capability
        let resp = send_recv(&mut stream, &init_with_batch());
        if let Response::InitResponse { capabilities, .. } = &resp {
            assert_eq!(capabilities.get("batch_step"), Some(&true));
        } else {
            panic!("expected InitResponse");
        }

        send_recv(&mut stream, &Request::Close);
        drop(stream);
        handle.join().unwrap();
    }
}
