//! TCP server for remote training communication.
//!
//! [`GymServer`] listens on a TCP port and handles one connection at a time,
//! dispatching [`Request`] messages to a [`GymEnv`].
//!
//! The protocol uses newline-delimited JSON: each message is a single line
//! of JSON followed by `\n`.

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};

use crate::env::GymEnv;
use crate::protocol::{Request, Response};

// ---------------------------------------------------------------------------
// GymServer
// ---------------------------------------------------------------------------

/// TCP server that exposes a [`GymEnv`] over the network.
///
/// Binds to a local address and handles one client connection at a time.
/// Each connection runs a request-response loop until the client sends
/// [`Close`](Request::Close) or disconnects.
pub struct GymServer {
    listener: TcpListener,
}

impl GymServer {
    /// Bind to the given address (e.g. `"127.0.0.1:9876"`).
    ///
    /// # Errors
    ///
    /// Returns an IO error if the address cannot be bound.
    pub fn bind(addr: &str) -> std::io::Result<Self> {
        let listener = TcpListener::bind(addr)?;
        Ok(Self { listener })
    }

    /// The local address the server is bound to.
    pub fn local_addr(&self) -> std::io::Result<std::net::SocketAddr> {
        self.listener.local_addr()
    }

    /// Accept one client connection and run the request-response loop.
    ///
    /// Blocks until a client connects. Processes messages until the client
    /// sends [`Close`](Request::Close) or the connection drops.
    ///
    /// # Errors
    ///
    /// Returns an IO error if accepting or communicating fails.
    pub fn serve_one(&self, env: &mut GymEnv) -> std::io::Result<()> {
        let (stream, _addr) = self.listener.accept()?;
        handle_connection(stream, env)
    }
}

// ---------------------------------------------------------------------------
// Connection handler
// ---------------------------------------------------------------------------

fn handle_connection(stream: TcpStream, env: &mut GymEnv) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut writer = stream;
    let mut line = String::new();

    loop {
        line.clear();
        let bytes_read = reader.read_line(&mut line)?;
        if bytes_read == 0 {
            // Client disconnected
            break;
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let response = match serde_json::from_str::<Request>(trimmed) {
            Ok(request) => dispatch(env, request),
            Err(e) => Response::error(format!("invalid request: {e}")),
        };

        let mut json = serde_json::to_string(&response).unwrap_or_else(|e| {
            format!(r#"{{"type":"error","message":"serialisation failed: {e}"}}"#)
        });
        json.push('\n');
        writer.write_all(json.as_bytes())?;
        writer.flush()?;

        if matches!(response, Response::Close) {
            break;
        }
    }

    Ok(())
}

fn dispatch(env: &mut GymEnv, request: Request) -> Response {
    match request {
        Request::Init { .. } => Response::error("handshake not implemented in legacy server"),
        Request::Spaces => Response::Spaces {
            observation_space: env.observation_space().clone(),
            action_space: env.action_space().clone(),
        },
        Request::Reset { seed } => Response::from_reset(env.reset(seed)),
        Request::Step { action } => Response::from_step(env.step(&action)),
        Request::Close => Response::Close,
        Request::Ping { timestamp } => Response::Pong {
            timestamp,
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
    use clankers_actuator::components::{Actuator, JointCommand, JointState, JointTorque};
    use clankers_core::traits::ActionApplicator;
    use clankers_core::types::{Action, ActionSpace, ObservationSpace};
    use clankers_env::prelude::*;
    use std::io::{BufRead, BufReader, Write};
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

    fn send_and_receive(stream: &mut TcpStream, msg: &str) -> String {
        let mut out = msg.to_string();
        out.push('\n');
        stream.write_all(out.as_bytes()).unwrap();
        stream.flush().unwrap();

        let mut reader = BufReader::new(stream.try_clone().unwrap());
        let mut response = String::new();
        reader.read_line(&mut response).unwrap();
        response
    }

    #[test]
    fn server_spaces_request() {
        let server = GymServer::bind("127.0.0.1:0").unwrap();
        let addr = server.local_addr().unwrap();

        let handle = std::thread::spawn(move || {
            let mut env = build_test_env();
            server.serve_one(&mut env).unwrap();
        });

        let mut stream = TcpStream::connect(addr).unwrap();

        // Send spaces request
        let resp = send_and_receive(&mut stream, r#"{"type":"spaces"}"#);
        let parsed: Response = serde_json::from_str(&resp).unwrap();
        assert!(matches!(parsed, Response::Spaces { .. }));

        // Close
        let resp = send_and_receive(&mut stream, r#"{"type":"close"}"#);
        let parsed: Response = serde_json::from_str(&resp).unwrap();
        assert!(matches!(parsed, Response::Close));

        drop(stream);
        handle.join().unwrap();
    }

    #[test]
    fn server_reset_and_step() {
        let server = GymServer::bind("127.0.0.1:0").unwrap();
        let addr = server.local_addr().unwrap();

        let handle = std::thread::spawn(move || {
            let mut env = build_test_env();
            server.serve_one(&mut env).unwrap();
        });

        let mut stream = TcpStream::connect(addr).unwrap();

        // Reset
        let resp = send_and_receive(&mut stream, r#"{"type":"reset","seed":42}"#);
        let parsed: Response = serde_json::from_str(&resp).unwrap();
        assert!(matches!(parsed, Response::Reset { .. }));

        // Step
        let resp = send_and_receive(&mut stream, r#"{"type":"step","action":{"Discrete":0}}"#);
        let parsed: Response = serde_json::from_str(&resp).unwrap();
        assert!(matches!(parsed, Response::Step { .. }));

        // Close
        send_and_receive(&mut stream, r#"{"type":"close"}"#);
        drop(stream);
        handle.join().unwrap();
    }

    #[test]
    fn server_invalid_request() {
        let server = GymServer::bind("127.0.0.1:0").unwrap();
        let addr = server.local_addr().unwrap();

        let handle = std::thread::spawn(move || {
            let mut env = build_test_env();
            server.serve_one(&mut env).unwrap();
        });

        let mut stream = TcpStream::connect(addr).unwrap();

        let resp = send_and_receive(&mut stream, "not valid json");
        let parsed: Response = serde_json::from_str(&resp).unwrap();
        assert!(matches!(parsed, Response::Error { .. }));

        send_and_receive(&mut stream, r#"{"type":"close"}"#);
        drop(stream);
        handle.join().unwrap();
    }

    #[test]
    fn server_handles_client_disconnect() {
        let server = GymServer::bind("127.0.0.1:0").unwrap();
        let addr = server.local_addr().unwrap();

        let handle = std::thread::spawn(move || {
            let mut env = build_test_env();
            // serve_one should return Ok when client disconnects
            server.serve_one(&mut env).unwrap();
        });

        let stream = TcpStream::connect(addr).unwrap();
        // Immediately drop — server should handle gracefully
        drop(stream);
        handle.join().unwrap();
    }
}
