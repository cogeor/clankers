# Clankers

Robotics simulation framework built on Bevy ECS.

## App

- `apps/clankers-app` — example headless simulation

## Crates

| Crate | Description |
|-------|-------------|
| `clankers-core` | System ordering, `SimTime`, traits (`Sensor`, `ActionApplicator`) |
| `clankers-noise` | Noise generators (Gaussian, Ornstein-Uhlenbeck, uniform) |
| `clankers-actuator-core` | Motor model traits and types |
| `clankers-actuator` | Joint components, PD control, dynamics systems |
| `clankers-env` | Episode lifecycle, observation pipeline, reward/termination |
| `clankers-urdf` | URDF parser and Bevy entity spawning |
| `clankers-policy` | Policy runner abstraction (scripted, random) |
| `clankers-domain-rand` | Actuator parameter randomisation per episode |
| `clankers-teleop` | Input-source-agnostic teleoperation mapping |
| `clankers-render` | Headless frame buffer, `RenderConfig`, `ImageSensor` |
| `clankers-gym` | Gymnasium-compatible `GymEnv`, TCP server, JSON protocol |
| `clankers-sim` | Meta-plugin integrating core + actuator + env, `SceneBuilder` |
| `clankers-test-utils` | Shared test helpers |

## Quick Start

```sh
cargo test --workspace
```
