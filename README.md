# Clankers Workspace

This workspace now follows the documented crate structure in `.delegate/doc/crates`.

## App Entry Point
- `apps/clankers-app`

## Crates
- `crates/clankers-core`
- `crates/clankers-noise`
- `crates/clankers-env`
- `crates/clankers-actuator-core`
- `crates/clankers-actuator`
- `crates/clankers-urdf`
- `crates/clankers-gym`
- `crates/clankers-domain-rand`
- `crates/clankers-policy`
- `crates/clankers-render`
- `crates/clankers-teleop`
- `crates/clankers-viz`
- `crates/clankers-sim`

## ROS2 (Opt-In)
ROS2 is disabled by default and enabled explicitly, matching the documented behavior inspired by IsaacLab.

Run without ROS:
```powershell
cargo run -p clankers-app -- --mode training --steps 25
```

Run with ROS2 (requires sourced ROS env, e.g. `ROS_DISTRO` set):
```powershell
cargo run -p clankers-app --features clankers-sim/ros2 -- --mode training --steps 25 --ros2-enable --ros2-topic /clankers/status
```
