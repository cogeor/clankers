"""Convert JSON episode files to MCAP format for replay.

Reads episode JSON from the packager output and writes MCAP files matching
the ``clankers-record`` schema (JointFrame, ActionFrame, RewardFrame).

Usage:
    python -m clankers_synthetic.mcap_writer input_episode.json output.mcap

Or programmatically:
    from clankers_synthetic.mcap_writer import episode_to_mcap
    episode_to_mcap("episode.json", "output.mcap")
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def episode_to_mcap(
    episode_path: str,
    output_path: str,
    *,
    control_dt: float = 0.02,
    joint_names: list[str] | None = None,
) -> None:
    """Convert a single JSON episode to MCAP.

    Parameters
    ----------
    episode_path : str
        Path to the JSON episode file.
    output_path : str
        Path for the output MCAP file.
    control_dt : float
        Timestep between control frames in seconds (default 0.02).
    joint_names : list[str] | None
        Joint names for JointFrame. If None, uses generic names.
    """
    try:
        from mcap.writer import Writer
    except ImportError as exc:
        raise ImportError("mcap package not installed. Install with: pip install mcap") from exc

    with open(episode_path) as f:
        episode = json.load(f)

    steps = episode.get("steps", [])
    if not steps:
        print(f"WARNING: no steps in {episode_path}")
        return

    # Infer joint count from obs or action dimensions
    n_action = len(steps[0].get("action", []))
    n_obs = len(steps[0].get("obs", []))
    n_joints = n_obs // 2 if n_obs > 0 else n_action

    if joint_names is None:
        joint_names = [f"joint_{i}" for i in range(n_joints)]

    dt_ns = int(control_dt * 1e9)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as out_f:
        writer = Writer(out_f)
        writer.start(profile="clankers", library="clankers_synthetic")

        # Register schemas
        joint_schema_id = writer.register_schema(
            name="clankers.JointFrame",
            encoding="jsonschema",
            data=json.dumps(
                {
                    "type": "object",
                    "properties": {
                        "timestamp_ns": {"type": "integer"},
                        "names": {"type": "array", "items": {"type": "string"}},
                        "positions": {"type": "array", "items": {"type": "number"}},
                        "velocities": {"type": "array", "items": {"type": "number"}},
                        "torques": {"type": "array", "items": {"type": "number"}},
                    },
                }
            ).encode(),
        )

        action_schema_id = writer.register_schema(
            name="clankers.ActionFrame",
            encoding="jsonschema",
            data=json.dumps(
                {
                    "type": "object",
                    "properties": {
                        "timestamp_ns": {"type": "integer"},
                        "data": {"type": "array", "items": {"type": "number"}},
                    },
                }
            ).encode(),
        )

        reward_schema_id = writer.register_schema(
            name="clankers.RewardFrame",
            encoding="jsonschema",
            data=json.dumps(
                {
                    "type": "object",
                    "properties": {
                        "timestamp_ns": {"type": "integer"},
                        "reward": {"type": "number"},
                    },
                }
            ).encode(),
        )

        # Register channels
        joint_channel_id = writer.register_channel(
            topic="/joint_states",
            message_encoding="json",
            schema_id=joint_schema_id,
        )

        action_channel_id = writer.register_channel(
            topic="/actions",
            message_encoding="json",
            schema_id=action_schema_id,
        )

        reward_channel_id = writer.register_channel(
            topic="/reward",
            message_encoding="json",
            schema_id=reward_schema_id,
        )

        for step_idx, step in enumerate(steps):
            timestamp_ns = step_idx * dt_ns

            # Joint states from obs (interleaved: [pos0, vel0, pos1, vel1, ...])
            obs = step.get("obs", [])
            if len(obs) >= 2 * n_joints:
                positions = obs[0 : 2 * n_joints : 2]
                velocities = obs[1 : 2 * n_joints : 2]
            else:
                positions = obs[:n_joints] if len(obs) >= n_joints else [0.0] * n_joints
                velocities = [0.0] * n_joints

            joint_frame = {
                "timestamp_ns": timestamp_ns,
                "names": joint_names,
                "positions": positions,
                "velocities": velocities,
                "torques": [0.0] * n_joints,
            }
            writer.add_message(
                channel_id=joint_channel_id,
                log_time=timestamp_ns,
                publish_time=timestamp_ns,
                data=json.dumps(joint_frame).encode(),
            )

            # Action
            action = step.get("action", [])
            action_frame = {
                "timestamp_ns": timestamp_ns,
                "data": action,
            }
            writer.add_message(
                channel_id=action_channel_id,
                log_time=timestamp_ns,
                publish_time=timestamp_ns,
                data=json.dumps(action_frame).encode(),
            )

            # Reward
            reward = step.get("reward", 0.0)
            reward_frame = {
                "timestamp_ns": timestamp_ns,
                "reward": reward,
            }
            writer.add_message(
                channel_id=reward_channel_id,
                log_time=timestamp_ns,
                publish_time=timestamp_ns,
                data=json.dumps(reward_frame).encode(),
            )

        writer.finish()

    print(f"Wrote {len(steps)} frames to {output_path}")


def main():
    """CLI entry point."""
    if len(sys.argv) < 3:
        print("Usage: python -m clankers_synthetic.mcap_writer <episode.json> <output.mcap>")
        print(
            "       python -m clankers_synthetic.mcap_writer <episode.json> <output.mcap> --dt 0.02"
        )
        sys.exit(1)

    episode_path = sys.argv[1]
    output_path = sys.argv[2]

    control_dt = 0.02
    if "--dt" in sys.argv:
        idx = sys.argv.index("--dt")
        if idx + 1 < len(sys.argv):
            control_dt = float(sys.argv[idx + 1])

    episode_to_mcap(episode_path, output_path, control_dt=control_dt)


if __name__ == "__main__":
    main()
