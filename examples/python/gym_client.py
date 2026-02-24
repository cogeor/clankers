#!/usr/bin/env python3
"""Gym client for the Clankers cart-pole server.

Connects to the TCP gym server started by `cartpole_gym` example,
runs a few episodes with random actions, and prints observations.

Usage:
    # Terminal 1: start server
    cargo run -p clankers-examples --bin cartpole_gym

    # Terminal 2: run this client
    python examples/python/gym_client.py
"""

from __future__ import annotations

import json
import random
import socket
import struct
import sys

# --- Wire format: 4-byte LE length prefix + JSON payload ---


def send_msg(sock: socket.socket, msg: dict) -> None:
    payload = json.dumps(msg).encode("utf-8")
    sock.sendall(struct.pack("<I", len(payload)) + payload)


def recv_msg(sock: socket.socket) -> dict:
    raw_len = _recv_exact(sock, 4)
    if not raw_len:
        raise ConnectionError("server closed connection")
    (length,) = struct.unpack("<I", raw_len)
    payload = _recv_exact(sock, length)
    return json.loads(payload.decode("utf-8"))


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("connection lost")
        buf += chunk
    return buf


# --- Main ---


def main() -> None:
    host = "127.0.0.1"
    port = 9877
    num_episodes = 3
    act_dim = 2  # cart_slide + pole_hinge

    print(f"Connecting to {host}:{port}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((host, port))
    except ConnectionRefusedError:
        print(
            "ERROR: Could not connect. Start the server first:\n"
            "  cargo run -p clankers-examples --bin cartpole_gym"
        )
        sys.exit(1)

    print("Connected!\n")

    # 1. Handshake
    send_msg(sock, {"type": "Init", "client_version": "0.1.0"})
    init_resp = recv_msg(sock)
    print(f"Init response: {json.dumps(init_resp, indent=2)}")
    print()

    for ep in range(num_episodes):
        print(f"--- Episode {ep + 1} ---")

        # 2. Reset
        send_msg(sock, {"type": "Reset", "seed": ep * 100})
        reset_resp = recv_msg(sock)
        obs = reset_resp.get("observation", [])
        print(f"  Reset obs ({len(obs)} dims): {_fmt_obs(obs)}")

        # 3. Step loop
        step = 0
        done = False

        while not done:
            # Simple proportional controller: push cart opposite to pole angle
            # obs = [cart_pos, cart_vel, pole_pos, pole_vel]
            if len(obs) >= 4:
                cart_force = -2.0 * obs[0] - 1.0 * obs[1]  # PD on cart position
                # No force on pole (it's passive)
                action = [max(-1.0, min(1.0, cart_force)), 0.0]
            else:
                action = [random.uniform(-1, 1) for _ in range(act_dim)]

            send_msg(sock, {"type": "Step", "action": action})
            step_resp = recv_msg(sock)

            obs = step_resp.get("observation", [])
            done = step_resp.get("done", False)
            truncated = step_resp.get("truncated", False)
            step += 1

            if step % 100 == 0 or done or truncated:
                print(
                    f"  step {step:4d}: obs={_fmt_obs(obs)}  "
                    f"done={done}  truncated={truncated}"
                )

            if done or truncated:
                break

        print(f"  Episode finished after {step} steps\n")

    # 4. Close
    send_msg(sock, {"type": "Close"})
    sock.close()
    print("Client disconnected. Done!")


def _fmt_obs(obs: list[float]) -> str:
    return "[" + ", ".join(f"{v:+.3f}" for v in obs) + "]"


if __name__ == "__main__":
    main()
