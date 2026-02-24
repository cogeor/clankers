"""Generate tiny ONNX test models for OnnxPolicy unit tests.

Requirements: pip install onnx numpy

Generates:
  - test_policy_none.onnx: 4-input, 1-output linear model, action_transform=none
  - test_policy_tanh.onnx: 4-input, 2-output linear model, action_transform=tanh
"""
import json
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def make_linear_policy(obs_dim, action_dim, filename, metadata=None):
    """Create a simple linear model: action = obs @ W_T + b.

    Uses identity-like weights so that action[i] = obs[i] for i < action_dim.
    """
    # Weight: identity-like (first action_dim rows of obs)
    W = np.eye(obs_dim, dtype=np.float32)[:action_dim]
    b = np.zeros(action_dim, dtype=np.float32)

    # W_T = W.T has shape (obs_dim, action_dim)
    W_T = W.T
    W_T_init = numpy_helper.from_array(W_T, name="W_T")
    b_init = numpy_helper.from_array(b, name="b")

    obs_input = helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, obs_dim])
    action_output = helper.make_tensor_value_info(
        "action", TensorProto.FLOAT, [1, action_dim]
    )

    matmul = helper.make_node("MatMul", ["obs", "W_T"], ["matmul_out"])
    add = helper.make_node("Add", ["matmul_out", "b"], ["action"])

    graph = helper.make_graph(
        [matmul, add],
        "test_policy",
        [obs_input],
        [action_output],
        initializer=[W_T_init, b_init],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    if metadata:
        for key, value in metadata.items():
            entry = model.metadata_props.add()
            entry.key = key
            entry.value = value

    onnx.checker.check_model(model)
    onnx.save(model, filename)
    print(f"  Saved {filename} ({obs_dim} -> {action_dim})")


if __name__ == "__main__":
    import os

    out_dir = os.path.dirname(os.path.abspath(__file__))

    # Basic model: 4 obs -> 1 action, no transform
    make_linear_policy(
        4,
        1,
        os.path.join(out_dir, "test_policy_none.onnx"),
        {
            "clanker_policy_version": "1.0.0",
            "action_transform": "none",
            "action_space": json.dumps(
                {"type": "Box", "shape": [1], "low": [-1], "high": [1]}
            ),
        },
    )

    # Tanh model: 4 obs -> 2 actions, tanh transform with scale
    make_linear_policy(
        4,
        2,
        os.path.join(out_dir, "test_policy_tanh.onnx"),
        {
            "clanker_policy_version": "1.0.0",
            "action_transform": "tanh",
            "action_space": json.dumps(
                {"type": "Box", "shape": [2], "low": [-2, -2], "high": [2, 2]}
            ),
            "action_scale": "[2.0, 2.0]",
            "action_offset": "[0.0, 0.0]",
        },
    )

    print("Done.")
