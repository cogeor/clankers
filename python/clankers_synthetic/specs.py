"""Pydantic data contracts for the clankers_synthetic pipeline.

All models used across the synthetic trajectory generation system are defined here.
This module is self-contained and does not import from the ``clankers`` package.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Scene models (spec section 4.1)
# ---------------------------------------------------------------------------


class SimulationSpec(BaseModel):
    """Physics and timing configuration for the simulation."""

    gravity: list[float] = Field(default_factory=lambda: [0.0, 0.0, -9.81])
    physics_dt: float = 0.001
    control_dt: float = 0.02
    max_episode_steps: int = 500
    seed: int = 42


class RobotSpec(BaseModel):
    """Robot configuration derived from URDF and user overrides."""

    name: str
    urdf_path: str
    base_position: list[float]
    base_orientation: list[float]
    fixed_base: bool = True
    control_mode: str = "position_pd"
    joint_names: list[str] = Field(default_factory=list)
    joint_limits: dict[str, list[float]] = Field(default_factory=dict)
    joint_types: dict[str, str] = Field(default_factory=dict)
    ee_link_name: str = "end_effector"
    pd_gains: dict[str, list[float]] = Field(default_factory=dict)


class ObjectSpec(BaseModel):
    """Specification for a single scene object."""

    name: str
    shape: str
    shape_params: dict = Field(default_factory=dict)
    position: list[float]
    orientation: list[float] = Field(default_factory=lambda: [0, 0, 0, 1])
    mass: float = 0.1
    friction: float = 0.8
    restitution: float = 0.1
    is_static: bool = False
    is_graspable: bool = False
    is_container: bool = False


class ConstraintSpec(BaseModel):
    """Workspace and safety constraints for plan validation."""

    workspace_bounds_min: list[float]
    workspace_bounds_max: list[float]
    max_ee_speed: float = 1.0
    max_joint_velocity: float = 2.0
    max_contact_force: float = 100.0
    keep_upright: list[str] = Field(default_factory=list)
    avoid_regions: list[dict] = Field(default_factory=list)
    no_self_collision: bool = True


class ObservationSpec(BaseModel):
    """Observation modality and dimensionality specification."""

    modalities: list[str] = Field(default_factory=lambda: ["proprio"])
    joint_state_dim: int = 0
    ee_pose_dim: int = 7
    object_pose_dim: int = 0
    cameras: list[dict] = Field(default_factory=list)


class SceneSpec(BaseModel):
    """Complete scene description consumed by the LLM prompt assembler."""

    scene_id: str
    units: dict = Field(default_factory=lambda: {"length": "m", "angle": "rad", "time": "s"})
    simulation: SimulationSpec
    robot: RobotSpec
    objects: list[ObjectSpec]
    constraints: ConstraintSpec
    observations: ObservationSpec


# ---------------------------------------------------------------------------
# Task models (spec section 4.2)
# ---------------------------------------------------------------------------


class SuccessCriterion(BaseModel):
    """One success condition that maps to a TerminationFn."""

    type: str
    params: dict


class TaskSpec(BaseModel):
    """Task description for LLM planning."""

    task_id: str
    task_text: str
    success_criteria: list[SuccessCriterion]
    reward_hint: str = ""
    preferences: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Plan models (spec sections 5.1, 6.4)
# ---------------------------------------------------------------------------


class GuardCondition(BaseModel):
    """Optional early-termination condition for a skill."""

    type: str
    body: str | None = None
    min_force: float | None = None
    from_body: str | None = None
    threshold: float | None = None
    steps: int | None = None


class SkillParams(BaseModel):
    """Union of all skill parameter fields.

    Each field is optional because different skills use different subsets.
    """

    target: dict | None = None
    orientation: dict | None = None
    speed_fraction: float | None = None
    tolerance: float | None = None
    direction: list[float] | None = None
    distance: float | None = None
    guard: dict | None = None
    frame: str | None = None
    delta: list[float] | None = None
    width: float | None = None
    force: float | None = None
    wait_settle_steps: int | None = None
    steps: int | None = None
    targets: dict[str, float] | None = None


class ProposedSkill(BaseModel):
    """A single skill as proposed by the LLM."""

    name: str
    params: dict = Field(default_factory=dict)
    comment: str | None = None


class LLMProposedPlan(BaseModel):
    """Raw plan output from the LLM before validation."""

    plan_id: str
    plan_type: str
    rationale: str
    assumptions: list[str] = Field(default_factory=list)
    uncertainty_flags: list[str] = Field(default_factory=list)
    skills: list[ProposedSkill]


class ResolvedSkill(BaseModel):
    """A skill with all references resolved to world-frame coordinates."""

    name: str
    target_world_position: list[float] | None = None
    target_orientation: list[float] | None = None
    params: dict = Field(default_factory=dict)
    guard: GuardCondition | None = None


class CanonicalPlan(BaseModel):
    """Validated plan with resolved references, ready for compilation."""

    plan_id: str
    skills: list[ResolvedSkill]
    assumptions: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class PlanRejection(BaseModel):
    """Structured rejection when a plan fails parsing/validation."""

    reasons: list[str]
    raw_plan: dict = Field(default_factory=dict)
    error_codes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Execution models (spec section 6.5)
# ---------------------------------------------------------------------------


class TraceStep(BaseModel):
    """One transition in an execution trace."""

    obs: list[float]
    action: list[float]
    next_obs: list[float]
    reward: float
    terminated: bool
    truncated: bool
    info: dict = Field(default_factory=dict)


class ExecutionTrace(BaseModel):
    """Complete execution trace of a plan through the simulator."""

    plan_id: str
    steps: list[TraceStep]
    total_reward: float
    terminated: bool
    truncated: bool
    final_info: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Validation models (spec section 6.6)
# ---------------------------------------------------------------------------


class ConstraintViolation(BaseModel):
    """A single constraint violation recorded during validation."""

    type: str
    step: int
    details: str


class ValidationMetrics(BaseModel):
    """Quantitative metrics computed over an execution trace."""

    total_steps: int
    max_contact_force: float
    max_joint_velocity: float
    max_ee_speed: float
    final_object_poses: dict[str, list[float]] = Field(default_factory=dict)
    success_at_step: int | None = None


class ValidationReport(BaseModel):
    """Result of validating an execution trace against task and constraints."""

    passed: bool
    task_success: bool
    constraint_violations: list[ConstraintViolation] = Field(default_factory=list)
    metrics: ValidationMetrics
    failure_reason: str | None = None


# ---------------------------------------------------------------------------
# LLM request model
# ---------------------------------------------------------------------------


class LLMRequest(BaseModel):
    """Structured LLM request containing assembled prompt messages."""

    system_message: str
    user_message: str
    model: str
    temperature: float
    max_tokens: int


# ---------------------------------------------------------------------------
# Dataset models (spec section 6.9)
# ---------------------------------------------------------------------------


class DatasetManifest(BaseModel):
    """Manifest written alongside a packaged dataset."""

    output_dir: str
    n_episodes: int
    n_original: int
    n_augmented: int
    schema_version: str
    scene_spec_hash: str
    task_spec_hash: str
    prompt_template_version: str
    llm_model: str
    stats: dict = Field(default_factory=dict)
    split_sizes: dict = Field(default_factory=dict)
