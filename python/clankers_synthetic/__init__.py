"""clankers_synthetic: LLM-driven synthetic trajectory generation for Clankers."""

from __future__ import annotations

from clankers_synthetic.specs import (
    CanonicalPlan,
    ConstraintSpec,
    ConstraintViolation,
    DatasetManifest,
    ExecutionTrace,
    GuardCondition,
    LLMProposedPlan,
    LLMRequest,
    ObjectSpec,
    ObservationSpec,
    PlanRejection,
    ProposedSkill,
    ResolvedSkill,
    RobotSpec,
    SceneSpec,
    SimulationSpec,
    SkillParams,
    SuccessCriterion,
    TaskSpec,
    TraceStep,
    ValidationMetrics,
    ValidationReport,
)

__all__ = [
    # Scene models
    "SimulationSpec",
    "RobotSpec",
    "ObjectSpec",
    "ConstraintSpec",
    "ObservationSpec",
    "SceneSpec",
    # Task models
    "SuccessCriterion",
    "TaskSpec",
    # Plan models
    "GuardCondition",
    "SkillParams",
    "ProposedSkill",
    "LLMProposedPlan",
    "ResolvedSkill",
    "CanonicalPlan",
    "PlanRejection",
    # Execution models
    "TraceStep",
    "ExecutionTrace",
    # Validation models
    "ConstraintViolation",
    "ValidationMetrics",
    "ValidationReport",
    # LLM request
    "LLMRequest",
    # Dataset models
    "DatasetManifest",
]

__version__ = "0.1.0"
