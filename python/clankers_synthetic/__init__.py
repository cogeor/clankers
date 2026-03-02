"""clankers_synthetic: LLM-driven synthetic trajectory generation for Clankers."""

from __future__ import annotations

from clankers_synthetic.cli import main
from clankers_synthetic.compiler import SkillCompiler
from clankers_synthetic.ik_solver import (
    DlsSolver,
    IKResult,
    JointInfo,
    KinematicChain,
)
from clankers_synthetic.openai_client import OpenAIClient, OpenAIClientError
from clankers_synthetic.packager import DatasetPackager
from clankers_synthetic.parser import VALID_SKILLS, PlanParser
from clankers_synthetic.pipeline import generate_dataset
from clankers_synthetic.planner import LLMPlanner, PromptAssembler
from clankers_synthetic.pvcb_refiner import PVCBRefiner
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
from clankers_synthetic.validator import SimValidator

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
    # OpenAI client
    "OpenAIClient",
    "OpenAIClientError",
    # Parser
    "PlanParser",
    "VALID_SKILLS",
    # IK solver
    "DlsSolver",
    "KinematicChain",
    "JointInfo",
    "IKResult",
    # Planner
    "PromptAssembler",
    "LLMPlanner",
    # Compiler
    "SkillCompiler",
    # PVCB Refiner
    "PVCBRefiner",
    # Packager
    "DatasetPackager",
    # Validator
    "SimValidator",
    # Pipeline
    "generate_dataset",
    # CLI
    "main",
]

__version__ = "0.1.0"
