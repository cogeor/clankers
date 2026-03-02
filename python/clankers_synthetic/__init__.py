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
    "VALID_SKILLS",
    "CanonicalPlan",
    "ConstraintSpec",
    # Validation models
    "ConstraintViolation",
    # Dataset models
    "DatasetManifest",
    # Packager
    "DatasetPackager",
    # IK solver
    "DlsSolver",
    "ExecutionTrace",
    # Plan models
    "GuardCondition",
    "IKResult",
    "JointInfo",
    "KinematicChain",
    "LLMPlanner",
    "LLMProposedPlan",
    # LLM request
    "LLMRequest",
    "ObjectSpec",
    "ObservationSpec",
    # OpenAI client
    "OpenAIClient",
    "OpenAIClientError",
    # PVCB Refiner
    "PVCBRefiner",
    # Parser
    "PlanParser",
    "PlanRejection",
    # Planner
    "PromptAssembler",
    "ProposedSkill",
    "ResolvedSkill",
    "RobotSpec",
    "SceneSpec",
    # Validator
    "SimValidator",
    # Scene models
    "SimulationSpec",
    # Compiler
    "SkillCompiler",
    "SkillParams",
    # Task models
    "SuccessCriterion",
    "TaskSpec",
    # Execution models
    "TraceStep",
    "ValidationMetrics",
    "ValidationReport",
    # Pipeline
    "generate_dataset",
    # CLI
    "main",
]

__version__ = "0.1.0"
