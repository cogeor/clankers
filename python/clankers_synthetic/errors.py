"""Typed exceptions for the clankers_synthetic compiler.

Per ``docs/plans/WS8-plan.md`` § 9 PR3 finding #6: ``move_relative`` must
fail loudly when no FK source is available rather than silently computing
``delta from world origin`` (the pre-W8 bug).
"""

from __future__ import annotations


class MoveRelativeWithoutFkError(RuntimeError):
    """Raised when ``move_relative`` is invoked without an FK source.

    The compiler needs the current end-effector pose to apply the
    requested Cartesian delta. An FK source is available when:

    1. The env exposes a callable ``forward_kinematics`` attribute, OR
    2. The most recent ``info`` dict contains an ``end_effector`` entry
       under ``body_poses``.

    Without either, the compiler would compute ``delta_from_origin`` (the
    pre-W8 behaviour) which silently corrupts trajectories.

    The legacy path is still reachable via
    ``SkillCompiler(legacy_move_relative=True)`` (and the
    ``--legacy-move-relative`` CLI flag), which emits
    ``DeprecationWarning`` for one release before the flag is removed.
    """
