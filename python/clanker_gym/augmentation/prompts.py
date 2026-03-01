"""Prompt templates for sim-to-real image generation.

Crafted to produce images that look like they were captured by a real robot
camera -- realistic lighting, sensor noise, non-ideal conditions.
"""

from __future__ import annotations

from enum import Enum


class SceneType(Enum):
    """Predefined scene contexts for prompt generation."""

    MANIPULATION = "manipulation"
    INDOOR_NAV = "indoor_nav"
    OUTDOOR = "outdoor"
    GENERIC = "generic"


# Base prompt components -- combined to create full prompts.
_BASE_REALISM = (
    "photorealistic image captured by a robot camera sensor, "
    "slight motion blur, natural indoor lighting with shadows, "
    "minor lens distortion, subtle sensor noise, realistic materials and textures"
)

_MANIPULATION_CONTEXT = (
    "robot arm workspace, tabletop scene, industrial robot camera view, "
    "overhead or angled camera perspective, lab or workshop environment, "
    "fluorescent and ambient lighting mix, slight reflections on surfaces"
)

_INDOOR_NAV_CONTEXT = (
    "indoor robot navigation view, hallway or room, floor-level camera perspective, "
    "mixed natural and artificial lighting, some overexposed windows, "
    "real building interior with wear and imperfections"
)

_OUTDOOR_CONTEXT = (
    "outdoor robot camera view, natural sunlight with shadows, "
    "slight lens flare, ground-level perspective, "
    "real outdoor environment with natural imperfections"
)

_GENERIC_CONTEXT = (
    "robot camera view, realistic environment, mixed lighting conditions, "
    "practical camera perspective, real-world setting"
)

_NEGATIVE_PROMPT = (
    "cartoon, anime, illustration, painting, drawing, art, sketch, "
    "unrealistic, 3d render, cgi, perfect lighting, studio lighting, "
    "oversaturated, overexposed, text, watermark, logo, blurry, "
    "low quality, deformed, distorted"
)

_CONTEXT_MAP = {
    SceneType.MANIPULATION: _MANIPULATION_CONTEXT,
    SceneType.INDOOR_NAV: _INDOOR_NAV_CONTEXT,
    SceneType.OUTDOOR: _OUTDOOR_CONTEXT,
    SceneType.GENERIC: _GENERIC_CONTEXT,
}


class PromptBuilder:
    """Builds prompts for sim-to-real image generation.

    Parameters
    ----------
    scene_type : SceneType
        The type of scene being generated.
    custom_suffix : str, optional
        Additional prompt text appended to the generated prompt.

    Examples
    --------
    >>> builder = PromptBuilder(SceneType.MANIPULATION)
    >>> prompt, negative = builder.build()
    >>> print(prompt[:40])
    'photorealistic image captured by a robot'
    """

    def __init__(
        self,
        scene_type: SceneType = SceneType.MANIPULATION,
        custom_suffix: str | None = None,
    ) -> None:
        self.scene_type = scene_type
        self.custom_suffix = custom_suffix

    def build(self) -> tuple[str, str]:
        """Build the positive and negative prompts.

        Returns
        -------
        tuple[str, str]
            (positive_prompt, negative_prompt)
        """
        context = _CONTEXT_MAP[self.scene_type]
        prompt = f"{_BASE_REALISM}, {context}"

        if self.custom_suffix:
            prompt = f"{prompt}, {self.custom_suffix}"

        return prompt, _NEGATIVE_PROMPT

    @staticmethod
    def negative_prompt() -> str:
        """Return the standard negative prompt."""
        return _NEGATIVE_PROMPT
