"""Tests for prompt generation."""

from __future__ import annotations

from clankers.augmentation.prompts import PromptBuilder, SceneType


class TestPromptBuilder:
    def test_all_scene_types_produce_prompts(self):
        """Every SceneType variant produces a (str, str) tuple, both non-empty."""
        for scene in SceneType:
            builder = PromptBuilder(scene)
            result = builder.build()
            assert isinstance(result, tuple)
            assert len(result) == 2
            prompt, negative = result
            assert isinstance(prompt, str) and len(prompt) > 0
            assert isinstance(negative, str) and len(negative) > 0

    def test_manipulation_prompt_contains_keywords(self):
        """MANIPULATION prompt contains expected robot/tabletop keywords."""
        builder = PromptBuilder(SceneType.MANIPULATION)
        prompt, _ = builder.build()
        prompt_lower = prompt.lower()
        assert "robot" in prompt_lower
        # Should mention tabletop or arm
        assert "tabletop" in prompt_lower or "arm" in prompt_lower

    def test_negative_prompt_contains_cartoon(self):
        """Negative prompt contains 'cartoon' and 'unrealistic'."""
        builder = PromptBuilder(SceneType.GENERIC)
        _, negative = builder.build()
        assert "cartoon" in negative.lower()
        assert "unrealistic" in negative.lower()

    def test_custom_suffix_appended(self):
        """custom_suffix is appended to the generated positive prompt."""
        builder = PromptBuilder(SceneType.MANIPULATION, custom_suffix="foggy weather")
        prompt, _ = builder.build()
        assert "foggy weather" in prompt

    def test_static_negative_prompt(self):
        """PromptBuilder.negative_prompt() returns a non-empty string."""
        neg = PromptBuilder.negative_prompt()
        assert isinstance(neg, str)
        assert len(neg) > 0
        assert "cartoon" in neg.lower()

    def test_no_suffix_when_none(self):
        """When custom_suffix is None the prompt does not end with a dangling comma."""
        builder = PromptBuilder(SceneType.GENERIC, custom_suffix=None)
        prompt, _ = builder.build()
        assert not prompt.rstrip().endswith(",")
