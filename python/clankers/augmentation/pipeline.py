"""SD 1.5 + ControlNet-seg inference pipeline for sim-to-real augmentation.

Wraps the HuggingFace diffusers library to provide a simple interface for
transforming segmentation images into photorealistic images.

Requires: diffusers>=0.25.0, transformers, torch, accelerate
"""

from __future__ import annotations

import contextlib

import numpy as np
from numpy.typing import NDArray

from clankers.augmentation.palette import PaletteRemapper
from clankers.augmentation.prompts import PromptBuilder, SceneType

# Default model identifiers
DEFAULT_SD_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DEFAULT_CONTROLNET_MODEL = "lllyasviel/control_v11p_sd15_seg"


class Sim2RealPipeline:
    """Transforms segmentation images into photorealistic robot camera images.

    Uses Stable Diffusion 1.5 with ControlNet segmentation conditioning.
    The pipeline:
    1. Remaps the input segmentation palette to ADE20K colors
    2. Generates a photorealistic image conditioned on the segmentation map
    3. Returns the result as a uint8 numpy array

    Parameters
    ----------
    device : str
        Torch device ("cuda", "cpu", "mps"). Default "cuda".
    dtype : str
        Model precision. "float16" for GPU, "float32" for CPU. Default "float16".
    sd_model : str
        Stable Diffusion model ID. Default "stable-diffusion-v1-5/stable-diffusion-v1-5".
    controlnet_model : str
        ControlNet model ID. Default "lllyasviel/control_v11p_sd15_seg".
    scene_type : SceneType
        Scene context for prompt generation. Default MANIPULATION.
    seed : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> pipeline = Sim2RealPipeline(device="cuda")
    >>> realistic = pipeline.generate(seg_image, num_inference_steps=20)
    >>> realistic.shape  # (H, W, 3), dtype uint8
    """

    def __init__(
        self,
        device: str = "cuda",
        dtype: str = "float16",
        sd_model: str = DEFAULT_SD_MODEL,
        controlnet_model: str = DEFAULT_CONTROLNET_MODEL,
        scene_type: SceneType = SceneType.MANIPULATION,
        seed: int | None = None,
    ) -> None:
        self.device = device
        self.dtype_str = dtype
        self.sd_model_id = sd_model
        self.controlnet_model_id = controlnet_model
        self.scene_type = scene_type
        self.seed = seed

        self._remapper = PaletteRemapper()
        self._prompt_builder = PromptBuilder(scene_type)
        self._pipe = None  # Lazy-loaded

    def _load_pipeline(self) -> None:
        """Lazy-load the diffusion pipeline (downloads models on first use)."""
        if self._pipe is not None:
            return

        import torch
        from diffusers import (
            ControlNetModel,
            StableDiffusionControlNetPipeline,
            UniPCMultistepScheduler,
        )

        torch_dtype = torch.float16 if self.dtype_str == "float16" else torch.float32

        controlnet = ControlNetModel.from_pretrained(
            self.controlnet_model_id,
            torch_dtype=torch_dtype,
        )

        self._pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.sd_model_id,
            controlnet=controlnet,
            torch_dtype=torch_dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )

        # Use efficient scheduler
        self._pipe.scheduler = UniPCMultistepScheduler.from_config(  # type: ignore[attr-defined]
            self._pipe.scheduler.config  # type: ignore[attr-defined]
        )

        self._pipe = self._pipe.to(self.device)  # type: ignore[attr-defined]

        # Enable memory-efficient attention if available
        with contextlib.suppress(ImportError, ModuleNotFoundError):
            self._pipe.enable_xformers_memory_efficient_attention()  # type: ignore[attr-defined]

    def generate(
        self,
        segmentation_image: NDArray[np.uint8],
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0,
        prompt_override: str | None = None,
        negative_prompt_override: str | None = None,
        seed: int | None = None,
    ) -> NDArray[np.uint8]:
        """Generate a photorealistic image from a segmentation map.

        Parameters
        ----------
        segmentation_image : NDArray[np.uint8], shape (H, W, 3)
            Input segmentation image with Clankers palette colors.
        num_inference_steps : int
            Number of denoising steps. More = higher quality but slower. Default 20.
        guidance_scale : float
            Classifier-free guidance scale. Higher = more prompt adherence. Default 7.5.
        controlnet_conditioning_scale : float
            ControlNet influence strength. 1.0 = full conditioning. Default 1.0.
        prompt_override : str, optional
            Custom positive prompt (overrides scene_type prompt).
        negative_prompt_override : str, optional
            Custom negative prompt.
        seed : int, optional
            Random seed for this generation (overrides pipeline-level seed).

        Returns
        -------
        NDArray[np.uint8], shape (H, W, 3)
            Photorealistic RGB image.
        """
        import torch
        from PIL import Image

        self._load_pipeline()

        # 1. Remap palette to ADE20K
        ade20k_image = self._remapper.remap(segmentation_image)

        # 2. Convert to PIL Image (diffusers expects PIL)
        control_image = Image.fromarray(ade20k_image)

        # 3. Build prompt
        if prompt_override:
            prompt = prompt_override
            negative = negative_prompt_override or PromptBuilder.negative_prompt()
        else:
            prompt, negative = self._prompt_builder.build()

        if negative_prompt_override and not prompt_override:
            negative = negative_prompt_override

        # 4. Set seed
        effective_seed = seed if seed is not None else self.seed
        generator = None
        if effective_seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(effective_seed)

        # 5. Run inference
        h, w = segmentation_image.shape[:2]
        # SD 1.5 UNet requires dimensions divisible by 64.
        # Clamp to at least 64 to avoid zero-size or too-small images.
        gen_h = max(64, (h // 64) * 64)
        gen_w = max(64, (w // 64) * 64)

        assert self._pipe is not None  # guaranteed by _load_pipeline()
        result = self._pipe(
            prompt=prompt,
            negative_prompt=negative,
            image=control_image.resize((gen_w, gen_h), Image.NEAREST),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator,
            height=gen_h,
            width=gen_w,
        )

        # 6. Convert back to numpy uint8
        output_pil = result.images[0]

        # Resize back to original dimensions if needed
        if (gen_h, gen_w) != (h, w):
            output_pil = output_pil.resize((w, h), Image.LANCZOS)

        return np.array(output_pil, dtype=np.uint8)

    def generate_batch(
        self,
        segmentation_images: list[NDArray[np.uint8]],
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0,
        seed: int | None = None,
    ) -> list[NDArray[np.uint8]]:
        """Generate photorealistic images from a batch of segmentation maps.

        Parameters
        ----------
        segmentation_images : list of NDArray[np.uint8]
            Input segmentation images.
        num_inference_steps : int
            Denoising steps per image.
        guidance_scale : float
            CFG scale.
        controlnet_conditioning_scale : float
            ControlNet strength.
        seed : int, optional
            Base seed. Each image gets seed + index for reproducibility.

        Returns
        -------
        list of NDArray[np.uint8]
            Photorealistic images.
        """
        results = []
        for i, seg_img in enumerate(segmentation_images):
            img_seed = (seed + i) if seed is not None else None
            result = self.generate(
                seg_img,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                seed=img_seed,
            )
            results.append(result)
        return results
