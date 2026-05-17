"""Qwen-VL crop captioner — drop-in replacement for MoondreamCaptioner.

Exposes `.caption(pil_crop, prompt=None) -> str` so it can be plugged into
CropwiseMoondreamPipeline without any pipeline-level changes.

Two loading paths:
  - unsloth/*   : FastVisionModel 4-bit (already downloaded, fits 16 GB with RT-DETR)
  - Qwen/*      : standard HF AutoModel (fp16/bf16, needs more VRAM)
"""

from __future__ import annotations

import torch
from PIL import Image


class QwenCropCaptioner:
    def __init__(
        self,
        model_id: str = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
        device: str = "cuda",
        prompt: str | None = None,
        max_new_tokens: int = 64,
    ) -> None:
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.prompt = prompt or (
            "Briefly describe the condition and action of the main subject in this image. "
            "Keep it very short, like a label (e.g., crashed car, injured person sitting, "
            "paramedic helping)."
        )
        self._unsloth = model_id.startswith("unsloth/")

        if self._unsloth:
            self._load_unsloth(model_id)
        else:
            self._load_hf(model_id)

    def _load_unsloth(self, model_id: str) -> None:
        from unsloth import FastVisionModel
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=model_id,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        FastVisionModel.for_inference(model)
        model.eval()
        self.model = model
        self.tokenizer = tokenizer

    def _load_hf(self, model_id: str) -> None:
        from transformers import AutoProcessor
        from vista.qwen import AutoModelQwenVL
        self.model = AutoModelQwenVL.from_pretrained(model_id)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def caption(self, crop: Image.Image, prompt: str | None = None) -> str:
        p = prompt or self.prompt
        if self._unsloth:
            return self._caption_unsloth(crop, p)
        return self._caption_hf(crop, p)

    def _caption_unsloth(self, crop: Image.Image, prompt: str) -> str:
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        input_text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            crop, input_text, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)
        out_ids = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, use_cache=True
        )
        gen_ids = out_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    def _caption_hf(self, crop: Image.Image, prompt: str) -> str:
        from qwen_vl_utils import process_vision_info
        messages = [{"role": "user", "content": [
            {"type": "image", "image": crop},
            {"type": "text", "text": prompt},
        ]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, return_tensors="pt"
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        out_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        gen_ids = out_ids[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(gen_ids, skip_special_tokens=True).strip()
