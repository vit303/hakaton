"""Загрузка Qwen VL и генерация ответа."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from rag.config import MAX_IMAGE_SIDE, MAX_NEW_TOKENS, QWEN_VL_MODEL
from rag.store import download_in_progress, weights_ready


class VisionModel:
    def __init__(self, model_name: str = QWEN_VL_MODEL) -> None:
        self.model_name = model_name
        self._model = None
        self._processor = None

    def caption(self, image_path: str) -> str:
        prompt = (
            "Опиши изображение для поискового индекса на русском языке. "
            "Если на нём есть текст, таблица, схема или числа — перепиши их. "
            "Если это фото или рисунок — кратко скажи, что изображено. Не больше 80 слов."
        )
        return self.generate(prompt, [image_path], system="Ты описываешь изображения точно и коротко.").strip()

    def generate(
        self,
        user_text: str,
        image_paths: list[str] | None = None,
        system: str = "",
        max_new_tokens: int | None = None,
    ) -> str:
        model, processor = self._load()
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel

        content: list[dict] = []
        for path in image_paths or []:
            content.append({"type": "image", "image": _prepare_image(path)})
        content.append({"type": "text", "text": user_text})
        messages = []
        if system:
            messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
        messages.append({"role": "user", "content": content})
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        if hasattr(inputs, "pop"):
            inputs.pop("token_type_ids", None)
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        prepared = {}
        items = inputs.items() if hasattr(inputs, "items") else []
        for key, value in items:
            if not hasattr(value, "to"):
                continue
            if getattr(value, "is_floating_point", lambda: False)():
                prepared[key] = value.to(device=device, dtype=dtype, non_blocking=True)
            else:
                prepared[key] = value.to(device=device, non_blocking=True)
        tokenizer = processor.tokenizer
        eos = getattr(tokenizer, "eos_token_id", None)
        pad = getattr(tokenizer, "pad_token_id", None) or eos
        limit = max_new_tokens or MAX_NEW_TOKENS
        with torch.inference_mode(), sdpa_kernel(
            [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
        ):
            generated = model.generate(
                **prepared,
                max_new_tokens=limit,
                do_sample=False,
                use_cache=True,
                eos_token_id=eos,
                pad_token_id=pad,
            )
        prompt_width = prepared["input_ids"].shape[-1]
        trimmed = generated[0, prompt_width:]
        return processor.batch_decode(
            [trimmed],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def _load(self):
        if self._model is not None:
            return self._model, self._processor
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA недоступна. Запустите скрипт в окружении: conda activate torch-cuda"
            )
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        if download_in_progress(self.model_name) and not weights_ready(self.model_name):
            raise RuntimeError(
                "Qwen VL ещё скачивается. Дождитесь окончания scripts\\download_models.py и повторите вопрос."
            )
        kwargs = {
            "dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
            "low_cpu_mem_usage": True,
        }
        try:
            model = Qwen3VLForConditionalGeneration.from_pretrained(self.model_name, **kwargs)
        except TypeError:
            kwargs["torch_dtype"] = kwargs.pop("dtype")
            model = Qwen3VLForConditionalGeneration.from_pretrained(self.model_name, **kwargs)
        model.to("cuda")
        model.eval()
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.use_cache = True
            model.generation_config.do_sample = False
        self._model = model
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        return self._model, self._processor


def _prepare_image(path: str) -> Image.Image:
    with Image.open(Path(path)) as source:
        image = source.convert("RGB")
        width, height = image.size
        scale = min(1.0, MAX_IMAGE_SIDE / max(width, height))
        if scale < 1:
            image = image.resize(
                (max(1, int(width * scale)), max(1, int(height * scale))),
                Image.Resampling.LANCZOS,
            )
        return image.copy()
