import os
import random
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Local model engine is optional; imported lazily
_LOCAL_ENGINE_CACHE = {}
_LOCAL_HF_ENGINE_CACHE = {}


class _LocalHFEngine:
    """Minimal local HF generation backend.

    Notes:
      - Uses AutoModelForCausalLM + AutoTokenizer.
      - Accepts a plain-text prompt (already contains any system/instruction formatting you want).
      - Keeps dependencies optional and imported lazily.
    """

    def __init__(self, *, model_path: str, device_map: str, torch_dtype: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = getattr(torch, torch_dtype, torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.model.eval()

    def _pick_device(self):
        import torch

        # If not sharded, model.device is reliable.
        dev = getattr(self.model, "device", None)
        if dev is not None and str(dev) not in ("meta", "cpu"):
            return dev

        # If sharded via accelerate, pick the first real device in the map.
        hf_map = getattr(self.model, "hf_device_map", None)
        if isinstance(hf_map, dict):
            for v in hf_map.values():
                if isinstance(v, str) and v not in ("cpu", "disk", "meta"):
                    return torch.device(v)

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate_text(self, prompt: str, *, max_new_tokens: int, temperature: float) -> str:
        import torch

        device = self._pick_device()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        do_sample = temperature is not None and float(temperature) > 0
        gen = self.model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=do_sample,
            temperature=float(temperature) if do_sample else None,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        # Decode only the newly generated part.
        new_tokens = gen[0][inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return (text or "").strip()


@dataclass(frozen=True)
class LocalModelSpec:
    model_path: str
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"


class LLMClient:
    """
    LLM client abstraction.

    Supported modes:
      - heuristic: no external calls; generates plausible counteroffers (fallback)
      - openai / openai_compatible: (kept for compatibility) chat.completions via OpenAI SDK
      - local_ministral: load a local Ministral/Mistral model (no API)
      - local_hf: load a local HuggingFace causal LM via transformers (no API)

    For local_ministral, pass:
      - model_path: local path to the model directory
      - device_map: e.g. 'auto'
      - torch_dtype: 'bfloat16' (default), 'float16', or 'float32'
    """

    def __init__(
        self,
        mode: str = "heuristic",
        model: str = "gpt-3.5-turbo",
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_s: float = 60.0,
        # local model args
        model_path: Optional[str] = None,
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
    ):
        self.mode = mode
        self.model = model
        self.base_url = base_url
        self.timeout_s = timeout_s

        self.model_path = model_path
        self.device_map = device_map
        self.torch_dtype = torch_dtype

        self._client = None
        self._sdk = None
        self._engine = None

        if self.mode == "local_ministral":
            if not self.model_path:
                raise ValueError("local_ministral mode requires model_path=...")

            spec = LocalModelSpec(self.model_path, self.device_map, self.torch_dtype)
            if spec not in _LOCAL_ENGINE_CACHE:
                from local_ministral_backend import LocalMinistralConfig, LocalMinistralEngine

                cfg = LocalMinistralConfig(
                    model_path=spec.model_path,
                    device_map=spec.device_map,
                    torch_dtype=spec.torch_dtype,
                )
                _LOCAL_ENGINE_CACHE[spec] = LocalMinistralEngine(cfg)
            self._engine = _LOCAL_ENGINE_CACHE[spec]
            return

        if self.mode == "local_hf":
            if not self.model_path:
                raise ValueError("local_hf mode requires model_path=...")

            spec = LocalModelSpec(self.model_path, self.device_map, self.torch_dtype)
            if spec not in _LOCAL_HF_ENGINE_CACHE:
                _LOCAL_HF_ENGINE_CACHE[spec] = _LocalHFEngine(
                    model_path=spec.model_path,
                    device_map=spec.device_map,
                    torch_dtype=spec.torch_dtype,
                )
            self._engine = _LOCAL_HF_ENGINE_CACHE[spec]
            return

        if self.mode in ("openai", "openai_compatible"):
            key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or os.getenv("MISTRAL_API_KEY")
            if not key:
                raise RuntimeError(
                    "No API key found. Set OPENAI_API_KEY (or API_KEY) for OpenAI, "
                    "or MISTRAL_API_KEY for Mistral, or pass api_key=... explicitly."
                )

            effective_base_url = self.base_url if self.mode == "openai_compatible" else (self.base_url or None)

            try:
                from openai import OpenAI  # type: ignore

                if effective_base_url:
                    self._client = OpenAI(api_key=key, base_url=effective_base_url, timeout=self.timeout_s)
                else:
                    self._client = OpenAI(api_key=key, timeout=self.timeout_s)
                self._sdk = "new"
            except Exception:
                import openai  # type: ignore

                openai.api_key = key
                if effective_base_url:
                    openai.api_base = effective_base_url
                self._client = openai
                self._sdk = "old"

    def complete_text(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """Return raw text completion."""
        if self.mode == "local_ministral":
            # max_tokens maps to max_new_tokens for local generation
            return self._engine.generate_text(prompt, max_new_tokens=max_tokens, temperature=temperature)

        if self.mode == "local_hf":
            return self._engine.generate_text(prompt, max_new_tokens=max_tokens, temperature=temperature)

        if self.mode in ("openai", "openai_compatible"):
            if self._sdk == "new":
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return (resp.choices[0].message.content or "").strip()

            resp = self._client.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp["choices"][0]["message"]["content"].strip()

        return self._heuristic_complete(prompt)

    def complete_prices(self, prompt: str, k: int, *, temperature: float = 0.7) -> List[int]:
        text = self.complete_text(prompt, max_tokens=64, temperature=temperature)
        nums = [int(x) for x in re.findall(r"\d+", text)]
        if len(nums) >= k:
            return nums[:k]

        # last-resort padding (should rarely trigger if your model follows instructions)
        lo, hi = 1, 1000000
        while len(nums) < k:
            nums.append(random.randint(lo, hi))
        return nums[:k]

    def _heuristic_complete(self, prompt: str) -> str:
        last_prices = [int(x) for x in re.findall(r"(?:Buyer|Seller):\s*(\d+)", prompt)]
        if last_prices and random.random() < 0.15:
            return str(last_prices[-1])
        return str(int(1000 + random.random() * 1000))
