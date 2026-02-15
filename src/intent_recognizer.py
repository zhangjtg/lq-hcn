from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class IntentOutput:
    pred_id: int
    label: str


class IntentRecognizer:
    """Intent recognizer for buyer utterances.

    The code is intentionally lightweight so it can be used both in:
      - run_LQ_negotiation.py (online human-buyer runtime)
      - offline simulation (Train.py / batch generation)
    """

    def __init__(
        self,
        *,
        base_model_path: str,
        adapter_path: Optional[str] = None,
        num_labels: int = 6,
        torch_dtype: str = "bfloat16",
        device: Optional[str] = None,
        label2id: Optional[Dict[str, int]] = None,
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        dtype = getattr(torch, torch_dtype, torch.bfloat16)
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model_path,
            num_labels=num_labels,
            torch_dtype=dtype,
            trust_remote_code=True,
        )

        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model.to(device)
        self.model.eval()

        # Prefer model config mapping if provided.
        cfg_id2label = getattr(getattr(self.model, "config", None), "id2label", None)
        use_cfg = False
        if isinstance(cfg_id2label, dict) and cfg_id2label:
            vals = [str(v) for v in cfg_id2label.values()]
            # Ignore generic mappings like "LABEL_0".
            if any(("谈判" in v or "讨价" in v) for v in vals) and not all(v.upper().startswith("LABEL") for v in vals):
                use_cfg = True

        if use_cfg:
            self.id2label = {int(k): str(v) for k, v in cfg_id2label.items()}
        else:
            # Fallback to the mapping described in the user request.
            if label2id is None:
                label2id = {"讨价还价": 0, "谈判失败": 1, "谈判成功": 3}
            self.id2label = {idx: label for label, idx in label2id.items()}

    def predict(self, text: str) -> IntentOutput:
        import torch

        inputs = self.tokenizer(text, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs)
            logits = out.logits
            pred = int(torch.argmax(logits, dim=-1).item())

        label = self.id2label.get(pred, "讨价还价")
        return IntentOutput(pred_id=pred, label=label)
