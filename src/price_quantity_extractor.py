import re
from typing import Any, Dict, Optional, Tuple, Union


Number = Union[int, float]


def load_price_model_and_tokenizer(
    *,
    base_model_path: str,
    tokenizer_path: Optional[str] = None,
    adapter_path: Optional[str] = None,
    device_map: str = "auto",
    torch_dtype: str = "bfloat16",
):
    """Load the price+quantity extraction model.

    Supports a PEFT adapter that can be merged (as described by the user):

      - tokenizer from tokenizer_path (often points to the fine-tuned adapter dir)
      - base model from base_model_path
      - adapter loaded from adapter_path, then merge_and_unload()
    """

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok_path = tokenizer_path or base_model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    dtype = getattr(torch, torch_dtype, torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map=device_map,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        # merge adapter weights into the base model for faster inference
        if hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


class PriceQuantityExtractor:
    """Wraps your existing price+quantity extraction model.

    This module is used by the WeChat human-in-the-loop negotiation loop:
      - input: the full dialogue transcript (string)
      - output: (unit_price, quantity)

    Notes
    -----
    * If you already have a class that implements `extra_price_quantity(self, tran_resp)`
      (as you pasted), you can:
        - copy that method into this class, OR
        - subclass this class and override `extra_price_quantity`.
    * For deployment robustness, we include a regex fallback when the model is not configured.
    """

    def __init__(
        self,
        *,
        case: Optional[Dict[str, Any]] = None,
        # If you want to use the neural extractor, pass these in (or set them later).
        price_model: Any = None,
        price_tokenizer: Any = None,
        # Or provide paths and let us load them.
        base_model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        adapter_path: Optional[str] = None,
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
        device: str = "cuda",
    ):
        self.case = case or {}

        if price_model is None and price_tokenizer is None and base_model_path:
            price_model, price_tokenizer = load_price_model_and_tokenizer(
                base_model_path=base_model_path,
                tokenizer_path=tokenizer_path,
                adapter_path=adapter_path,
                device_map=device_map,
                torch_dtype=torch_dtype,
            )

        self.price_model = price_model
        self.price_tokenizer = price_tokenizer
        self.device = device

    # ----------------------
    # 1) Neural extraction
    # ----------------------
    def extra_price_quantity(self, tran_resp: str) -> Tuple[Union[Number, str], Union[Number, str]]:
        """Extract (unit_price, quantity) from the buyer's dialogue.

        Return format is kept identical to your pasted implementation:
          - price: float or 'null'
          - quantity: float or 'null'

        If `price_model/price_tokenizer` are not configured, falls back to regex.
        """

        # If your model isn't wired in yet, do not crash the whole negotiation thread.
        if self.price_model is None or self.price_tokenizer is None:
            return self._regex_fallback(tran_resp)

        # Lazy import to keep this module importable even in minimal environments.
        import torch

        case = self.case or {}
        prompt = f"""是谈判对话识别助手，任务是分析对话，识别当前轮次买家提及或接受的单价和商品数量，并且严格按照指定格式输出。

【识别规则】
1. 单价识别规则：
- 买家直接提及单价（\"X元/台\"、\"单价X\"）
- 买家接受卖家价格（\"X元可以\"、\"同意这个价\"）
- 基于总价和数量计算（总价÷当前数量 或 总价÷上下文数量）
- 无法直接判断价格类型时，通过与初始价、保留价或市场价的对比及上下文语义推断其含义。
- 如提及折扣，计算折后单价
- 如间接引用之前的价格
- 当前轮次完全无价格信息则为null

2. 数量识别规则：
- 优先使用当前轮次买家明确提及的数量
- 若未明确提及但基于前轮已确定数量继续讨论，则推断该数量
- 当买家提出价格但未提及数量时：
  • 如上下文有明确数量则使用上下文数量
  • 如上下文无数量信息则默认数量为1
- 无法确定则为null

【输出格式】
当前单价：[数值/null]
商品数量：[数值/null]

【输入数据】
产品信息：{case.get('product_name', '')}（{case.get('seller_item_description', '')}）
初始价：{case.get('init_price', '')}
对话记录：{tran_resp}

【输出】
"""

        inputs = self.price_tokenizer(prompt, return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)
        else:
            # transformers returns a dict-like BatchEncoding
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self.price_model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0,
                do_sample=False,
                early_stopping=True,
                eos_token_id=getattr(self.price_tokenizer, "eos_token_id", None),
            )

        # decode only the newly generated part
        input_len = inputs["input_ids"].shape[1] if isinstance(inputs, dict) else inputs.input_ids.shape[1]
        response = self.price_tokenizer.decode(output[0][input_len:], skip_special_tokens=True)

        price_match = re.search(r"当前单价[：:]\s*([\d,]+\.?\d*|null)", response)
        if price_match:
            price_str = price_match.group(1).lower().replace(",", "")
            price: Union[Number, str] = float(price_str) if price_str != "null" else "null"
        else:
            price = "null"

        quantity_match = re.search(r"商品数量[：:]\s*([\d,]+\.?\d*|null)", response)
        if quantity_match:
            quantity_str = quantity_match.group(1).lower().replace(",", "")
            quantity: Union[Number, str] = float(quantity_str) if quantity_str != "null" else "null"
        else:
            quantity = "null"

        return price, quantity

    # ----------------------
    # 2) Convenience wrappers
    # ----------------------
    def set_case(self, case: Dict[str, Any]) -> None:
        self.case = case or {}

    def extract(self, tran_resp: str, *, case: Optional[Dict[str, Any]] = None) -> Tuple[Optional[int], Optional[float]]:
        """Returns (unit_price_int, quantity_float) with None meaning unknown."""
        if case is not None:
            self.set_case(case)

        price, qty = self.extra_price_quantity(tran_resp)

        price_out: Optional[int]
        qty_out: Optional[float]

        price_out = None if price == "null" else int(round(float(price)))
        qty_out = None if qty == "null" else float(qty)
        return price_out, qty_out

    # ----------------------
    # Fallback
    # ----------------------
    def _regex_fallback(self, tran_resp: str) -> Tuple[Union[Number, str], Union[Number, str]]:
        """Very simple fallback: extract first 1-2 numbers.

        This is only used when your neural extractor isn't wired in.
        """
        s = str(tran_resp)
        nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s)]
        if not nums:
            return "null", "null"

        # Heuristic: first number is price, second number (if any) is quantity.
        price = nums[0]
        qty = nums[1] if len(nums) > 1 else "null"
        return price, qty
