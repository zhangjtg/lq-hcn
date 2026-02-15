from __future__ import annotations

from typing import Any, List, Sequence, Tuple

from llm_client import LLMClient


def _format_dialogue_history(dialogue: Sequence[Tuple[str, str]], *, max_turns: int = 12) -> str:
    """Format dialogue to Chinese speaker tags for the seller NLG prompt."""
    if not dialogue:
        return "（暂无对话）"

    clipped = dialogue[-max_turns:]
    lines: List[str] = []
    for role, text in clipped:
        if role.lower() in ("buyer", "human", "user", "买家"):
            lines.append(f"买家：{text}")
        else:
            lines.append(f"卖家：{text}")
    return "\n".join(lines)


def build_seller_prompt(
    *,
    case: Any,
    dialogue: Sequence[Tuple[str, str]],
    buyer_bid: int,
    quantity: int,
    strategy: str,
    price: int,
) -> str:
    """Build the exact Chinese prompt requested by the user for Qwen3 generation."""

    dia_history = _format_dialogue_history(dialogue)

    if strategy == "Accept":
        output_require = "回复中不得出现任何数字或价格（包括单价、数量、折扣等）。"
    else:
        output_require = f"回复中必须包含且仅包含一个数字，该数字必须等于 {float(price):g}（允许小数），不得包含其它数字。"

    return f"""你是一名专业的二手商品卖家。请仔细阅读以下信息，并生成一段连贯的卖家回复，用来实现本回合的策略决策。回复必须遵守输出要求，并且与谈判策略模型给出的动作与出价严格一致。

【商品信息】
- 名称：{case['product_name']}
- 描述：{case['seller_item_description']}
- 初始价格：{case['init_price']}
- 底价：{case['seller_reserve_price']}

【对话历史】
{dia_history}

【谈判上下文（当前回合）】
- 买家出价：{buyer_bid}
- 数量：{quantity}

【策略决策（必须严格遵循）】
- 动作： {strategy}（取值之一：Counteroffer，Accept）
- 卖家出价：{price if strategy == 'Counteroffer' else 'null'}（策略计算得到的卖方报价；若动作为Accept则为null）

【输出要求】
- 动作一致性：回复必须体现给定的 {strategy}，不得与之矛盾。
- 数值一致性：{output_require}
- 结合上下文：回复需基于对话历史，礼貌且符合真实交易语境。
- 简洁：仅用 1～2 句简短中文表达。
- 格式：不要重复提示词、背景信息或规则说明。只输出卖家的回复内容。

【你的回复】
Seller:"""


class SellerResponseGenerator:
    """Generate Chinese seller utterances using an LLM (recommended: Qwen3-4B-Instruct-2507)."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def generate(
        self,
        *,
        case: Any,
        dialogue: Sequence[Tuple[str, str]],
        buyer_bid: int,
        quantity: int,
        strategy: str,
        price: int,
        max_tokens: int = 128,
        temperature: float = 0.2,
    ) -> str:
        prompt = build_seller_prompt(
            case=case,
            dialogue=dialogue,
            buyer_bid=buyer_bid,
            quantity=quantity,
            strategy=strategy,
            price=price,
        )
        out = self.llm.complete_text(prompt, max_tokens=max_tokens, temperature=temperature)
        # Some chat models may echo the "Seller:" prefix; strip it.
        out = out.strip()
        if out.lower().startswith("seller:"):
            out = out.split(":", 1)[-1].strip()
        return out
