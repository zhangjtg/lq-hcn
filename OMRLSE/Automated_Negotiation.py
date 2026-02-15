# Automated_Negotiation.py


from __future__ import annotations
import json
import math
import random
import re
import os
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict

import numpy as np
import torch



try:
    from transformers import MistralCommonBackend, Mistral3ForConditionalGeneration
except Exception:
    MistralCommonBackend = None
    Mistral3ForConditionalGeneration = None


# =========================
# Fallback LLM生成函数
# =========================

def _fallback_llm_generate_text(
    tokenizer, 
    model, 
    prompt, 
    role: str = "", 
    max_new_tokens: int = 128, 
    temperature: float = 0.0, 
    do_sample: bool = False
):
    """
    最小化的LLM生成函数
    支持字符串或消息列表作为输入
    """
    if model is None or tokenizer is None:
        raise RuntimeError("llm_generate_text不可用，且未提供(tokenizer, model)")
    
    # 处理输入
    if isinstance(prompt, list):
        s = "\n".join([f"{m.get('role', '')}: {m.get('content', '')}" for m in prompt])
    else:
        s = str(prompt)
    
    # 设备放置
    device = None
    try:
        device = next(model.parameters()).device
    except Exception:
        device = getattr(model, "device", None)
    
    inputs = tokenizer(s, return_tensors="pt")
    if device is not None:
        try:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        except Exception:
            pass
    
    # 生成
    gen = model.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        do_sample=bool(do_sample),
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
    )
    
    out = tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return out.strip()


# =========================
# LLM生成函数（主函数）
# =========================

def llm_generate_text(
    tokenizer, 
    model, 
    prompt, 
    role: str = 'default', 
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    do_sample: bool = True
):
    """通用LLM文本生成函数"""
    try:
        # 尝试使用chat template
        if isinstance(prompt, list):
            inputs = tokenizer.apply_chat_template(prompt, return_tensors="pt", add_generation_prompt=True)
            inputs = torch.tensor(inputs).unsqueeze(0).to(model.device)
        else:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else tokenizer.pad_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return response.strip()
    except Exception as e:
        # 回退到fallback
        return _fallback_llm_generate_text(
            tokenizer, model, prompt, role, max_new_tokens, temperature, do_sample
        )


# =========================
# 数据模型
# =========================

@dataclass(frozen=True)
class CommodityRecord:
    """商品记录"""
    product_id: str
    product_name: str
    seller_item_description: str
    init_price: float
    buyer_reserve_price: float
    seller_reserve_price: float
    quantity: float


@dataclass
class DialogueTurn:
    """对话轮次"""
    role: str
    text: str
    intent: Optional[str] = None
    extracted_price: Optional[float] = None


@dataclass
class DialogueState:
    """对话状态"""
    turns: List[DialogueTurn] = field(default_factory=list)

    def add(self, role: str, text: str, intent: Optional[str] = None, extracted_price: Optional[float] = None):
        self.turns.append(DialogueTurn(role=role, text=text, intent=intent, extracted_price=extracted_price))

    def to_text(self, max_turns: int = 12) -> str:
        """转换为文本（含元数据）"""
        ts = self.turns[-max_turns:]
        out = []
        role_map = {"seller": "卖家", "buyer": "买家", "system": "系统"}
        for t in ts:
            meta = []
            if t.intent:
                meta.append(f"意图={t.intent}")
            if t.extracted_price is not None:
                meta.append(f"价格={t.extracted_price}")
            meta_s = f" [{ ' '.join(meta) }]" if meta else ""
            role_zh = role_map.get((t.role or "").lower(), str(t.role))
            out.append(f"{role_zh}：{t.text}{meta_s}")
        return "\n".join(out)
    
    def to_conversation_text(self, max_turns: int = 12) -> str:
        """转换为对话文本（不含元数据）"""
        ts = self.turns[-max_turns:]
        out = []
        role_map = {"seller": "卖家", "buyer": "买家"}
        for t in ts:
            role_zh = role_map.get((t.role or "").lower(), str(t.role))
            out.append(f"{role_zh}: {t.text}")
        return "\n".join(out)


# =========================
# JSON加载
# =========================

def load_commodities_from_json(path: str) -> List[CommodityRecord]:
    """从JSON加载商品数据"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    items: List[CommodityRecord] = []
    for it in raw:
        items.append(
            CommodityRecord(
                product_id=str(it["product_id"]),
                product_name=str(it["product_name"]),
                seller_item_description=str(it["seller_item_description"]),
                init_price=float(it["init_price"]),
                buyer_reserve_price=float(it["buyer_reserve_price"]),
                seller_reserve_price=float(it["seller_reserve_price"]),
                quantity=float(it.get("quantity", 1)),
            )
        )
    return items


# =========================
# 价格分层编码器
# =========================

@dataclass
class PriceTierEncoder:
    """价格分层编码器"""
    edges: np.ndarray

    @staticmethod
    def fit(init_prices: List[float], n_tiers: int = 8) -> "PriceTierEncoder":
        p = np.array(init_prices, dtype=float)
        qs = np.linspace(0.0, 1.0, n_tiers + 1)
        edges = np.quantile(p, qs)
        edges = np.unique(edges)
        if len(edges) < 2:
            mn = float(np.min(p))
            mx = float(np.max(p))
            edges = np.array([mn, mx + 1e-6], dtype=float)
        return PriceTierEncoder(edges=edges)

    def n_tiers(self) -> int:
        return max(1, len(self.edges) - 1)

    def transform(self, init_price: float) -> int:
        tier = int(np.searchsorted(self.edges, float(init_price), side="right") - 1)
        return int(np.clip(tier, 0, self.n_tiers() - 1))


# =========================
# 卖家价格网格计算
# =========================

def compute_product_grid_seller_only(
        init_price: float,
        seller_reserve_price: float,
        max_rounds: int = 8,
        headroom_pct: float = 0.00,
        currency_step: float = 0.50,
        pct_step: float = 0.01,
        desired_steps: int = 24,
        min_bins: int = 21,
        max_bins: int = 151,
        money_tick: float = 0.01,
) -> np.ndarray:
    """计算卖家价格网格"""
    p_low = float(seller_reserve_price)
    p_high = float(init_price) * (1.0 + float(headroom_pct))

    if p_high <= p_low + 1e-9:
        grid = np.array([p_low, p_low + max(money_tick, 1e-6)], dtype=float)
        if money_tick > 0:
            grid = np.round(grid / money_tick) * money_tick
        return np.unique(grid)

    delta = max(float(currency_step), float(pct_step) * float(init_price))
    steps_target = max(int(desired_steps), 3 * int(max_rounds))
    max_delta = (p_high - p_low) / float(steps_target)
    if max_delta > 1e-9:
        delta = min(delta, max_delta)

    n_bins = 1 + math.ceil((p_high - p_low) / delta)
    n_bins = max(min_bins, min(max_bins, int(n_bins)))

    grid = np.linspace(p_low, p_high, n_bins)
    if money_tick > 0:
        grid = np.round(grid / money_tick) * money_tick
    grid = np.unique(grid)

    if len(grid) < 2:
        grid = np.array([p_low, p_low + max(money_tick, 1e-6)], dtype=float)

    grid[0] = max(grid[0], p_low)
    grid[-1] = min(grid[-1], p_high)
    return grid


# =========================
# 对手建模
# =========================

@dataclass
class OpponentSignals:
    """对手信号"""
    aa_strength: float
    p_positive: float


class OpponentModel:
    """对手模型"""
    def get_signals(self, seller_id: str, buyer_id: str, item: CommodityRecord,
                    dialogue: DialogueState) -> OpponentSignals:
        return OpponentSignals(aa_strength=0.0, p_positive=0.5)


# =========================
# 历史文本转换
# =========================

def history_to_zh_text(dia_history, max_turns: int = 12) -> str:
    """将对话历史转换为中文文本"""
    if dia_history is None:
        return ""
    
    if isinstance(dia_history, str):
        return dia_history.strip()
    
    if isinstance(dia_history, list):
        h = dia_history[-max_turns:]
        lines = []
        for m in h:
            if isinstance(m, dict):
                role = (m.get("role") or "").lower()
                content = (m.get("content") or "").strip()
            else:
                role = ""
                content = str(m).strip()
            
            if not content:
                continue
            
            if role in ("user", "buyer"):
                lines.append(f"买家：{content}")
            else:
                lines.append(f"卖家：{content}")
        
        return "\n".join(lines).strip()
    
    return str(dia_history).strip()


# =========================
# 卖家对话生成（严格约束版本）
# =========================

def seller_generate_dialogue(
        seller_tokenizer,
        seller_model,
        case: CommodityRecord,
        dia_history,
        strategy: str,  # "Accept" 或 "Counteroffer"
        buyer_bid: Optional[float],
        quantity: Optional[float],
        price: Optional[float],
) -> str:
    """
    生成卖家对话，严格对齐意图与价格
    
    硬约束：
    - Accept: 输出不能包含数字，必须表达成交
    - Counteroffer: 输出必须只包含一个数字，且等于price
    """
    # 无模型时的回退
    if seller_tokenizer is None or seller_model is None:
        if (strategy or "").lower().startswith("accept"):
            return "好的，成交！我马上为您安排。"
        if price is None:
            return "我可以再优惠一些，您再给个合适的单价吧。"
        return f"我可以给到单价{float(price):g}元/件，您看可以吗？"
    
    # 构建历史文本
    history_text = history_to_zh_text(dia_history, max_turns=16)
    
    # 标准化策略
    st = (strategy or "").strip()
    if st.lower() in ("accept", "成交", "同意"):
        st = "Accept"
    else:
        st = "Counteroffer"
    
    # 输出要求
    if st == "Accept":
        output_require = "回复中不得出现任何数字或价格（包括单价、数量、折扣等）。"
    else:
        output_require = f"回复中必须包含且仅包含一个数字，该数字必须等于 {float(price):g}（允许小数），不得包含其它数字。"
    buyer_bid_str = "未出价" if buyer_bid is None else f"{int(round(float(buyer_bid)))}"
    # 构建prompt（Accept时不包含数字字段，防止泄露）

    prompt = f"""你是一名专业的二手商品卖家。请仔细阅读以下信息，并生成一段连贯的卖家回复。

【商品信息】
- 名称：{case.product_name}
- 描述：{case.seller_item_description}
- 初始价格：{case.init_price}
- 底价：{case.seller_reserve_price}

【对话历史】
{history_text if history_text else "（无）"}

【谈判上下文（当前回合）】
- 买家出价：{buyer_bid_str}
- 数量：{quantity}

【策略决策（必须严格遵循）】
- 动作： {st}（取值之一：Counteroffer，Accept）
- 卖家出价：{price if strategy == 'Counteroffer' else 'null'}（策略计算得到的卖方报价；若动作为Accept则为null）

【输出要求】
- 动作一致性：回复必须体现给定的 {strategy}，不得与之矛盾。
- 数值一致性：{output_require}
- 结合上下文：回复需基于对话历史，礼貌且符合真实交易语境。
- 简洁：仅用 1～2 句简短中文表达。
- 格式：不要重复提示词、背景信息或规则说明。只输出卖家的回复内容。

【你的回复】
Seller:"""

    messages = [{"role": "system", "content": prompt}]
    
    try:
        out = llm_generate_text(
            seller_tokenizer, seller_model, messages,
            role="seller", max_new_tokens=96, temperature=0.0, do_sample=False
        )
    except Exception:
        out = _fallback_llm_generate_text(
            seller_tokenizer, seller_model, messages,
            role="seller", max_new_tokens=96, temperature=0.0, do_sample=False
        )
    
    # 规范化输出
    out = re.sub(r"\s+", " ", out).strip()
    out = re.sub(r"^\s*(?:Seller|卖家)\s*[:：]\s*", "", out, flags=re.IGNORECASE).strip()
    
    # ========== 验证硬约束 ==========
    if st == "Accept":
        # Accept约束：不能有数字
        if re.search(r"\d", out):
            return "好的，成交！我马上为您安排。"
        # 必须包含成交关键词
        if not any(k in out for k in ["成交", "同意", "可以", "没问题", "达成", "确定", "就按", "安排"]):
            return "好的，成交！我马上为您安排。"
        return out
    
    # Counteroffer约束
    if price is None:
        return "我可以再优惠一些，您再给个合适的单价吧。"
    
    # 必须只有一个数字
    nums = re.findall(r"\d+(?:\.\d+)?", out)
    if len(nums) != 1:
        return f"我可以给到单价{float(price):g}元/件，您看可以吗？"
    
    try:
        v = float(nums[0])
    except Exception:
        return f"我可以给到单价{float(price):g}元/件，您看可以吗？"
    
    # 数字必须等于目标价格
    if abs(v - float(price)) > 1e-3:
        return f"我可以给到单价{float(price):g}元/件，您看可以吗？"
    
    # 必须包含报价关键词
    if not any(k in out for k in ["可以给", "给到", "单价", "报价", "出到", "最多", "我这边", "我能"]):
        return f"我可以给到单价{float(price):g}元/件，您看可以吗？"
    
    return out


def seller_generate_welcome_message(
        seller_tokenizer,
        seller_model,
        item: CommodityRecord,
) -> str:
    """生成卖家欢迎消息（不包含价格，避免被误认为卖家出价）"""
    prompt = f"""你是一位经验丰富的二手商品卖家。有买家来咨询你的商品，请生成欢迎消息。

【商品信息】
- 商品名称：{item.product_name}
- 商品描述：{item.seller_item_description}

【回复要求】
1. 热情地欢迎买家
2. 简要介绍商品特点
3. 询问买家的期望价格和数量
4. 简洁友好，2-3句话
5. 不要在欢迎消息中报出具体价格

请直接生成你的回复："""

    messages = [{"role": "system", "content": prompt}]
    response = llm_generate_text(seller_tokenizer, seller_model, messages, role='seller', max_new_tokens=128)
    return response.strip()


# =========================
# 意图分类
# =========================

# 意图标签映射
LABEL2ID = {'讨价还价': 0, '谈判失败': 1, '谈判成功': 3}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


def infer_buyer_intent_keywords(buyer_text: str, extracted_price: Optional[float], last_seller_offer: Optional[float]) -> str:
    """
    轻量级意图推断（基于关键词）
    用于快速判断，不依赖模型
    """
    t = (buyer_text or "").strip().lower()
    
    # 离开关键词
    leave_kw = ["不买", "不需要", "算了", "退出", "结束", "取消", "放弃", "再见", "bye", "quit", "leave", "stop"]
    if any(k in t for k in leave_kw):
        return "LEAVE"
    
    # 接受关键词
    accept_kw = ["成交", "可以", "同意", "接受", "就这个", "没问题", "行", "ok", "deal", "accept", "agree"]
    if any(k in t for k in accept_kw):
        # 如果买家给出了具体价格，检查是否接近卖家出价
        if last_seller_offer is not None and extracted_price is not None:
            if abs(float(extracted_price) - float(last_seller_offer)) <= max(1e-6, 0.01 * float(last_seller_offer)):
                return "ACCEPT"
            # 买家说"可以"但给了不同价格 -> 继续谈判
            return "NEGOTIATE"
        # "可以"没有数字通常是接受上一次卖家出价
        return "ACCEPT"
    
    return "NEGOTIATE"


def classify_buyer_intent_llm(
        tokenizer, model,
        buyer_message: str,
        dialogue: DialogueState,
        item: CommodityRecord
) -> str:
    """使用LLM分类买家意图"""
    prompt = f"""【意图识别】
请将【买家消息】严格归类为以下三类之一：
1）讨价还价：继续议价、提出/修改价格或条件。
2）谈判成功：明确表达同意成交/接受卖家条件。
3）谈判失败：明确表示不买/放弃/结束谈判。

【输出要求】
仅输出一个类别名称：讨价还价 或 谈判成功 或 谈判失败。不要输出其它文字。

【商品信息】
名称：{item.product_name}
初始价：{item.init_price}

【对话历史】
{dialogue.to_text(max_turns=8) if dialogue.turns else "（空）"}

【买家消息】
{buyer_message}
"""
    
    out = llm_generate_text(tokenizer, model, prompt, role="intent", max_new_tokens=16).strip()
    
    if "谈判成功" in out:
        return "ACCEPT"
    if "谈判失败" in out:
        return "LEAVE"
    return "NEGOTIATE"


def predict_buyer_intent_model(intent_tokenizer, intent_model, human_response_text: str) -> str:
    """
    使用独立分类模型预测买家意图
    
    返回：
    - BARGAIN (讨价还价)
    - FAIL (谈判失败)
    - SUCCESS (谈判成功)
    """
    if intent_tokenizer is None or intent_model is None:
        return "BARGAIN"  # 默认继续谈判
    
    if not human_response_text:
        return "BARGAIN"
    
    inputs = intent_tokenizer(human_response_text, truncation=True, return_tensors="pt")
    
    # 设备放置
    try:
        device = next(intent_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
    except Exception:
        pass
    
    with torch.no_grad():
        out = intent_model(**inputs)
        logits = getattr(out, "logits", None)
        if logits is None:
            return "BARGAIN"
        pred = int(torch.argmax(logits, dim=-1).item())
    
    if pred == LABEL2ID['讨价还价']:
        return "BARGAIN"
    if pred == LABEL2ID['谈判失败']:
        return "FAIL"
    if pred == LABEL2ID['谈判成功']:
        return "SUCCESS"
    
    return "BARGAIN"


# =========================
# 价格提取
# =========================

def extract_price_quantity_llm(
        tokenizer, model,
        tran_resp: str,
        case: dict
) -> Tuple[Optional[float], Optional[float]]:
    """使用LLM提取价格和数量"""
    prompt = f"""是谈判对话识别助手，任务是分析对话，识别当前轮次买家提及或接受的单价和商品数量，并且严格按照指定格式输出。

【识别规则】
1. 单价识别规则：
- 买家直接提及单价（"X元/台"、"单价X"）
- 买家接受卖家价格（"X元可以"、"同意这个价"）
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
    
    # 设备放置
    try:
        device = next(model.parameters()).device
    except Exception:
        device = getattr(model, "device", None)
    
    inputs = tokenizer(prompt, return_tensors="pt")
    if device is not None:
        try:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        except Exception:
            pass
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=32,
            temperature=0.0,
            do_sample=False,
            early_stopping=True,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
        )
    
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    
    def _to_float_or_none(s: str):
        s = s.strip().lower().replace(",", "")
        if s == "null":
            return None
        try:
            return float(s)
        except Exception:
            return None
    
    # 提取价格
    price_match = re.search(r"当前单价[：:]\s*([\d,]+\.?\d*|null)", response, re.IGNORECASE)
    if not price_match:
        price_match = re.search(r"Current\s*Unit\s*Price[：:]\s*([\d,]+\.?\d*|null)", response, re.IGNORECASE)
    price = _to_float_or_none(price_match.group(1)) if price_match else None
    
    # 提取数量
    qty_match = re.search(r"商品数量[：:]\s*([\d,]+\.?\d*|null)", response, re.IGNORECASE)
    if not qty_match:
        qty_match = re.search(r"Product\s*Quantity[：:]\s*([\d,]+\.?\d*|null)", response, re.IGNORECASE)
    quantity = _to_float_or_none(qty_match.group(1)) if qty_match else None
    
    return price, quantity


# =========================
# 卖家配置
# =========================

@dataclass
class SellerConfig:
    """卖家配置"""
    max_rounds: int = 8
    alpha: float = 0.15
    epsilon: float = 0.10
    n_action_levels: int = 21
    u_bins: int = 21
    k_cost: float = 0.10
    cost_baseline: float = 1.0
    omega1: float = 0.5
    omega2: float = 0.5
    headroom_pct: float = 0.00
    currency_step: float = 0.50
    pct_step: float = 0.01
    desired_steps: int = 24
    min_bins: int = 21
    max_bins: int = 151
    money_tick: float = 0.01
    enforce_buyer_reserve: bool = True
    fail_penalty: float = 0.0
    accept_bonus: float = 0.0


# =========================
# 卖家智能体
# =========================

class PaperAlignedSellerOptionA:
    """卖家智能体：基于强化学习的谈判策略"""

    def __init__(self, cfg: SellerConfig, tier_encoder: PriceTierEncoder, opponent_model: OpponentModel):
        self.cfg = cfg
        self.tier_encoder = tier_encoder
        self.opponent_model = opponent_model
        self.Q: Dict[Tuple[int, int, int], np.ndarray] = {}
        self.grid_cache: Dict[str, np.ndarray] = {}
        self._prev_buyer_offer: Optional[float] = None
        self._prev_buyer_u: Optional[float] = None

    def save(self, save_path: str) -> None:
        """保存Q表和配置"""
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        payload = {
            "format": "PaperAlignedSellerOptionA-v1",
            "cfg": asdict(self.cfg),
            "tier_edges": np.asarray(self.tier_encoder.edges, dtype=float),
            "Q": self.Q,
        }
        torch.save(payload, save_path)

    @classmethod
    def load(cls, save_path: str, opponent_model: Optional["OpponentModel"] = None) -> "PaperAlignedSellerOptionA":
        """加载卖家模型"""
        try:
            payload = torch.load(save_path, map_location="cuda", weights_only=False)
        except TypeError:
            payload = torch.load(save_path, map_location="cuda")

        cfg = SellerConfig(**payload["cfg"])
        tier_encoder = PriceTierEncoder(edges=np.asarray(payload["tier_edges"], dtype=float))
        seller = cls(cfg=cfg, tier_encoder=tier_encoder, opponent_model=opponent_model or OpponentModel())

        Q_loaded: Dict[Tuple[int, int, int], np.ndarray] = {}
        for k, v in payload["Q"].items():
            key = tuple(k) if isinstance(k, (list, tuple)) else k
            Q_loaded[key] = np.asarray(v, dtype=float)

        seller.Q = Q_loaded
        seller.grid_cache = {}
        return seller

    def _grid_for_item(self, item: CommodityRecord) -> np.ndarray:
        if item.product_id not in self.grid_cache:
            self.grid_cache[item.product_id] = compute_product_grid_seller_only(
                init_price=item.init_price,
                seller_reserve_price=item.seller_reserve_price,
                max_rounds=self.cfg.max_rounds,
                headroom_pct=self.cfg.headroom_pct,
                currency_step=self.cfg.currency_step,
                pct_step=self.cfg.pct_step,
                desired_steps=self.cfg.desired_steps,
                min_bins=self.cfg.min_bins,
                max_bins=self.cfg.max_bins,
                money_tick=self.cfg.money_tick,
            )
        return self.grid_cache[item.product_id]

    @staticmethod
    def _snap_to_grid(grid: np.ndarray, p: float) -> float:
        idx = int(np.argmin(np.abs(grid - float(p))))
        return float(grid[int(np.clip(idx, 0, len(grid) - 1))])

    @staticmethod
    def _utility_norm(price: float, seller_reserve: float, max_ask: float) -> float:
        denom = max(float(max_ask) - float(seller_reserve), 1e-9)
        u = (float(price) - float(seller_reserve)) / denom
        return float(np.clip(u, 0.0, 1.0))

    def _utility_to_bin(self, u: float) -> int:
        ub = int(round(float(u) * (self.cfg.u_bins - 1)))
        return int(np.clip(ub, 0, self.cfg.u_bins - 1))

    def _Qrow(self, state: Tuple[int, int, int]) -> np.ndarray:
        if state not in self.Q:
            self.Q[state] = np.zeros(self.cfg.n_action_levels, dtype=float)
        if len(self.Q[state]) != self.cfg.n_action_levels:
            self.Q[state] = np.zeros(self.cfg.n_action_levels, dtype=float)
        return self.Q[state]

    def _select_action(self, qrow: np.ndarray) -> int:
        if random.random() < self.cfg.epsilon:
            return int(np.random.randint(0, len(qrow)))
        return int(np.argmax(qrow))

    def _c_of_a(self, aa_strength: float) -> float:
        return float(self.cfg.cost_baseline + 0.5 * self.cfg.k_cost * (float(aa_strength) ** 2))

    @staticmethod
    def _gamma_star(p_positive: float) -> float:
        return float(np.clip(p_positive, 0.0, 1.0))

    def _update_lambda_components(self, item: CommodityRecord, max_ask: float, buyer_offer: float) -> Tuple[float, float]:
        curr_u = self._utility_norm(buyer_offer, item.seller_reserve_price, max_ask)

        if self._prev_buyer_u is None:
            lambda_e = 1.0
        else:
            shock = abs(curr_u - self._prev_buyer_u)
            lambda_e = float(np.clip(1.0 - shock, 0.0, 1.0))

        if self._prev_buyer_offer is None:
            lambda_theta = 1.0
        else:
            delta = float(buyer_offer - self._prev_buyer_offer)
            scale = max(float(item.init_price), 1e-6)
            lambda_theta = float(np.clip(1.0 + math.tanh(delta / scale), 0.5, 1.5))

        self._prev_buyer_offer = float(buyer_offer)
        self._prev_buyer_u = float(curr_u)
        return lambda_e, lambda_theta

    def _lambda_combined(self, lambda_e: float, lambda_theta: float) -> float:
        return float(self.cfg.omega1 * lambda_e + self.cfg.omega2 * lambda_theta)

    def _offer_from_eq13_shape(
            self,
            seller_reserve: float,
            max_ask: float,
            round_idx: int,
            gamma_star: float,
            c_a: float,
            lam: float,
            q_target: float,
    ) -> float:
        remaining = max(1, self.cfg.max_rounds - round_idx)
        if abs(gamma_star - 1.0) < 1e-9:
            disc_avg = q_target
        else:
            disc_sum = (1.0 - (gamma_star ** remaining)) / (1.0 - gamma_star)
            disc_avg = (disc_sum / remaining) * q_target

        frac = float(np.clip(c_a * lam * disc_avg, 0.0, 1.0))
        return float(seller_reserve) + frac * (float(max_ask) - float(seller_reserve))

    def reset_episode_memory(self):
        """重置回合记忆"""
        self._prev_buyer_offer = None
        self._prev_buyer_u = None

    def compute_seller_offer(
            self,
            item: CommodityRecord,
            buyer_offer: float,
            round_idx: int,
    ) -> Tuple[float, bool]:
        """
        计算卖家出价和是否接受
        
        Returns:
            (卖家出价, 是否接受买家出价)
        """
        grid = self._grid_for_item(item)
        seller_reserve = float(item.seller_reserve_price)
        max_ask = float(grid[-1])
        
        tier = self.tier_encoder.transform(item.init_price)
        
        u = self._utility_norm(buyer_offer, seller_reserve, max_ask)
        u_bin = self._utility_to_bin(u)
        
        state = (tier, round_idx, u_bin)
        qrow = self._Qrow(state)
        action = self._select_action(qrow)
        
        q_target = float(action) / float(max(1, self.cfg.n_action_levels - 1))
        
        signals = self.opponent_model.get_signals("seller", "buyer", item, DialogueState())
        gamma_star = self._gamma_star(signals.p_positive)
        c_a = self._c_of_a(signals.aa_strength)
        
        lambda_e, lambda_theta = self._update_lambda_components(item, max_ask, buyer_offer)
        lam = self._lambda_combined(lambda_e, lambda_theta)
        
        planned_offer_raw = self._offer_from_eq13_shape(
            seller_reserve=seller_reserve,
            max_ask=max_ask,
            round_idx=round_idx,
            gamma_star=gamma_star,
            c_a=c_a,
            lam=lam,
            q_target=q_target,
        )
        planned_offer = float(max(seller_reserve, self._snap_to_grid(grid, planned_offer_raw)))
        
        if float(buyer_offer) >= float(planned_offer):
            return buyer_offer, True
        
        return planned_offer, False
