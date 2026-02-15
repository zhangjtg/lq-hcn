"""run_mipn_negotiation.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional


_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if _CUR_DIR not in sys.path:
    sys.path.insert(0, _CUR_DIR)

import torch

from src.calculator_negotiation_metrics import calculate_and_print_metrics
from src.intent_recognizer import IntentRecognizer
from src.price_quantity_extractor import PriceQuantityExtractor
from mipn import ProductConfig, MiPNSellerAgent



MAX_TURNS: int = int(os.environ.get("NEGOTIATION_MAX_TURNS", "8"))
MAX_WECHAT_SESSIONS: int = int(os.environ.get("MAX_WECHAT_SESSIONS", "2000"))

# MiPN ckpt（卖家策略网络）
MIPN_AGENT_CKPT: str = os.environ.get("MIPN_AGENT_CKPT", "trained_mipn_seller_agent.pth")

# 意图识别模型（TinyLlama + LoRA）
INTENT_BASE_MODEL: str = os.environ.get("INTENT_BASE_MODEL", "/home/zjt/local/model_bin/TinyLlama_v1.1_chinese")
INTENT_ADAPTER: str = os.environ.get("INTENT_ADAPTER", "../output/intent/TinyLlama_v1.1_chinese-classification83")

# 价格数量抽取模型（Qwen2.5-1.5B + LoRA）
PRICE_TOKENIZER_PATH: str = os.environ.get("PRICE_TOKENIZER_PATH", "../output/price_model/Qwen2p5-1p5B-instruct-a6-l4-618")
PRICE_BASE_MODEL: str = os.environ.get("PRICE_BASE_MODEL", "/home/zjt/local/model_bin/Qwen2.5-1.5B-Instruct")
PRICE_ADAPTER: str = os.environ.get("PRICE_ADAPTER", "../output/price_model/Qwen2p5-1p5B-instruct-a5-l4-618")

# 卖家话术模型（Qwen3-4B）
SELLER_LLM_PATH: str = os.environ.get("SELLER_LLM_PATH", "/home/zjt/local/model_bin/Qwen3-4B-Instruct-2507")

# 指标输出
METRICS_DIR: str = os.environ.get("NEGOTIATION_METRICS_DIR", "results")
METRICS_FILENAME: str = os.environ.get(
    "NEGOTIATION_METRICS_FILE",
    f"wechat_negotiation_metrics_{int(time.time())}.txt",
)
RESULTS_JSONL: str = os.environ.get(
    "NEGOTIATION_METRICS_JSONL",
    os.path.join(METRICS_DIR, f"wechat_session_results_{int(time.time())}.jsonl"),
)


# ============================================================
# 1) 工具：格式化历史（给抽取模型/卖家话术模型）
# ============================================================

def format_dialogue_history(history: List[Dict[str, str]], keep_last_n: int = 12) -> str:
    """将 history 转成中文对话串。

    history item:
      - {"role": "user", "content": "..."}
      - {"role": "assistant", "content": "..."}
    """
    h = history[-keep_last_n:]
    lines: List[str] = []
    for m in h:
        who = "买家" if m.get("role") == "user" else "卖家"
        lines.append(f"{who}：{m.get('content','')}".strip())
    return "\n".join([x for x in lines if x])


def _append_jsonl(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ============================================================
# 2) 卖家话术生成（Qwen3-4B） + 一致性校验 + 兜底
# ============================================================


class SellerDialogueGenerator:
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 延迟导入 transformers（避免仅做策略推理时被强依赖）
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(SELLER_LLM_PATH, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            SELLER_LLM_PATH,
            device_map="auto" if self.device.startswith("cuda") else None,
            torch_dtype=torch.bfloat16 if self.device.startswith("cuda") else None,
            trust_remote_code=True,
        )
        self.model.eval()

    @staticmethod
    def _extract_prices_with_unit(text: str) -> List[float]:
        # 只提取“带货币单位”的数字，避免把数量/时间当成价格
        hits = re.findall(r"(\d+(?:\.\d+)?)\s*(?:元|￥|rmb)", text, flags=re.I)
        out: List[float] = []
        for h in hits:
            try:
                out.append(float(h))
            except Exception:
                pass
        return out

    @staticmethod
    def _fallback(strategy: str, *, deal_price: Optional[float], counter_price: Optional[float], quantity: Optional[float]) -> str:
        # 模板兜底：保证“策略+价格”绝对对齐
        if strategy == "Accept":
            if deal_price is not None:
                qtxt = f"，数量{quantity:g}" if quantity is not None else ""
                return f"好的，可以成交，就按{deal_price:g}元/件{qtxt}。"
            return "好的，可以成交。"

        p = counter_price if counter_price is not None else 0.0
        qtxt = f"，数量{quantity:g}" if quantity is not None else ""
        return f"这个价格我这边接受不了，我可以给到{p:g}元/件{qtxt}，您看可以吗？"

    @classmethod
    def _validate_alignment(
        cls,
        text: str,
        *,
        strategy: str,
        deal_price: Optional[float],
        counter_price: Optional[float],
        tol: float = 1e-3,
    ) -> bool:
        t = (text or "").strip()
        if not t:
            return False

        prices = cls._extract_prices_with_unit(t)

        if strategy == "Accept":
            # Accept：允许不报价格；若报价格，只能报成交价
            if not prices:
                return True
            if deal_price is None:
                return False
            if len(prices) != 1:
                return False
            return abs(prices[0] - float(deal_price)) <= tol

        # Counteroffer：必须且只能出现一个价格，且必须等于 counter_price
        if counter_price is None:
            return False
        if len(prices) != 1:
            return False
        return abs(prices[0] - float(counter_price)) <= tol

    @torch.no_grad()
    def generate(
        self,
        *,
        case: dict,
        dia_history: str,
        buyer_bid: Optional[float],
        quantity: Optional[float],
        strategy: str,  # "Counteroffer" or "Accept"
        price: Optional[float],  # Counteroffer 时为卖家出价；Accept 时为 None
        deal_price: Optional[float],  # Accept 时的成交价（用于一致性校验与可选复述）
    ) -> str:
        if strategy == "Accept":
            output_require = "回复中不得出现任何数字或价格（包括单价、数量、折扣等）。"
        else:
            output_require = f"回复中必须包含且仅包含一个数字，该数字必须等于 {float(price):g}（允许小数），不得包含其它数字。"
        buyer_bid_str = "未出价" if buyer_bid is None else f"{int(round(float(buyer_bid)))}"
        prompt = f"""你是一名专业的二手商品卖家。请仔细阅读以下信息，并生成一段连贯的卖家回复，用来实现本回合的策略决策。回复必须遵守输出要求，并且与谈判策略模型给出的动作与出价严格一致。

【商品信息】
- 名称：{case.get('product_name')}
- 描述：{case.get('seller_item_description')}
- 初始价格：{case.get('init_price')}
- 底价：{case.get('seller_reserve_price')}

【对话历史】
{dia_history}

【谈判上下文（当前回合）】
- 买家出价：{buyer_bid_str}
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

        messages = [{"role": "system", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt

        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        out_ids = self.model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0,
            do_sample=False,
            eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
        )
        gen = self.tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        gen = re.sub(r"^\s*Seller\s*[:：]\s*", "", gen).strip()

        if self._validate_alignment(gen, strategy=strategy, deal_price=deal_price, counter_price=price):
            return gen

        return self._fallback(strategy=strategy, deal_price=deal_price, counter_price=price, quantity=quantity)


# ============================================================
# 3) 单会话执行器：微信人机谈判（真人买家）
# ============================================================


@dataclass
class SessionResult:
    success: bool
    final_price: Optional[float]
    turns: int
    seller_bottom_price: float
    buyer_max_price: float
    quantity: Optional[float]


class WeChatNegotiationRunner:
    def __init__(
        self,
        *,
        intent_recognizer: IntentRecognizer,
        extractor: PriceQuantityExtractor,
        seller_llm: SellerDialogueGenerator,
        mipn_ckpt_path: str,
        device: str,
    ):
        self.intent_recognizer = intent_recognizer
        self.extractor = extractor
        self.seller_llm = seller_llm
        self.mipn_ckpt_path = mipn_ckpt_path
        self.device = device

        # 预加载 MiPN 权重（只加载一次），避免每个会话都读硬盘
        self._mipn_state_dict: Optional[dict] = None
        if os.path.exists(self.mipn_ckpt_path):
            try:
                self._mipn_state_dict = torch.load(self.mipn_ckpt_path, map_location="cpu")
                print(f"[run_LQ_negotiation] Loaded MiPN ckpt: {self.mipn_ckpt_path}")
            except Exception as e:
                print(f"[run_LQ_negotiation WARNING] Failed to load MiPN ckpt: {e} -> 将使用随机初始化")
                self._mipn_state_dict = None
        else:
            print(f"[run_LQ_negotiation WARNING] MiPN ckpt not found: {self.mipn_ckpt_path} -> 将使用随机初始化")

    def run_one_session(self, wechat_service, *, user_id: str, product: dict) -> SessionResult:
        # 绑定商品给价格抽取器
        self.extractor.set_case(product)

        cfg = ProductConfig(data_dict=product)
        seller_agent = MiPNSellerAgent(
            config=cfg,
            device=self.device,
            ckpt_path=None,  # 用 state_dict 更快
            state_dict=self._mipn_state_dict,
        )

        seller_bottom_price = float(product.get("seller_reserve_price", 0.0))
        buyer_max_price = float(product.get("buyer_reserve_price", float(product.get("init_price", 0.0))))

        history: List[Dict[str, str]] = []
        buyer_turns = 0
        seller_turns = 0
        turns = 0

        last_quantity: Optional[float] = None
        last_seller_price: Optional[float] = float(product.get("init_price", 0.0))

        # Welcome（不算回合）
        welcome = (
            f"您好，我是卖家。\n"
            f"商品：{product.get('product_name')}\n"
            f"描述：{product.get('seller_item_description','')}\n"
            f"标价：{float(product.get('init_price',0)):.2f}元/件。\n"
            f"请告诉我数量，并给出您的出价（例如：单价800元，数量10）。"
        )
        wechat_service.send_message(user_id, {"action": "continue", "response": welcome})
        history.append({"role": "assistant", "content": welcome})

        initialized = False

        while buyer_turns < MAX_TURNS:
            # 1) 等待买家消息
            buyer_text = wechat_service.wait_for_user_message(user_id, timeout=1200)
            if buyer_text is None:
                end_msg = "等待您的回复超时，本次谈判结束。"
                wechat_service.send_message(user_id, {"action": "end", "response": end_msg, "deal_price": None})
                return SessionResult(False, None, turns, seller_bottom_price, buyer_max_price, last_quantity)

            buyer_text = str(buyer_text).strip()
            if not buyer_text:
                wechat_service.send_message(user_id, {"action": "continue", "response": "我没收到有效内容，请再说一次。"})
                continue

            buyer_turns += 1
            history.append({"role": "user", "content": buyer_text})

            # 2) 意图识别
            intent_out = self.intent_recognizer.predict(buyer_text)
            intent_label = intent_out.label

            # 3) 终止意图
            if intent_label == "谈判失败":
                end_msg = "好的，理解您的意思。本次谈判结束，期待下次合作。"
                wechat_service.send_message(user_id, {"action": "end", "response": end_msg, "deal_price": None})
                return SessionResult(False, None, turns, seller_bottom_price, buyer_max_price, last_quantity)

            dia_history_str = format_dialogue_history(history, keep_last_n=12)

            # 4) 价格/数量抽取（无论成功/失败都可以抽一下，用于成交价兜底）
            price_raw, qty_raw = self.extractor.extra_price_quantity(dia_history_str)
            buyer_price: Optional[float] = None if price_raw == "null" else float(price_raw)

            if qty_raw == "null":
                quantity = last_quantity if last_quantity is not None else 1.0
            else:
                quantity = float(qty_raw)
                last_quantity = quantity

            # 5) 买家“谈判成功”：视为买家接受卖家上一口价
            if intent_label == "谈判成功":
                deal_price = last_seller_price
                if deal_price is None:
                    deal_price = buyer_price
                if deal_price is None:
                    deal_price = float(product.get("init_price", 0.0))

                seller_reply = self.seller_llm.generate(
                    case=product,
                    dia_history=dia_history_str,
                    buyer_bid=deal_price,
                    quantity=quantity,
                    strategy="Accept",
                    price=None,
                    deal_price=deal_price,
                )
                history.append({"role": "assistant", "content": seller_reply})
                wechat_service.send_message(
                    user_id,
                    {"action": "end", "response": seller_reply, "deal_price": deal_price, "quantity": quantity},
                )
                turns += 1
                return SessionResult(True, deal_price, turns, seller_bottom_price, buyer_max_price, quantity)

            # 6) 讨价还价：必须有价格
            if buyer_price is None:
                if buyer_turns >= MAX_TURNS:
                    end_msg = "已到最大谈判轮次（8轮），但仍未识别到有效出价，本次结束。"
                    wechat_service.send_message(user_id, {"action": "end", "response": end_msg, "deal_price": None})
                    return SessionResult(False, None, turns, seller_bottom_price, buyer_max_price, quantity)

                wechat_service.send_message(
                    user_id,
                    {"action": "continue", "response": "我没识别到您的出价，请按“单价X元，数量Y”再说一次。"},
                )
                continue

            # 7) 更新 MiPN 会话状态
            if not initialized:
                seller_agent.reset(opening_buyer_price=buyer_price)
                initialized = True
            else:
                seller_agent.update_buyer(buyer_price)

            # 8) 卖家决策（若卖家已用完8次，则直接结束）
            if seller_turns >= MAX_TURNS:
                end_msg = "已到最大谈判轮次（卖家已给出8次报价），本次未能达成一致。"
                wechat_service.send_message(user_id, {"action": "end", "response": end_msg, "deal_price": None})
                return SessionResult(False, None, turns, seller_bottom_price, buyer_max_price, quantity)

            decision = seller_agent.decide(seller_turn_count=seller_turns, max_turns=MAX_TURNS, deterministic=True)

            # 9) 卖家输出
            if decision.strategy == "Accept":
                deal_price = buyer_price
                seller_reply = self.seller_llm.generate(
                    case=product,
                    dia_history=dia_history_str,
                    buyer_bid=buyer_price,
                    quantity=quantity,
                    strategy="Accept",
                    price=None,
                    deal_price=deal_price,
                )
                history.append({"role": "assistant", "content": seller_reply})
                wechat_service.send_message(
                    user_id,
                    {"action": "end", "response": seller_reply, "deal_price": deal_price, "quantity": quantity},
                )
                seller_turns += 1
                turns += 1
                return SessionResult(True, deal_price, turns, seller_bottom_price, buyer_max_price, quantity)

            # Counteroffer
            proposed_price = float(decision.price) if decision.price is not None else float(product.get("init_price", 0.0))
            proposed_price = round(proposed_price, 1)
            last_seller_price = proposed_price
            seller_agent.update_seller(proposed_price)

            seller_reply = self.seller_llm.generate(
                case=product,
                dia_history=dia_history_str,
                buyer_bid=buyer_price,
                quantity=quantity,
                strategy="Counteroffer",
                price=proposed_price,
                deal_price=None,
            )
            history.append({"role": "assistant", "content": seller_reply})
            wechat_service.send_message(
                user_id,
                {"action": "continue", "response": seller_reply, "seller_price": proposed_price, "quantity": quantity},
            )

            seller_turns += 1
            turns += 1

            # 如果卖家已给出第8次报价且仍未成交，按“最多8次”规则结束
            if seller_turns >= MAX_TURNS:
                end_msg = "已到最大谈判轮次（8轮），本次未能达成一致。"
                wechat_service.send_message(user_id, {"action": "end", "response": end_msg, "deal_price": None})
                return SessionResult(False, None, turns, seller_bottom_price, buyer_max_price, quantity)

        # 理论兜底
        wechat_service.send_message(user_id, {"action": "end", "response": "谈判结束。", "deal_price": None})
        return SessionResult(False, None, turns, seller_bottom_price, buyer_max_price, last_quantity)


# ============================================================
# 4) app.py 调用入口：run_training
# ============================================================


def run_training(wechat_service, mode: str = "train", output_csv: str = "custom_results.csv"):
    """后台线程入口（app.py 会启动一个 daemon Thread 调用它）。

    参数 mode/output_csv 仅为兼容 app.py，不影响本文件人机谈判流程。
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    intent_recognizer = IntentRecognizer(
        base_model_path=INTENT_BASE_MODEL,
        adapter_path=INTENT_ADAPTER,
        device=device,
    )

    extractor = PriceQuantityExtractor(
        base_model_path=PRICE_BASE_MODEL,
        tokenizer_path=PRICE_TOKENIZER_PATH,
        adapter_path=PRICE_ADAPTER,
        device=device,
        device_map="auto" if device.startswith("cuda") else "cpu",
        torch_dtype="bfloat16",
    )

    seller_llm = SellerDialogueGenerator(device=device)

    runner = WeChatNegotiationRunner(
        intent_recognizer=intent_recognizer,
        extractor=extractor,
        seller_llm=seller_llm,
        mipn_ckpt_path=MIPN_AGENT_CKPT,
        device=device,
    )

    all_results: List[dict] = []
    completed_sessions = 0

    print("[run_LQ_negotiation] Ready. Waiting for new sessions...")

    while True:
        user_id, product_id = wechat_service.wait_for_new_session(timeout=None)
        if not user_id:
            time.sleep(0.05)
            continue

        # 达到2000次后，不再接收新会话（仍给前端回一条结束信息避免卡住）
        if completed_sessions >= MAX_WECHAT_SESSIONS:
            msg = f"系统已完成 {MAX_WECHAT_SESSIONS} 次完整人机谈判采集，当前不再接受新会话。"
            wechat_service.send_message(user_id, {"action": "end", "response": msg, "deal_price": None})
            wechat_service.episode_processing_lock.clear()
            continue

        wechat_service.episode_processing_lock.set()
        try:
            product = wechat_service.get_product_by_id(product_id)
            if not product:
                wechat_service.send_message(user_id, {"action": "end", "response": "商品未找到，无法开始谈判。", "deal_price": None})
                continue

            session = runner.run_one_session(wechat_service, user_id=user_id, product=product)
            completed_sessions += 1

            res = {
                "success": bool(session.success),
                "final_price": float(session.final_price) if session.final_price is not None else 0.0,
                "seller_bottom_price": float(session.seller_bottom_price),
                "buyer_max_price": float(session.buyer_max_price),
                "turns": int(session.turns),
                "quantity": float(session.quantity) if session.quantity is not None else 1.0,
                "user_id": str(user_id),
                "product_id": int(product_id) if product_id is not None else -1,
                "timestamp": int(time.time()),
            }
            all_results.append(res)
            _append_jsonl(RESULTS_JSONL, res)

            print(
                f"[run_LQ_negotiation] Session completed: {completed_sessions}/{MAX_WECHAT_SESSIONS}  "
                f"success={res['success']} final_price={res['final_price']} turns={res['turns']}"
            )

            if completed_sessions == MAX_WECHAT_SESSIONS:
                os.makedirs(METRICS_DIR, exist_ok=True)
                calculate_and_print_metrics(all_results, output_dir=METRICS_DIR, filename=METRICS_FILENAME)
                print(f"[run_LQ_negotiation] Metrics written: {os.path.join(METRICS_DIR, METRICS_FILENAME)}")

        except Exception as e:
            wechat_service.send_message(user_id, {"action": "end", "response": f"后台发生错误，谈判结束：{e}", "deal_price": None})
            print(f"[run_LQ_negotiation ERROR] session failed: {e}")
        finally:
            wechat_service.episode_processing_lock.clear()
