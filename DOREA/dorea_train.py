# -*- coding: utf-8 -*-
"""
dorea_train.py
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from dorea_main import DOREAFramework
from dorea_negotiation import (
    HumanOpponentNegotiationEnvironment,
    create_price_quantity_domain_from_product,
    nearest_outcome_for_price_quantity,
    offer_to_numbers,
)

from intent_recognizer import IntentRecognizer
from price_quantity_extractor import PriceQuantityExtractor


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def append_pickle(path: str, items: List[Dict[str, Any]]) -> None:
    """Append list items into a pickle list file."""
    ensure_dir(path)
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
    else:
        data = []
    data.extend(items)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    ensure_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def build_transcript(history: List[Dict[str, str]]) -> str:
    lines = []
    for h in history:
        role = h.get("role", "")
        content = h.get("content", "")
        if role == "buyer":
            lines.append(f"买家：{content}")
        else:
            lines.append(f"卖家：{content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Baseline behavior policy π_beta for collecting D_off
# ---------------------------------------------------------------------

def time_dependent_baseline(state: np.ndarray, *, reservation_value: float = 0.0) -> np.ndarray:
    """
    A simple time-dependent concession policy in utility space, used as behavior policy
    for offline dataset collection (D_off).
    """
    t_norm = float(state[0])
    # start at 0.98 and linearly concede to max(reservation, 0.55)
    floor = max(float(reservation_value), 0.55)
    target = 0.98 - (0.98 - floor) * t_norm
    # small exploration noise
    target += np.random.normal(0.0, 0.01)
    target = float(np.clip(target, floor, 1.0))
    return np.array([target], dtype=np.float32)


# ---------------------------------------------------------------------
# Seller utterance generator (Qwen3-4B-Instruct-2507)
# ---------------------------------------------------------------------

@dataclass
class SellerLLMConfig:
    model_path: str = "Qwen/Qwen3-4B-Instruct-2507"
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"
    max_new_tokens: int = 96
    temperature: float = 0.2


class QwenSellerGenerator:
    """
    Generate seller reply in Chinese with strict action/price consistency.

    If the model cannot be loaded, it will fall back to a deterministic template.
    """

    def __init__(self, cfg: SellerLLMConfig):
        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        self._try_load()

    def _try_load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_path, trust_remote_code=True)
            dtype = getattr(torch, self.cfg.torch_dtype, torch.bfloat16)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.cfg.model_path,
                device_map=self.cfg.device_map,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            self.model.eval()
        except Exception as e:
            print(f"[QwenSellerGenerator] WARNING: failed to load model: {e}")
            self.model = None
            self.tokenizer = None

    def _post_enforce_numbers(self, text: str, *, price: float, quantity: int) -> str:
        # Ensure price/quantity appear; otherwise append a strict clause.
        t = str(text).strip().replace("Seller:", "").strip()
        has_price = re.search(rf"{int(round(price))}\s*元", t) is not None
        has_qty = re.search(rf"{int(quantity)}\s*(件|个|台)", t) is not None or re.search(rf"数量\s*{int(quantity)}", t) is not None
        if not (has_price and has_qty):
            t = t.rstrip("。") + f"。单价{int(round(price))}元，数量{int(quantity)}。"
        return t

    def generate(
        self,
        *,
        case: Dict[str, Any],
        dia_history: str,
        buyer_bid: Optional[float],
        quantity: int,
        strategy: str,   # Counteroffer / Accept
        price: float,    # seller offer (Counteroffer) or deal price (Accept)
    ) -> str:
        output_require = ""
        if strategy == "Accept":
            output_require = "回复中不得出现任何数字或价格（包括单价、数量、折扣等）。"
        else:
            output_require = f"回复中必须包含且仅包含一个数字，该数字必须等于 {float(price):g}（允许小数），不得包含其它数字。"

        buyer_bid_str = "未出价" if buyer_bid is None else f"{int(round(float(buyer_bid)))}"

        prompt = f"""你是一名专业的二手商品卖家。请仔细阅读以下信息，并生成一段连贯的卖家回复，用来实现本回合的策略决策。回复必须遵守输出要求，并且与谈判策略模型给出的动作与出价严格一致。

【商品信息】
- 名称：{case.get('product_name', '')}
- 描述：{case.get('seller_item_description', '')}
- 初始价格：{case.get('init_price', '')}
- 底价：{case.get('seller_reserve_price', case.get('reserve_price', case.get('seller_bottom_price', '')))}

【对话历史】
{dia_history}

【谈判上下文（当前回合）】
- 买家出价：{buyer_bid_str}
- 数量：{int(quantity)}

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

        # fallback template
        def template() -> str:
            if strategy == "Accept":
                return f"可以，就按单价{int(round(price))}元，数量{int(quantity)}成交。"
            return f"我这边最多只能做到单价{int(round(price))}元，数量{int(quantity)}，您看可以吗？"

        if self.model is None or self.tokenizer is None:
            return template()

        try:
            import torch

            messages = [{"role": "system", "content": prompt}]
            if hasattr(self.tokenizer, "apply_chat_template"):
                input_ids = self.tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
                ).to(self.model.device)
                gen = self.model.generate(
                    input_ids,
                    max_new_tokens=self.cfg.max_new_tokens,
                    temperature=self.cfg.temperature,
                    do_sample=self.cfg.temperature > 0,
                    eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
                )
                out = self.tokenizer.decode(gen[0][input_ids.shape[1]:], skip_special_tokens=True)
            else:
                # very old tokenizers
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                gen = self.model.generate(**inputs, max_new_tokens=self.cfg.max_new_tokens, temperature=self.cfg.temperature)
                out = self.tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            out = self._post_enforce_numbers(out, price=price, quantity=quantity)

            # avoid empty outputs
            if not out.strip():
                return template()
            return out.strip()
        except Exception as e:
            print(f"[QwenSellerGenerator] WARNING: generation failed: {e}")
            return template()


# ---------------------------------------------------------------------
# Intent normalization
# ---------------------------------------------------------------------

def normalize_intent(intent_label: str) -> str:
    """
    Map model labels into: 'accept' | 'fail' | 'bargain'
    """
    s = str(intent_label or "").strip()
    if any(k in s for k in ["失败", "不买", "算了", "放弃"]):
        return "fail"
    if any(k in s for k in ["成功", "接受", "同意", "成交", "可以"]):
        return "accept"
    return "bargain"


# ---------------------------------------------------------------------
# WeChat episode (human buyer)
# ---------------------------------------------------------------------

def run_one_wechat_episode(
    *,
    wechat_service,
    user_id: str,
    product: Dict[str, Any],
    intent_recognizer: IntentRecognizer,
    price_extractor: PriceQuantityExtractor,
    seller_policy_fn: Callable[[np.ndarray], np.ndarray],   # returns 1-d action (target utility)
    seller_llm: QwenSellerGenerator,
    max_seller_offers: int = 8,
    max_buyer_offers: int = 8,
    buyer_timeout: float = 5000.0,
    episode_log_path: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    One complete human-machine negotiation episode for WeChat.

    Returns:
      - transitions: list of (s, a, r, s') dicts
      - episode_record: full episode info for debugging/analysis
    """
    # Normalize product keys for prompt + domain
    case = dict(product)
    # unify reserve price key
    if "seller_reserve_price" not in case:
        if "reserve_price" in case:
            case["seller_reserve_price"] = case["reserve_price"]
        elif "seller_bottom_price" in case:
            case["seller_reserve_price"] = case["seller_bottom_price"]

    domain, seller_pref, meta = create_price_quantity_domain_from_product(case, max_qty=int(case.get("max_qty") or case.get("quantity") or 10))
    env = HumanOpponentNegotiationEnvironment(
        domain,
        seller_pref,
        max_rounds=max_seller_offers,
        reservation_value=float(meta.get("reservation_utility", 0.0)),
        meta=meta,
    )

    history: List[Dict[str, str]] = []
    transitions: List[Dict[str, Any]] = []

    # Track context quantity; default 1 until extracted
    context_qty = 1

    # reset state and send initial seller offer
    state = env.reset()

    raw_action = seller_policy_fn(state)
    action = np.array([float(np.clip(float(raw_action[0]), env.reservation_value, 1.0))], dtype=np.float32)
    pending_offer = env.action_to_offer(action)

    offer_p, offer_q = offer_to_numbers(pending_offer)
    context_qty = int(offer_q)

    # initial seller message (counts as seller offer #1)
    dia_history = build_transcript(history)
    seller_text = seller_llm.generate(
        case=case,
        dia_history=dia_history if dia_history else "（尚无对话）",
        buyer_bid=None,
        quantity=context_qty,
        strategy="Counteroffer",
        price=float(offer_p),
    )
    history.append({"role": "seller", "content": seller_text})
    wechat_service.send_message(user_id, {"action": "continue", "response": seller_text})

    seller_offer_count = 1
    buyer_offer_count = 0

    done = False
    final_info: Dict[str, Any] = {"agreement": False, "outcome": None}

    while not done:
        # stop if reached offer limits
        if seller_offer_count >= max_seller_offers and buyer_offer_count >= max_buyer_offers:
            # force failure
            next_state, reward, done, info = env.step_with_human_reply(
                action=action,
                agent_offer=pending_offer,
                opponent_offer=None,
                opponent_accepts_agent_offer=False,
                force_fail=True,
            )
            transitions.append({
                "state": state, "action": action, "reward": reward, "next_state": next_state, "done": done,
                "meta": {"reason": "max_offers_reached", "user_id": user_id, "product_id": case.get("product_id")}
            })
            wechat_service.send_message(user_id, {"action": "end", "response": "本次出价次数已达上限，谈判结束。", "deal_price": None})
            final_info = info
            break

        buyer_msg = wechat_service.wait_for_user_message(user_id, timeout=buyer_timeout)
        if buyer_msg is None:
            # timeout -> failure
            next_state, reward, done, info = env.step_with_human_reply(
                action=action,
                agent_offer=pending_offer,
                opponent_offer=None,
                opponent_accepts_agent_offer=False,
                force_fail=True,
            )
            transitions.append({
                "state": state, "action": action, "reward": reward, "next_state": next_state, "done": done,
                "meta": {"reason": "buyer_timeout", "user_id": user_id, "product_id": case.get("product_id")}
            })
            wechat_service.send_message(user_id, {"action": "end", "response": "长时间未收到消息，谈判结束。", "deal_price": None})
            final_info = info
            break

        buyer_msg = str(buyer_msg).strip()
        history.append({"role": "buyer", "content": buyer_msg})

        # 1) intent
        try:
            pred = intent_recognizer.predict(buyer_msg)
            intent_label = pred.get("label", "")
        except Exception as e:
            print(f"[intent_recognizer] WARNING: {e}")
            intent_label = ""
        intent = normalize_intent(intent_label)

        # 2) accept / fail
        if intent == "accept":
            next_state, reward, done, info = env.step_with_human_reply(
                action=action,
                agent_offer=pending_offer,
                opponent_offer=None,
                opponent_accepts_agent_offer=True,
            )
            transitions.append({
                "state": state, "action": action, "reward": reward, "next_state": next_state, "done": done,
                "meta": {"intent": intent_label, "accepted": "seller_offer", "offer": pending_offer}
            })

            deal_p, deal_q = offer_to_numbers(pending_offer)
            dia_history = build_transcript(history)
            seller_text = seller_llm.generate(
                case=case,
                dia_history=dia_history,
                buyer_bid=float(deal_p),
                quantity=int(deal_q),
                strategy="Accept",
                price=float(deal_p),
            )
            history.append({"role": "seller", "content": seller_text})
            wechat_service.send_message(user_id, {"action": "end", "response": seller_text, "deal_price": float(deal_p)})
            final_info = info
            break

        if intent == "fail":
            next_state, reward, done, info = env.step_with_human_reply(
                action=action,
                agent_offer=pending_offer,
                opponent_offer=None,
                opponent_accepts_agent_offer=False,
                force_fail=True,
            )
            transitions.append({
                "state": state, "action": action, "reward": reward, "next_state": next_state, "done": done,
                "meta": {"intent": intent_label, "accepted": None, "reason": "buyer_fail"}
            })
            wechat_service.send_message(user_id, {"action": "end", "response": "好的，感谢沟通，祝您生活愉快。", "deal_price": None})
            final_info = info
            break

        # 3) bargain -> extract (price, qty) from full transcript
        transcript = build_transcript(history)
        price_extractor.set_case(case)
        unit_price, qty = price_extractor.extract(transcript, case=case)

        # If cannot extract price: ask for explicit price (does NOT consume an offer count)
        if unit_price is None:
            wechat_service.send_message(user_id, {"action": "continue", "response": "方便给个明确的单价（元）吗？也可以说明数量。"})
            continue

        # Quantity logic
        if qty is None:
            qty = float(context_qty) if context_qty else 1.0
        context_qty = int(round(float(qty))) if qty else 1
        context_qty = max(1, context_qty)

        buyer_offer_count += 1
        opponent_offer = nearest_outcome_for_price_quantity(domain, float(unit_price), float(context_qty))

        # 4) step env with buyer counteroffer
        next_state, reward, done, info = env.step_with_human_reply(
            action=action,
            agent_offer=pending_offer,
            opponent_offer=opponent_offer,
            opponent_accepts_agent_offer=False,
        )

        transitions.append({
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "meta": {
                "intent": intent_label,
                "buyer_utterance": buyer_msg,
                "buyer_offer": opponent_offer,
                "seller_offer": pending_offer,
                "round": env.current_round,
            },
        })

        state = next_state

        if done:
            final_info = info
            if info.get("agreement") and info.get("outcome") is not None:
                deal_p, deal_q = offer_to_numbers(info["outcome"])
                dia_history = build_transcript(history)
                seller_text = seller_llm.generate(
                    case=case,
                    dia_history=dia_history,
                    buyer_bid=float(unit_price),
                    quantity=int(deal_q),
                    strategy="Accept",
                    price=float(deal_p),
                )
                history.append({"role": "seller", "content": seller_text})
                wechat_service.send_message(user_id, {"action": "end", "response": seller_text, "deal_price": float(deal_p)})
            else:
                wechat_service.send_message(user_id, {"action": "end", "response": "很遗憾，本次未能达成一致，谈判结束。", "deal_price": None})
            break

        # 5) choose next action -> next seller offer -> send
        if seller_offer_count >= max_seller_offers:
            # cannot counteroffer anymore -> fail
            next_state2, reward2, done2, info2 = env.step_with_human_reply(
                action=action,
                agent_offer=pending_offer,
                opponent_offer=None,
                opponent_accepts_agent_offer=False,
                force_fail=True,
            )
            transitions.append({
                "state": state, "action": action, "reward": reward2, "next_state": next_state2, "done": done2,
                "meta": {"reason": "max_seller_offers_reached"}
            })
            wechat_service.send_message(user_id, {"action": "end", "response": "卖家出价次数已达上限，谈判结束。", "deal_price": None})
            final_info = info2
            break

        raw_action = seller_policy_fn(state)
        action = np.array([float(np.clip(float(raw_action[0]), env.reservation_value, 1.0))], dtype=np.float32)
        pending_offer = env.action_to_offer(action)
        offer_p, offer_q = offer_to_numbers(pending_offer)
        context_qty = int(offer_q)

        dia_history = build_transcript(history)
        seller_text = seller_llm.generate(
            case=case,
            dia_history=dia_history,
            buyer_bid=float(unit_price),
            quantity=context_qty,
            strategy="Counteroffer",
            price=float(offer_p),
        )
        history.append({"role": "seller", "content": seller_text})
        wechat_service.send_message(user_id, {"action": "continue", "response": seller_text})
        seller_offer_count += 1

    # episode record
    ep = {
        "user_id": user_id,
        "product_id": case.get("product_id"),
        "product_name": case.get("product_name"),
        "max_seller_offers": max_seller_offers,
        "max_buyer_offers": max_buyer_offers,
        "history": history,
        "final_info": final_info,
        "transitions": len(transitions),
        "timestamp": time.time(),
    }
    if episode_log_path:
        append_jsonl(episode_log_path, ep)

    return transitions, ep


# ---------------------------------------------------------------------
# WeChat worker (runs forever, or until max_episodes reached)
# ---------------------------------------------------------------------

def run_dorea_wechat_worker(
    wechat_service,
    *,
    mode: str = "collect_offline",
    max_episodes: int = 2000,
    offline_path: str = "./data/D_off_wechat.pkl",
    online_path: str = "./data/D_on_wechat.pkl",
    episode_log_path: str = "./data/wechat_episodes.jsonl",
    model_path: str = "./data/dorea_model.pt",
    max_seller_offers: int = 8,
    max_buyer_offers: int = 8,
    # intent/price model paths
    intent_base_model_path: Optional[str] = None,
    intent_adapter_path: Optional[str] = None,
    price_base_model_path: Optional[str] = None,
    price_tokenizer_path: Optional[str] = None,
    price_adapter_path: Optional[str] = None,
    # seller llm
    seller_llm_model_path: str = "Qwen/Qwen3-4B-Instruct-2507",
):
    """
    mode:
      - collect_offline: use baseline policy to collect D_off
      - train_offline: train DOREA on D_off and save model
      - online_finetune: load model, negotiate with humans, write D_on and fine-tune
      - serve: load model, negotiate with humans, write D_on (optional) but do not update

    To keep app.py unchanged, you can map your app's "train" to "collect_offline".
    """

    # ---- models (loaded once)
    intent_recognizer = IntentRecognizer(
        base_model_path=intent_base_model_path or "Qwen/Qwen2.5-1.5B-Instruct",
        adapter_path=intent_adapter_path,
        device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "auto",
    )

    price_extractor = PriceQuantityExtractor(
        base_model_path=price_base_model_path,
        tokenizer_path=price_tokenizer_path,
        adapter_path=price_adapter_path,
    )

    seller_llm = QwenSellerGenerator(SellerLLMConfig(model_path=seller_llm_model_path))

    dorea = DOREAFramework(state_dim=7, action_dim=1, n_ensemble=5, batch_size=256, cql_alpha=1.0)

    if mode == "train_offline":
        if not os.path.exists(offline_path):
            raise FileNotFoundError(f"offline dataset not found: {offline_path}")
        with open(offline_path, "rb") as f:
            ds = pickle.load(f)
        for t in ds:
            dorea.add_offline_data(t["state"], t["action"], t["reward"], t["next_state"], t["done"])
        dorea.train_offline(n_steps=200000)  # adjust as needed
        dorea.save(model_path)
        print(f"[train_offline] saved model to {model_path}")
        return

    if mode in ["online_finetune", "serve"]:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"model not found: {model_path}")
        dorea.load(model_path)

        # optionally also load offline data into buffer for balanced replay
        if os.path.exists(offline_path):
            with open(offline_path, "rb") as f:
                ds = pickle.load(f)
            for t in ds:
                dorea.add_offline_data(t["state"], t["action"], t["reward"], t["next_state"], t["done"])

    print(f"[wechat_worker] mode={mode} max_episodes={max_episodes} listening ...")

    episodes_done = 0
    while True:
        user_id, product_id = wechat_service.wait_for_new_session(timeout=1.0)
        if not user_id:
            continue

        if episodes_done >= max_episodes:
            wechat_service.send_message(user_id, {"action": "end", "response": "数据采集已达到目标次数，暂不再接单。", "deal_price": None})
            wechat_service.end_session(user_id)
            continue

        try:
            wechat_service.episode_processing_lock.set()

            product = wechat_service.get_product_by_id(product_id)
            if not product:
                wechat_service.send_message(user_id, {"action": "end", "response": "商品信息未找到，无法开始谈判。", "deal_price": None})
                wechat_service.end_session(user_id)
                continue

            # policy fn
            if mode == "collect_offline":
                # baseline in utility space; reservation value will be known inside episode
                policy_fn = lambda s: time_dependent_baseline(s, reservation_value=0.0)
            else:
                policy_fn = lambda s: dorea.select_action(s, deterministic=False)

            transitions, ep = run_one_wechat_episode(
                wechat_service=wechat_service,
                user_id=user_id,
                product=product,
                intent_recognizer=intent_recognizer,
                price_extractor=price_extractor,
                seller_policy_fn=policy_fn,
                seller_llm=seller_llm,
                max_seller_offers=max_seller_offers,
                max_buyer_offers=max_buyer_offers,
                episode_log_path=episode_log_path,
            )

            if mode == "collect_offline":
                append_pickle(offline_path, transitions)
                print(f"[collect_offline] episode={episodes_done+1} transitions={len(transitions)} user={user_id}")
            else:
                append_pickle(online_path, transitions)
                for t in transitions:
                    dorea.add_online_data(t["state"], t["action"], t["reward"], t["next_state"], t["done"])
                print(f"[online] episode={episodes_done+1} transitions={len(transitions)} user={user_id}")

                if mode == "online_finetune":
                    dorea.finetune(n_steps=2000)
                    dorea.save(model_path)
                    print(f"[online_finetune] updated model saved to {model_path}")

            episodes_done += 1
            wechat_service.end_session(user_id)

        except Exception as e:
            try:
                wechat_service.send_message(user_id, {"action": "end", "response": f"后台发生错误，谈判结束：{e}", "deal_price": None})
                wechat_service.end_session(user_id)
            except Exception:
                pass
            print(f"[wechat_worker ERROR] {e}")
        finally:
            wechat_service.episode_processing_lock.clear()


# ---------------------------------------------------------------------
# Offline evaluation on HLQ scenes
# ---------------------------------------------------------------------

def _safe_float(x, default=None):
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip().lower() in ["null", "none", "nan", ""]:
            return default
        return float(x)
    except Exception:
        return default

def _safe_int(x, default=None):
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip().lower() in ["null", "none", "nan", ""]:
            return default
        return int(float(x))
    except Exception:
        return default

def load_hlq_scenes(json_path: str) -> List[Dict[str, Any]]:
    """Load HLQ_negotiation_scence_test_data.json (structure-robust)."""
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        scenes = obj
    elif isinstance(obj, dict):
        scenes = obj.get("data") or obj.get("scenes") or obj.get("items") or []
    else:
        scenes = []
    return scenes


def normalize_scene(scene: Dict[str, Any], idx: int = 0) -> Dict[str, Any]:
    """Normalize keys used by evaluation buyer simulator."""
    s = dict(scene)
    s["product_id"] = s.get("product_id", idx)
    s["product_name"] = s.get("product_name", s.get("title", f"item_{idx}"))
    s["seller_bottom_price"] = _safe_float(s.get("seller_bottom_price") or s.get("seller_reserve_price") or s.get("reserve_price") or s.get("bottom_price"), 0.0)
    s["init_price"] = _safe_float(s.get("init_price") or s.get("initial_seller_price") or s.get("start_price"), s["seller_bottom_price"] * 1.2)
    s["buyer_max_price"] = _safe_float(s.get("buyer_max_price") or s.get("buyer_reserve_price") or s.get("max_price"), s["init_price"])
    s["initial_buyer_price"] = _safe_float(s.get("initial_buyer_price") or s.get("buyer_init_price") or s.get("buyer_start_price"), s["buyer_max_price"] * 0.6)
    s["quantity"] = _safe_int(s.get("quantity") or s.get("max_qty") or 1, 1)
    return s


class SimulatedBuyer:
    """A simple non-LLM buyer for evaluation only (NOT used for WeChat data collection)."""

    def __init__(self, buyer_max_price: float, initial_buyer_price: float, quantity: int):
        self.buyer_max_price = float(buyer_max_price)
        self.initial_buyer_price = float(initial_buyer_price)
        self.quantity = int(quantity)

    def propose_price(self, turn: int, max_turns: int) -> float:
        # linear concession from initial to max
        if max_turns <= 1:
            return self.buyer_max_price
        frac = min(1.0, max(0.0, turn / float(max_turns - 1)))
        return float(self.initial_buyer_price + frac * (self.buyer_max_price - self.initial_buyer_price))

    def respond(self, seller_unit_price: float, turn: int, max_turns: int) -> Tuple[bool, float, int]:
        if float(seller_unit_price) <= self.buyer_max_price:
            return True, float(seller_unit_price), self.quantity
        return False, self.propose_price(turn, max_turns), self.quantity


def run_one_scene_episode(
    agent_policy_fn: Callable[[np.ndarray], np.ndarray],
    scene: Dict[str, Any],
    max_rounds: int = 8,
    deterministic: bool = True,
) -> Dict[str, Any]:
    s = normalize_scene(scene, idx=0)
    product = {
        "product_id": s["product_id"],
        "product_name": s["product_name"],
        "init_price": s["init_price"],
        "seller_reserve_price": s["seller_bottom_price"],
        "max_qty": s["quantity"],
    }
    domain, seller_pref, meta = create_price_quantity_domain_from_product(product)
    env = HumanOpponentNegotiationEnvironment(
        domain,
        seller_pref,
        max_rounds=max_rounds,
        reservation_value=float(meta.get("reservation_utility", 0.0)),
        meta=meta,
    )

    buyer = SimulatedBuyer(s["buyer_max_price"], s["initial_buyer_price"], s["quantity"])
    state = env.reset()

    agreement = False
    final_price = float("nan")
    final_qty = s["quantity"]
    turns = 0

    for t in range(max_rounds):
        turns = t + 1
        raw_action = agent_policy_fn(state)
        action = np.array([float(np.clip(float(raw_action[0]), env.reservation_value, 1.0))], dtype=np.float32)
        seller_offer = env.action_to_offer(action)
        seller_p, _ = offer_to_numbers(seller_offer)

        accept, buyer_p, buyer_q = buyer.respond(seller_p, t, max_rounds)
        if accept:
            next_state, reward, done, info = env.step_with_human_reply(
                action=action, agent_offer=seller_offer, opponent_offer=None, opponent_accepts_agent_offer=True
            )
        else:
            opp_offer = nearest_outcome_for_price_quantity(domain, buyer_p, buyer_q)
            next_state, reward, done, info = env.step_with_human_reply(
                action=action, agent_offer=seller_offer, opponent_offer=opp_offer, opponent_accepts_agent_offer=False
            )

        state = next_state
        if done:
            agreement = bool(info.get("agreement", False))
            if agreement and info.get("outcome") is not None:
                p, q = offer_to_numbers(info["outcome"])
                final_price, final_qty = float(p), int(q)
            break

    return {
        "success": bool(agreement),
        "final_price": float(final_price),
        "seller_bottom_price": float(s["seller_bottom_price"]),
        "buyer_max_price": float(s["buyer_max_price"]),
        "turns": int(turns),
        "quantity": int(final_qty),
        "product_id": s["product_id"],
        "product_name": s["product_name"],
        "initial_buyer_price": float(s["initial_buyer_price"]),
        "initial_seller_price": float(s["init_price"]),
    }


def evaluate_dorea_on_scenes(
    scenes_json_path: str,
    *,
    model_path: str = "./data/dorea_model.pt",
    output_dir: str = "results",
    metrics_filename: str = "dorea_metrics.txt",
    results_filename: str = "dorea_results.json",
    max_rounds: int = 8,
    deterministic: bool = True,
) -> List[Dict[str, Any]]:
    scenes = load_hlq_scenes(scenes_json_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model not found: {model_path}")

    dorea = DOREAFramework(state_dim=7, action_dim=1, n_ensemble=5, batch_size=256, cql_alpha=1.0)
    dorea.load(model_path)

    def policy(s: np.ndarray) -> np.ndarray:
        return dorea.select_action(s, deterministic=deterministic)

    results: List[Dict[str, Any]] = []
    for sc in scenes:
        results.append(run_one_scene_episode(policy, sc, max_rounds=max_rounds, deterministic=deterministic))

    ensure_dir(os.path.join(output_dir, "dummy.txt"))
    res_path = os.path.join(output_dir, results_filename)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # compute metrics using the provided module
    from calculator_negotiation_metrics import calculate_and_print_metrics
    calculate_and_print_metrics(results, output_dir, metrics_filename)

    return results


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train_offline")
    p_train.add_argument("--offline_path", default="./data/D_off_wechat.pkl")
    p_train.add_argument("--model_path", default="./data/dorea_model.pt")
    p_train.add_argument("--n_steps", type=int, default=200000)

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--scenes", required=True, help="HLQ_negotiation_scence_test_data.json")
    p_eval.add_argument("--model_path", default="./data/dorea_model.pt")
    p_eval.add_argument("--output_dir", default="results")
    p_eval.add_argument("--max_rounds", type=int, default=8)

    args = p.parse_args()

    if args.cmd == "train_offline":
        if not os.path.exists(args.offline_path):
            raise FileNotFoundError(args.offline_path)
        with open(args.offline_path, "rb") as f:
            ds = pickle.load(f)
        dorea = DOREAFramework(state_dim=7, action_dim=1, n_ensemble=5, batch_size=256, cql_alpha=1.0)
        for t in ds:
            dorea.add_offline_data(t["state"], t["action"], t["reward"], t["next_state"], t["done"])
        dorea.train_offline(n_steps=args.n_steps)
        dorea.save(args.model_path)
        print(f"saved model to {args.model_path}")

    elif args.cmd == "eval":
        evaluate_dorea_on_scenes(args.scenes, model_path=args.model_path, output_dir=args.output_dir, max_rounds=args.max_rounds)
        print("evaluation finished")


if __name__ == "__main__":
    main()
