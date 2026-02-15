

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from ecommerce_dataset import ProductContext, as_public_context, as_seller_context
from Game import NegotiationGame
from llm_client import LLMClient
from Model import ValueModel
from QMCTS import NegotiationQMCTS
from src.price_quantity_extractor import PriceQuantityExtractor
from src.intent_recognizer import IntentRecognizer
from seller_response_generator import SellerResponseGenerator
from src.wechat_service import WeChatService


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v is not None and v != "" else default


def _to_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _make_llm(prefix: str, *, fallback_mode: str = "heuristic") -> LLMClient:
    """Create an LLMClient from environment variables.

    Supported vars (prefix = SELLER or BUYER_SIM):
      - {prefix}_LLM_MODE: heuristic | openai | openai_compatible | local_ministral | local_hf
      - {prefix}_LLM_MODEL: model name for OpenAI-compatible modes
      - {prefix}_BASE_URL: base_url for openai_compatible
      - {prefix}_MODEL_PATH: local model path for local_ministral / local_hf
      - DEVICE_MAP / TORCH_DTYPE: used for local_ministral
    """

    mode = _env(f"{prefix}_LLM_MODE", fallback_mode)
    model = _env(f"{prefix}_LLM_MODEL", "gpt-3.5-turbo")
    base_url = _env(f"{prefix}_BASE_URL", None)
    model_path = _env(f"{prefix}_MODEL_PATH", None)
    device_map = _env("DEVICE_MAP", "auto")
    torch_dtype = _env("TORCH_DTYPE", "bfloat16")

    if mode == "local_ministral":
        if not model_path:
            # If missing, fall back to heuristic (so service still runs)
            return LLMClient(mode="heuristic")
        return LLMClient(mode="local_ministral", model_path=model_path, device_map=device_map, torch_dtype=torch_dtype)

    if mode == "local_hf":
        if not model_path:
            return LLMClient(mode="heuristic")
        return LLMClient(mode="local_hf", model_path=model_path, device_map=device_map, torch_dtype=torch_dtype)

    if mode in ("openai", "openai_compatible"):
        return LLMClient(mode=mode, model=model, base_url=base_url)

    return LLMClient(mode="heuristic")


def _product_to_ctx(prod: Dict[str, Any]) -> ProductContext:
    # ProductContext fields vary by your dataset. Keep it defensive.
    return ProductContext(
        product_id=_to_int(prod.get("product_id"), 0),
        product_name=str(prod.get("product_name", "")),
        seller_item_description=str(prod.get("seller_item_description", "")),
        init_price=_to_int(prod.get("init_price"), 0),
        buyer_reserve_price=_to_int(prod.get("buyer_reserve_price"), _to_int(prod.get("init_price"), 0)),
        seller_reserve_price=_to_int(prod.get("seller_reserve_price"), 0),
        currency=str(prod.get("currency", "CNY")),
        quantity=float(prod.get("quantity", 1.0)),
    )


def _render_transcript(lines: List[Tuple[str, str]], max_turns: int = 12) -> str:
    """Convert [(role, text), ...] to a compact transcript string."""
    keep = lines[-max_turns:] if max_turns > 0 else lines
    return "\n".join([f"{r}: {t}" for r, t in keep])


@dataclass
class HumanSession:
    user_id: str
    product: Dict[str, Any]
    quantity: float = 1.0
    dialogue: List[Tuple[str, str]] = None

    def __post_init__(self):
        if self.dialogue is None:
            self.dialogue = []


def _format_seller_message(*, seller_price: int, quantity: float, currency: str, accepted: bool) -> str:
    if accepted:
        return f"好的，我们成交：{seller_price} {currency}/台，数量 {quantity:g}。"
    return (
        f"我这边可以给到 {seller_price} {currency}/台，数量按 {quantity:g} 计算。\n"
        f"如果您接受，请回复：\"{seller_price}可以\"（或直接发 {seller_price}）。"
    )


def _format_need_price_message() -> str:
    return "我没识别到您的出价。请在消息里明确单价和数量，例如：\"2800元/台，10台\"。"


def run_training(wechat_service: WeChatService, mode: str = "train", results_path: str = "custom_results.csv"):
    """Background loop started by `app.py`.

    Parameters are kept for compatibility with the original entrypoint used in `app.py`.
    """

    # --- Build components once ---
    seller_llm = _make_llm("SELLER", fallback_mode="heuristic")
    buyer_sim_llm = _make_llm("BUYER_SIM", fallback_mode="heuristic")

    # 卖家自然语言回复（推荐本地 Qwen3-4B-Instruct-2507，使用 local_hf 方式加载）
    seller_nlg_llm = _make_llm("SELLER_NLG", fallback_mode="local_hf")
    seller_nlg = SellerResponseGenerator(seller_nlg_llm)

    # 买家意图识别模型（TinyLlama 分类模型 + PEFT adapter）
    intent_base = _env("INTENT_BASE_MODEL_PATH", "/home/zjt/local/model_bin/TinyLlama_v1.1_chinese")
    intent_adapter = _env("INTENT_ADAPTER_PATH", "../output/intent/TinyLlama_v1.1_chinese-classification83")
    intent_device = _env("INTENT_DEVICE", None)
    intent_num_labels = _to_int(_env("INTENT_NUM_LABELS", "6"), 6)
    try:
        intent_recognizer = IntentRecognizer(
            base_model_path=intent_base,
            adapter_path=intent_adapter,
            num_labels=intent_num_labels,
            device=intent_device,
        )
        print(f"[run_LQ_negotiation] Loaded intent model base={intent_base} adapter={intent_adapter}")
    except Exception as e:
        intent_recognizer = None
        print(f"[run_LQ_negotiation] Failed to load intent model: {e}")

    turn_limit = _to_int(_env("TURN_LIMIT", "16"), 16)
    memory_size = _to_int(_env("MEMORY_SIZE", "20"), 20)

    game = NegotiationGame(
        seller_llm=seller_llm,
        buyer_llm=buyer_sim_llm,  # used only for opponent simulation inside seller MCTS
        memory_size=memory_size,
        turn_limit=turn_limit,
    )

    # MCTS config
    num_searches = _to_int(_env("NUM_SEARCHES", "10"), 10)
    device = _env("VALUE_DEVICE", "cpu")
    state_dim = 5 + memory_size
    value_model = ValueModel(state_dim=state_dim, hidden_dim=256, device=device)
    weights = _env("VALUE_MODEL_PATH", None)
    if weights and os.path.exists(weights):
        try:
            value_model.load_state_dict(torch.load(weights, map_location=value_model.device))
            value_model.eval()
            print(f"[run_LQ_negotiation] Loaded value model weights from {weights}")
        except Exception as e:
            print(f"[run_LQ_negotiation] Failed to load VALUE_MODEL_PATH={weights}: {e}")

    mcts = NegotiationQMCTS(game, value_model, args={"C": 2.0, "num_searches": num_searches})

    # 价格/数量识别模型（Qwen2.5 1.5B + PEFT adapter，按用户提供的加载方式支持 merge_and_unload）
    # 推荐环境变量：
    #   PRICE_BASE_MODEL_PATH=/home/zjt/local/model_bin/Qwen2.5-1.5B-Instruct
    #   PRICE_TOKENIZER_PATH=../output/price_model/Qwen2p5-1p5B-instruct-a5-l4-618
    #   PRICE_ADAPTER_PATH=../output/price_model/Qwen2p5-1p5B-instruct-a5-l4-618
    price_device = _env("PRICE_DEVICE", "cuda")
    price_base = _env("PRICE_BASE_MODEL_PATH", None) or _env("PRICE_MODEL_PATH", None)
    price_tok = _env("PRICE_TOKENIZER_PATH", None)
    price_adapter = _env("PRICE_ADAPTER_PATH", None)
    extractor = PriceQuantityExtractor(
        device=price_device,
        base_model_path=price_base,
        tokenizer_path=price_tok,
        adapter_path=price_adapter,
        device_map=_env("PRICE_DEVICE_MAP", "auto"),
        torch_dtype=_env("PRICE_TORCH_DTYPE", "bfloat16"),
    )
    if price_base:
        print(f"[run_LQ_negotiation] Loaded price extractor base={price_base} adapter={price_adapter or '(none)'}")

    print("[run_LQ_negotiation] WeChat negotiation loop started.")

    while True:
        user_id, product_id = wechat_service.wait_for_new_session(timeout=1.0)
        if not user_id:
            continue

        # Mark busy (app.py uses this to block concurrent start requests)
        wechat_service.episode_processing_lock.set()

        try:
            product = wechat_service.get_product_by_id(product_id)
            if not product:
                wechat_service.send_message(user_id, {"action": "end", "response": "商品信息读取失败，请重新开始。", "deal_price": None})
                continue

            env_ctx = _product_to_ctx(product)
            seller_ctx = as_seller_context(env_ctx)
            public_ctx = as_public_context(env_ctx)

            # session state
            sess = HumanSession(user_id=user_id, product=product, quantity=float(getattr(env_ctx, "quantity", 1.0)))
            extractor.set_case(product)

            state = game.get_initial_state(public_ctx)

            # --- Send welcome ---
            welcome = (
                f"您好！我是卖家，我们来谈 {env_ctx.product_name} 的价格。\n"
                f"商品：{env_ctx.seller_item_description}\n"
                f"标价：{env_ctx.init_price} {env_ctx.currency}/台。\n"
                "请先给出您的出价（单价+数量），例如：\"2800元/台，10台\"。"
            )
            wechat_service.send_message(user_id, {"action": "continue", "response": welcome})
            sess.dialogue.append(("Seller", welcome))

            # --- Main negotiation loop ---
            while not game.is_terminal(state):
                if int(state[1]) >= int(state[2]):
                    wechat_service.send_message(user_id, {"action": "end", "response": "已到最大轮次，暂未达成一致。", "deal_price": None})
                    break

                # Ensure it's buyer's turn (human)
                if int(state[0]) != -1:
                    state = game.change_perspective(state, -1)

                user_msg = wechat_service.wait_for_user_message(user_id, timeout=1200)
                if user_msg is None:
                    wechat_service.send_message(user_id, {"action": "end", "response": "等待您的消息超时，本次谈判结束。", "deal_price": None})
                    break

                # User asked to quit
                if re.search(r"(退出|结束|不谈了|取消)", str(user_msg)):
                    wechat_service.send_message(user_id, {"action": "end", "response": "好的，已结束本次谈判。", "deal_price": None})
                    break

                sess.dialogue.append(("Buyer", str(user_msg)))
                transcript = _render_transcript(sess.dialogue)

                # 1) 先做买家意图识别：讨价还价 / 谈判成功 / 谈判失败
                if intent_recognizer is not None:
                    intent = intent_recognizer.predict(str(user_msg)).label
                else:
                    intent = "讨价还价"

                if intent == "谈判失败":
                    wechat_service.send_message(user_id, {"action": "end", "response": "好的，感谢沟通，我们下次再聊。", "deal_price": None})
                    break

                if intent == "谈判成功":
                    # 买家表达成交：默认接受“卖家上一轮出价”。
                    so = game.get_seller_offer(state)
                    if so is None:
                        # 如果没有卖家出价（极少见），尝试从文本里解析一个价格。
                        price_int, qty = extractor.extract(transcript, case=product)
                        if price_int is None:
                            wechat_service.send_message(user_id, {"action": "continue", "response": "我理解您想成交，但我还没报过价。请您先给个具体单价或让我先出价。"})
                            continue
                    else:
                        price_int, qty = int(so), None

                    if qty is not None:
                        sess.quantity = float(qty)

                    state = game.get_next_state(state, (0, int(price_int)), player=-1)
                    deal_price = int(game.get_buyer_offer(state) or price_int)
                    wechat_service.send_message(
                        user_id,
                        {"action": "end", "response": f"好的，成交价：{deal_price} {env_ctx.currency}/台，数量 {sess.quantity:g}。", "deal_price": deal_price},
                    )
                    break

                # 2) 讨价还价：需要从对话里识别单价 + 数量
                price_int, qty = extractor.extract(transcript, case=product)

                if qty is not None:
                    sess.quantity = float(qty)

                if price_int is None or price_int <= 0:
                    wechat_service.send_message(user_id, {"action": "continue", "response": _format_need_price_message()})
                    continue

                # Apply buyer offer to the game
                state = game.get_next_state(state, (0, int(price_int)), player=-1)

                # Buyer might be accepting last seller price -> terminal immediately.
                if game.is_terminal(state):
                    deal_price = int(game.get_buyer_offer(state) or price_int)
                    wechat_service.send_message(
                        user_id,
                        {
                            "action": "end",
                            "response": f"好的，成交价：{deal_price} {env_ctx.currency}/台，数量 {sess.quantity:g}。",
                            "deal_price": deal_price,
                        },
                    )
                    break

                # Seller turn
                state = game.change_perspective(state, 1)

                res = mcts.search(state, seller_ctx, public_ctx)
                seller_action = res.best_action
                seller_price = int(seller_action[1])

                state = game.get_next_state(state, seller_action, player=1)
                accepted = game.is_terminal(state)

                strategy = "Accept" if accepted else "Counteroffer"
                try:
                    seller_text = seller_nlg.generate(
                        case=product,
                        dialogue=sess.dialogue,
                        buyer_bid=int(game.get_buyer_offer(state) or price_int),
                        quantity=int(round(sess.quantity)),
                        strategy=strategy,
                        price=seller_price,
                    )
                except Exception:
                    # Fallback to deterministic template if NLG model is not available.
                    seller_text = _format_seller_message(
                        seller_price=seller_price,
                        quantity=sess.quantity,
                        currency=env_ctx.currency,
                        accepted=accepted,
                    )
                sess.dialogue.append(("Seller", seller_text))

                payload = {"action": "end" if accepted else "continue", "response": seller_text}
                if accepted:
                    payload["deal_price"] = seller_price
                wechat_service.send_message(user_id, payload)

                if accepted:
                    break

        except Exception as e:
            print(f"[run_LQ_negotiation ERROR] user={user_id} product={product_id} err={e}")
            try:
                wechat_service.send_message(user_id, {"action": "end", "response": f"服务端异常：{e}", "deal_price": None})
            except Exception:
                pass
        finally:
            # release busy flag
            wechat_service.episode_processing_lock.clear()

        # avoid tight loop
        time.sleep(0.05)
