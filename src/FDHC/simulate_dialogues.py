"""Offline  negotiation dialogue simulation.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import torch

from ecommerce_dataset import as_buyer_context, as_public_context, as_seller_context, load_products_json, sample_product
from Game import NegotiationGame
from src.intent_recognizer import IntentRecognizer
from llm_client import LLMClient
from Model import ValueModel
from src.price_quantity_extractor import PriceQuantityExtractor
from QMCTS import NegotiationQMCTS
from seller_response_generator import SellerResponseGenerator


def _make_llm(
    *,
    mode: str,
    model: str,
    base_url: Optional[str],
    model_path: Optional[str],
    device_map: str,
    torch_dtype: str,
) -> LLMClient:
    if mode == "local_ministral":
        if not model_path:
            raise ValueError("local_ministral requires --*_model_path")
        return LLMClient(mode="local_ministral", model_path=model_path, device_map=device_map, torch_dtype=torch_dtype)
    if mode == "local_hf":
        if not model_path:
            raise ValueError("local_hf requires --*_model_path")
        return LLMClient(mode="local_hf", model_path=model_path, device_map=device_map, torch_dtype=torch_dtype)
    if mode in ("openai", "openai_compatible"):
        return LLMClient(mode=mode, model=model, base_url=base_url)
    return LLMClient(mode="heuristic")


def _render_transcript(lines: List[Tuple[str, str]], max_turns: int = 12) -> str:
    keep = lines[-max_turns:] if max_turns > 0 else lines
    return "\n".join([f"{r}: {t}" for r, t in keep])


def _buyer_utterance(price: int, quantity: float, *, accept: bool = False) -> str:
    q = f"{quantity:g}" if quantity is not None else "1"
    if accept:
        return f"行，就按{price}元/台成交，数量{q}。"
    return f"我出{price}元/台，数量{q}，可以吗？"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--products", type=str, required=True, help="JSON list of products (see ecommerce_dataset.load_products_json)")
    parser.add_argument("--out", type=str, default="dialogues_2000.jsonl")
    parser.add_argument("--num_episodes", type=int, default=2000)
    parser.add_argument("--turn_limit", type=int, default=16)
    parser.add_argument("--memory_size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    # Seller policy model (for candidate generation inside MCTS)
    parser.add_argument("--seller_llm_mode", type=str, default="local_ministral",
                        choices=["heuristic", "openai", "openai_compatible", "local_ministral", "local_hf"])
    parser.add_argument("--seller_llm_model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--seller_base_url", type=str, default=None)
    parser.add_argument("--seller_model_path", type=str, default=None)

    # Buyer simulator model
    parser.add_argument("--buyer_llm_mode", type=str, default="heuristic",
                        choices=["heuristic", "openai", "openai_compatible", "local_ministral", "local_hf"])
    parser.add_argument("--buyer_llm_model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--buyer_base_url", type=str, default=None)
    parser.add_argument("--buyer_model_path", type=str, default=None)

    # Seller NLG (recommended Qwen3)
    parser.add_argument("--seller_nlg_mode", type=str, default="local_hf",
                        choices=["heuristic", "openai", "openai_compatible", "local_ministral", "local_hf"])
    parser.add_argument("--seller_nlg_model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--seller_nlg_base_url", type=str, default=None)
    parser.add_argument("--seller_nlg_model_path", type=str, default=None)

    # Shared local loading
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])

    # Value model
    parser.add_argument("--value_device", type=str, default="cpu")
    parser.add_argument("--value_weights", type=str, default=None)
    parser.add_argument("--num_searches", type=int, default=10)

    # Intent model
    parser.add_argument("--intent_base", type=str, default="/home/zjt/local/model_bin/TinyLlama_v1.1_chinese")
    parser.add_argument("--intent_adapter", type=str, default="../output/intent/TinyLlama_v1.1_chinese-classification83")
    parser.add_argument("--intent_num_labels", type=int, default=6)
    parser.add_argument("--intent_device", type=str, default=None)

    # Price extractor
    parser.add_argument("--price_device", type=str, default="cuda")
    parser.add_argument("--price_base", type=str, default="/home/zjt/local/model_bin/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--price_tokenizer", type=str, default="../output/price_model/Qwen2p5-1p5B-instruct-a5-l4-618")
    parser.add_argument("--price_adapter", type=str, default="../output/price_model/Qwen2p5-1p5B-instruct-a5-l4-618")

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    products = load_products_json(args.products)

    seller_llm = _make_llm(
        mode=args.seller_llm_mode,
        model=args.seller_llm_model,
        base_url=args.seller_base_url,
        model_path=args.seller_model_path,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
    )
    buyer_llm = _make_llm(
        mode=args.buyer_llm_mode,
        model=args.buyer_llm_model,
        base_url=args.buyer_base_url,
        model_path=args.buyer_model_path,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
    )
    seller_nlg_llm = _make_llm(
        mode=args.seller_nlg_mode,
        model=args.seller_nlg_model,
        base_url=args.seller_nlg_base_url,
        model_path=args.seller_nlg_model_path,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
    )

    game = NegotiationGame(
        seller_llm=seller_llm,
        buyer_llm=buyer_llm,
        memory_size=args.memory_size,
        turn_limit=args.turn_limit,
    )

    # Value model (optional weights)
    state_dim = 5 + args.memory_size
    value_model = ValueModel(state_dim=state_dim, hidden_dim=256, device=args.value_device)
    if args.value_weights and os.path.exists(args.value_weights):
        value_model.load_state_dict(torch.load(args.value_weights, map_location=value_model.device))
        value_model.eval()

    mcts = NegotiationQMCTS(game, value_model, args={"C": 2.0, "num_searches": int(args.num_searches)})

    intent = IntentRecognizer(
        base_model_path=args.intent_base,
        adapter_path=args.intent_adapter,
        num_labels=int(args.intent_num_labels),
        device=args.intent_device,
    )

    extractor = PriceQuantityExtractor(
        device=args.price_device,
        base_model_path=args.price_base,
        tokenizer_path=args.price_tokenizer,
        adapter_path=args.price_adapter,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
    )

    seller_nlg = SellerResponseGenerator(seller_nlg_llm)

    with open(args.out, "w", encoding="utf-8") as f:
        for epi in range(int(args.num_episodes)):
            env_ctx = sample_product(products)
            seller_ctx = as_seller_context(env_ctx)
            buyer_ctx = as_buyer_context(env_ctx)
            public_ctx = as_public_context(env_ctx)

            # For NLG prompt building we also need a raw dict (keep consistent with run_LQ_negotiation)
            case: Dict[str, Any] = {
                "product_id": env_ctx.product_id,
                "product_name": env_ctx.product_name,
                "seller_item_description": env_ctx.seller_item_description,
                "init_price": env_ctx.init_price,
                "buyer_reserve_price": env_ctx.buyer_reserve_price,
                "seller_reserve_price": env_ctx.seller_reserve_price,
                "currency": env_ctx.currency,
                "quantity": env_ctx.quantity,
            }
            extractor.set_case(case)

            state = game.get_initial_state(public_ctx)
            dialogue: List[Tuple[str, str]] = []
            qty = float(env_ctx.quantity)

            # 开场白（卖家）
            opening = (
                f"您好！我们来谈 {env_ctx.product_name} 的价格。"
                f"商品：{env_ctx.seller_item_description}。"
                f"标价：{env_ctx.init_price} {env_ctx.currency}/台。"
            )
            dialogue.append(("Seller", opening))

            deal_price: Optional[int] = None
            end_reason: str = "turn_limit"

            while not game.is_terminal(state):
                if int(state[1]) >= int(state[2]):
                    end_reason = "turn_limit"
                    break

                # ---- Buyer turn ----
                if int(state[0]) != -1:
                    state = game.change_perspective(state, -1)

                buyer_action = game.buyer_offer(state, buyer_ctx)
                so = game.get_seller_offer(state)
                accept = (buyer_action[0] == 6) or (so is not None and int(buyer_action[1]) == int(so))
                buyer_msg = _buyer_utterance(int(buyer_action[1]), qty, accept=accept)
                dialogue.append(("Buyer", buyer_msg))

                transcript = _render_transcript(dialogue)
                intent_label = intent.predict(buyer_msg).label

                if intent_label == "谈判失败":
                    end_reason = "buyer_fail"
                    break

                if intent_label == "谈判成功":
                    # 默认接受卖家上一轮出价
                    if so is not None:
                        buyer_price = int(so)
                    else:
                        buyer_price, parsed_qty = extractor.extract(transcript, case=case)
                        buyer_price = int(buyer_price or buyer_action[1])
                        if parsed_qty is not None:
                            qty = float(parsed_qty)
                    state = game.get_next_state(state, (0, int(buyer_price)), player=-1)
                    if game.is_terminal(state):
                        deal_price = int(buyer_price)
                        end_reason = "deal"
                    else:
                        end_reason = "buyer_success_no_deal"
                    break

                # 讨价还价：从对话里抽取价格/数量
                buyer_price, parsed_qty = extractor.extract(transcript, case=case)
                if parsed_qty is not None:
                    qty = float(parsed_qty)
                if buyer_price is None:
                    buyer_price = int(buyer_action[1])
                state = game.get_next_state(state, (0, int(buyer_price)), player=-1)
                if game.is_terminal(state):
                    deal_price = int(buyer_price)
                    end_reason = "deal"
                    break

                # ---- Seller turn ----
                state = game.change_perspective(state, 1)
                res = mcts.search(state, seller_ctx, public_ctx)
                seller_action = res.best_action
                seller_price = int(seller_action[1])
                state = game.get_next_state(state, seller_action, player=1)
                accepted = game.is_terminal(state)
                strategy = "Accept" if accepted else "Counteroffer"

                try:
                    seller_msg = seller_nlg.generate(
                        case=case,
                        dialogue=dialogue,
                        buyer_bid=int(buyer_price),
                        quantity=int(round(qty)),
                        strategy=strategy,
                        price=seller_price,
                    )
                except Exception:
                    seller_msg = f"我这边报价 {seller_price} 元/台，数量 {qty:g}。" if not accepted else f"好，{seller_price} 元/台成交。"
                dialogue.append(("Seller", seller_msg))

                if accepted:
                    deal_price = int(seller_price)
                    end_reason = "deal"
                    break

            rec = {
                "episode": epi,
                "product_id": env_ctx.product_id,
                "deal_price": deal_price,
                "end_reason": end_reason,
                "turn_limit": int(args.turn_limit),
                "transcript": dialogue,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
