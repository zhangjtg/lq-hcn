
import argparse
import random
from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from ecommerce_dataset import (
    load_products_json,
    sample_product,
    as_buyer_context,
    as_public_context,
    as_seller_context,
    ProductContext,
)
from llm_client import LLMClient
from Game import NegotiationGame
from QMCTS import NegotiationQMCTS
from Model import ValueModel
import csv
import json
import os
import ast
@dataclass
class EpisodeMemory:
    states: List[np.ndarray]
    final_state: np.ndarray
    seller_reward: float
    buyer_reward: float


def run_episode(
    game: NegotiationGame,
    mcts: NegotiationQMCTS,
    env_ctx: ProductContext,
    seller_ctx: Any,
    buyer_ctx: Any,
    public_ctx: Any,
    max_steps: int,
) -> EpisodeMemory:
    state = game.get_initial_state(public_ctx)
    traj_states: List[np.ndarray] = []

    while not game.is_terminal(state) and int(state[1]) < max_steps:
        traj_states.append(state.copy())

        if int(state[0]) == 1:
            # seller: best-response style search (MCTS selects among LLM-proposed candidate prices)
            res = mcts.search(state, seller_ctx, public_ctx)
            action = res.best_action
            state = game.get_next_state(state, action, player=1)
        else:
            # buyer: independent LLM bid (strategy prompt; must not exceed buyer reserve)
            action = game.buyer_offer(state, buyer_ctx)
            state = game.get_next_state(state, action, player=-1)

    seller_r = game.fairness_reward(state, env_ctx, player=1)
    buyer_r = game.fairness_reward(state, env_ctx, player=-1)
    return EpisodeMemory(traj_states, state, seller_r, buyer_r)


def build_dataset(memories: List[EpisodeMemory]) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    for ep in memories:
        for st in ep.states:
            xs.append(st.astype(np.float32))
            ys.append([ep.seller_reward, ep.buyer_reward])
    x = torch.tensor(np.stack(xs), dtype=torch.float32)
    y = torch.tensor(np.stack(ys), dtype=torch.float32)
    return x, y


def _make_llm(
    *,
    mode: str,
    model: str,
    base_url: str | None,
    model_path: str | None,
    device_map: str,
    torch_dtype: str,
) -> LLMClient:
    if mode == "local_ministral":
        if not model_path:
            raise ValueError("--model_path must be provided when using --*_llm_mode local_ministral")
        return LLMClient(mode="local_ministral", model_path=model_path, device_map=device_map, torch_dtype=torch_dtype)
    if mode == "local_hf":
        if not model_path:
            raise ValueError("--model_path must be provided when using --*_llm_mode local_hf")
        return LLMClient(mode="local_hf", model_path=model_path, device_map=device_map, torch_dtype=torch_dtype)
    if mode in ("openai", "openai_compatible"):
        return LLMClient(mode=mode, model=model, base_url=base_url)
    return LLMClient(mode="heuristic")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--products", type=str, default="../../../../data/negotiation_data/HLQ_Scene_train_english.json")
    parser.add_argument("--device", type=str, default="cpu")

    # Shared local model settings (used if seller_model_path/buyer_model_path not given)
    parser.add_argument("--model_path", type=str, default="/home/zjt/local/zjt/model-bin/Ministral-3-14B-Instruct-2512", help="Local path for Ministral-3-14B-Instruct-2512 (or similar).")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])

    # Seller LLM (candidate generation for seller actions)
    parser.add_argument("--seller_llm_mode", type=str, default="local_ministral",
                        choices=["heuristic", "openai", "openai_compatible", "local_ministral", "local_hf"])
    parser.add_argument("--seller_llm_model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--seller_base_url", type=str, default=None)
    parser.add_argument("--seller_model_path", type=str, default=None, help="Optional: override local model path for seller.")

    # Buyer simulator LLM (ONLY for self-play training; NOT used in WeChat human-buyer runtime)
    parser.add_argument("--buyer_llm_mode", type=str, default="heuristic",
                        choices=["heuristic", "openai", "openai_compatible", "local_ministral", "local_hf"])
    parser.add_argument("--buyer_llm_model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--buyer_base_url", type=str, default=None)
    parser.add_argument("--buyer_model_path", type=str, default=None, help="Optional: override local model path for buyer.")


    # 需求：至少完成 2000 次完整谈判 episode（每个 episode 最多 16 个 turn：买家 8 + 卖家 8）
    parser.add_argument("--num_episodes", type=int, default=2000)
    parser.add_argument("--turn_limit", type=int, default=16)
    parser.add_argument("--memory_size", type=int, default=20)
    parser.add_argument("--num_searches", type=int, default=10)#50
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)

    # Requirement: 400 epochs
    parser.add_argument("--epochs", type=int, default=400)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="models/materials_model.pt")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    products = load_products_json(args.products)

    # Resolve model paths
    seller_path = args.seller_model_path or args.model_path
    buyer_path = args.buyer_model_path or args.model_path

    seller_llm = _make_llm(
        mode=args.seller_llm_mode,
        model=args.seller_llm_model,
        base_url=args.seller_base_url,
        model_path=seller_path,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
    )
    buyer_sim_llm = _make_llm(
        mode=args.buyer_llm_mode,
        model=args.buyer_llm_model,
        base_url=args.buyer_base_url,
        model_path=buyer_path,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
    )

    game = NegotiationGame(
        seller_llm=seller_llm,
        buyer_llm=buyer_sim_llm,
        memory_size=args.memory_size,
        turn_limit=args.turn_limit,
    )

    state_dim = 5 + args.memory_size
    model = ValueModel(state_dim=state_dim, hidden_dim=256, device=args.device)
    mcts = NegotiationQMCTS(game, model, args={"C": 2.0, "num_searches": args.num_searches})
    memories: List[EpisodeMemory] = []
    csv_filename=  f"training_data_search.csv"
    memories, start_episode_idx = load_memories_from_csv(csv_filename)
    pbar = trange(start_episode_idx, args.num_episodes, desc="Self-play episodes", initial=start_episode_idx,
                  total=args.num_episodes)
    for i in pbar:
        env_ctx = sample_product(products)
        seller_ctx = as_seller_context(env_ctx)
        buyer_ctx = as_buyer_context(env_ctx)
        public_ctx = as_public_context(env_ctx)

        ep = run_episode(game, mcts, env_ctx, seller_ctx, buyer_ctx, public_ctx, max_steps=args.turn_limit)
        memories.append(ep)
        save_episode_to_csv(csv_filename, i, ep)


    x, y = build_dataset(memories)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in trange(args.epochs, desc="Training epochs"):
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(model.device)
            yb = yb.to(model.device)

            pred = model(xb)
            loss = F.mse_loss(pred, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.detach().cpu().item())

        if (epoch + 1) % 100 == 0:
            avg_loss = total_loss / max(1, len(loader))
            print(f"Epoch {epoch+1}/{args.epochs}  loss={avg_loss:.6f}")

    import os
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"Saved model to: {args.save_path}")


def save_episode_to_csv(filepath: str, episode_idx: int, ep: EpisodeMemory):
    """
    将单个 EpisodeMemory 追加写入 CSV 文件。
    每一行代表一个状态步 (Step)，包含 Episode ID 和最终奖励。
    """
    file_exists = os.path.isfile(filepath)

    with open(filepath, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # 如果文件不存在，先写入表头
        if not file_exists:
            writer.writerow(["episode_id", "state_json", "seller_reward", "buyer_reward"])

        # 将 trajectory 中的每个状态写入一行
        # 注意：EpisodeMemory 中的 reward 是整局结束后的奖励，对于该局的所有 step 都是一样的
        for state in ep.states:
            # state 是 numpy array，转为 list 再转 json string 存入 csv
            state_str = json.dumps(state.tolist())
            writer.writerow([episode_idx, state_str, ep.seller_reward, ep.buyer_reward])


def load_memories_from_csv(filepath: str) -> Tuple[List[EpisodeMemory], int]:
    """
    从 CSV 读取数据并重建 memories 列表。
    返回: (memories 列表, 下一个应该开始的 episode_idx)
    """
    if not os.path.exists(filepath):
        print("未发现现有数据文件，将从头开始生成。")
        return [], 0

    print(f"发现数据文件 {filepath}，正在加载以进行断点续传...")

    # 使用字典临时存储重建的数据: {episode_id: {'states': [], 'rewards': (s_r, b_r)}}
    temp_data = {}

    with open(filepath, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        max_id = -1

        for row in reader:
            ep_id = int(row["episode_id"])
            state_list = json.loads(row["state_json"])
            state_arr = np.array(state_list, dtype=np.int32)  # 确保类型与 get_initial_state 一致

            s_reward = float(row["seller_reward"])
            b_reward = float(row["buyer_reward"])

            if ep_id not in temp_data:
                temp_data[ep_id] = {
                    'states': [],
                    'seller_reward': s_reward,
                    'buyer_reward': b_reward
                }

            temp_data[ep_id]['states'].append(state_arr)
            # 更新 max_id 用于确定下一步从哪里开始
            if ep_id > max_id:
                max_id = ep_id

    # 将字典转换回 List[EpisodeMemory]
    restored_memories = []
    # 确保按 ID 顺序排序
    sorted_ids = sorted(temp_data.keys())
    for ep_id in sorted_ids:
        data = temp_data[ep_id]
        # 注意：EpisodeMemory 需要 final_state，这里我们简单取 states 的最后一个作为 final_state
        # (虽然训练时只用了 states list, 但为了保持结构完整)
        final_st = data['states'][-1] if data['states'] else None

        memory = EpisodeMemory(
            states=data['states'],
            final_state=final_st,
            seller_reward=data['seller_reward'],
            buyer_reward=data['buyer_reward']
        )
        restored_memories.append(memory)

    next_episode_idx = max_id + 1
    print(f"已恢复 {len(restored_memories)} 个 Episodes，将从 Episode {next_episode_idx} 继续。")
    return restored_memories, next_episode_idx
if __name__ == "__main__":
    main()
