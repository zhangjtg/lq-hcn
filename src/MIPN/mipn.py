"""mipn.py

"""

from __future__ import annotations

import collections
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


# ==========================================
# 1) CONFIGURATION & DOMAIN
# ==========================================


class ProductConfig:
    """Loads negotiation context from JSON/data_dict and builds discrete price grid."""

    def __init__(
        self,
        json_path: Optional[str] = None,
        data_dict: Optional[dict] = None,
        num_bins: int = 50,
    ):
        if data_dict is not None:
            data = data_dict
        elif json_path is not None:
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"Config file {json_path} not found.")
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            raise ValueError("Either json_path or data_dict must be provided.")

        self.name = data.get("product_name", "Unknown Product")
        self.display_price = float(data.get("init_price"))

        # Hidden info
        self.seller_res = float(data.get("seller_reserve_price"))
        self.buyer_res = float(data.get("buyer_reserve_price", self.display_price))

        # DISCRETIZATION
        self.num_bins = int(num_bins)
        # 注意：训练时一般是 linspace(seller_res, display_price, num_bins)
        self.price_grid = np.linspace(self.seller_res, self.display_price, self.num_bins).tolist()


# ==========================================
# 2) NEURAL NETWORK (MiPN Architecture)
# ==========================================


class MiPN_SingleIssue(nn.Module):
    """Multi-Issue Policy Network adapted for single dimension.

    State = [Agent_Hist_t-1, Agent_Hist_t, Opp_Hist_t-1, Opp_Hist_t, Time]
    这里与训练脚本保持一致：输入维度 (num_bins*4)+1
    """

    def __init__(self, num_price_bins: int, hidden_dim: int = 64):
        super().__init__()

        input_dim = (num_price_bins * 4) + 1

        self.shared_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.accept_head = nn.Linear(hidden_dim, 2)
        self.price_head = nn.Linear(hidden_dim, num_price_bins)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor):
        feats = self.shared_net(state)
        return self.accept_head(feats), self.price_head(feats), self.value_head(feats)

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        acc_logits, price_logits, value = self.forward(state)

        acc_dist = Categorical(logits=acc_logits)
        acc_action = acc_dist.probs.argmax() if deterministic else acc_dist.sample()

        price_dist = Categorical(logits=price_logits)
        price_action = price_dist.probs.argmax() if deterministic else price_dist.sample()

        action = {"accept": int(acc_action.item()), "price_idx": int(price_action.item())}
        log_probs = {"accept": acc_dist.log_prob(acc_action), "price": price_dist.log_prob(price_action)}

        return action, log_probs, value


# ==========================================
# 3) ONLINE STATE ENCODER
# ==========================================


class MiPNStateEncoder:
    """把最近2次卖家报价 + 最近2次买家报价 + 时间 编码为网络输入。

    对齐你原始环境 PriceNegotiationEnv._get_state() 的编码顺序：
      for p in agent_hist: extend(encode(p))   # agent_hist deque(maxlen=2)
      for p in opp_hist:   extend(encode(p))
      append(round/max_rounds)

    注意：deque 迭代顺序为“从旧到新”。
    """

    def __init__(self, price_grid: List[float]):
        self.price_grid = np.array(price_grid, dtype=np.float32)
        self.num_bins = int(len(price_grid))

        self.agent_hist = collections.deque(maxlen=2)
        self.opp_hist = collections.deque(maxlen=2)

        self.reset(opening_buyer_price=None)

    def reset(self, opening_buyer_price: Optional[float]):
        self.agent_hist.clear()
        self.agent_hist.extend([None, None])

        self.opp_hist.clear()
        self.opp_hist.extend([None, None])

        if opening_buyer_price is not None:
            self.opp_hist.append(float(opening_buyer_price))

    def update_buyer(self, price: float):
        self.opp_hist.append(float(price))

    def update_seller(self, price: float):
        self.agent_hist.append(float(price))

    def last_seller_offer(self) -> Optional[float]:
        for p in reversed(self.agent_hist):
            if p is not None:
                return float(p)
        return None

    def _encode_price(self, price: Optional[float]) -> List[float]:
        vec = np.zeros(self.num_bins, dtype=np.float32)
        if price is None:
            return vec.tolist()
        idx = int(np.abs(self.price_grid - float(price)).argmin())
        vec[idx] = 1.0
        return vec.tolist()

    def get_state(self, seller_turn_count: int, max_turns: int) -> torch.FloatTensor:
        state: List[float] = []
        for p in self.agent_hist:
            state.extend(self._encode_price(p))
        for p in self.opp_hist:
            state.extend(self._encode_price(p))
        t = 0.0 if max_turns <= 0 else float(seller_turn_count) / float(max_turns)
        state.append(t)
        return torch.FloatTensor(state)


# ==========================================
# 4) ONLINE SELLER AGENT WRAPPER
# ==========================================


@dataclass
class MiPNDecision:
    strategy: str                  # "Accept" or "Counteroffer"
    price: Optional[float] = None  # Counteroffer 时有效


class MiPNSellerAgent:
    """在线推理封装：不包含任何买家模型。

    用法：
      cfg = ProductConfig(data_dict=case)
      agent = MiPNSellerAgent(cfg, ckpt_path=..., device='cuda')
      agent.reset(opening_buyer_price)
      agent.update_buyer(new_buyer_price)
      decision = agent.decide(seller_turn_count, max_turns)
      if decision.strategy == 'Counteroffer': agent.update_seller(decision.price)
    """

    def __init__(
        self,
        config: ProductConfig,
        device: Optional[str] = None,
        ckpt_path: Optional[str] = None,
        state_dict: Optional[dict] = None,
    ):
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = MiPN_SingleIssue(num_price_bins=self.config.num_bins).to(self.device)
        self.model.eval()

        # 加载权重（优先 state_dict，其次 ckpt_path）
        sd = state_dict
        if sd is None and ckpt_path:
            if os.path.exists(ckpt_path):
                sd = torch.load(ckpt_path, map_location="cpu")

        if sd is not None:
            try:
                self.model.load_state_dict(sd)
            except Exception:
                # 有些 ckpt 是 {'model': ...} 包装
                if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
                    self.model.load_state_dict(sd["model"])
                else:
                    raise

        self.encoder = MiPNStateEncoder(self.config.price_grid)

    def reset(self, opening_buyer_price: float):
        self.encoder.reset(opening_buyer_price=opening_buyer_price)

    def update_buyer(self, buyer_price: float):
        self.encoder.update_buyer(buyer_price)

    def update_seller(self, seller_price: float):
        self.encoder.update_seller(seller_price)

    @torch.no_grad()
    def decide(self, seller_turn_count: int, max_turns: int = 8, deterministic: bool = True) -> MiPNDecision:
        """返回卖家本回合的策略动作与价格。"""

        state = self.encoder.get_state(seller_turn_count=seller_turn_count, max_turns=max_turns)
        state = state.unsqueeze(0).to(self.device)

        action, _, _ = self.model.get_action(state, deterministic=deterministic)

        if int(action.get("accept", 0)) == 1:
            return MiPNDecision(strategy="Accept", price=None)

        idx = int(action.get("price_idx", 0))
        idx = max(0, min(idx, self.config.num_bins - 1))
        price = float(self.config.price_grid[idx])
        return MiPNDecision(strategy="Counteroffer", price=price)
