import math
import random
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class SearchResult:
    best_action: Tuple[int, int]
    root: Any


class NegotiationNode:
    def __init__(
        self,
        game,
        state: np.ndarray,
        seller_context: Any,
        opponent_sim_context: Any,
        buyer_max_sampler=None,
        num_buyer_particles: int = 3,
        parent: Optional["NegotiationNode"] = None,
        action: Optional[Tuple[int, int]] = None,
    ):
        self.game = game
        self.state = state
        self.player = int(state[0])

        # Privacy-safe contexts:
        # - seller_context: contains seller reserve only
        # - opponent_sim_context: contains no reserves (public)
        self.seller_context = seller_context
        self.opponent_sim_context = opponent_sim_context

        # Belief-based opponent ceiling sampler (NOT the true buyer reserve)
        self.buyer_max_sampler = buyer_max_sampler
        self.num_buyer_particles = int(num_buyer_particles)

        self.parent = parent
        self.action = action  # (action_id, price)
        self.children: List["NegotiationNode"] = []

        self.visit_count = 0
        self.value_sum_p1 = 0.0
        self.value_sum_p2 = 0.0
        self.expanded = False

    def q(self, player: int) -> float:
        if self.visit_count == 0:
            return 0.0
        return (self.value_sum_p1 if player == 1 else self.value_sum_p2) / self.visit_count

    def expand(self):
        if self.expanded:
            return

        if self.player == 1:
            moves = self.game.neural_valid_moves(self.state, self.seller_context)
        else:
            # Buyer-node expansion under info asymmetry:
            # - We MUST NOT pass buyer_ctx (omniscience).
            # - But using pure public_ctx causes overly high bids.
            # -> Resolve by sampling a plausible buyer_max ceiling (belief) and
            #    constraining buyer candidate generation with buyer_max_override.
            moves: List[Tuple[int, int]] = []
            seen_prices = set()

            if self.buyer_max_sampler is not None:
                particles = max(1, self.num_buyer_particles)
                max_branch = max(10, int(getattr(self.game, "action_size", 7)) * 2)

                for _ in range(particles):
                    buyer_max = int(self.buyer_max_sampler(self.state, self.opponent_sim_context))
                    cand = self.game.neural_valid_moves(
                        self.state,
                        self.opponent_sim_context,
                        buyer_max_override=buyer_max,
                    )
                    for act in cand:
                        price = int(act[1])
                        if price not in seen_prices:
                            moves.append(act)
                            seen_prices.add(price)
                        if len(moves) >= max_branch:
                            break
                    if len(moves) >= max_branch:
                        break

            if not moves:
                moves = self.game.neural_valid_moves(self.state, self.opponent_sim_context)

        for act in moves:
            child_state = self.game.get_next_state(self.state, act, self.player)
            child_state = self.game.change_perspective(child_state, self.game.get_opponent(self.player))
            self.children.append(
                NegotiationNode(
                    self.game,
                    child_state,
                    self.seller_context,
                    self.opponent_sim_context,
                    buyer_max_sampler=self.buyer_max_sampler,
                    num_buyer_particles=self.num_buyer_particles,
                    parent=self,
                    action=act,
                )
            )

        self.expanded = True


class NegotiationQMCTS:
    def __init__(self, game, model, args: dict):
        self.game = game
        self.model = model
        self.args = args

    def _sample_buyer_max(self, state: np.ndarray, public_ctx: Any) -> int:
        """
        Sample a plausible buyer max willingness-to-pay (belief), using ONLY public/observable info:
          - init_price (public)
          - buyer past offers (observable lower bound)

        This is NOT the true buyer reservation price.
        """
        if isinstance(public_ctx, dict):
            init_price = int(public_ctx.get("init_price", 1) or 1)
        else:
            init_price = int(getattr(public_ctx, "init_price", 1) or 1)

        init_price = max(1, init_price)

        bo = self.game.get_buyer_offer(state)
        floor_ratio = float(self.args.get("buyer_max_floor_ratio", 0.60))
        floor = int(bo) if bo is not None else int(init_price * floor_ratio)

        low_ratio = float(self.args.get("buyer_max_ratio_low", 0.75))
        high_ratio = float(self.args.get("buyer_max_ratio_high", 0.95))

        low = max(floor, int(init_price * low_ratio))
        high = max(low, int(init_price * high_ratio))

        cap = int(self.args.get("buyer_max_cap", init_price))
        high = min(high, max(low, cap))

        return random.randint(int(low), int(high))

    def _ucb(self, parent: NegotiationNode, child: NegotiationNode) -> float:
        C = float(self.args.get("C", 2.0))
        q = child.q(parent.player)
        u = C * math.sqrt(math.log(parent.visit_count + 1.0) / (child.visit_count + 1.0))
        return q + u

    @torch.no_grad()
    def _eval_state(self, state: np.ndarray) -> Tuple[float, float]:
        device = self.model.device if hasattr(self.model, "device") else torch.device("cpu")
        x = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        out = self.model(x).squeeze(0).detach().cpu().numpy().tolist()
        return float(out[0]), float(out[1])

    def _backprop(self, node: NegotiationNode, v_p1: float, v_p2: float):
        while node is not None:
            node.visit_count += 1
            node.value_sum_p1 += v_p1
            node.value_sum_p2 += v_p2
            node = node.parent

    def search(self, state: np.ndarray, seller_context: Any, opponent_sim_context: Any) -> SearchResult:
        root = NegotiationNode(
            self.game,
            state,
            seller_context,
            opponent_sim_context,
            buyer_max_sampler=self._sample_buyer_max,
            num_buyer_particles=int(self.args.get("num_buyer_particles", 3)),
        )
        root.visit_count = 1

        num_searches = int(self.args.get("num_searches", 50))

        for _ in range(num_searches):
            node = root

            # selection
            while node.expanded and node.children:
                node = max(node.children, key=lambda c: self._ucb(node, c))

            # expansion
            if not self.game.is_terminal(node.state):
                node.expand()

            # evaluate (no env reserves used here)
            v_p1, v_p2 = self._eval_state(node.state)
            self._backprop(node, v_p1, v_p2)

        if not root.children:
            moves = self.game.neural_valid_moves(state, seller_context)
            return SearchResult(best_action=moves[0], root=root)

        best_child = max(root.children, key=lambda c: c.visit_count)
        return SearchResult(best_action=best_child.action, root=root)
