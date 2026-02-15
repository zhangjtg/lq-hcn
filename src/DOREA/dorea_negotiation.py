# -*- coding: utf-8 -*-
"""
dorea_negotiation.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math
import numpy as np


# ---------------------------------------------------------------------
# Domain (finite outcome space)
# ---------------------------------------------------------------------

@dataclass
class PriceQuantityDomain:
    """Finite discrete outcomes for (unit_price, quantity)."""

    prices: List[float]          # discrete unit prices
    quantities: List[int]        # discrete quantities

    def all_outcomes(self) -> List[Dict[str, Any]]:
        outs: List[Dict[str, Any]] = []
        for q in self.quantities:
            for p in self.prices:
                outs.append({"unit_price": float(p), "quantity": int(q)})
        return outs

    def nearest_outcome(self, unit_price: float, quantity: float) -> Dict[str, Any]:
        p = float(unit_price)
        q = int(round(float(quantity)))
        p_near = min(self.prices, key=lambda x: abs(float(x) - p))
        q_near = min(self.quantities, key=lambda x: abs(int(x) - q))
        return {"unit_price": float(p_near), "quantity": int(q_near)}


# ---------------------------------------------------------------------
# Utility (seller preference)
# ---------------------------------------------------------------------

@dataclass
class SellerPreferencePQ:
    """Simple additive utility for seller on (unit_price, quantity)."""

    init_price: float
    reserve_price: float
    min_qty: int = 1
    max_qty: int = 10
    w_price: float = 0.8
    w_qty: float = 0.2

    def utility(self, offer: Dict[str, Any]) -> float:
        p = float(offer["unit_price"])
        q = int(offer["quantity"])

        # price utility: 0 at reserve, 1 at init (clip)
        denom = max(1e-6, self.init_price - self.reserve_price)
        p_u = (p - self.reserve_price) / denom
        p_u = float(np.clip(p_u, 0.0, 1.0))

        # quantity utility: 0 at min_qty, 1 at max_qty
        if self.max_qty <= self.min_qty:
            q_u = 0.0
        else:
            q_u = (q - self.min_qty) / float(self.max_qty - self.min_qty)
            q_u = float(np.clip(q_u, 0.0, 1.0))

        u = self.w_price * p_u + self.w_qty * q_u
        return float(np.clip(u, 0.0, 1.0))


# ---------------------------------------------------------------------
# Opponent model (frequency-based, standard in ANAC literature)
# ---------------------------------------------------------------------

class FrequencyOpponentModelPQ:
    """
    Very lightweight frequency-based opponent model:
    - Counts how often buyer offers each price/quantity value
    - Estimates issue weights by entropy/variance proxy
    - Estimates value evaluations by normalized frequencies
    """

    def __init__(self, domain: PriceQuantityDomain):
        self.domain = domain
        self.price_counts = {float(p): 0 for p in domain.prices}
        self.qty_counts = {int(q): 0 for q in domain.quantities}
        self.total = 0

    def update(self, offer: Dict[str, Any]) -> None:
        p = float(offer["unit_price"])
        q = int(offer["quantity"])
        if p not in self.price_counts:
            # robust: snap to nearest
            p = float(min(self.domain.prices, key=lambda x: abs(float(x) - p)))
        if q not in self.qty_counts:
            q = int(min(self.domain.quantities, key=lambda x: abs(int(x) - q)))
        self.price_counts[p] += 1
        self.qty_counts[q] += 1
        self.total += 1

    def _norm_freq(self, counts: Dict[Any, int]) -> Dict[Any, float]:
        if self.total <= 0:
            return {k: 1.0 / max(1, len(counts)) for k in counts}
        vals = np.array([counts[k] for k in counts], dtype=np.float32)
        # add-one smoothing
        vals = vals + 1.0
        vals = vals / float(np.sum(vals))
        return {k: float(v) for k, v in zip(counts.keys(), vals)}

    def estimate_issue_weights(self) -> Tuple[float, float]:
        """
        Simple variability proxy:
        - If buyer varies a lot on price but not on qty, weight(price) > weight(qty)
        """
        if self.total <= 0:
            return 0.5, 0.5

        p_freq = np.array(list(self._norm_freq(self.price_counts).values()), dtype=np.float32)
        q_freq = np.array(list(self._norm_freq(self.qty_counts).values()), dtype=np.float32)

        # entropy -> lower entropy means stronger preference -> higher weight
        def entropy(x: np.ndarray) -> float:
            x = np.clip(x, 1e-9, 1.0)
            return float(-np.sum(x * np.log(x)))

        p_ent = entropy(p_freq)
        q_ent = entropy(q_freq)
        # map to preference strength
        p_strength = 1.0 / (p_ent + 1e-6)
        q_strength = 1.0 / (q_ent + 1e-6)
        s = p_strength + q_strength
        return float(p_strength / s), float(q_strength / s)

    def estimate_utility(self, offer: Dict[str, Any]) -> float:
        p = float(offer["unit_price"])
        q = int(offer["quantity"])
        p_freq = self._norm_freq(self.price_counts)
        q_freq = self._norm_freq(self.qty_counts)

        # For buyer, lower price is usually better; but frequency model doesn't know direction.
        # We treat higher frequency as higher preference.
        p_eval = p_freq.get(p, 1e-6)
        q_eval = q_freq.get(q, 1e-6)
        w_p, w_q = self.estimate_issue_weights()
        u = w_p * p_eval + w_q * q_eval
        return float(np.clip(u, 0.0, 1.0))


# ---------------------------------------------------------------------
# Environment (for DOREA state/action/reward)
# ---------------------------------------------------------------------

class HumanOpponentNegotiationEnvironment:
    """
    Environment step is aligned with DOREA paper's abstraction:

      s_t = [t/T_max, u_s(ω_o^{t-3}), u_s(ω_s^{t-3}), ..., u_s(ω_o^{t-1}), u_s(ω_s^{t-1})]
      a_t ∈ [u_r, 1] (target utility for next agent offer)
      reward:
        - 0 if not terminal
        - U_s(agreement) if agreement
        - -1 if failure (deadline or explicit)

    The buyer is human; this env expects the outer loop to provide:
      - opponent_accepts_agent_offer: bool
      - opponent_offer: dict (if counteroffer)
    """

    def __init__(
        self,
        domain: PriceQuantityDomain,
        seller_preference: SellerPreferencePQ,
        *,
        max_rounds: int = 8,
        reservation_value: float = 0.0,
        meta: Optional[Dict[str, Any]] = None,
        inverse_tol: float = 0.02,
    ):
        self.domain = domain
        self.seller_pref = seller_preference
        self.max_rounds = int(max_rounds)
        self.reservation_value = float(reservation_value)
        self.meta = meta or {}
        self.inverse_tol = float(inverse_tol)

        self.opp_model = FrequencyOpponentModelPQ(domain)

        self.current_round = 0
        self.seller_offers: List[Dict[str, Any]] = []
        self.buyer_offers: List[Dict[str, Any]] = []

    # ---------------- core helpers ----------------

    def reset(self) -> np.ndarray:
        self.current_round = 0
        self.seller_offers = []
        self.buyer_offers = []
        self.opp_model = FrequencyOpponentModelPQ(self.domain)
        return self._get_state()

    def seller_utility(self, offer: Dict[str, Any]) -> float:
        return self.seller_pref.utility(offer)

    def action_to_offer(self, action: np.ndarray) -> Dict[str, Any]:
        """Inverse utility: pick offer ≈ target utility and maximize estimated opponent utility."""
        target = float(action[0])
        target = float(np.clip(target, self.reservation_value, 1.0))

        candidates: List[Tuple[float, float, Dict[str, Any]]] = []
        best = None
        best_dist = float("inf")

        for offer in self.domain.all_outcomes():
            u_s = self.seller_utility(offer)
            dist = abs(u_s - target)
            if dist < best_dist:
                best_dist = dist
                best = offer

            if dist <= self.inverse_tol:
                u_o_hat = self.opp_model.estimate_utility(offer)
                candidates.append((u_o_hat, u_s, offer))

        if candidates:
            # max opponent estimated utility; tie-breaker: closer to target utility
            candidates.sort(key=lambda x: (x[0], -abs(x[1] - target)), reverse=True)
            return candidates[0][2]

        # fallback: closest utility
        assert best is not None
        return best

    def step_with_human_reply(
        self,
        *,
        action: np.ndarray,
        agent_offer: Dict[str, Any],
        opponent_offer: Optional[Dict[str, Any]],
        opponent_accepts_agent_offer: bool,
        force_fail: bool = False,
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Advance ONE DOREA step corresponding to ONE seller offer + buyer reply.

        Returns: next_state, reward, done, info
        """
        info: Dict[str, Any] = {"agreement": False, "outcome": None}

        # If user explicitly quits or timeout
        if force_fail:
            reward = -1.0
            done = True
            info["agreement"] = False
            info["outcome"] = None
            return self._terminal_state(), reward, done, info

        # record seller offer for this round
        self.current_round += 1
        self.seller_offers.append(agent_offer)

        # buyer accepts seller offer
        if opponent_accepts_agent_offer:
            outcome = agent_offer
            reward = float(self.seller_utility(outcome))
            done = True
            info["agreement"] = True
            info["outcome"] = outcome
            return self._terminal_state(), reward, done, info

        # buyer counteroffers (or keeps bargaining)
        if opponent_offer is not None:
            self.buyer_offers.append(opponent_offer)
            self.opp_model.update(opponent_offer)

            # acceptance strategy (paper): accept opponent offer if it's >= intended own offer
            if self.seller_utility(opponent_offer) >= self.seller_utility(agent_offer):
                outcome = opponent_offer
                reward = float(self.seller_utility(outcome))
                done = True
                info["agreement"] = True
                info["outcome"] = outcome
                return self._terminal_state(), reward, done, info

        # deadline failure
        if self.current_round >= self.max_rounds:
            reward = -1.0
            done = True
            info["agreement"] = False
            info["outcome"] = None
            return self._terminal_state(), reward, done, info

        # otherwise continue
        reward = 0.0
        done = False
        return self._get_state(), reward, done, info

    # ---------------- state ----------------

    def _get_state(self) -> np.ndarray:
        t_norm = self.current_round / float(max(1, self.max_rounds))
        # get last 3 pairs (buyer, seller) utilities
        pairs: List[float] = []
        # align from the end; buyer may have fewer offers than seller if accept happened (terminal only)
        for k in range(3, 0, -1):
            idx = -k
            u_b = 0.0
            u_s = 0.0
            if len(self.buyer_offers) >= k:
                u_b = self.seller_utility(self.buyer_offers[idx])
            if len(self.seller_offers) >= k:
                u_s = self.seller_utility(self.seller_offers[idx])
            pairs.extend([u_b, u_s])

        state = np.array([t_norm] + pairs, dtype=np.float32)
        # state dim must be 7
        if state.shape[0] != 7:
            state = np.resize(state, (7,)).astype(np.float32)
        return state

    def _terminal_state(self) -> np.ndarray:
        # terminal state keeps same dim; set t_norm=1
        s = self._get_state()
        s[0] = 1.0
        return s


# ---------------------------------------------------------------------
# Utility helpers for training loop
# ---------------------------------------------------------------------

def create_price_quantity_domain_from_product(
    product: Dict[str, Any],
    *,
    price_steps: int = 41,
    max_qty: Optional[int] = None,
) -> Tuple[PriceQuantityDomain, SellerPreferencePQ, Dict[str, Any]]:
    """
    Build a discretized (price, qty) domain from product info.

    Expected fields in `product` (robustly handled):
      - init_price
      - seller_reserve_price OR reserve_price OR seller_bottom_price

    Returns:
      domain, seller_pref, meta
    """
    init_price = float(product.get("init_price") or product.get("initial_seller_price") or 0.0)
    reserve_price = float(
        product.get("seller_reserve_price")
        if product.get("seller_reserve_price") is not None
        else product.get("reserve_price")
        if product.get("reserve_price") is not None
        else product.get("seller_bottom_price")
        if product.get("seller_bottom_price") is not None
        else init_price * 0.7
    )

    # sanity
    if init_price <= 0:
        init_price = max(1.0, reserve_price * 1.2)
    if reserve_price <= 0:
        reserve_price = init_price * 0.7
    if reserve_price > init_price:
        reserve_price, init_price = init_price, reserve_price

    if max_qty is None:
        max_qty = int(product.get("max_qty") or product.get("quantity") or 10)
    max_qty = max(1, int(max_qty))

    # build price grid (inclusive)
    steps = max(2, int(price_steps))
    prices = np.linspace(reserve_price, init_price, num=steps, dtype=np.float32).tolist()
    # more human-friendly: round to integer if prices are close to int
    prices = [float(int(round(p))) if abs(p - round(p)) < 1e-3 else float(p) for p in prices]
    # ensure unique, sorted
    prices = sorted(set(prices))

    quantities = list(range(1, max_qty + 1))

    domain = PriceQuantityDomain(prices=prices, quantities=quantities)
    seller_pref = SellerPreferencePQ(
        init_price=init_price,
        reserve_price=reserve_price,
        min_qty=1,
        max_qty=max_qty,
        w_price=float(product.get("w_price", 0.8)),
        w_qty=float(product.get("w_qty", 0.2)),
    )

    meta = {
        "init_price": init_price,
        "reserve_price": reserve_price,
        "max_qty": max_qty,
        # reservation utility at (reserve_price, 1)
        "reservation_utility": seller_pref.utility({"unit_price": reserve_price, "quantity": 1}),
        "price_steps": len(prices),
    }
    return domain, seller_pref, meta


def nearest_outcome_for_price_quantity(domain: PriceQuantityDomain, price: float, quantity: float) -> Dict[str, Any]:
    return domain.nearest_outcome(price, quantity)


def offer_to_numbers(offer: Dict[str, Any]) -> Tuple[float, int]:
    return float(offer["unit_price"]), int(offer["quantity"])


def offer_to_text_cn(offer: Dict[str, Any], *, product_name: str = "") -> str:
    p, q = offer_to_numbers(offer)
    if product_name:
        return f"我这边{product_name}可以给到：单价{p:.0f}元，数量{q}。您看可以吗？"
    return f"我这边可以给到：单价{p:.0f}元，数量{q}。您看可以吗？"
