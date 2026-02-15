import json
import re
from typing import Any, List, Optional, Tuple

import numpy as np

from llm_client import LLMClient


def _get(ctx: Any, key: str, default=None):
	if isinstance(ctx, dict):
		return ctx.get(key, default)
	return getattr(ctx, key, default)


def _extract_first_json(text: str) -> Optional[dict]:
	"""
    Robustly extracts the first valid JSON object found in the text.
    1) Prefer ```json ... ``` blocks.
    2) Otherwise scan for first balanced {...}.
    """
	s = str(text)

	m = re.search(r"```json\s*(\{.*?\})\s*```", s, re.DOTALL)
	if m:
		try:
			return json.loads(m.group(1))
		except Exception:
			pass

	idx = 0
	while True:
		start = s.find("{", idx)
		if start == -1:
			break
		bal = 0
		for i in range(start, len(s)):
			if s[i] == "{":
				bal += 1
			elif s[i] == "}":
				bal -= 1
				if bal == 0:
					cand = s[start: i + 1]
					try:
						return json.loads(cand)
					except Exception:
						break
		idx = start + 1

	return None


class NegotiationGame:
	"""
    Single-issue price negotiation game.

    Privacy model:
      - State contains ONLY public info + offer history.
      - Seller reserve is only in SellerContext.
      - Buyer reserve is only in BuyerContext.
      - PublicContext contains no reserves and is used for opponent simulation inside seller MCTS.

    State layout:
      idx 0: player_to_move (1=seller, -1=buyer)
      idx 1: step_count
      idx 2: turn_limit
      idx 3: init_price (public)
      idx 4: reserved (unused; always 0)
      idx 5.. : offer memory (buyer bids first: buyer at 5,7,...; seller at 6,8,...)
    """

	def __init__(
			self,
			seller_llm: Optional[LLMClient] = None,
			buyer_llm: Optional[LLMClient] = None,
			memory_size: int = 20,
			turn_limit: int = 16,
	):
		self.memory_size = int(memory_size)
		self.turn_limit = int(turn_limit)
		self.action_size = 7

		self.seller_llm = seller_llm or LLMClient(mode="heuristic")
		self.buyer_llm = buyer_llm or self.seller_llm

	# -------------
	# State helpers
	# -------------

	def get_initial_state(self, public_context: Any) -> np.ndarray:
		init_price = int(_get(public_context, "init_price"))
		return np.array([-1, 0, self.turn_limit, init_price, 0] + [0] * self.memory_size, dtype=np.int32)

	def change_perspective(self, state: np.ndarray, player_to_move: int) -> np.ndarray:
		st = state.copy()
		st[0] = int(player_to_move)
		return st

	def get_opponent(self, player: int) -> int:
		return -1 if int(player) == 1 else 1

	def _offers(self, state: np.ndarray) -> List[int]:
		offers: List[int] = []
		for v in state[5:]:
			if int(v) == 0:
				break
			offers.append(int(v))
		return offers

	def get_seller_offer(self, state: np.ndarray) -> Optional[int]:
		offers = self._offers(state)
		for i in range(len(offers) - 1, -1, -1):
			# buyer indices: 0,2,4,... (buyer bids first)
			# seller indices: 1,3,5,...
			if i % 2 == 1:
				return offers[i]
		return None

	def get_buyer_offer(self, state: np.ndarray) -> Optional[int]:
		offers = self._offers(state)
		for i in range(len(offers) - 1, -1, -1):
			if i % 2 == 0:
				return offers[i]
		return None

	def is_terminal(self, state: np.ndarray) -> bool:
		if int(state[1]) >= int(state[2]):
			return True
		bo = self.get_buyer_offer(state)
		so = self.get_seller_offer(state)
		return (bo is not None and so is not None and int(bo) == int(so))

	def get_next_state(self, state: np.ndarray, action: Tuple[int, int], player: int) -> np.ndarray:
		st = state.copy()
		pos = np.where(st[5:] == 0)[0]
		if len(pos) == 0:
			st[5:-1] = st[6:]
			st[-1] = 0
			pos = np.where(st[5:] == 0)[0]
		idx = int(pos[0] + 5)

		_, price = action
		st[idx] = int(price)
		st[1] = int(st[1]) + 1
		st[0] = int(self.get_opponent(player))
		return st

	def _history_lines(self, state: np.ndarray) -> str:
		offers = self._offers(state)
		if not offers:
			return "（暂无出价）"
		lines = []
		for i, p in enumerate(offers):
			who = "买家" if i % 2 == 0 else "卖家"
			lines.append(f"{who}: {int(p)}")
		return "\n".join(lines)

	# -----------------------------
	# Buyer: independent LLM bidding
	# -----------------------------

	def _buyer_strategy_prompt(
			self,
			state: np.ndarray,
			buyer_context: Any,
			*,
			k: int,
			buyer_max_override: Optional[int] = None,
	) -> str:
		init_price = int(_get(buyer_context, "init_price"))
		currency = str(_get(buyer_context, "currency", "CNY"))
		history = self._history_lines(state)
		so = self.get_seller_offer(state)

		true_buyer_reserve = _get(buyer_context, "buyer_reserve_price")
		buyer_max = buyer_max_override if buyer_max_override is not None else true_buyer_reserve

		# Prompt wording (Chinese):
		# - 如果有真实买家最高价（BuyerContext），告诉它这是私密信息、不能泄露。
		# - 如果是模拟（PublicContext 或 buyer_max_override），只强制预算上限，但不要声称这是“真实保留价”。
		if buyer_max is None:
			buyer_max = init_price
			private_line = (
				"你有一个私密的最高可接受价格，但这里并未提供。"
				"请把标价当作一个安全的预算上限，不要臆造任何私密约束。"
			)
		else:
			buyer_max = int(buyer_max)
			if true_buyer_reserve is not None and buyer_max_override is None:
				private_line = (
					f"你的私密最高可接受单价（reservation price）是 {buyer_max} {currency}。"
					f"绝对不要出价高于 {buyer_max}。不要在对话中泄露这个数字。"
				)
			else:
				private_line = (
					f"在本次模拟中，你的出价不得高于 {buyer_max} {currency}。"
					"把它当作预算上限，不要在对话中提及或解释这个上限。"
				)

		buyer_max = int(buyer_max)
		last_seller = f"{int(so)} {currency}" if so is not None else "(none yet)"

		product_name = _get(buyer_context, "product_name", "Unknown")
		desc = _get(buyer_context, "seller_item_description", "")

		return f"""你是一名二手商品买家，正在与卖家协商单价。

【商品信息】
- 名称：{product_name}
- 描述：{desc}
- 标价：{init_price} {currency}/台

【约束】
{private_line}

【卖家当前报价】
{last_seller}

【谈判历史（仅包含数字出价，最近在下方）】
{history}

【策略建议（仅作参考，你可自行决定）】
- 先给出一个“偏低但合理”的锚定出价。
- 逐步让步，除非为了成交，不要出现过大的跳价。
- 如果你想“接受成交”，请把卖家上一轮的价格原样作为出价（数字完全一致）。
- 语气礼貌，但坚持自己的立场。

【输出格式要求】
只返回严格合法的 JSON（不要输出任何解释），schema 固定为：
{{"offers":[...]}}

其中 offers 必须只包含整数单价。
""".strip()

	def buyer_offer(
			self,
			state: np.ndarray,
			buyer_context: Any,
			*,
			buyer_max_override: Optional[int] = None,
	) -> Tuple[int, int]:
		prompt = self._buyer_strategy_prompt(state, buyer_context, k=1, buyer_max_override=buyer_max_override)

		true_buyer_reserve = _get(buyer_context, "buyer_reserve_price")
		buyer_max = buyer_max_override if buyer_max_override is not None else true_buyer_reserve
		if buyer_max is None:
			buyer_max = int(_get(buyer_context, "init_price"))
		buyer_max = int(buyer_max)

		for _ in range(3):
			out = self.buyer_llm.complete_text(prompt, max_tokens=64, temperature=0)
			obj = _extract_first_json(out)

			offer = None
			if isinstance(obj, dict) and isinstance(obj.get("offers"), list) and obj["offers"]:
				try:
					offer = int(obj["offers"][0])
				except Exception:
					offer = None
			if offer is None:
				nums = [int(x) for x in re.findall(r"\d+", str(out))]
				offer = nums[0] if nums else None

			if offer is not None and 1 <= int(offer) <= buyer_max:
				return (0, int(offer))

			prompt = prompt + f"\n\n上次输出不符合要求：出价必须是 <= {buyer_max} 的整数。请严格只输出 JSON。"

		# last-resort fallback (still within buyer_max)
		so = self.get_seller_offer(state)
		if so is not None and int(so) <= buyer_max:
			return (6, int(so))
		return (0, int(buyer_max))

	def buyer_candidate_offers(
			self,
			state: np.ndarray,
			buyer_context: Any,
			k: int = 5,
			*,
			buyer_max_override: Optional[int] = None,
	) -> List[Tuple[int, int]]:
		prompt = self._buyer_strategy_prompt(state, buyer_context, k=k, buyer_max_override=buyer_max_override)

		true_buyer_reserve = _get(buyer_context, "buyer_reserve_price")
		buyer_max = buyer_max_override if buyer_max_override is not None else true_buyer_reserve
		if buyer_max is None:
			buyer_max = int(_get(buyer_context, "init_price"))
		buyer_max = int(buyer_max)

		cleaned: List[int] = []
		seen = set()

		for _attempt in range(2):
			out = self.buyer_llm.complete_text(prompt, max_tokens=64, temperature=0)
			obj = _extract_first_json(out)

			offers: List[int] = []
			if isinstance(obj, dict) and isinstance(obj.get("offers"), list):
				for x in obj["offers"]:
					try:
						offers.append(int(x))
					except Exception:
						pass
			if not offers:
				offers = [int(x) for x in re.findall(r"\d+", str(out))]

			cleaned = []
			seen = set()
			for n in offers:
				if 1 <= int(n) <= buyer_max and int(n) not in seen:
					cleaned.append(int(n))
					seen.add(int(n))
				if len(cleaned) >= k:
					break

			if len(cleaned) >= max(1, k // 2):
				break

			prompt = prompt + f"\n\nProvide {k} valid offers <= {buyer_max} as JSON ONLY."

		acts = [(i, int(cleaned[i])) for i in range(len(cleaned[:k]))]

		so = self.get_seller_offer(state)
		if so is not None and int(so) <= buyer_max:
			acts.append((6, int(so)))

		return acts if acts else [(0, min(buyer_max, int(_get(buyer_context, "init_price"))))]

	# ----------------------------
	# Seller move generators
	# ----------------------------

	def _seller_ceiling_belief(self, state: np.ndarray, public_context: Any) -> int:
		init_price = int(_get(public_context, "init_price"))
		bo = self.get_buyer_offer(state)
		return int(max(init_price, bo if bo is not None else 0))

	def _seller_midpoint_target(self, seller_reserve: int, ceiling_belief: int) -> int:
		if ceiling_belief <= seller_reserve:
			return seller_reserve
		return int((seller_reserve + ceiling_belief) // 2)

	def average(self, state: np.ndarray, role_context: Any) -> List[Tuple[int, int]]:
		role = "seller" if int(state[0]) == 1 else "buyer"
		if role == "buyer":
			return [self.buyer_offer(state, role_context)]

		init_price = int(_get(role_context, "init_price"))
		currency = str(_get(role_context, "currency", "CNY"))
		seller_reserve = _get(role_context, "seller_reserve_price")
		if seller_reserve is None:
			raise ValueError("SellerContext must include seller_reserve_price")
		seller_reserve = int(seller_reserve)

		ceiling = self._seller_ceiling_belief(state, role_context)
		midpoint = self._seller_midpoint_target(seller_reserve, ceiling)

		so = self.get_seller_offer(state)
		bo = self.get_buyer_offer(state)

		prev = so if so is not None else max(init_price, seller_reserve)
		bound_low = max(midpoint, bo if bo is not None else 0, seller_reserve)
		bound_high = max(prev, bound_low)

		prompt = f"Return ONE integer price between {bound_low} and {bound_high} (inclusive). Output only the integer."
		out = self.seller_llm.complete_text(prompt, max_tokens=64, temperature=0)
		nums = [int(x) for x in re.findall(r"\d+", str(out))]
		proposal = nums[0] if nums else (bound_low + bound_high) // 2
		proposal = int(max(bound_low, min(bound_high, int(proposal))))

		midpoint_offer = int((bound_low + bound_high) // 2)
		midpoint_offer = int(max(bound_low, min(bound_high, midpoint_offer)))

		return [(0, proposal), (5, midpoint_offer)]

	def neural_valid_moves(
			self,
			state: np.ndarray,
			role_context: Any,
			*,
			buyer_max_override: Optional[int] = None,
	) -> List[Tuple[int, int]]:
		role = "seller" if int(state[0]) == 1 else "buyer"
		if role == "buyer":
			return self.buyer_candidate_offers(state, role_context, k=5, buyer_max_override=buyer_max_override)

		init_price = int(_get(role_context, "init_price"))
		seller_reserve = _get(role_context, "seller_reserve_price")
		product_name = _get(role_context, "product_name", "Unknown")
		desc = _get(role_context, "seller_item_description", "")
		history = self._history_lines(state)
		if seller_reserve is None:
			raise ValueError("SellerContext must include seller_reserve_price")
		seller_reserve = int(seller_reserve)

		ceiling = self._seller_ceiling_belief(state, role_context)
		midpoint = self._seller_midpoint_target(seller_reserve, ceiling)

		so = self.get_seller_offer(state)
		bo = self.get_buyer_offer(state)

		prev = so if so is not None else max(init_price, seller_reserve)
		bound_low = max(midpoint, bo if bo is not None else 0, seller_reserve)
		bound_high = max(prev, bound_low)

		last_buyer = self.get_buyer_offer(state)
		accept_clause = ""
		# Defaults keep the prompt robust even if seller ever moves first (or history is empty).
		ban_init_clause = ""
		structure_clause = ""
		if last_buyer is not None:
			accept_clause = f"- 如果你愿意接受买家出价，请把 {int(last_buyer)} 原样作为 5 个候选之一。"
			if int(init_price) != int(last_buyer):
				ban_init_clause = f"- 除非你要接受成交（即匹配买家最后一次出价），否则不要把标价 {int(init_price)} 放入候选列表。"
			structure_clause = f"""请严格按下面结构生成 **恰好 5 个** 候选报价（全部为整数），并且严格递减且互不相同：

1）明显让步：相比你上一轮卖家出价（{bound_high}）有清晰下降（不要太小）。
   - 让步幅度要明显（例如当前价位的 3%～6% 左右），但仍需高于你的底价。
   - 如果之前没有卖家出价，请给出一个高但可谈的价格（必须高于底价），不要围绕标价锚定。
2）中等让步：比（1）再降一档，幅度适中，仍偏卖家有利。
3）较大让步：比（2）明显更低，向成交更靠近，但不是“甩卖”。
4）强但可控的让步：本轮最低的还价，更接近成交，但不是“送”。
5）接受选项：
   - 如果你可以接受买家最后一次出价，请把它原样写入。
   - 否则就重复（4）。

硬性约束：
- 所有候选报价都必须 >= 你的底价。
- 让步要逐步进行：相邻候选之间不要出现过大的断崖式下降。
- 不要围绕标价做锚定。
""".strip()

		prompt = f"""你是一名二手商品卖家，正在与买家协商单价。

【商品信息】
- 名称：{product_name}
- 描述：{desc}
- 标价：{init_price} {currency}/台

【私密约束】
- 你的私密底价（最低可接受单价）是 {seller_reserve} {currency}。绝对不要低于该价格。
- 不要在对话中提及或暗示“底价/最低价”。

【谈判历史（仅包含数字出价，最近在下方）】
{history}

【任务】
生成“下一步卖家报价/还价”的候选单价，必须严格输出 5 个整数。

【规则】
- 每个候选报价必须落在 [{bound_low}, {bound_high}]（含端点）。
- 候选报价要像真人谈判一样逐步让步。
{ban_init_clause}
{structure_clause}
{accept_clause}

【输出格式】
只输出严格合法的 JSON，不要输出任何解释文字：{{"offers":[...]}}
""".strip()

		out = self.seller_llm.complete_text(prompt, max_tokens=64, temperature=0)
		obj = _extract_first_json(out)

		offers: List[int] = []
		if isinstance(obj, dict) and isinstance(obj.get("offers"), list):
			for x in obj["offers"]:
				try:
					offers.append(int(x))
				except Exception:
					pass
		if not offers:
			offers = [int(x) for x in re.findall(r"\d+", str(out))]
		if last_buyer is not None and int(init_price) != int(last_buyer):
			offers = [p for p in offers if int(p) != int(init_price)]
			if not offers:
				offers = [bound_high, (bound_low + bound_high) // 2, bound_low]
		cleaned: List[int] = []
		seen = set()
		for n in offers:
			n = int(n)
			if n < bound_low or n > bound_high:
				continue
			if n not in seen:
				cleaned.append(n)
				seen.add(n)
			if len(cleaned) >= 5:
				break

		if len(cleaned) < 5:
			mid = int((bound_low + bound_high) // 2)
			for n in [bound_high, mid, bound_low]:
				if len(cleaned) >= 5:
					break
				if n not in seen and bound_low <= n <= bound_high:
					cleaned.append(n)
					seen.add(n)

		moves = [(i, cleaned[i]) for i in range(len(cleaned))]
		moves.append((5, int((bound_low + bound_high) // 2)))
		if bo is not None:
			moves.append((6, int(bo)))
		return moves

	# --------------------------------------
	# Environment-only reward (training)
	# --------------------------------------

	def fairness_reward(self, state: np.ndarray, env_context: Any, player: int) -> float:
		bo = self.get_buyer_offer(state)
		so = self.get_seller_offer(state)

		if bo is None or so is None:
			return -0.01
		if int(bo) != int(so):
			return -0.01

		deal = float(so)
		br = float(_get(env_context, "buyer_reserve_price"))
		sr = float(_get(env_context, "seller_reserve_price"))

		if br <= sr:
			return -0.05
		if deal < sr or deal > br:
			return -0.05

		p1_share = (deal - sr) / (br - sr)

		if int(player) == 1:
			return float((1 - p1_share) if p1_share >= 0.5 else (-p1_share))
		return float((-p1_share) if p1_share >= 0.5 else (p1_share))
