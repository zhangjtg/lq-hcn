import numpy as np
from enum import Enum
class BuyerProfile(Enum):
	COMPROMISING = "easy"  # 妥协型 - p1
	EXPLORER = "linear"  # 探索型 - p2
	STUBBORN = "stud"  # 固执型 - p3
'''
def time_belief(t, t_max, alpha=0.5):
    t = np.minimum(t, t_max)
    return (1 - (t - 1) / (t_max - 1)) ** alpha
'''
def calculate_buyer_profile_type(buyer_first_bid,buyer_last_bid,user_history_price,current_concession):
	"""根据让步模式判断买家类型"""
	if (buyer_first_bid is None or buyer_last_bid is None or
			len(user_history_price) < 3):
		# 数据不足时返回默认类型
		return BuyerProfile.EXPLORER
	#if len(user_history_price) - 2 <= 0:  # Not enough history for average concession
		#historical_avg_concession = current_concession  # Or handle differently
	# 计算历史平均让步幅度: (p_b^(t-1) - p_b^(1)) / (t-2)
	concessions = [
		user_history_price[i-1] - user_history_price[i - 2]
		for i in range(2, len(user_history_price))
	]
	historical_avg_concession = sum(concessions) / len(concessions)

	# 计算当前预期让步幅度 (基于最近的让步趋势)
	# 妥协型判断: 历史平均让步 < 当前让步 (加速让步)
	if historical_avg_concession < current_concession:
		return BuyerProfile.COMPROMISING
	# 固执型判断: 让步幅度很小或负值
	elif historical_avg_concession > current_concession:#current_concession <= 0.02 * self.display_price:
		return BuyerProfile.STUBBORN
	else:
		return BuyerProfile.EXPLORER
'''
@dataclass
class NegotiationState:
	"""谈判状态数据类"""
	seller_price: float  # 卖家当前价格 P_s^(t-1)
	buyer_last_bid: Optional[float]  # 买家上次具体出价 P_b^(t-1)
	buyer_first_bid: Optional[float]  # 买家首次出价 P_b^(1)
	buyer_avg_concession: float  # 买家历史平均让步幅度 δ_u^(t-1)
	current_round: int  # 当前轮次 t
	negotiation_distance: float  # 谈判差距 θ_2
	time_factor: float  # 时间信念因子 bθ^c(t)
	buyer_profile: BuyerProfile  # 买家档案
	#utterance_type: str  # 模糊话语类型
'''

class FuzzyNegotiationSystem:
	"""模糊谈判推理系统"""

	def __init__(self, display_price: float):
		self.display_price = display_price
		self.default_pcm_base_ratio = 0.02  # 默认PCM基准为显示价格的2%
	def calculate_profile_factor(self, buyer_first_bid,buyer_last_bid,user_history_price,current_concession,negotiation_distance):
		"""计算买家档案因子 f_profile"""
		# 动态判断买家类型
		actual_profile = calculate_buyer_profile_type(buyer_first_bid,buyer_last_bid,user_history_price,current_concession)

		if actual_profile == BuyerProfile.COMPROMISING:
			# 妥协型：让步幅度高于平均，显示出加速让步的意愿
			# 基础因子较高，但会根据谈判进程调整
			base_factor = np.random.uniform(1.25, 1.55) #(1.1, 1.3)  #profile #parameter experiment

			# 如果已经让步较多，可能会减缓让步速度
			if buyer_first_bid and buyer_last_bid:
				total_concession_rate = (buyer_last_bid - buyer_first_bid) / buyer_first_bid
				if total_concession_rate > 0.2:  # 已让步超过20%
					base_factor *= 0.8  # 适当降低继续让步的幅度
			return base_factor

		elif actual_profile == BuyerProfile.EXPLORER:
			# 探索型：让步幅度接近平均，保持稳定的谈判节奏
			return np.random.uniform(0.80, 1.10)  #(0.9, 1.1) #parameter experiment

		elif actual_profile == BuyerProfile.STUBBORN:
			# 固执型：让步幅度远小于平均，抗拒让步
			base_factor = np.random.uniform(0.10, 0.45) #(0.3, 0.6) #parameter experiment
			# 如果谈判差距很大，固执型买家让步更少
			#relative_gap = abs(negotiation_distance) / self.display_price
			#if relative_gap > 0.3:
				#base_factor *= 0.5

			return base_factor

		return 1.0
	def calculate_gap_factor(self, negotiation_distance: float) -> float:
		"""计算谈判差距因子 f_gap"""
		relative_gap = abs(negotiation_distance) / self.display_price

		if relative_gap < 0.04: #g1  0.05:  # 差距很小 #parameter experiment
			return np.random.uniform(0.45, 0.75) #(0.6, 0.8)  # 较小的让步 #parameter experiment
		elif relative_gap < 0.12:#g2  0.15:  # 中等差距 #parameter experiment
			return np.random.uniform(0.95, 1.25) #(0.9, 1.1)  # 标准让步 #parameter experiment
		else:  # 差距很大
			return np.random.uniform(1.35,1.90)  #(1.2, 1.5)  # 需要更大让步 #parameter experiment

	def calculate_pcm_base(self, buyer_avg_concession) :
		"""计算PCM基准值"""
		if buyer_avg_concession > 0:
			# 如果有历史让步数据，使用历史平均
			return buyer_avg_concession
		else:
			# 否则使用默认启发值
			return self.default_pcm_base_ratio * self.display_price

	def calculate_pcm(self, buyer_avg_concession,buyer_first_bid,buyer_last_bid,user_history_price,current_concession,negotiation_distance,f_time) :
		"""计算预测让步幅度 PCM_t"""
		# 2. 计算基准PCM
		pcm_base = self.calculate_pcm_base(buyer_avg_concession)

		# 3. 计算各种调整因子
		f_profile = self.calculate_profile_factor(buyer_first_bid,buyer_last_bid,user_history_price,current_concession,negotiation_distance)
		f_gap = self.calculate_gap_factor(negotiation_distance)

		# 4. 综合计算PCM
		pcm_t = pcm_base * f_profile * f_time * f_gap #* f_utterance

		# 5. 应用合理性约束
		#max_reasonable_concession_abs  = 0.3 * self.display_price  # 最大30%的让步
		max_reasonable_concession_gap = (0.70 # lamada_1 #0.5
										 * abs(negotiation_distance)) if negotiation_distance != 0 else 0 #parameter experiment
		#pcm_t = min(pcm_t, max_reasonable_concession_abs, max_reasonable_concession_gap)
		pcm_t = np.clip(pcm_t, 0,max_reasonable_concession_gap)  # #parameter experiment

		return pcm_t

	def predict_buyer_offer(self, buyer_avg_concession,buyer_first_bid,buyer_last_bid,user_history_price,current_concession,negotiation_distance,time_factor,seller_last_price=0) -> float:
		"""预测买家的隐含新出价"""
		if buyer_last_bid ==0:
			# 有历史出价的情况
			initial_offer_estimate = self.display_price * np.random.uniform(0.18, 0.30)#u (0.3, 0.35) #parameter experiment
			# 确保不低于最低阈值
			#predicted_offer = max(initial_offer_estimate, 0.65 * self.display_price)
			#predicted_offer = min(predicted_offer, self.display_price)  # 不可能高于展示价
			return round(initial_offer_estimate,1)

		pcm_t = self.calculate_pcm(buyer_avg_concession, buyer_first_bid, buyer_last_bid, user_history_price,
									   current_concession, negotiation_distance, time_factor)
		predicted_offer = buyer_last_bid + pcm_t
		predicted_offer = max(predicted_offer, buyer_last_bid)  # 由 pcm_t >= 0 保证
		if seller_last_price is not None and seller_last_price >0:
			predicted_offer = min(predicted_offer, seller_last_price)  # 买方不太可能出价高于卖家要价
		# 确保预测价格在合理范围内
		#predicted_offer = max(predicted_offer, 0.5 * self.display_price)  #
		predicted_offer = min(predicted_offer, self.display_price)  # 不高于显示价格

		return round(predicted_offer,1)

'''
def demo_usage():
	"""使用示例"""
	# 初始化系统
	display_price = 1000.0
	system = FuzzyNegotiationSystem(display_price)

	# 场景1：妥协型买家询问"能再便宜点吗？"
	print("=== 场景1：妥协型买家一般性折扣询问 ===")
	state1 = NegotiationState(
		seller_price=950.0,
		buyer_last_bid=800.0,
		buyer_first_bid=750.0,  # 首次出价
		buyer_avg_concession=30.0,  # 当前让步幅度
		current_round=4,
		negotiation_distance=150.0,  # 950 - 800 当前轮次卖家与上一轮买家出价的差价
		time_factor=0.8,
		buyer_profile=BuyerProfile.COMPROMISING,
		#utterance_type=""
	)

	# 验证妥协型条件：历史平均让步 = (800-750)/(4-2) = 25 < 30 ✓
	historical_avg = (state1.buyer_last_bid - state1.buyer_first_bid) / (state1.current_round - 2)
	print(f"历史平均让步: {historical_avg:.1f}, 当前让步幅度: {state1.buyer_avg_concession:.1f}")
	print(f"妥协型条件: {historical_avg < state1.buyer_avg_concession} (历史平均 < 当前让步)")

	utterance1 = "能再便宜点吗？"
	predicted_offer1, details1 = system.predict_buyer_offer(state1, utterance1)

	print(f"买家话语: {utterance1}")
	print(f"预测新出价: {predicted_offer1:.2f}")
	print(f"预测让步幅度: {details1['pcm_t']:.2f}")
	print(f"推理详情: {details1['reasoning']}")
	print()

	# 场景2：探索型买家询问"买两件能优惠些吗？"
	print("=== 场景2：探索型买家批量折扣询问 ===")
	state2 = NegotiationState(
		seller_price=950.0,
		buyer_last_bid=850.0,
		buyer_first_bid=800.0,
		buyer_avg_concession=20.0,
		current_round=4,
		negotiation_distance=100.0,
		time_factor=1.0,
		buyer_profile=BuyerProfile.EXPLORER,
		utterance_type=""
	)

	# 验证探索型条件：历史平均让步 = (850-800)/(4-2) = 25 > 20
	historical_avg2 = (state2.buyer_last_bid - state2.buyer_first_bid) / (state2.current_round - 2)
	print(f"历史平均让步: {historical_avg2:.1f}, 当前让步幅度: {state2.buyer_avg_concession:.1f}")
	print(f"探索型特征: 稳定的让步节奏")

	utterance2 = "买两件能优惠些吗？"
	predicted_offer2, details2 = system.predict_buyer_offer(state2, utterance2)

	print(f"买家话语: {utterance2}")
	print(f"预测新出价: {predicted_offer2:.2f}")
	print(f"推理详情: {details2['reasoning']}")
	print()

	# 场景3：固执型买家询问最低价
	print("=== 场景3：固执型买家询问最低价 ===")
	state3 = NegotiationState(
		seller_price=900.0,
		buyer_last_bid=750.0,
		buyer_first_bid=730.0,
		buyer_avg_concession=5.0,  # 很小的让步
		current_round=5,
		negotiation_distance=150.0,
		time_factor=0.6,
		buyer_profile=BuyerProfile.STUBBORN,
		utterance_type=""
	)

	# 验证固执型条件：让步幅度很小
	historical_avg3 = (state3.buyer_last_bid - state3.buyer_first_bid) / (state3.current_round - 2)
	print(f"历史平均让步: {historical_avg3:.1f}, 当前让步幅度: {state3.buyer_avg_concession:.1f}")
	print(f"固执型特征: 让步幅度极小，抗拒让步")

	utterance3 = "这是你能给的最低价了吗？"
	predicted_offer3, details3 = system.predict_buyer_offer(state3, utterance3)

	print(f"买家话语: {utterance3}")
	print(f"预测新出价: {predicted_offer3:.2f}")
	print(f"各因子分解:")
	for factor, value in details3['factors'].items():
		if factor != 'utterance_type':
			print(f"  {factor}: {value:.3f}")
'''

'''
if __name__ == "__main__":
	demo_usage()
'''