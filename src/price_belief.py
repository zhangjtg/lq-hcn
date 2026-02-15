import numpy as np
class SellerPriceBelief:
	"""
	使用 Beta 分布为卖家 Agent 模拟价格信念的模型。

	该模型追踪卖家对最终成交价格在其报价区间内概率分布的认知。
	"""

	def __init__(self, p_min: float, p_max: float,  initial_alpha: float = 1.0, initial_beta: float = 1.0,learning_rate=0.1, confidence_decay=0.95):
		"""
		初始化价格信念模型。

		Args:
			p_min (float): 卖家的最低可接受售价（底线）。
			p_max (float): 卖家的初始报价或期望的最高售价。
			k (float): 敏感度因子，控制信念更新的幅度。
			initial_alpha (float, optional): 初始 alpha 参数。默认为 1.0。
			initial_beta (float, optional): 初始 beta 参数。默认为 1.0。
		"""
		if p_min >= p_max:
			raise ValueError("p_min必须小于p_max")

		self.p_min = p_min
		self.p_max = p_max
		#self.k = k
		self.alpha = initial_alpha #可以理解为"支持高价的虚拟观测次数"
		self.beta = initial_beta #可以理解为"支持低价的虚拟观测次数"
		self.learning_rate = learning_rate ## 适中的学习率
		self.confidence_decay = confidence_decay #缓慢的置信度衰减
		self.update_count = 0
	def _normalize_price(self, price: float) -> float:
		"""将实际价格归一化到 [0, 1] 区间。"""
		# 裁剪价格以确保它在定义的区间内
		clipped_price = np.clip(price, self.p_min, self.p_max)
		return (clipped_price - self.p_min) / (self.p_max - self.p_min)

	'''
	def _denormalize_price(self, x: float) -> float:
		"""将归一化的值转换回实际价格。"""
		return self.p_min + x * (self.p_max - self.p_min)
	
	def get_expected_price(self) -> float:
		"""
		计算当前信念下的期望成交价格。

		Returns:
			float: 当前的期望成交价。
		"""
		# Beta 分布的期望值是 alpha / (alpha + beta)
		expected_x = self.alpha / (self.alpha + self.beta)
		return self._denormalize_price(expected_x)
	
	def update(self, opponent_offer: float):
		"""
		根据对手（买家）的出价更新信念。

		Args:
			opponent_offer (float): 买家提出的新出价。
		"""
		print(f"\n--- 收到新出价: ${opponent_offer:.2f} ---")

		# 获取更新前的期望价格
		current_expected_x = self.alpha / (self.alpha + self.beta)

		# 归一化对手出价
		offer_x = self._normalize_price(opponent_offer)

		# 计算信念更新量
		delta_alpha = self.k * max(0, offer_x - current_expected_x)
		delta_beta = self.k * max(0, current_expected_x - offer_x)

		# 更新信念参数
		self.alpha += delta_alpha
		self.beta += delta_beta

		print(f"信念更新: Δα = {delta_alpha:.3f}, Δβ = {delta_beta:.3f}")
		print(f"当前信念参数: α = {self.alpha:.3f}, β = {self.beta:.3f}")
	'''

	def update_belief(self, observed_price, weight=1.0):
		"""
		更新价格信念

		Args:
			observed_price: 观测到的归一化价格 [0,1]
			weight: 观测权重，反映观测的重要性
		"""
		current_expectation = self.get_expected_price()

		# 计算偏差和更新强度
		deviation = observed_price - current_expectation
		update_strength = self.learning_rate * weight

		# 应用置信度衰减
		if self.update_count > 0:
			self.alpha *= self.confidence_decay
			self.beta *= self.confidence_decay

		# 基于偏差更新参数
		if deviation > 0:
			# 观测价格高于期望，增加α
			alpha_increment = update_strength * (1 + abs(deviation))
			self.alpha += alpha_increment
		else:
			# 观测价格低于期望，增加β
			beta_increment = update_strength * (1 + abs(deviation))
			self.beta += beta_increment

		# 确保参数不会过小
		self.alpha = max(self.alpha, 0.1)
		self.beta = max(self.beta, 0.1)

		self.update_count += 1

	def get_expected_price(self):
		"""获取期望价格"""
		return self.alpha / (self.alpha + self.beta)
	'''
	def get_variance(self):
		"""获取方差"""
		ab_sum = self.alpha + self.beta
		return (self.alpha * self.beta) / (ab_sum ** 2 * (ab_sum + 1))
	'''
	def get_state(self, buyer_offer):
		# 归一化对手出价
		offer_x = self._normalize_price(buyer_offer)
		prior_mean = self.get_expected_price()
		self.update_belief(offer_x)
		mean_x = self.get_expected_price()
		fluctuate_x = abs(offer_x - prior_mean)
		return  mean_x, fluctuate_x
	'''
	def plot_belief(self, title: str):
		"""
		可视化当前的价格信念分布。
		"""
		# 创建价格轴
		price_range = np.linspace(self.p_min, self.p_max, 500)
		# 将价格轴归一化
		x_range = self._normalize_price(price_range)

		# 计算 Beta 分布的概率密度函数 (PDF)
		pdf_values = beta.pdf(x_range, self.alpha, self.beta)

		# 绘制图形
		plt.figure(figsize=(10, 6))
		plt.plot(price_range, pdf_values, lw=2, label=f'Belief (α={self.alpha:.2f}, β={self.beta:.2f})')

		# 标记期望价格
		expected_price = self.get_expected_price()
		plt.axvline(expected_price, color='r', linestyle='--', label=f'Expected Price: ${expected_price:.2f}')

		plt.title(title)
		plt.xlabel("Price ($)")
		plt.ylabel("Probability Density")
		plt.legend()
		plt.grid(True, linestyle='--', alpha=0.6)
		plt.show()
	'''
'''
# --- 主程序：运行与之前例子完全相同的场景 ---
if __name__ == "__main__":
	# 1. 初始化场景：卖家小王卖二手笔记本
	print("场景: 卖家小王出售二手笔记本")
	p_min_seller = 400.0
	p_max_seller = 600.0
	sensitivity = 4.0

	# 创建小王的价格信念模型实例
	agent_belief = SellerPriceBelief(p_min=p_min_seller, p_max=p_max_seller, k=sensitivity)

	# 2. 初始信念 (t=0)
	print("\n--- t=0: 谈判开始前 ---")
	print(f"初始信念参数: α = {agent_belief.alpha:.3f}, β = {agent_belief.beta:.3f}")
	print(f"初始期望价格: ${agent_belief.get_expected_price():.2f}")
	agent_belief.plot_belief("t=0: Initial Belief (Uniform Distribution)")

	# 3. 第一次更新 (t=1)
	# 收到买家首次出价: $420
	first_offer = 420.0
	agent_belief.update(first_offer)
	print(f"更新后期望价格: ${agent_belief.get_expected_price():.2f}")
	agent_belief.plot_belief("t=1: Belief After First Offer ($420)")

	# 4. 第二次更新 (t=2)
	# 经过一番还价，收到买家第二次出价: $480
	second_offer = 480.0
	agent_belief.update(second_offer)
	print(f"更新后期望价格: ${agent_belief.get_expected_price():.2f}")
	agent_belief.plot_belief("t=2: Belief After Second Offer ($480)")
'''