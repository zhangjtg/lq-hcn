import gym
from gym import spaces
import numpy as np
from collections import deque
import re,torch
from  predict_buyer_price import  FuzzyNegotiationSystem,calculate_buyer_profile_type,BuyerProfile
from transformers import AutoTokenizer, AutoModel,AutoModelForCausalLM,AutoModelForSequenceClassification,BertTokenizer,StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
from price_belief import SellerPriceBelief
from wechat_service import WeChatService
from threading import Lock, Condition
import math
import zhconv
# --- Define State Space Dimensions and Normalization Constants ---
HISTORY_EXCHANGES_N = 3  # Number of past (Agent Offer, Opponent Offer) exchanges
# Each exchange: agent_unit_price, agent_assumed_qty, opponent_unit_price, opponent_proposed_qty, opponent_offer_fuzzy
FEATURES_PER_EXCHANGE = 5  # Added opponent_offer_fuzzy flag
HISTORY_DIM = HISTORY_EXCHANGES_N * FEATURES_PER_EXCHANGE

# Other state components dimensions
CURRENT_OPPONENT_OFFER_DIM = 3  # unit_price, quantity, is_fuzzy_offer
#AGENT_LAST_OFFER_DIM = 2  # unit_price, assumed_quantity for that price
TIME_BELIEF_DIM = 1
PRICE_BELIEF_DIM = 2  # For agent's belief about final unit price distribution
USER_BEHAVIOR_DIM = 3  # �O�(EasyToCompromise),  ,��(Normal), }�'(Stubborn)

# Total Observation Dimension
OBSERVATION_DIM = (HISTORY_DIM +
				   CURRENT_OPPONENT_OFFER_DIM +
				   #AGENT_LAST_OFFER_DIM +
				   TIME_BELIEF_DIM +
				   PRICE_BELIEF_DIM +
				   USER_BEHAVIOR_DIM)

PADDING_NORMALIZED_VALUE = -1  # For missing history or uninitialized values (if normalizing to [0,1])

# Action Space (Agent primarily controls unit price adjustment)
NUM_DISCRETE_ACTIONS = 3  # Propose/Increase Price, Counter Price, Accept
PARAM_MIN_NORMALIZED = 0  # For price adjustment parameter
PARAM_MAX_NORMALIZED = 1.0

def denormalize_price(P_norm, P_min, P_max):
    """
    将归一化价格转换回实际价格
    :param P_norm: 归一化价格，标量或数组，范围[0,1]
    :param P_min: 价格区间最小值
    :param P_max: 价格区间最大值
    :return: 实际价格
    """
    P_norm = np.array(P_norm)
    P = P_norm * (P_max - P_min) + P_min
    return P
def norm_quantity(q, scale=5.0,default_on_none=PADDING_NORMALIZED_VALUE):
	if q is None:
		return default_on_none
	return 1 - math.exp(-q / scale)
def normalize_feature(value, min_val, max_val, clip_value=True, default_on_none=PADDING_NORMALIZED_VALUE):
	"""
	Normalizes a feature to the [0,1] range.
	Includes optional clipping for values outside [min_val, max_val].
	"""
	if value is None:
		return default_on_none
	if max_val == min_val:  # Avoid division by zero
		return 0.5 if value == min_val else default_on_none

	value_to_normalize = value
	if clip_value:  # Clip the value to the defined min/max range before normalization
		value_to_normalize = np.clip(value, min_val, max_val)

	# If not clipping, values outside [min_val, max_val] will result in normalized values outside [0,1]
	return (value_to_normalize - min_val) / (max_val - min_val)

class StopOnWords(StoppingCriteria):
    def __init__(self, tokenizer, stop_words, device="cuda"):
        super().__init__()
        # 将停止词（字符串）转换为它们对应的token ID
        # 注意：一个词可能被分解为多个token，所以结果是一个列表的列表
        self.tokenizer = tokenizer
        self.stop_token_ids = [
            tokenizer.encode(word, add_special_tokens=False, return_tensors='pt').to(device)
            for word in stop_words
        ]
        #print(f"停止词的Token IDs: {[ids[0].tolist() for ids in self.stop_token_ids]}")

    # __call__ 方法在模型每生成一个新token后被自动调用
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # input_ids 是到目前为止生成的所有token的ID
        # 遍历我们定义的所有停止词token序列
        for stop_ids in self.stop_token_ids:
            # 检查当前生成的文本的末尾部分是否与某个停止词序列完全匹配
            # .shape[1] 是序列的长度
            if input_ids.shape[1] >= stop_ids.shape[1]:
                # 获取生成文本的尾部，长度与停止词序列相同
                tail = input_ids[0, -stop_ids.shape[1]:]
                if torch.equal(tail, stop_ids[0]):
                    print(f"\n[INFO] 触发停止词: '{ self.tokenizer.decode(tail)}'")
                    return True # 如果匹配，返回True，模型将停止生成
        return False # 如果没有匹配，返回False，模型继续生成

class NegotiationEnv(gym.Env):
	metadata = {'render.modes': ['human', 'ansi']}

	def __init__(self, env_config,max_turn,type, wechat_service: WeChatService):
		super(NegotiationEnv, self).__init__()
		cfg = env_config if env_config is not None else {}
		self.agent_is_seller = cfg.get("agent_is_seller", True)  # True if agent is seller, False if buyer
		#self.dataset = dataset
		self.max_turn = max_turn
		self.type = type

		self.fuzzyNeg =None
		self.deal_base_reward = 3  # 达成交易的基础奖励
		#self.gamma = 1  # 交易速度奖励的折扣系数
		self.time_penalty = 0.05  # 时间惩罚
		self.price_change_threshold_ratio = 0.2 # 价格相对变化比例阈值，
		self.extreme_penalty = 1 #  # 极端让步惩罚
		self.price_tokenizer = AutoTokenizer.from_pretrained('../output/price_model/Qwen2p5-1p5B-instruct-a5-l4-618')
		self.price_model = AutoModelForCausalLM.from_pretrained(
			'/home/zjt/local/model_bin/Qwen2.5-1.5B-Instruct',
			device_map="auto",
			torch_dtype=torch.bfloat16
		)
		self.price_model = PeftModel.from_pretrained(self.price_model,
													 '../output/price_model/Qwen2p5-1p5B-instruct-a5-l4-618')
		self.price_model = self.price_model.merge_and_unload()  # 合并适配器
		self.price_model.eval()  # 设置为评估模式

		self.intent_tokenizer =  AutoTokenizer.from_pretrained('/home/zjt/local/model_bin/TinyLlama_v1.1_chinese')
		self.intent_model = AutoModelForSequenceClassification.from_pretrained(
			'/home/zjt/local/model_bin/TinyLlama_v1.1_chinese',
			num_labels=6, torch_dtype=torch.bfloat16
		).half().cuda()
		self.intent_model = PeftModel.from_pretrained(self.intent_model, '../output/intent/TinyLlama_v1.1_chinese-classification83')

		self.agent_tokenizer = AutoTokenizer.from_pretrained("/home/zjt/local/model_bin/Qwen3-4B-Instruct-2507",torch_dtype=torch.bfloat16)
		self.agent_model = AutoModelForCausalLM.from_pretrained("/home/zjt/local/model_bin/Qwen3-4B-Instruct-2507",torch_dtype=torch.bfloat16).half().cuda()
		self.agent_model.eval()
		self.num_user_behavior_types = USER_BEHAVIOR_DIM #cfg.get("num_user_behavior_types", USER_BEHAVIOR_DIM)
		self.observation_space = spaces.Box(low=-0.1, high=1.1,  # Allow slight overshoot for padding, most in [0,1]
											shape=(OBSERVATION_DIM,), dtype=np.float32)

		# Action space definition
		action_space_elements = [spaces.Discrete(NUM_DISCRETE_ACTIONS)]
		action_space_elements.append(
			spaces.Box(low=PARAM_MIN_NORMALIZED, high=PARAM_MAX_NORMALIZED, shape=(1,), dtype=np.float32))  # 1st param
		action_space_elements.append(
			spaces.Box(low=np.array([], dtype=np.float32), high=np.array([], dtype=np.float32), shape=(0,),
					   dtype=np.float32))

		action_space_elements.append(
			spaces.Box(low=np.array([], dtype=np.float32), high=np.array([], dtype=np.float32), shape=(0,),
					   dtype=np.float32))  # Additional param placeholders

		self.action_space = spaces.Tuple(tuple(action_space_elements))

		# Internal state variables
		self.offer_history = deque(maxlen=HISTORY_EXCHANGES_N)
		self.reward_p = 4
		self.reward_q = 2
		self.reward_t = 1
		self.time_w =  0.7
		self.current_opponent_unit_price = None
		self.current_opponent_quantity = None
		self.agent_last_unit_price = None
		self.agent_last_assumed_quantity = None  # Quantity agent's last price was based on
		self.user_history_price = None
		self.agent_history_price = None
		self.list_price = None
		self.user_behavior_type = None  # Integer index for user behavior

		# Time belief indicating probability of acceptance
		self.agent_reservation_price = None
		self.buyer_reserve_price = None
		self.max_consecutive_fuzzy_offers = cfg.get("max_consecutive_fuzzy_offers", 3)
		self.wechat_service = wechat_service
		stop_words = ["买家:", "\n买家:"]
		stop_on_words = StopOnWords(self.agent_tokenizer, stop_words, device="cuda")
		self.agent_stopping_criteria = StoppingCriteriaList([stop_on_words])


	def _get_encoded_history(self):
		encoded_history = np.full(HISTORY_DIM, PADDING_NORMALIZED_VALUE, dtype=np.float32)
		history_entries = list(self.offer_history)  # Get a list copy to iterate from oldest to newest
		num_entries_to_encode = len(history_entries)
		start_vector_idx = HISTORY_DIM - (num_entries_to_encode * FEATURES_PER_EXCHANGE)

		current_vector_ptr = start_vector_idx
		for i in range(num_entries_to_encode):
			entry = history_entries[i]  # Iterates from older to newer among the selected N
			op_fuzzy,op_price, op_qty, ag_price, ag_qty  = entry
			encoded_history[current_vector_ptr + 0] = float(op_fuzzy)  # 1.0 if fuzzy, 0.0 if specific
			encoded_history[current_vector_ptr + 1] = normalize_feature(op_price, self.agent_reservation_price,
																		self.list_price,clip_value=False)
			encoded_history[current_vector_ptr + 2] = norm_quantity(op_qty)
			encoded_history[current_vector_ptr + 3] = normalize_feature(ag_price, self.agent_reservation_price,
																		self.list_price)
			encoded_history[current_vector_ptr + 4] = norm_quantity(ag_qty)

			current_vector_ptr += FEATURES_PER_EXCHANGE

		return encoded_history

	def _get_state(self):
		history_component_vec = self._get_encoded_history()

		norm_curr_opp_price = normalize_feature(self.current_opponent_unit_price, self.agent_reservation_price,
												self.list_price,clip_value=False)
		norm_curr_opp_qty = norm_quantity(self.current_opponent_quantity)
		curr_opp_fuzzy = np.float32(float(self.current_opponent_offer_is_fuzzy))
		current_opponent_offer_vec = np.array([curr_opp_fuzzy,norm_curr_opp_price, norm_curr_opp_qty],
											  dtype=np.float32)

		self._update_time_belief()
		time_belief_val = np.array([self.time_belief], dtype=np.float32)
		if self.current_opponent_unit_price is not None:
			price_belief_mean,price_belief_variance  = self.price_belief.get_state(self.current_opponent_unit_price)
		else:
			price_belief_mean, price_belief_variance = 0.5, 0.0
		price_belief_component = np.array([price_belief_mean, price_belief_variance], dtype=np.float32)
		if len(self.user_history_price) >= 3:
			current_concession =  self.user_history_price[-1] - self.user_history_price[-2]
			self.user_behavior_type = calculate_buyer_profile_type(self.user_history_price[0], self.user_history_price[-2], self.user_history_price, current_concession)
			if self.user_behavior_type  == BuyerProfile.COMPROMISING:
				self.user_behavior_type = 0
			elif self.user_behavior_type  == BuyerProfile.EXPLORER:
				self.user_behavior_type = 1
			elif self.user_behavior_type == BuyerProfile.STUBBORN:
				self.user_behavior_type = 2
		else:
			self.user_behavior_type = 1
		user_behavior_vec = np.zeros(self.num_user_behavior_types, dtype=np.float32)
		if self.user_behavior_type is not None:  # Should always be set in reset
			user_behavior_vec[self.user_behavior_type] = 1.0

		state_parts = [
			history_component_vec,current_opponent_offer_vec,
			time_belief_val, price_belief_component, user_behavior_vec
		]
		# Filter out empty arrays if some features are disabled (e.g. price_belief_component)
		state_parts_filtered = [part for part in state_parts if part.size > 0]

		state = np.concatenate(state_parts_filtered).astype(np.float32)

		# Safety check for dimension
		if state.shape[0] != self.observation_space.shape[0]:
			raise ValueError(
				f"Constructed state dimension {state.shape[0]} does not match observation_space dimension {self.observation_space.shape[0]}")
		return state

	def _update_time_belief(self,alpha=1.5):
		"""
		Update the time belief based on negotiation progress.
        This represents agent's belief about probability of acceptance at current stage.
        """
        # Basic time decay - acceptance gets more likely as time passes (urgency increases)
		self.time_belief=(self.current_round/ self.max_turn) ** alpha
		# Adjust based on how close the last offers were
		if self.agent_last_unit_price is not None and self.current_opponent_unit_price is not None:
			price_gap = abs(self.agent_last_unit_price - self.current_opponent_unit_price)
			price_range = self.list_price - self.agent_reservation_price
			if price_range > 0:
				normalized_gap = price_gap / price_range
				# The smaller the gap, the higher the acceptance probability
				normalized_gap = np.clip(normalized_gap,0,1)
				gap_factor = 1.0 - normalized_gap
				self.time_belief = self.time_w * self.time_belief + (1-self.time_w) * gap_factor
		# Ensure time_belief stays in [0,1]
		self.time_belief = np.clip(self.time_belief, 0.0, 1.0)

	def step(self, action):
		info = {}
		done = False
		reward = 0.0
		action_type, price_param = action
		agent_unit_price_proposal = None
		pre_seller_price_proposal = self.agent_last_unit_price if self.agent_last_unit_price is not None else self.list_price
		agent_assumed_quantity_for_proposal = self.current_opponent_quantity if self.current_opponent_quantity is not None else 1
		if action_type == -100:
			if price_param==1:
				buyer_intent='问候'
			elif price_param==4:
				buyer_intent = '材料咨询'
			elif price_param==5:
				buyer_intent = '物流咨询'
			non_bargaining_messages = non_bargaining_CBMessages(self.case,self.conversation,intent=buyer_intent)
			seller_llm_response_text = self.generate_response(messages=non_bargaining_messages)
			seller_llm_response_text = process_seller_response(seller_llm_response_text)
			self.conversation.append({"role": '卖家', "content": re.sub(r"^卖家[:：]", "", seller_llm_response_text)})
			if self.current_round >= self.max_turn and not done:
				done = True
			self._send_structured_message(seller_llm_response_text, done, deal_price=None)

		else:
			if  self.current_opponent_offer_is_fuzzy==0 and self.current_opponent_unit_price is not None:
				self.pre_user_not_fuzzy_price = self.current_opponent_unit_price
				self.pre_user_offer_not_fuzzy = self.current_opponent_offer_is_fuzzy
			pre_user_price = self.current_opponent_unit_price
			pre_user_qty =  self.current_opponent_quantity if self.current_opponent_quantity is not None else 1
			seller_llm_response_text = ""  # 存储代理LLM生成的回复
			# Process different action types
			if action_type == 0:  #  Counter Price
				agent_unit_price_proposal = denormalize_price(P_norm=price_param[0], P_min=self.agent_reservation_price, P_max=self.list_price)
				agent_unit_price_proposal = round(agent_unit_price_proposal,1)
				# Ensure price is within allowed range
				agent_unit_price_proposal = np.clip(agent_unit_price_proposal, self.agent_reservation_price, self.list_price)
				# Assume quantity based on opponent's last quantity or a default
				self.agent_history_price.append(agent_unit_price_proposal)
				messages = CBMessages(self.case,'seller',self.conversation,action=0,price=agent_unit_price_proposal,
									  quantity=agent_assumed_quantity_for_proposal,buyer_bid=pre_user_price,time_pressure=self.time_belief)
				seller_llm_response_text = self.generate_response( messages=messages)
				seller_llm_response_text = process_seller_response(seller_llm_response_text)

			elif action_type == 1:  # insist Price
				# Similar to Propose but focuses on countering the opponent's last offer
				if self.agent_last_unit_price is None:
					agent_unit_price_proposal = self.list_price
				else:
					agent_unit_price_proposal = self.agent_last_unit_price
				self.agent_history_price.append(agent_unit_price_proposal)
				messages = CBMessages(self.case, 'seller', self.conversation, action=1,
									  price=agent_unit_price_proposal ,quantity=agent_assumed_quantity_for_proposal,
									  buyer_bid=pre_user_price,time_pressure=self.time_belief)
				seller_llm_response_text = self.generate_response(messages=messages)
				seller_llm_response_text = process_seller_response(seller_llm_response_text)

			elif action_type == 2:  # Accept current offer
				if self.current_opponent_unit_price is not None and  self.current_opponent_offer_is_fuzzy==0:
					# Check if the price is acceptable to agent
					acceptable_price = self.current_opponent_unit_price
					agent_unit_price_proposal = acceptable_price
					self.agent_history_price.append(agent_unit_price_proposal)
					done = True
					if acceptable_price >= self.agent_reservation_price:
						reward += self._calculate_deal_reward(acceptable_price,agent_assumed_quantity_for_proposal)
						info["deal_made"] = True
						info["deal_price"] = acceptable_price
						info["deal_quantity"] = agent_assumed_quantity_for_proposal

					else:
						# 错误地接受了低于底价的报价
						print(f"致命错误：智能体接受了低于底价的价格 {acceptable_price} (底价: {self.agent_reservation_price})")
						penalty_score = self._calculate_violation_penalty(acceptable_price,agent_assumed_quantity_for_proposal)
						reward += -10.0 + penalty_score  # 应用基础惩罚和距离惩罚
						info["error"] = "Agent accepted an offer below its reservation price."
						info["violation"] = True
					#seller_llm_response_text = f"好的，成交，单价为{acceptable_price}，数量:{agent_assumed_quantity_for_proposal}"
					messages = CBMessages(self.case, 'seller', self.conversation, action=2,price=acceptable_price,quantity=agent_assumed_quantity_for_proposal,
										  buyer_bid=pre_user_price)
					seller_llm_response_text = self.generate_response(messages=messages)
					seller_llm_response_text = process_seller_response(seller_llm_response_text)

				else :# 无法接受模糊或不存在的报价
					print("致命错误：智能体尝试接受一个模糊或不存在的报价。")
					done = True
					reward -= 10.0  # 惩罚无效动作
					info["error"] = "Agent tried to accept a fuzzy or non-existent offer."
					info["violation"] = True
					seller_llm_response_text = "抱歉，本次商议先到这里吧。"
					self.agent_history_price.append(self.current_opponent_unit_price)#self.agent_last_unit_price

			# 更新代理上一次的提议，用于下一轮或计算奖励
			if agent_unit_price_proposal is not None:
				self.agent_last_unit_price = agent_unit_price_proposal
			if agent_assumed_quantity_for_proposal is not None:
				self.agent_last_assumed_quantity = agent_assumed_quantity_for_proposal
			if pre_user_price is not None or self.agent_last_unit_price is not None:  # 只有当代理有明确动作时才记录
				self.offer_history.append((
					self.current_opponent_offer_is_fuzzy,
					pre_user_price, pre_user_qty,
					self.agent_last_unit_price, self.agent_last_assumed_quantity
				))
			user_counter_price = None
			user_counter_quantity = None
			# 检查是否达到最大回合数
			if self.current_round >= self.max_turn and not done:
				done = True
				info["max_steps_reached"] = True
				reward -= 3  # 未在规定回合内完成交易的惩罚
				self.conversation.append({"role": '卖家', "content": "时间到了，我们今天就先谈到这里吧。"})
				self._send_structured_message("时间到了，我们今天就先谈到这里吧。", done=True, deal_price=None)
			else:
				# 将代理(卖家)的回复添加到对话历史并发送给小程序用户
				if len(seller_llm_response_text)>3:
					self.conversation.append({"role": '卖家', "content": re.sub(r"^卖家[:：]", "", seller_llm_response_text)})
					seller_acceptable_price = agent_unit_price_proposal if done else None
					self._send_structured_message(re.sub(r"^卖家[:：]", "", seller_llm_response_text), done, deal_price=seller_acceptable_price)
				else:
					if done:
						seller_fail_send_text = f"""我同意以总价{self.agent_last_unit_price * agent_assumed_quantity_for_proposal}成交"""
					else:
						seller_fail_send_text = f"""价格低了，给你总计{self.agent_last_unit_price * agent_assumed_quantity_for_proposal}吧"""
					self.conversation.append({"role": '卖家', "content":seller_fail_send_text})
					seller_acceptable_price = agent_unit_price_proposal if done else None
					self._send_structured_message(seller_fail_send_text, done,deal_price=seller_acceptable_price)

		# --- Simulate user reaction based on agent's action ---
		intent_code = None
		if not done:  # 如果代理的动作没有直接结束对话
			self.current_round += 1
			user_reply = self.wechat_service.wait_for_user_message(self.current_wechat_user_id)
			if user_reply is None:
				# 处理超时 - 用户未响应
				print(f"[ENV] 用户 {self.current_wechat_user_id} 响应超时。结束当前回合。")
				reward -= 0
				done = True
				info["user_timeout"] = True
				return self._get_state(), reward, done, info
			self.conversation.append({"role": "买家", "content": user_reply})
			user_counter_price, user_counter_quantity, intent_code = self._get_human_user_reaction(agent_unit_price_proposal,agent_assumed_quantity_for_proposal,user_reply)
			if user_counter_price is not None:
				self.user_history_price.append(user_counter_price)
			if intent_code == 3:  # 人类用户接受了代理的提议
				if agent_unit_price_proposal is not None:
					reward += self._calculate_deal_reward(agent_unit_price_proposal, agent_assumed_quantity_for_proposal,self.pre_user_offer_not_fuzzy)
					info["deal_made"] = True
					info["deal_price"] = agent_unit_price_proposal
					info["deal_quantity"] = agent_assumed_quantity_for_proposal
					done = True
					seller_response_text = f"好的，我们就定在 ¥{agent_unit_price_proposal:.2f} 吧！成交！"
					self._send_structured_message(seller_response_text, done=True, deal_price=agent_unit_price_proposal)
				else:  # 用户接受了，但代理本轮没有明确出价 (例如代理尝试接受但失败了)
					reward += self._calculate_deal_reward(pre_seller_price_proposal, agent_assumed_quantity_for_proposal)
					info["deal_made"] = True
					info["deal_price"] = pre_seller_price_proposal
					info["deal_quantity"] = agent_assumed_quantity_for_proposal
					done = True  # 结束这个混乱的回合
					seller_response_text = f"好的，我们就定在 ¥{pre_seller_price_proposal:.2f} 吧！成交！"
					self._send_structured_message(seller_response_text, done=True, deal_price=pre_seller_price_proposal)
					#info["error"] = "User accepted, but agent had no clear proposal this turn."
			elif intent_code == 2:  # 人类用户拒绝并不再议价
				reward -= 3.0  # 较大惩罚，因为谈判失败
				info["user_rejected_deal_explicitly"] = True
				done = True
				seller_response_text = "明白了。很遗憾没能达成协议，下次再见！"
				self._send_structured_message(seller_response_text, done=True, deal_price=None)
			elif intent_code == 1 or  intent_code == 4 or intent_code == 5:
				if self.current_opponent_unit_price is None:
					self.current_opponent_quantity = None
					self.current_opponent_offer_is_fuzzy = -1
			else:  # intent_code == 0 (用户还价或继续商议)
				if agent_unit_price_proposal is not None:  # 如果代理本轮出价了
					reward += self._calculate_step_reward(agent_unit_price_proposal,self.pre_user_offer_not_fuzzy)
				else:  # 代理本轮未出价 (例如尝试接受但失败)
					reward -= 0  # 轻微惩罚
		# 更新对手的当前报价状态
		if user_counter_price is not None:
			self.current_opponent_unit_price = user_counter_price
		if user_counter_quantity is not None:
			self.current_opponent_quantity = user_counter_quantity
		# 如果用户没有提供数量，则沿用上一次的数量
		elif self.current_opponent_quantity is None and self.current_opponent_unit_price is not None:
			self.current_opponent_quantity = 1

		if self.consecutive_fuzzy_offers >= self.max_consecutive_fuzzy_offers and not done:
			info["fuzzy_deadlock"] = True
			reward -= 1.5
			done = True
			seller_response_text = "由于多次模糊出价，商议未能达成明确结果，我们下次再议吧。"
			self._send_structured_message(seller_response_text, done=True, deal_price=None)

		next_state = self._get_state()
		return next_state, reward, done, info,intent_code

	def reset(self,case,user_id):
		"""Reset the environment for a new episode"""
		self.current_round = 0
		self.agent_history_price = []
		self.user_history_price = []
		self.user_history_fuzzy_offers = []
		self.offer_history.clear()
		self.consecutive_fuzzy_offers = 0
		self.time_belief = 1.0
		self.current_opponent_unit_price = None
		self.current_opponent_quantity = None
		self.current_opponent_offer_is_fuzzy = 1
		# Reset agent's last offer
		self.agent_last_unit_price = None
		self.agent_last_assumed_quantity = None
		# Initialize with random user behavior type
		self.user_behavior_type = 1  # 一般线型 -
		self.case = case
		self.current_wechat_user_id = user_id
		self.conversation = []
		self.pre_user_not_fuzzy_price = None
		self.pre_user_offer_not_fuzzy = 1
		self.κ1_t = 0.5

		# Reset user prices based on behavior type
		if self.agent_is_seller:  # Agent is seller, user is buyer
			# More stubborn users have lower reservation prices
			self.buyer_reserve_price = round(self.case.get("buyer_reserve_price"),1)
			self.agent_reservation_price = round(self.case.get("seller_reserve_price"),1)
			self.list_price = round(self.case['init_price'],1)
		else:  # Agent is buyer, user is seller
			# More stubborn users have higher reservation prices
			raise ValueError('agent role must be seller')

		self.fuzzyNeg = FuzzyNegotiationSystem(self.case['init_price'])
		self.price_belief = SellerPriceBelief(p_min=self.agent_reservation_price, p_max=self.case['init_price'],
											  learning_rate=0.15, confidence_decay=0.98)

		# 等待用户第一条消息
		user_first_msg = self.wechat_service.wait_for_user_message(self.current_wechat_user_id)

		if user_first_msg is None:
			self._send_structured_message("由于您长时间未响应，本次会话已自动结束。", done=True)
			return None,None  # 超时处理
		# 处理用户第一条消息
		self.agent_last_unit_price = self.list_price
		self.agent_last_assumed_quantity = 1  # 默认为最小数量
		self.agent_history_price.append(self.list_price)
		self.conversation.append({"role": "买家", "content": user_first_msg})
		self.current_round += 1
		first_user_price, first_user_qty, first_intent = self._get_human_user_reaction(
			self.agent_last_unit_price, self.agent_last_assumed_quantity,user_first_msg
		)
		if first_intent == 2 and first_user_price is None:  # 用户直接拒绝且未出价
			self._send_structured_message("明白了。很遗憾没能达成协议，下次再见！", done=True)
			return None,first_intent
		elif first_intent==1 or first_intent== 4 or first_intent==5:
			self.current_opponent_unit_price = None
			self.current_opponent_quantity = None
			self.current_opponent_offer_is_fuzzy = -1
		else:
			self.current_opponent_unit_price = first_user_price
			self.current_opponent_quantity = first_user_qty if first_user_qty is not None else 1  # 默认数量
			if self.current_opponent_unit_price is not None:
				self.user_history_price.append(self.current_opponent_unit_price)

		return self._get_state(),first_intent

	def _get_human_user_reaction(self,agent_price,agent_qty,human_response_text):
		label2id = {'讨价还价': 0, '问候': 1, '谈判失败': 2, '谈判成功': 3, '材料咨询': 4, '物流咨询': 5}
		id2label = {idx: label for label, idx in label2id.items()}
		human_response_text_remove_stop = preprocess_text_for_negotiation(human_response_text)
		intent_input_ids = self.intent_tokenizer(human_response_text_remove_stop, truncation=True, return_tensors="pt").to(self.intent_model.device)
		with torch.no_grad():
			output_ids = self.intent_model(**intent_input_ids)
		# 获取预测的logits
		logits = output_ids.logits
		# 获取类别预测（最大logit对应的类别）
		prediction = torch.argmax(logits, dim=-1)
		pred_value = prediction.item()  # 提取原始预测值

		user_counter_price = None
		user_new_quantity = None  # 默认为None，如果用户未提及则保持上一次的数量或默认值
		#intent_code = 0  # 默认为 "报价" (继续商议)
		if id2label[pred_value] == '问候' :
			intent_code = 1
			return None,None,intent_code
		elif  id2label[pred_value] == '材料咨询':
			intent_code = 4
			return None, None, intent_code
		elif  id2label[pred_value] == '物流咨询':
			intent_code = 5
			return None, None, intent_code
		elif id2label[pred_value] =='谈判成功':
			intent_code = 3
			user_counter_price = agent_price#self.agent_last_unit_price  # 用户同意了代理上一次的报价
			user_new_quantity = agent_qty#self.agent_last_assumed_quantity
		elif id2label[pred_value] =='谈判失败':
			intent_code = 2

		else:  # "报价" 或其他继续议价的情况
			intent_code = 0
			# 提取用户报价中的具体价格和数量
			# tran_resp_for_extraction 应该和 tran_resp_for_intent 相同
			tran_resp_for_intent = translate_response(self.conversation)  # tran_resp现在包含用户的最新回复
			extracted_price, extracted_quantity = self.extra_price_quantity(tran_resp_for_intent)
			if extracted_price != 'null':
				user_counter_price = round(float(extracted_price),1)
				self.current_opponent_offer_is_fuzzy = 0
				self.consecutive_fuzzy_offers = 0
				if agent_price is not None and user_counter_price==round(agent_price,1):
					intent_code = 3
			else:
				self.current_opponent_offer_is_fuzzy = 1
				self.consecutive_fuzzy_offers += 1
				avg_concession = self.calculate_average_concession()
				buyer_first_b = self.user_history_price[0] if self.user_history_price else 0
				buyer_last_b = self.user_history_price[-1] if self.user_history_price else 0
				current_concession = (self.user_history_price[-1] - self.user_history_price[-2]) if len(
					self.user_history_price) >= 2 else 0
				neg_dist = (self.agent_last_unit_price - buyer_last_b) if self.agent_last_unit_price is not None else 0
				if self.fuzzyNeg is not None:
					f_time=self._calculate_kappa_1()# 确保 fuzzyNeg 已初始化
					user_counter_price = self.fuzzyNeg.predict_buyer_offer(
						buyer_avg_concession=avg_concession,
						buyer_first_bid=buyer_first_b,
						buyer_last_bid=buyer_last_b,
						user_history_price=self.user_history_price,
						current_concession=current_concession,
						negotiation_distance=neg_dist,
						time_factor=f_time,
						seller_last_price=self.agent_last_unit_price
					)
					user_counter_price = round(user_counter_price, 1)
				else:  # Fallback if fuzzyNeg is not available
					user_counter_price = self.current_opponent_unit_price  # Or some other logic
			if extracted_quantity != 'null':
				user_new_quantity = round(float(extracted_quantity),1)
		if user_new_quantity is None:
			user_new_quantity = self.current_opponent_quantity if self.current_opponent_quantity is not None else 1
		return user_counter_price, user_new_quantity, intent_code

	def _calculate_step_reward(self,price,pre_user_offer_is_fuzzy):
		reward = 0
		reward -= self.time_penalty
		if self.pre_user_not_fuzzy_price is not None and  pre_user_offer_is_fuzzy==0:
			if price < self.pre_user_not_fuzzy_price:
				reward -= 3
		# --- 3. 基于历史的核心奖惩逻辑 ---
		history_len = len(self.agent_history_price)
		# 必须有至少2条历史记录才能进行比较
		if history_len < 2:
			return reward
		current_price = float(price)
		last_p = float(self.agent_history_price[-2])
		if current_price > last_p:
			reward -= 3
		relative_change = abs(current_price - last_p) / (abs(last_p) + 1e-8)
		if relative_change > self.price_change_threshold_ratio:
			reward -= self.extreme_penalty * (relative_change / self.price_change_threshold_ratio)
		# 必须有至少3条历史记录才能判断僵持和收敛
		if history_len < 3:
			return reward
		prev_p = float(self.agent_history_price[-3])
		# d) 惩罚连续三次僵持
		if (current_price == last_p == prev_p) and current_price != self.agent_reservation_price:
			reward -= 3
		# 只有在价格实际发生变化时才计算，避免与僵持逻辑冲突
		'''if abs(current_price - last_p) > 1e-8:
			last_diff = abs(last_p - prev_p) / (abs(prev_p) + 1e-8)
			current_diff = abs(current_price - last_p) / (abs(last_p) + 1e-8)
			if current_diff < last_diff:
				reward += (last_diff - current_diff)  # 奖励收敛
			else:
				reward -= (current_diff - last_diff)  # 惩罚发散'''
		return reward

	def _calculate_violation_penalty(self, unacceptable_price,deal_quantity):
		"""
		【新增】计算因接受低于底价的报价而产生的惩罚。
		这个函数会在智能体犯下“错误接受”这个致命错误时被调用。
		"""
		# 价格范围，用于归一化惩罚力度
		price_range = self.list_price - self.agent_reservation_price
		if price_range <= 0:
			# 避免除以零的错误，返回一个固定的惩罚
			return -5.0

		# 计算接受价格与底价的差距
		gap = self.agent_reservation_price - unacceptable_price
		norm_deal_quantity=norm_quantity(deal_quantity)
		# 将差距归一化，差距越大，惩罚越重
		# 我们乘以一个惩罚系数（例如10），让这个错误信号更加强烈
		penalty_score = - (gap / price_range) * self.reward_p - norm_deal_quantity * self.reward_q

		return penalty_score

	def _calculate_deal_reward(self,deal_price,deal_quantity,pre_user_offer_is_fuzzy = 1):
		if self.pre_user_not_fuzzy_price is not None and  pre_user_offer_is_fuzzy==0:
			if deal_price < self.pre_user_not_fuzzy_price:
				return -10
		# 达成交易时的奖励函数
		reward = self.deal_base_reward
		# 计算利润奖励（越接近底价越好）
		price_factor  = (deal_price - self.agent_reservation_price) / (self.list_price - self.agent_reservation_price)
		price_factor = np.clip(price_factor, 0, 1)
		norm_deal_quantity = norm_quantity(deal_quantity)
		reward += self.reward_p * price_factor + norm_deal_quantity * self.reward_q
		# 计算速度奖励（越早成交越好）
		reward += self.reward_t * (1 - self.current_round / self.max_turn)
		if pre_user_offer_is_fuzzy==1:
			last_p = float(self.agent_history_price[-2])
			relative_change = abs(deal_price - last_p) / (abs(last_p) + 1e-8)
			if relative_change > self.price_change_threshold_ratio:
				reward -= self.extreme_penalty * (relative_change / self.price_change_threshold_ratio)
		return reward

	def calculate_average_concession(self):
		if len(self.user_history_price) < 2:
			return 0.0  # 少于两轮没有让步

		concessions = [
			self.user_history_price[i] - self.user_history_price[i - 1]
			for i in range(1,  len(self.user_history_price))
		]
		average_concession = sum(concessions) / len(concessions)
		return average_concession
	def generate_response(self, messages):
		text = self.agent_tokenizer.apply_chat_template(messages,
														tokenize=False,
														add_generation_prompt=True, enable_thinking=False)
		inputs = self.agent_tokenizer(text, return_tensors="pt").to(self.agent_model.device)

		# 生成输出
		with torch.no_grad():
			output = self.agent_model.generate(
				**inputs,
				max_new_tokens=64,
				temperature=0.1,
				top_p=0.9,
				do_sample=True,
				repetition_penalty=1.2,
				early_stopping=True,
				stopping_criteria=self.agent_stopping_criteria,
			)

		# 解析结果
		output = self.agent_tokenizer.decode(output[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
		return output

	def extra_price_quantity(self,tran_resp):
		# 转换为目标格式字符串
		prompt = f"""是谈判对话识别助手，任务是分析对话，识别当前轮次买家提及或接受的单价和商品数量，并且严格按照指定格式输出。

【识别规则】
1. 单价识别规则：
   - 买家直接提及单价（"X元/台"、"单价X"）
   - 买家接受卖家价格（"X元可以"、"同意这个价"）
   - 基于总价和数量计算（总价÷当前数量 或 总价÷上下文数量）
   - 无法直接判断价格类型时，通过与初始价、保留价或市场价的对比及上下文语义推断其含义。
   - 如提及折扣，计算折后单价
   - 如间接引用之前的价格
   - 当前轮次完全无价格信息则为null

2. 数量识别规则：
   - 优先使用当前轮次买家明确提及的数量
   - 若未明确提及但基于前轮已确定数量继续讨论，则推断该数量
    - 当买家提出价格但未提及数量时：
        • 如上下文有明确数量则使用上下文数量
        • 如上下文无数量信息则默认数量为1
   - 无法确定则为null

【输出格式】
当前单价：[数值/null]
商品数量：[数值/null]

【输入数据】
产品信息：{self.case['product_name']}（{self.case['seller_item_description']}）
初始价：{self.case['init_price']}
对话记录：{tran_resp}
【输出】
"""
		input = self.price_tokenizer(prompt, return_tensors="pt").to('cuda')
		with torch.no_grad():  # 推理时不需要计算梯度
			output = self.price_model.generate(
				**input,  # 输入数据
				max_new_tokens=32,  # 最大生成的新 token 数量
				temperature=0,  # 降低随机性,
				# top_p=0.7,
				# num_return_sequences=1,  # 仅返回一个结果
				# no_repeat_ngram_size=2,  # 避免重复短语（如 "当前轮次买家的要价"）
				do_sample=False,
				early_stopping=True,
				# length_penalty=2.0,  # 惩罚过长生成（参考知识库[1]
				eos_token_id=self.price_tokenizer.eos_token_id)  # 禁用采样，使用贪婪搜索
		response = self.price_tokenizer.decode(output[0][input.input_ids.shape[1]:], skip_special_tokens=True)
		# 使用正则表达式提取
		price_match = re.search(r"当前单价[：:]\s*([\d,]+\.?\d*|null)", response)
		if price_match:
			price_str = price_match.group(1).lower().replace(',', '')   # 获取捕获组内容
			price = float(price_str) if price_str != 'null' else 'null'  # 转换为浮点数或 None
		else:
			price = 'null'

		# 提取商品数量，可以是数字或null
		quantity_match = re.search(r"商品数量[：:]\s*([\d,]+\.?\d*|null)", response)
		if quantity_match:
			quantity_str = quantity_match.group(1).lower().replace(',', '')  # 获取捕获组内容
			quantity = float(quantity_str) if quantity_str != 'null' else 'null'  # 转换为浮点数或 None
		else:
			quantity = 'null'  # 如果未匹配到，设置为 None
		return price, quantity


	def _send_structured_message(self, text, done=False, deal_price=None):
		"""向 WeChat 服务发送结构化消息"""
		response_payload = {
			"action": "end" if done else "continue",
			"response": text,
			"deal_price": deal_price
		}
		self.wechat_service.send_message(self.current_wechat_user_id, response_payload)


	def _calculate_kappa_1(self, lambda_2=0.02,C_underline= 0.01):
		"""
        计算自适应参数 kappa_1_t (κ1^t)。
        对应于公式 (16) 。

        参数:
        t (int): 当前谈判轮次 (t >= 3)
        T_max (int): 最大谈判轮次
        p_b_history (list): 买方历史出价 [p_b^1, p_b^2, ..., p_b^{t-1}]
        p_s_t_minus_1 (float): 卖方上一轮的出价 p_s^{t-1}
        kappa_1_t_minus_1 (float): 上一轮的 kappa_1 值 (κ1^{t-1})
        """
		if self.current_round <= 3:
			kappa_1_t = 0.5  # 前两轮使用固定值
		else:
			C_b_t_minus_1 = max(self.user_history_price[-1] - self.user_history_price[-2], 0) # 买家在t-1轮的让步
			C_b_t_minus_2 = max(self.user_history_price[-2] - self.user_history_price[-3], 0) # 买家在t-2轮的让步
			psi = (C_b_t_minus_1 + lambda_2*self.list_price) / (C_b_t_minus_2 + lambda_2*self.list_price)
			if psi < 1:
				kappa_1_t = np.clip(psi, 0.1, 20)
			elif psi > 1:
				price_gap =self.agent_last_unit_price - self.user_history_price[-1]  # p_s^{t-1} - p_b^{t-1}
				relative_concession = C_b_t_minus_1 / price_gap
				C_hat_b = np.clip(psi * relative_concession, C_underline, 1.0)
				time_term = np.log(1.0 - self.current_round / self.max_turn)
				ln_C_hat_b = np.log(C_hat_b)
				ln_C_underline = np.log(C_underline)
				R = ln_C_hat_b / ln_C_underline
				# 外层对数: ln(R)
				if R <= 1e-9:
					# R 趋近于 0 (即 C_hat_b 趋近于 1)
					# 此时 ln(R) 趋近于 -infinity
					unclipped_kappa1 = np.inf
				elif R >= 1.0 - 1e-9:
					# R 趋近于 1 (即 C_hat_b 趋近于 C_underline)
					# 此时 ln(R) 趋近于 0
					unclipped_kappa1 = 0.0
				else:
					# 正常计算
					log_R = np.log(R)
					unclipped_kappa1 = log_R / time_term

				kappa_1_t = np.clip(unclipped_kappa1, 0.1, 20.0)
			else:
				kappa_1_t = np.clip(self.κ1_t,0.1,20)
		buyer_time_belief=(self.current_round / self.max_turn) ** (1 / kappa_1_t)
		self.κ1_t = kappa_1_t
		return buyer_time_belief

def concession_sufficiency(initial_price,current_price,buyer_target_price):
    """计算让步充分度"""
    achieved = initial_price - current_price
    expected = (initial_price - buyer_target_price) * 0.85
    return min(1.0, max(0, achieved / expected))
def CBMessages(case, role, conversation, action=None,price=None,quantity=None,buyer_bid=None,time_pressure=None):
	if role == 'seller':
		if conversation is not None:
			dia_history = translate_response(conversation)
		else:
			dia_history = "（这是第一轮对话）\n"
		if action == 0:
			strategy = "Counteroffer"
			output_require = f"""回复中必须明确给出 {price} 作为卖家的报价（必须原样写出该数字）。"""
		elif action ==1:
			strategy = "Insist"
			output_require = f"""不要提出任何新价格；仅坚持并重申当前报价 {price}。"""
		else:
			strategy = "Accept"
			output_require = f"""确认同意当前条件成交，不要提出新价格。"""
		if time_pressure:
			if time_pressure <=0.3:
				pressure = "低紧迫度"
			elif time_pressure <=0.7:
				pressure = "中等紧迫度"
			else:
				pressure =  "高紧迫度"
		else:
			pressure = "无"
		prompt = f"""你是一名专业的二手商品卖家。请仔细阅读以下信息，并生成一段连贯的卖家回复，用来实现本回合的策略决策。回复必须遵守输出要求，并且与谈判策略模型给出的动作与出价严格一致。

【商品信息】
- 名称：{case['product_name']}
- 描述：{case['seller_item_description']}
- 初始价格：{case['init_price']}
- 底价：{case["seller_reserve_price"]}

【对话历史】
{dia_history}

【谈判上下文（当前回合）】
- 买家出价：{buyer_bid}
- 数量：{quantity} 
- 时间压力：{pressure}

【策略决策（必须严格遵循）】
- 动作： {strategy}（取值之一：Counteroffer，Insist，Accept）
- 卖家出价：{price}（策略计算得到的卖方报价）

【输出要求】
- 动作一致性：回复必须体现给定的 {strategy}，不得与之矛盾。
- 数值一致性：{output_require}
- 结合上下文：回复需基于对话历史，礼貌且符合真实交易语境。
- 简洁：仅用 1～2 句简短中文表达。
- 格式：不要重复提示词、背景信息或规则说明。只输出卖家的回复内容。

【你的回复】
Seller:"""

	messages = [{"role": "system", "content": prompt}]
	return messages

def non_bargaining_CBMessages(case,  conversation, intent):
	if conversation is not None:
		dia_history = translate_response(conversation)
	else:
		dia_history = "（这是第一轮对话）\n"
	prompt = f"""#角色定义
你是一位经验丰富的二手商品卖家。你的任务是仔细阅读以下信息，并提供一个连贯的回复，该回复必须严格遵循输出要求，并有效回应买家的意图。
	
[输入信息]
## 1. 商品详情
- **商品名称:** {case['product_name']}
- **商品描述:** {case['seller_item_description']}
- **你的初始要价:** ¥{case['init_price']}

## 2. 对话历史
{dia_history}

## 3. 回复指引
- **买家意图:** {intent}

## 4. 输出要求
- **意图对齐:** 你的回复必须直接且准确地回应买家意图所体现的目标。例如，若意图是提问，则给出明确答案；若是问候，则礼貌回应。
- **贴合语境且信息充分:** 回复应有帮助、精准，并基于对话历史。当买家意图为咨询时，请结合商品信息（如名称、描述、价格）回答问题或确认相关细节，确保回复自然承接上文。
- **禁止议价:** 在任何情况下都不进行价格讨论或还价协商，如遇价格相关询问，应礼貌引导至其他商品细节。
- **简洁性:** 回复应简明扼要，通常为1至2句话。
- **格式:** 无需重复背景信息或指令，直接以“卖家：”开头输出你的回复内容。

[你的回复]
卖家:"""
	messages = [{"role": "system", "content": prompt}]
	return messages


def format_dialogue_history(conversation):
    history_lines = []
    for turn in conversation:
        # role一般是 '买家' 或 '卖家'
        role = turn['role']
        content = turn['content']
        history_lines.append(f"{role}：{content}")
    return "\n".join(history_lines)

def Qwen_prompt(messages, role):
    seps = [' ', ' </s><s>']
    ret = messages[0]['content'] + seps[0]
    for i, message in enumerate(messages[1:]):
        if message['role'] == role:
            role_text = 'ASSISTANT'
        elif message['role'] != role:
            role_text = 'USER'
        role_text = message['role']
        ret += role_text + "：" + message['content'] + seps[i % 2]
    ret += '%s' % role+ "："
    return ret


def postprocess_response(response,role):
	# 移除 </s><s>[角色]：...... 或者类似部分的内容
	if role in response:
		response = response.split(role)[0].strip()    # 去掉最后的卖家/买家部分

	# 分句
	#sents = nltk.sent_tokenize(response, language='chinese')
	# 如果是单句
	response = re.sub(r'<[\/]?s>', '', response)
	if len(response) > 0:
		if response[-1] not in ['。', '！', '？', '：']:  # 中文的标点符号
			return response + '。'  # 添加中文句号
	else:
		print(response)
	return response.strip()

def translate_response(response):

	# 用来存储转换后的数据
	formatted_data = []

	# 遍历原始数据并格式化
	for i, entry in enumerate(response):
		role = entry['role']
		content = entry['content']
		formatted_data.append(f"{role}：{content}\n")
		'''
		if role=='卖家':
			# 创建每轮的格式
			formatted_data.append(f"{role}：{content} </s><s>")
		else:
			formatted_data.append(f"{role}：{content}")
		# 卖家和买家的对话是一对一的，买家之后是卖家的回话，所以下一轮开始时加1
		'''
	return ''.join(formatted_data)

def process_seller_response(response):
	seller_patterns = [
		r'\*\*卖家\*\*\s*[：:]\s*',  # **卖家:**
		r'#{1,3}\s*卖家\s*[：:]\s*',  # ### 卖家：
		r'-{2,}\s*卖家\s*[：:]\s*',  # --- 卖家:
		r'卖家\s*[：:]\s*',  # 卖家: 或 卖家：
	]

	# 买家/用户标识符的正则表达式模式（作为结束标记）
	buyer_patterns = [
		r'Human\s*[：:]\s*',  # Human:
		r'买家\s*[：:]\s*',  # 买家:
		r'buyer\s*[：:]\s*',  # buyer:
		r'user\s*[：:]\s*',  # user:
		r'用户\s*[：:]\s*',  # 用户:
	]

	# 合并卖家模式
	seller_combined_pattern = '|'.join(f'({pattern})' for pattern in seller_patterns)

	# 合并买家模式
	buyer_combined_pattern = '|'.join(f'({pattern})' for pattern in buyer_patterns)

	# 找到所有卖家匹配
	seller_matches = list(re.finditer(seller_combined_pattern, response, re.IGNORECASE))

	if not seller_matches:
		return clean_llm_response(response.strip())

	# 获取第一个卖家匹配
	first_seller_match = seller_matches[0]
	start_pos = first_seller_match.end()

	# 找到所有买家匹配（从第一个卖家匹配之后开始）
	text_after_first_seller = response[start_pos:]
	buyer_matches = list(re.finditer(buyer_combined_pattern, text_after_first_seller, re.IGNORECASE))

	# 确定结束位置
	end_pos = len(response)  # 默认到文本结尾

	# 检查是否有第二个卖家匹配
	if len(seller_matches) > 1:
		second_seller_pos = seller_matches[1].start()
		end_pos = min(end_pos, second_seller_pos)

	# 检查是否有买家匹配
	if buyer_matches:
		# 找到第一个买家匹配的位置（相对于原文本的位置）
		first_buyer_pos = start_pos + buyer_matches[0].start()
		end_pos = min(end_pos, first_buyer_pos)

	# 提取第一句并清理
	first_sentence = response[start_pos:end_pos].strip()

	# 移除可能的尾部标记符号
	first_sentence = re.sub(r'\s*[-*#]+\s*$', '', first_sentence)

	return clean_llm_response(first_sentence)


def preprocess_text_for_negotiation(text):
	"""
    针对议价场景的文本预处理函数
    """
	# 2. 格式标准化
	# text = converter.convert(text) # 繁简转简体
	text = zhconv.convert(text, 'zh-cn')  # 使用zhconv进行繁简转换

	# 全角转半角
	text = re.sub(r'[\uFF00-\uFFEF]', lambda x: chr(ord(x.group(0)) - 65248), text)

	# 3. 内容过滤
	# 去除URL和邮箱
	#text = re.sub(r'http\S+|www\.\S+|\S+@\S+', '', text)
	# 去除HTML标签
	text = re.sub(r'<[^>]+>', '', text)

	# 处理标点符号：将非保留标点替换为空格
	# 定义要保留的所有中英文标点符号
	preserve_punct = {
		# 中文标点 (全角)
		'？', '！', '，', '。', '；', '：', '、', '…', '（', '）', '【', '】', '《', '》', '“', '”', '‘', '’', '—', '·',
		# 英文标点 (半角)
		'?', '!', ',', '.', ';', ':', '(', ')', '[', ']', '{', '}', '-', '_', '+', '=', '*', '/', '\\', '|', '~', '`',
		'@', '#', '$', '%', '^', '&', '<', '>'
	}

	processed_chars = []
	for char in text:
		# 如果字符是我们要保留的标点符号，直接保留
		if char in preserve_punct:
			processed_chars.append(char)
		# 如果是其他特殊符号（非字母数字、非汉字、非保留标点），替换为空格
		# 例如：©, ®, ™, 或者其他一些不常见的符号
		elif not char.isalnum() and not '\u4e00' <= char <= '\u9fff':
			processed_chars.append(' ')
		# 其他情况（字母、数字、汉字）直接保留
		else:
			processed_chars.append(char)

	text = ''.join(processed_chars)

	# 4. 规范化空格
	text = re.sub(r'\s+', ' ', text).strip()

	return text


def clean_llm_response(text):
	"""
    清理语言模型生成的回复，移除JSON格式和多余内容

    Args:
        text (str): 原始回复文本

    Returns:
        str: 清理后的文本
    """

	# 移除JSON代码块
	match = re.match(r'^(?!```json)[\s\S]*?(?=```json|$)', text)
	if match:
		text = match.group(0).strip()

	# 移除多余的引号
	text = re.sub(r'^["\']\s*|\s*["\']$','',text)
	return text