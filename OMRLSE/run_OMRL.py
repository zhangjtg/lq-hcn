# run_OMRL.py


import os
import sys
import json
import threading
import time
import torch
import random
import argparse
from datetime import datetime
from typing import Optional, Dict, Any, List

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入核心模块
from Automated_Negotiation import (
    PaperAlignedSellerOptionA,
    SellerConfig,
    PriceTierEncoder,
    OpponentModel,
    CommodityRecord,
    DialogueState,
    load_commodities_from_json,
    seller_generate_dialogue,
    seller_generate_welcome_message,
    llm_generate_text,
    # 意图分类
    infer_buyer_intent_keywords,
    classify_buyer_intent_llm,
    predict_buyer_intent_model,
    # 价格提取
    extract_price_quantity_llm,
    # 历史转换
    history_to_zh_text,
)

# 尝试导入独立的意图识别器和价格提取器
try:
    from intent_recognizer import IntentRecognizer
    HAS_INTENT_RECOGNIZER = True
except ImportError:
    HAS_INTENT_RECOGNIZER = False
    IntentRecognizer = None

try:
    from price_quantity_extractor import PriceQuantityExtractor
    HAS_PRICE_EXTRACTOR = True
except ImportError:
    HAS_PRICE_EXTRACTOR = False
    PriceQuantityExtractor = None


# =========================
# 意图标签映射
# =========================

INTENT_LABEL_MAP = {
    "讨价还价": "NEGOTIATE",
    "谈判失败": "LEAVE", 
    "谈判成功": "ACCEPT",
    "negotiate": "NEGOTIATE",
    "accept": "ACCEPT",
    "leave": "LEAVE",
    "bargain": "NEGOTIATE",
    "fail": "LEAVE",
    "success": "ACCEPT",
}


# =========================
# 微信谈判服务类
# =========================

class WeChatNegotiationService:


    def __init__(
        self,
        train_json_path: str,
        seller_checkpoint_path: Optional[str] = None,
        model_name: str = "Ministral-3-8B-Instruct-2512",
        n_tiers: int = 8,
        seed: int = 7,
        # 独立模型配置
        intent_model_path: Optional[str] = None,
        intent_adapter_path: Optional[str] = None,
        price_model_path: Optional[str] = None,
        price_adapter_path: Optional[str] = None,
        # 错误处理配置
        max_invalid_inputs: int = 3,
        user_timeout: int = 300,
    ):
        """初始化谈判服务"""
        self.train_json_path = train_json_path
        self.seller_checkpoint_path = seller_checkpoint_path
        self.model_name = model_name
        self.n_tiers = n_tiers
        self.seed = seed
        self.max_invalid_inputs = max_invalid_inputs
        self.user_timeout = user_timeout
        
        random.seed(seed)
        torch.manual_seed(seed)
        
        # 初始化
        self._init_models(model_name)
        self._init_seller()
        self._init_intent_recognizer(intent_model_path, intent_adapter_path)
        self._init_price_extractor(price_model_path, price_adapter_path)
        
        # 会话管理
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_lock = threading.Lock()
        
        # 统计
        self.total_negotiations = 0
        self.successful_negotiations = 0
        self.results: List[Dict[str, Any]] = []

    def _init_models(self, model_name: str):
        """初始化主模型"""
        print(f"[服务] 正在加载模型: {model_name}")
        
        try:
            from transformers import MistralCommonBackend, Mistral3ForConditionalGeneration
            
            self.tokenizer = MistralCommonBackend.from_pretrained(model_name)
            self.model = Mistral3ForConditionalGeneration.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
        except Exception as e:
            print(f"[服务] 使用备用模型加载: {e}")
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
        
        print("[服务] 模型加载完成")
        self.seller_tokenizer = self.tokenizer
        self.seller_model = self.model

    def _init_seller(self):
        """初始化卖家智能体"""
        train_items = load_commodities_from_json(self.train_json_path)
        
        self.tier_encoder = PriceTierEncoder.fit(
            [x.init_price for x in train_items],
            n_tiers=self.n_tiers
        )
        
        self.seller_config = SellerConfig(max_rounds=8)
        self.opponent_model = OpponentModel()
        
        if self.seller_checkpoint_path and os.path.exists(self.seller_checkpoint_path):
            print(f"[服务] 加载卖家模型: {self.seller_checkpoint_path}")
            self.seller = PaperAlignedSellerOptionA.load(
                self.seller_checkpoint_path,
                self.opponent_model
            )
        else:
            print("[服务] 创建新的卖家智能体")
            self.seller = PaperAlignedSellerOptionA(
                self.seller_config,
                self.tier_encoder,
                self.opponent_model
            )

    def _init_intent_recognizer(self, model_path: Optional[str], adapter_path: Optional[str]):
        """初始化意图识别器"""
        self.intent_recognizer = None
        
        if not HAS_INTENT_RECOGNIZER:
            print("[服务] 未安装独立意图识别器，将使用LLM")
            return
        
        if model_path and os.path.exists(model_path):
            print(f"[服务] 加载意图识别器: {model_path}")
            try:
                self.intent_recognizer = IntentRecognizer(
                    base_model_path=model_path,
                    adapter_path=adapter_path,
                    num_labels=6,
                )
                print("[服务] 意图识别器加载完成")
            except Exception as e:
                print(f"[服务] 意图识别器加载失败: {e}")
                self.intent_recognizer = None
        else:
            print("[服务] 未配置意图识别器，将使用LLM")

    def _init_price_extractor(self, model_path: Optional[str], adapter_path: Optional[str]):
        """初始化价格提取器"""
        self.price_extractor = None
        
        if not HAS_PRICE_EXTRACTOR:
            print("[服务] 未安装独立价格提取器，将使用LLM")
            return
        
        if model_path and os.path.exists(model_path):
            print(f"[服务] 加载价格提取器: {model_path}")
            try:
                self.price_extractor = PriceQuantityExtractor(
                    base_model_path=model_path,
                    adapter_path=adapter_path,
                )
                print("[服务] 价格提取器加载完成")
            except Exception as e:
                print(f"[服务] 价格提取器加载失败: {e}")
                self.price_extractor = None
        else:
            print("[服务] 未配置价格提取器，将使用LLM")

    def classify_intent(
        self, 
        buyer_message: str, 
        dialogue: DialogueState, 
        item: CommodityRecord,
        extracted_price: Optional[float] = None,
        last_seller_offer: Optional[float] = None,
    ) -> str:
        """
        分类买家意图（多级回退）
        
        优先级：
        1. 独立意图识别模型
        2. LLM意图分类
        3. 关键词推断
        """
        # 1. 尝试独立模型
        if self.intent_recognizer is not None:
            result = self.intent_recognizer.predict(buyer_message)
            intent = INTENT_LABEL_MAP.get(result.label, None)
            if intent:
                return intent
        
        # 2. 尝试LLM
        try:
            return classify_buyer_intent_llm(
                self.tokenizer, self.model, buyer_message, dialogue, item
            )
        except Exception:
            pass
        
        # 3. 关键词推断
        return infer_buyer_intent_keywords(buyer_message, extracted_price, last_seller_offer)

    def extract_price_quantity(self, tran_resp: str, case: dict) -> tuple:
        """
        提取价格和数量（多级回退）
        
        优先级：
        1. 独立价格提取模型
        2. LLM价格提取
        """
        # 1. 尝试独立模型
        if self.price_extractor is not None:
            try:
                self.price_extractor.set_case(case)
                return self.price_extractor.extra_price_quantity(tran_resp)
            except Exception:
                pass
        
        # 2. LLM提取
        return extract_price_quantity_llm(self.tokenizer, self.model, tran_resp, case)

    def start_new_negotiation(self, wechat_service, user_id: str, product_id: int) -> Dict[str, Any]:
        """
        开始新的谈判会话
        
        注意：欢迎消息不包含价格，避免被误认为卖家出价
        """
        product = wechat_service.get_product_by_id(product_id)
        if not product:
            return {"success": False, "error": "商品不存在"}
        
        item = CommodityRecord(
            product_id=str(product["product_id"]),
            product_name=product["product_name"],
            seller_item_description=product["seller_item_description"],
            init_price=float(product["init_price"]),
            buyer_reserve_price=float(product.get("buyer_reserve_price", product["init_price"] * 1.2)),
            seller_reserve_price=float(product["seller_reserve_price"]),
            quantity=float(product.get("quantity", 1)),
        )
        
        self.seller.reset_episode_memory()
        
        # 创建会话（包含无效输入计数器）
        session = {
            "item": item,
            "dialogue": DialogueState(),
            "conversation_history": [],
            "round_idx": 0,
            "last_seller_offer": float(item.init_price),  # 初始价格作为第一次卖家出价
            "buyer_offers": [],
            "seller_offers": [],
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "invalid_cnt": 0,  # 无效输入计数器
            "initial_buyer_price": None,
        }
        
        with self.session_lock:
            self.active_sessions[user_id] = session
        
        # 生成欢迎消息（不包含价格）
        welcome_message = seller_generate_welcome_message(
            self.seller_tokenizer, self.seller_model, item
        )
        
        wechat_service.send_message(user_id, {
            "action": "negotiate",
            "response": welcome_message,
            "round": 0
        })
        
        session["conversation_history"].append({"role": "assistant", "content": welcome_message})
        session["dialogue"].add("seller", welcome_message, intent="INIT")
        
        return {
            "success": True,
            "welcome_message": welcome_message,
            "product_info": {
                "id": item.product_id,
                "name": item.product_name,
                "init_price": item.init_price,
            }
        }

    def process_buyer_message(
        self, 
        wechat_service, 
        user_id: str, 
        buyer_message: str, 
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        处理买家消息（包含无效输入处理）
        """
        with self.session_lock:
            if user_id not in self.active_sessions:
                return {"action": "error", "response": "会话不存在"}
            session = self.active_sessions[user_id]
        
        item = session["item"]
        dialogue = session["dialogue"]
        conversation_history = session["conversation_history"]
        round_idx = session["round_idx"]
        invalid_cnt = session["invalid_cnt"]
        
        # 检查是否超过最大轮次
        if round_idx >= self.seller_config.max_rounds:
            return self._end_negotiation(
                wechat_service, user_id, False, None, 
                "谈判轮次已达上限，很遗憾未能达成交易。"
            )
        
        # 添加买家消息到历史
        conversation_history.append({"role": "user", "content": buyer_message})
        
        # 构建对话上下文
        history_text = history_to_zh_text(conversation_history, max_turns=16)
        case = {
            "product_name": item.product_name,
            "seller_item_description": item.seller_item_description,
            "init_price": item.init_price,
        }
        
        # 提取价格和数量
        extracted_price, extracted_quantity = self.extract_price_quantity(history_text, case)
        
        # 分类意图
        buyer_intent = self.classify_intent(
            buyer_message, dialogue, item,
            extracted_price, session["last_seller_offer"]
        )
        
        # 处理提取的价格
        buyer_offer = None
        if extracted_price is not None:
            buyer_offer = float(extracted_price)
        
        dialogue.add("buyer", buyer_message, intent=buyer_intent, extracted_price=buyer_offer)
        
        if verbose:
            print(f"[轮次 {round_idx + 1}] 意图: {buyer_intent}, 价格: {buyer_offer}")
        
        # ========== 处理终端意图 ==========
        if buyer_intent == "LEAVE":
            return self._end_negotiation(
                wechat_service, user_id, False, None,
                "好的，感谢您的咨询。如果之后还想谈，随时联系我。"
            )
        
        if buyer_intent == "ACCEPT":
            # 买家接受当前卖家出价
            accept_price = session["last_seller_offer"]
            if accept_price >= item.seller_reserve_price:
                accept_message = seller_generate_dialogue(
                    self.seller_tokenizer, self.seller_model, item, conversation_history,
                    "Accept", buyer_offer, None, accept_price
                )
                conversation_history.append({"role": "assistant", "content": accept_message})
                return self._end_negotiation(wechat_service, user_id, True, accept_price, accept_message)
            else:
                return self._end_negotiation(
                    wechat_service, user_id, False, None,
                    "抱歉，这个价格低于我的成本，无法成交。"
                )
        
        # ========== 处理谈判意图 ==========
        if buyer_offer is None:
            # 无效输入处理
            invalid_cnt += 1
            session["invalid_cnt"] = invalid_cnt
            
            if invalid_cnt >= self.max_invalid_inputs:
                return self._end_negotiation(
                    wechat_service, user_id, False, None,
                    "多次未识别到您的出价，本次谈判结束。您可以重新开始并直接给出单价和数量。"
                )
            
            # 请求澄清
            clarify = "我没识别到明确的单价/数量。请直接给出您的单价（元/件）和数量，例如：800元/件，50件。"
            wechat_service.send_message(user_id, {
                "action": "negotiate",
                "response": clarify,
                "round": round_idx
            })
            conversation_history.append({"role": "assistant", "content": clarify})
            dialogue.add("seller", clarify, intent="CLARIFY")
            
            return {"action": "negotiate", "response": clarify, "round": round_idx}
        
        # 重置无效计数
        session["invalid_cnt"] = 0
        
        # 记录买家出价
        if session["initial_buyer_price"] is None:
            session["initial_buyer_price"] = buyer_offer
        session["buyer_offers"].append(round(buyer_offer, 1))
        
        # 计算卖家出价
        seller_offer, accepted = self.seller.compute_seller_offer(item, buyer_offer, round_idx)
        session["round_idx"] = round_idx + 1
        
        if accepted:
            # 卖家接受买家出价
            accept_message = seller_generate_dialogue(
                self.seller_tokenizer, self.seller_model, item, conversation_history,
                "Accept", buyer_offer, None, buyer_offer
            )
            conversation_history.append({"role": "assistant", "content": accept_message})
            return self._end_negotiation(wechat_service, user_id, True, buyer_offer, accept_message)
        else:
            # 卖家反报价
            session["seller_offers"].append(round(seller_offer, 1))
            session["last_seller_offer"] = seller_offer
            
            counter_message = seller_generate_dialogue(
                self.seller_tokenizer, self.seller_model, item, conversation_history,
                "Counteroffer", buyer_offer, None, seller_offer
            )
            
            conversation_history.append({"role": "assistant", "content": counter_message})
            dialogue.add("seller", counter_message, intent="COUNTER", extracted_price=seller_offer)
            
            wechat_service.send_message(user_id, {
                "action": "negotiate",
                "response": counter_message,
                "seller_offer": seller_offer,
                "round": session["round_idx"]
            })
            
            return {
                "action": "negotiate",
                "response": counter_message,
                "seller_offer": seller_offer
            }

    def _end_negotiation(
        self, 
        wechat_service, 
        user_id: str, 
        success: bool, 
        final_price: Optional[float], 
        message: str
    ) -> Dict[str, Any]:
        """结束谈判"""
        with self.session_lock:
            if user_id in self.active_sessions:
                session = self.active_sessions[user_id]
                
                result = {
                    "user_id": user_id,
                    "product_id": session["item"].product_id,
                    "product_name": session["item"].product_name,
                    "success": success,
                    "final_price": final_price,
                    "turns": session["round_idx"] + 1,
                    "seller_bottom_price": session["item"].seller_reserve_price,
                    "buyer_max_price": session["item"].buyer_reserve_price,
                    "initial_seller_price": session["item"].init_price,
                    "initial_buyer_price": session["initial_buyer_price"],
                    "buyer_offers": session["buyer_offers"],
                    "seller_offers": session["seller_offers"],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                
                self.total_negotiations += 1
                if success:
                    self.successful_negotiations += 1
                self.results.append(result)
                
                del self.active_sessions[user_id]
        
        wechat_service.send_message(user_id, {
            "action": "end",
            "response": message,
            "deal_price": final_price
        })
        
        return {"action": "end", "response": message, "deal_price": final_price}

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计"""
        return {
            "total_negotiations": self.total_negotiations,
            "successful_negotiations": self.successful_negotiations,
            "success_rate": self.successful_negotiations / max(1, self.total_negotiations),
        }

    def save_results(self, output_dir: str = "results"):
        """保存结果"""
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, "negotiation_results.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2, default=str)
        print(f"[服务] 结果已保存: {filepath}")


# =========================
# 主服务循环
# =========================

def run_training(wechat_service, mode: str = 'train', output_file: str = 'custom_results.csv'):
    """
    主训练/服务循环（与app.py兼容）
    """
    print(f"[主程序] 启动谈判服务，模式: {mode}")
    
    train_json_path = "../../../data/HLQ_negotiation_scence_train_data.json"
    seller_checkpoint_path = "./checkpoints/seller_rl.pt"
    
    # 创建服务
    service = WeChatNegotiationService(
        train_json_path=train_json_path,
        seller_checkpoint_path=seller_checkpoint_path,
    )
    
    print("[主程序] 等待微信小程序连接...")
    
    while True:
        try:
            user_id, product_id = wechat_service.wait_for_new_session(timeout=None)
            
            if user_id and product_id:
                print(f"[主程序] 新会话: 用户 {user_id}")
                
                wechat_service.episode_processing_lock.set()
                
                try:
                    result = service.start_new_negotiation(wechat_service, user_id, product_id)
                    
                    if not result.get("success"):
                        continue
                    
                    while user_id in service.active_sessions:
                        user_message = wechat_service.wait_for_user_message(user_id, timeout=service.user_timeout)
                        
                        if user_message is None:
                            print(f"[主程序] 用户 {user_id} 等待超时")
                            break
                        
                        response = service.process_buyer_message(
                            wechat_service, user_id, user_message, verbose=True
                        )
                        
                        if response.get("action") == "end":
                            stats = service.get_statistics()
                            print(f"[主程序] 统计: 总计={stats['total_negotiations']}, "
                                  f"成功={stats['successful_negotiations']}, "
                                  f"成功率={stats['success_rate']:.2%}")
                            
                            if stats['total_negotiations'] % 100 == 0:
                                service.save_results()
                            break
                            
                finally:
                    wechat_service.episode_processing_lock.clear()
                    
        except Exception as e:
            print(f"[主程序] 错误: {e}")
            import traceback
            traceback.print_exc()
            wechat_service.episode_processing_lock.clear()


# =========================
# 批量测试
# =========================

def run_batch_test(
    train_json_path: str,
    seller_checkpoint_path: Optional[str],
    num_negotiations: int = 2000,
    output_dir: str = "results",
    verbose: bool = False,
):
    """运行批量测试"""
    print(f"[批量测试] 开始 {num_negotiations} 次谈判")
    
    service = WeChatNegotiationService(
        train_json_path=train_json_path,
        seller_checkpoint_path=seller_checkpoint_path,
    )
    
    # 模拟微信服务
    class MockWeChat:
        def __init__(self, products):
            self.product_database = {p["product_id"]: p for p in products}
        def get_product_by_id(self, pid):
            return self.product_database.get(pid)
        def send_message(self, uid, msg):
            pass
    
    products = load_commodities_from_json(train_json_path)
    product_list = [{
        "product_id": int(p.product_id),
        "product_name": p.product_name,
        "seller_item_description": p.seller_item_description,
        "init_price": p.init_price,
        "buyer_reserve_price": p.buyer_reserve_price,
        "seller_reserve_price": p.seller_reserve_price,
        "quantity": p.quantity,
    } for p in products]
    
    mock_wechat = MockWeChat(product_list)
    
    for i in range(num_negotiations):
        product = random.choice(product_list)
        user_id = f"test_user_{i+1}"
        
        # 开始谈判
        service.start_new_negotiation(mock_wechat, user_id, product["product_id"])
        
        # 模拟买家
        session = service.active_sessions.get(user_id)
        if session:
            item = session["item"]
            # 买家目标价格
            buyer_target = item.seller_reserve_price + (item.buyer_reserve_price - item.seller_reserve_price) * 0.4
            
            for round_idx in range(8):
                # 模拟买家出价（逐步让步）
                if round_idx == 0:
                    offer = buyer_target * 0.75
                else:
                    last_offer = session["buyer_offers"][-1] if session["buyer_offers"] else buyer_target * 0.75
                    gap = buyer_target - last_offer
                    offer = last_offer + gap * 0.3
                
                offer = max(item.seller_reserve_price * 0.8, min(offer, buyer_target))
                buyer_msg = f"我出价{offer:.1f}元，可以吗？"
                
                response = service.process_buyer_message(mock_wechat, user_id, buyer_msg, verbose)
                
                if response.get("action") == "end":
                    break
        
        # 定期打印进度
        if (i + 1) % 100 == 0:
            stats = service.get_statistics()
            print(f"[批量测试] 进度: {i+1}/{num_negotiations}, 成功率: {stats['success_rate']:.2%}")
    
    # 最终统计
    stats = service.get_statistics()
    print(f"\n{'='*50}")
    print(f"[批量测试] 完成")
    print(f"总计: {stats['total_negotiations']}")
    print(f"成功: {stats['successful_negotiations']}")
    print(f"成功率: {stats['success_rate']:.2%}")
    print(f"{'='*50}\n")
    
    service.save_results(output_dir)
    
    return service.results


def main():
    parser = argparse.ArgumentParser(description="微信小程序谈判运行器")
    parser.add_argument("--mode", type=str, default="serve", choices=["serve", "batch"])
    parser.add_argument("--train_json_path", type=str, required=True)
    parser.add_argument("--seller_checkpoint", type=str, default=None)
    parser.add_argument("--intent_model_path", type=str, default=None)
    parser.add_argument("--price_model_path", type=str, default=None)
    parser.add_argument("--num_negotiations", type=int, default=2000)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    if args.mode == "batch":
        run_batch_test(
            train_json_path=args.train_json_path,
            seller_checkpoint_path=args.seller_checkpoint,
            num_negotiations=args.num_negotiations,
            output_dir=args.output_dir,
            verbose=args.verbose,
        )
    else:
        print("服务模式需要通过 app.py 启动")


if __name__ == "__main__":
    main()
