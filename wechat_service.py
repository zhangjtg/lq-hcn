# wechat_service.py
import json
import threading
from collections import defaultdict, deque
from flask import Flask


class WeChatService:
    def __init__(self, app: Flask, sessions: dict):
        self.app = app
        #self.lock = threading.Lock()
        self.sessions = sessions
        # 条件变量 1：通知 RL 有新会话
        self.new_session_cv = threading.Condition()
        # 条件变量 2：通知 Flask 有智能体回复
        self.agent_response_cv = threading.Condition()
        # 条件变量 3：通知 RL 有用户消息
        self.user_message_cv = threading.Condition()
        self.user_message_queues = defaultdict(deque)
        self.session_queue = deque()
        self.user_outbox_queues = defaultdict(deque)  # 存储发送给用户的消息
        self.product_database = self.load_product_database()
        # 新增一个Event来标记训练线程是否正在处理一个完整的episode
        self.episode_processing_lock = threading.Event()

    def end_session(self, user_id: str):
        """
        彻底结束并清理一个用户的会话。
        这将被 RL 训练循环在谈判结束后调用。
        """
        if user_id in self.sessions:
            try:
                # 使用 pop 安全地移除键，如果键不存在也不会报错
                self.sessions.pop(user_id, None)
                print(f"[WeChatService] 已清理用户 {user_id} 的会话。")
            except Exception as e:
                print(f"[WeChatService ERROR] 清理用户 {user_id} 会话失败: {e}")

    def load_product_database(self):
        try:
            with open('../../../data/HLQ_negotiation_scence_train_data.json', 'r', encoding='utf-8') as f:
                products = json.load(f)
            return {product['product_id']: product for product in products}
        except Exception as e:
            print(f"加载商品数据失败: {e}")
            return {}

    def get_product_by_id(self, product_id):
        return self.product_database.get(int(product_id))

    def new_session_available(self, user_id, product_id):
        """通知 RL 线程有一个新会话"""
        with self.new_session_cv:
            self.session_queue.append((user_id, product_id))
            self.new_session_cv.notify_all()# 唤醒监听新会话的线程

    def wait_for_new_session(self, timeout=None):
        """RL 线程等待新会话"""
        with self.new_session_cv:
            while not self.session_queue:
                if not self.new_session_cv.wait(timeout=timeout):
                    return None, None
            return self.session_queue.popleft()

    def put_user_message(self, user_id, message):
        """存储来自用户的聊天消息"""
        with self.user_message_cv:
            self.user_message_queues[user_id].append(message)
            self.user_message_cv.notify_all()

    def wait_for_user_message(self, user_id, timeout=5000):
        """RL 线程等待用户消息"""
        with self.user_message_cv:
            while not self.user_message_queues.get(user_id):
                if not self.user_message_cv.wait(timeout=timeout):
                    return None
            return self.user_message_queues[user_id].popleft()
        #print(f"[Thread-{threading.get_ident()}] 返回用户 {user_id} 的响应")
    def send_message(self, user_id, message):
        """RL 线程发送消息给用户"""
        with self.agent_response_cv:
            self.user_outbox_queues[user_id].append(message)
            self.agent_response_cv.notify_all()


    def wait_for_agent_response(self, user_id, timeout=60):
        """Flask 线程等待 RL 消息"""
        with self.agent_response_cv:
            while not self.user_outbox_queues.get(user_id):
                if not self.agent_response_cv.wait(timeout=timeout):
                    return []  # 超时返回空列表

            messages = list(self.user_outbox_queues[user_id])
            self.user_outbox_queues[user_id].clear()
            return messages
        #print(f"[Thread-{threading.get_ident()}] 返回用户 {user_id} 的响应")
    '''
        def get_user_messages(self, user_id,timeout=60):
            """获取发送给用户的消息"""
            with self.lock:
                if user_id in self.user_outbox_queues and self.user_outbox_queues[user_id]:
                    messages = list(self.user_outbox_queues[user_id])
                    self.user_outbox_queues[user_id].clear()
                    return messages
                    # 等待消息到达
                if self.condition.wait(timeout=timeout):
                    # 被唤醒，检查消息
                    if user_id in self.user_outbox_queues and self.user_outbox_queues[user_id]:
                        messages = list(self.user_outbox_queues[user_id])
                        self.user_outbox_queues[user_id].clear()
                        return messages

                # 超时或没有消息
                return []
        '''