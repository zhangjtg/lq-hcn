from flask import Flask, request, jsonify
import threading
from flask_cors import CORS
import json
from datetime import datetime
import os
from run_LQ_negotiation import run_training
from wechat_service import WeChatService

app = Flask(__name__)
CORS(app)  # 允许跨域请求
# 全局环境实例
global_env = None
env_lock = threading.Lock()
negotiation_sessions = {}
wechat_service = WeChatService(app,negotiation_sessions)


def save_negotiation_result(user_id,deal_price):
    """保存谈判结果到JSON文件"""
    if user_id not in negotiation_sessions:
        return False

    dia_session = negotiation_sessions[user_id]
    result = {
        "user_id": user_id,
        "product_info": dia_session['product'],
        "negotiation_history": dia_session['history'],
        "start_time": dia_session['start_time'],
        "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "deal_price": deal_price if deal_price else 'no deal'
    }

    # 创建目录
    os.makedirs('../output/negotiation_records', exist_ok=True)
    # 生成文件名
    filename = f"../output/negotiation_records/{user_id}.json"
    try:
        # 检查文件是否存在
        if not os.path.exists(filename):
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("[\n")
        with open(filename, 'a', encoding='utf-8') as f:
            # 需要处理逗号和换行，确保 JSON 数组格式正确
            f.write(f" {json.dumps(result, ensure_ascii=False)},\n")
        return True
    except Exception as e:
        print(f"保存谈判记录失败: {e}")
        return False
    finally:
        # 清理会话
        wechat_service.end_session(user_id)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_input = data.get('message', '')
        user_id = data.get('user_id', '')
        if not user_id:
            return jsonify({"error": "缺少用户ID"}), 400
            # 检查会话是否存在
        if not user_input:
            return jsonify({"error": "请输入有效内容"}), 400

        if user_id not in negotiation_sessions:
            return jsonify({
                "action": "end",
                "response": "谈判会话已过期，请重新开始谈判"
            })
        dia_session = negotiation_sessions[user_id]
        # 添加用户消息到历史
        dia_session['history'].append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        # 转发用户消息到环境
        wechat_service.put_user_message(user_id, user_input)
        # 等待 RL 的回复
        messages = wechat_service.wait_for_agent_response(user_id, timeout=300)

        if messages:
            agent_response_data  = messages[0]
            # 添加AI回复到历史
            dia_session['history'].append({
                "role": "bot",
                "content": agent_response_data['response'],
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            if agent_response_data['action'] =='end':
                save_negotiation_result(user_id,agent_response_data.get('deal_price'))
            # 判断是否谈判结束（示例逻辑）
            return jsonify(agent_response_data)
        else:
            save_negotiation_result(user_id, None)  # 清理会话
            return jsonify({"error": "等待智能体回复超时"}), 504

    except Exception as e:
        print(f"[Flask ERROR] /chat 请求失败: {e}")
        return jsonify({"error": f"服务器内部错误: {str(e)}"}), 500


# 注册API路由
@app.route('/start_negotiation', methods=['POST'])
def start_negotiation():
    try:
        # 在处理请求前，检查后台是否正忙
        if wechat_service.episode_processing_lock.is_set():
            return jsonify({
                "error": "智能体当前正忙，请您稍后几秒再试"
            }), 503  # 503 Service Unavailable 是一个很贴切的状态码
        data = request.json
        user_id = data.get('user_id', '')
        product_id = data.get('product_id')

        if not user_id:
            return jsonify({"error": "缺少用户ID"}), 400
        if product_id is None:
            return jsonify({"error": "找不到该商品"}), 400
        product_id = int(product_id)
        # 如果该用户已有会话，先清理
        if user_id in negotiation_sessions:
            wechat_service.end_session(user_id)
            print(f"用户 {user_id} 的旧会话已清理，准备开始新会话。")
        # 通知有新会话
        wechat_service.new_session_available(user_id, product_id)
        # 2. 不再立即获取消息，而是等待智能体响应
        print(f"[Flask] 正在等待用户 {user_id} 的欢迎消息...")
        messages = wechat_service.wait_for_agent_response(user_id, timeout=1200)
        # 初始化谈判会话

        if messages:
            welcome_message = messages[0]
            product = wechat_service.get_product_by_id(product_id)
            if not product:
                return jsonify({"error": "商品未找到"}), 404
            negotiation_sessions[user_id] = {
                'product': product,  # 完整的商品信息（包含最低价格）
                'history': [],
                'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            return jsonify({
                "status": "started",
                "welcome_message": welcome_message["response"],
                "product_info": {
                    "id": product["product_id"],
                    "name": product["product_name"],
                    "base_price": product["init_price"],
                    "description": product["seller_item_description"]
                }
            })
        else:
            print(f"[Flask ERROR] 等待用户 {user_id} 的欢迎消息超时。")
            # 如果超时，也需要确保后台标志被重置，以防万一
            wechat_service.episode_processing_lock.clear()
            return jsonify({"error": "等待智能体欢迎消息超时"}), 504
    except Exception as e:
        print(f"启动谈判失败: {e}")
        return jsonify({"error": f"启动谈判失败: {str(e)}"}), 500

# 在main中添加
if __name__ == '__main__':
    training_thread = threading.Thread(
        target=run_training,
        args=(wechat_service, 'train', 'custom_results.csv'),
        daemon=True  # El hilo daemon se cerrará cuando el programa principal termine
    )
    training_thread.start()
    print("启动Flask服务器...")
    app.run(host='0.0.0.0', port=5000, ssl_context='adhoc')