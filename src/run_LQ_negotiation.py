# run_negotiation.py
import os
from db_logger import TrainingLogger
import torch
import numpy as np
from LQ_negotiation_env import NegotiationEnv  # Your environment
from MP_DQN import MultiPassPDQNAgent  # The agent to use
import json,csv
from calculator_negotiation_metrics import calculate_and_print_metrics
import time
def run_training(wechat_service,run_type = 'train',csv_filename='human_agent_negotiation.csv'):
    logger = TrainingLogger(db_path='../output/training_data.db')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Environment Configuration
    env_config = {
        "agent_is_seller": True,
        "max_consecutive_fuzzy_offers": 3,
    }
    env = NegotiationEnv(env_config=env_config,max_turn=8,type=run_type,wechat_service=wechat_service)
    # Agent Hyperparameters
    # These kwargs are passed to PDQNAgent constructor, which MultiPassPDQNAgent calls via super()
    agent_params = {
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "actor_kwargs": {'hidden_layers': (128, 64), 'activation': 'relu'},
        "actor_param_kwargs": {'hidden_layers': (128, 64), 'activation': 'relu'},
        "batch_size": 128,
        "learning_rate_actor": 1e-3,  # For Q-network (MultiPassQActor)
        "learning_rate_actor_param": 1e-4,  # For ParamActor
        "gamma": 0.99,
        "tau_actor": 0.1,
        "tau_actor_param": 0.001,
        "epsilon_initial": 1.0,
        "epsilon_final": 0.01,
        "epsilon_steps": 6000,  # Total environment steps for epsilon decay
        "replay_memory_size": 100000,
        "initial_memory_threshold": 2000,#2000,  # Start learning after this many experiences
        "use_ornstein_noise": True,
        "clip_grad": 10.0,  # Gradient clipping
        "inverting_gradients": True,  # From DDPG, often helpful
        "device": device,
        "seed": 42,
        "start_epsilon" : 0
    }

    agent = MultiPassPDQNAgent(**agent_params)
    total_env_steps = 0
    print("Agent Initialized.")
    start_episode = 1

    CHECKPOINT_PATH = '../output/Param_QLearning/negotiation_checkpoint'
    model_file='checkpoint.pt'
    # 检查是否存在检查点文件
    if os.path.exists(os.path.join(CHECKPOINT_PATH, model_file)):
        print("发现已存在的检查点，正在加载...")
        try:
            start_episode, total_env_steps = agent.load_checkpoint(CHECKPOINT_PATH,model_file,load_type='test')
            agent._episode = start_episode
            start_episode += 1  # 从下一个 episode 开始
        except Exception as e:
            print(f"加载检查点失败: {e}。将从头开始训练。")
            start_episode = 1
            total_env_steps = 0
    else:
        print("未找到检查点，将从头开始训练。")
    #len_dataset = len(dataset)
    num_episodes = 2000
    print_every_episodes = 20
    save_every_episodes = 100
    last_saved_episode = start_episode - 1
    try:
        if run_type =='test':
            results = []
            output_dir = "../output/human_agent_negotiation_evaluate"
            output_result_filename = "human_agent_negotiation_evaluate_result"
            file_exists = os.path.isfile(csv_filename)
            if not file_exists:
                fieldnames = ['success', 'final_price', 'initial_seller_price',
                              'seller_bottom_price', 'initial_buyer_price',
                              'buyer_max_price', 'turns', 'quantity']
                with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
            else:
                with open(csv_filename, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        result_entry = {
                            'success': row.get('success', '').strip() == 'True',  # 直接比较 'True'
                            'final_price': float(row.get('final_price', 0)) if row.get('final_price') else 0.0,
                            'initial_seller_price': float(row.get('initial_seller_price', 0)) if row.get(
                                'initial_seller_price') else 0.0,
                            'seller_bottom_price': float(row.get('seller_bottom_price', 0)) if row.get(
                                'seller_bottom_price') else 0.0,
                            'initial_buyer_price': float(row.get('initial_buyer_price', 0)) if row.get(
                                'initial_buyer_price') else 0.0,
                            'buyer_max_price': float(row.get('buyer_max_price', 0)) if row.get(
                                'buyer_max_price') else 0.0,
                            'turns': int(row.get('turns', 0)) if row.get('turns') else 0,
                            'quantity': int(row.get('quantity', 1)) if row.get('quantity') else 1,
                        }
                        results.append(result_entry)
        for episode in range(start_episode, num_episodes + 1):
            # 在开始等待新会话前，确保标志是清除的（表示空闲）
            wechat_service.episode_processing_lock.clear()
            print(f"Agent is ready for a new session.")
            # 等待新用户连接
            user_id, product_id = wechat_service.wait_for_new_session()
            if user_id is None:
                time.sleep(1)  # 如果没有新会话，稍等一下避免空转
                continue
            product = wechat_service.get_product_by_id(product_id)
            # 一旦获取到新会话，立刻设置“忙碌”标志
            wechat_service.episode_processing_lock.set()
            print(f"Agent is now busy with user {user_id}.")
            # 直接在训练循环中生成并发送欢迎语
            welcome_message_text = (
                f"您好！欢迎参与{product.get('product_name')}的谈判。"
                f"商品描述: {product.get('seller_item_description')}，"
                f"销售价格：¥{product.get('init_price')}。请提出您的报价或询问商品详情。"
            )
            # 构造前端期望的结构化消息体
            welcome_payload = {
                "action": "continue",  # 表示谈判尚未结束
                "response": welcome_message_text,
                "deal_price": None
            }

            wechat_service.send_message(user_id, welcome_payload)
            state,buyer_intent = env.reset(product, user_id)
            if state is None and (buyer_intent is None or buyer_intent== 2):
                wechat_service.end_session(user_id)
                if run_type=='test':
                    '''
                    results.append({
                        'success': False,
                        'final_price': None,
                        'initial_seller_price': product.get('init_price'),
                        'seller_bottom_price': product.get('seller_reserve_price'),
                        'initial_buyer_price': None,
                        'buyer_max_price': product.get('buyer_reserve_price'),
                        'turns': env.current_round,
                        'quantity': product.get('quantity')
                    })
                    '''
                continue
            agent.start_episode()
            episode_reward = 0
            episode_steps = 0
            done = False
            last_saved_episode = episode  # 更新最后处理的回合
            while not done:
                if buyer_intent ==1 or buyer_intent ==4 or buyer_intent ==5:
                    action_for_env=(-100,buyer_intent)
                    state, _, done, _,buyer_intent = env.step(action_for_env)
                    if run_type =='train':
                        episode_steps += 1
                        total_env_steps += 1
                        if agent.epsilon_steps > 0:
                            if total_env_steps < agent.epsilon_steps:
                                agent.epsilon = agent.epsilon_initial - (agent.epsilon_initial - agent.epsilon_final) * (
                                            total_env_steps / agent.epsilon_steps)
                            else:
                                agent.epsilon = agent.epsilon_final
                else:
                    discrete_act_idx, params_for_chosen_action, all_actual_params_vector = agent.act(state,_eval=True)
                    action_for_env = (discrete_act_idx, params_for_chosen_action)
                    next_state, reward, done, info,buyer_intent = env.step(action_for_env)
                    episode_reward += reward
                    if run_type =='train':
                        episode_steps += 1
                        total_env_steps += 1
                        if agent.epsilon_steps > 0:
                            if total_env_steps < agent.epsilon_steps:
                                agent.epsilon = agent.epsilon_initial - (agent.epsilon_initial - agent.epsilon_final) * (
                                            total_env_steps / agent.epsilon_steps)
                            else:
                                agent.epsilon = agent.epsilon_final

                        # Prepare next_action for replay buffer
                        if not done:
                            next_discrete_act_idx, next_params_for_chosen_action, next_all_actual_params_vector = agent.act(
                                next_state)  # Get next action for storage
                            next_action_for_buffer = (next_discrete_act_idx, next_all_actual_params_vector)
                        else:
                            dummy_next_discrete_act = 0
                            # all_actual_params_vector is already numpy array from agent.act
                            dummy_next_all_params = np.zeros_like(
                                all_actual_params_vector) if agent.action_parameter_size > 0 else np.array([])
                            next_action_for_buffer = (dummy_next_discrete_act, dummy_next_all_params)

                        # Store experience in agent's replay buffer
                        agent.step(state=state,
                                   action=(discrete_act_idx, all_actual_params_vector),
                                   reward=reward,
                                   next_state=next_state,
                                   next_action=next_action_for_buffer,
                                   terminal=done)
                        '''
                        logger.log_step(
                            state=state,
                            action=(discrete_act_idx, all_actual_params_vector),
                            reward=reward,
                            next_state=next_state,
                            next_action=next_action_for_buffer,
                            done=done
                        )
                        '''
                    state = next_state

            agent.end_episode()
            if run_type == 'test':
                initial_buyer_price = env.user_history_price[0] if env.user_history_price else None
                result_entry = {
                    'success': info.get('deal_made', False),
                    'final_price': info.get('deal_price'),
                    'initial_seller_price': product.get('init_price'),
                    'seller_bottom_price': product.get('seller_reserve_price'),
                    'initial_buyer_price': initial_buyer_price,
                    'buyer_max_price': product.get('buyer_reserve_price'),
                    'turns': env.current_round,
                    'quantity': product.get('quantity')
                }
                with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
                    fieldnames = ['success', 'final_price', 'initial_seller_price',
                                  'seller_bottom_price', 'initial_buyer_price',
                                  'buyer_max_price',  'turns', 'quantity']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    # 确保结果条目包含所有必需的字段
                    writer.writerow(result_entry)
                    results.append(result_entry)
            if run_type =='train' and episode % print_every_episodes == 0:
                print(
                    f"Ep: {episode}/{num_episodes} | Steps: {episode_steps} | Total Env Steps: {total_env_steps} | Reward: {episode_reward:.2f} | Eps: {agent.epsilon:.3f}")
                '''
                if info.get("deal_made"):
                    print(f"  DEAL @ {info.get('agreed_price', 'N/A')} in {info.get('steps', 'N/A')} steps.")
                elif info.get("error"):
                    print(info.get("error"))
                elif info.get("max_steps_reached") :
                    print(f"  NO DEAL (max steps reached).")
                elif info.get("fuzzy_deadlock"):
                    print(f"  NO DEAL (fuzzy deadlock).")
                else:
                    print(f"  NO DEAL (negotiation failed).")
                '''
            if run_type =='train' and episode % save_every_episodes == 0:
                agent.save_checkpoint(CHECKPOINT_PATH, episode, total_env_steps,mode="saved_data")
                pass

            wechat_service.end_session(user_id)
    except Exception as e:
        print(f"Caught exception: {e}")
    finally:
        if run_type =='train' and last_saved_episode > 0:
            agent.save_checkpoint(CHECKPOINT_PATH, last_saved_episode, total_env_steps,model_name='final_checkpoint.pt',mode="unsaved_data")
            print("Training finished.")
            pass
        if run_type =="test":
            calculate_and_print_metrics(results, output_dir=output_dir,
                                        filename=output_result_filename)
        logger.close()
        # 确保在任何情况下退出时，都将标志清除
        wechat_service.episode_processing_lock.clear()



