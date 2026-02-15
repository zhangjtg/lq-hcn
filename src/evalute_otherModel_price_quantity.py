import torch
import argparse
import os , json,gc,re
from transformers import AutoTokenizer, AutoModelForCausalLM, PhiConfig
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np


def custom_metrics(y_true_list, y_pred_list, tolerance=0.01,price_flag=True):
    """
    计算允许误差范围内的准确率、精确率、召回率和 F1 分数。

    参数:
        y_true (list or array): 真实值列表。
        y_pred (list or array): 预测值列表。
        tolerance (float): 允许的最大误差，默认为 0.01。

    返回:
        dict: 包含准确率、精确率、召回率和 F1 分数的字典。
    """
    y_true_binary = []
    y_pred_binary = []
    for gt, pred in zip(y_true_list, y_pred_list):
        y_true_binary.append(1)  # 真实标签：希望预测在容忍范围内
        # 处理真实值为 null 的情况
        if gt == 'null' or pred == 'null':
            if gt == pred:
                y_pred_binary.append(1)
            else:
                y_pred_binary.append(0)
        else:
            # 处理数值比较
            if price_flag:
                is_correct = abs(round(gt,0) - round(pred,0)) <= tolerance
            else:
                is_correct = abs(gt - pred) <= tolerance
            y_pred_binary.append(1 if is_correct else 0)

    # 计算指标
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# Evaluate performance
def evaluate(predictions, ground_truth):
    # Prepare data for metrics calculation
    y_true_price = []
    y_pred_price = []
    y_true_quantity = []
    y_pred_quantity = []

    # Combined prediction (both price and quantity correct)
    y_true_combined = []
    y_pred_combined = []

    for gt, pred in zip(ground_truth, predictions):
        # Price evaluation
        true_price = gt['price']
        pred_price = pred['price']
        # If both are not None, check if they match
        price_match = False
        if true_price !='null' and pred_price !='null':
            # Allow small floating point differences
            price_match = abs(round(true_price,0) - round(pred_price,0)) < 0.1
        elif true_price =='null' and pred_price =='null':
            price_match = True

        y_true_price.append(true_price)
        y_pred_price.append(pred_price)

        # Quantity evaluation
        true_quantity = gt['quantity']
        pred_quantity = pred['quantity']

        quantity_match = False
        if true_quantity != 'null' and pred_quantity != 'null':
            quantity_match = (true_quantity == pred_quantity)
        elif true_quantity == 'null' and pred_quantity == 'null':
            quantity_match = True

        y_true_quantity.append(true_quantity)
        y_pred_quantity.append(pred_quantity)

        # Combined evaluation (both price and quantity must match)
        combined_match = price_match and quantity_match
        y_true_combined.append(1)  # Ground truth is always correct
        y_pred_combined.append(1 if combined_match else 0)

    # Calculate metrics
    results = {}
    results['price'] = custom_metrics(y_true_price, y_pred_price, tolerance=0.1)
    results['quantity'] = custom_metrics(y_true_quantity,y_pred_quantity,tolerance=0.1,price_flag=False)
    # Combined metrics
    combined_accuracy = accuracy_score(y_true_combined, y_pred_combined)
    combined_precision = precision_score(y_true_combined, y_pred_combined)
    combined_recall = recall_score(y_true_combined, y_pred_combined)
    combined_f1 = f1_score(y_true_combined, y_pred_combined)

    results['combined'] = {
        'accuracy': combined_accuracy,
        'precision': combined_precision,
        'recall': combined_recall,
        'f1': combined_f1
    }

    return results


def extract_price_and_quantity(text):
    # 匹配要价
    price_pattern = r'当前单价[：:]\s*(\d+\.?\d*|null)'
    price_match = re.search(price_pattern, text)
    if price_match:
        price_str = price_match.group(1)  # 获取捕获组内容
        price = float(price_str) if price_str!='null' else 'null'  # 转换为浮点数或 None
    else:
        print(text)
        raise Exception("无法匹配")
        price = 'null'  # 如果未匹配到，设置为 None

    # 匹配商品数量
    quantity_pattern = r'商品数量[：:]\s*(\d+\.?\d*|null)'
    quantity_match = re.search(quantity_pattern, text)
    if quantity_match:
        quantity_str = quantity_match.group(1)  # 获取捕获组内容
        quantity = float(quantity_str) if quantity_str!='null'  else 'null'  # 转换为浮点数或 None
    else:
        print(text)
        quantity = 'null'  # 如果未匹配到，设置为 None

    return price,quantity

def main(args):
	# 设置参数
	# 手动调用垃圾回收器
    gc.collect()
    # 清理 CUDA 缓存
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(args.load_mode)
    '''
    config = PhiConfig.from_pretrained(args.load_mode)
    config.rope_scaling = {"type": "dynamic", "factor": 64}
    model = AutoModelForCausalLM.from_pretrained(
        args.load_mode,
        config=config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    '''
    model = AutoModelForCausalLM.from_pretrained(
        args.load_mode,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.eval()  # 设置为评估模式
    predictions = []
    ground_truth = []
    with open(args.test_file, 'r', encoding='utf-8') as f:
        datas = json.load(f)
    for d in datas:
        prompt = f"""你是谈判对话识别助手，你的任务是分析对话，识别当前轮次买家提及或接受的单价和商品数量，并且严格按照指定格式输出当前轮次买家的单价和数量要求。

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
严格按照以下格式输出，不要添加任何额外文字或解释：
当前单价：[数值/null]
商品数量：[数值/null]

【示例1】
【输入数据】
产品信息：
- 名称：高级办公椅
- 描述：人体工学设计，透气网布
- 初始价：1200元/把
对话记录：
买家：您好，1200元一把太贵了，我要订50把，能不能给我更优惠一点？
卖家：您订这么多，我可以给到900元每把的价格。
买家：我看了一下，市场价也就800元左右，900还是高了。我们就按800元一把，一共50把，这个价格可以吗？

【示例1输出】
当前单价：800.0
商品数量：50.0

【示例2】
【输入数据】
产品信息：
- 名称：智能手表
- 描述：高清触屏，支持健康监测
- 初始价：1300元/块
对话记录：
买家：你们这个表1300太贵了，我出1100元可以吗？

【示例2输出】
当前单价：1100.0
商品数量：1.0

【示例3】
【输入数据】
产品信息：
- 名称：高性能笔记本电脑
- 描述：16GB内存，1TB固态硬盘
- 初始价：6000元/台
对话记录：
买家：我打算订购10台笔记本，不过6000元一台太贵了。
卖家：要是数量这么多，我可以给你5500元一台。
买家：能不能5000元一台？

【示例3输出】
当前单价：5000.0
商品数量：10.0

现在，请根据以上规则和示例，处理下面的输入数据：

【输入数据】
产品信息：
- 名称：{d['product_name']}
- 描述：{d['description']}
- 初始价：{d['initial_price']}
对话记录：
{d['ret']}

【输出】
"""
        messages = [
            {"role": "system", "content": prompt}
        ]
        input_ids  = tokenizer.apply_chat_template(messages,tokenize=True,return_tensors="pt",add_generation_prompt=True,enable_thinking=True ).to(model.device) # Switches between thinking and non-thinking modes. Default is True.
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=50,
            #max_new_tokens=2048,
            #temperature=0.6,
            #top_p=0.95,
            #top_k=20,
            #min_p=0,
            repetition_penalty=1.2,
            early_stopping=True,
            eos_token_id=tokenizer.eos_token_id,
            #do_sample=True
            do_sample=False
        )
        response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        '''
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        output = model.generate(
            input_ids,
            max_new_tokens=50,
            do_sample=False,
            temperature=0,
            early_stopping=True,
            eos_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
        '''
        price,quantity = extract_price_and_quantity(response)
        predictions.append({'price':price,'quantity':quantity})
        ground_truth.append({'price':d['price'],'quantity':d['quantity']})
    results = evaluate(predictions, ground_truth)
    # 将结果以 JSON 格式写入文件
    with open(args.output_result, "w") as file:
        json.dump(results, file, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str,default='../../../data/price_quantity/HQL_price_test_0811.json')
    parser.add_argument('--load_mode', type=str,default='/home/zjt/local/model_bin/Qwen2.5-3B-Instruct')
    parser.add_argument('--output_result', type=str,
                        default='../output/price_model/price_evaluate/Qwen2.5-3B-Instruct_price_quantity_evaluate_98')
    args = parser.parse_args()
    main(args)