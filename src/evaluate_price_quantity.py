import torch
import argparse
import os , json,gc,re
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
def custom_metrics(y_true_list, y_pred_list, tolerance=0.01,price_flag=True):
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

def main(args):
	# 设置参数
	# 手动调用垃圾回收器
	gc.collect()
	# 清理 CUDA 缓存
	torch.cuda.empty_cache()
	tokenizer = AutoTokenizer.from_pretrained(args.lora_path)
	model = AutoModelForCausalLM.from_pretrained(
		args.load_mode,
		device_map="auto",
		torch_dtype=torch.bfloat16
	)
	model = PeftModel.from_pretrained(model, args.lora_path)
	model = model.merge_and_unload()  # 合并适配器
	model.eval()  # 设置为评估模式
	with open(args.test_file, 'r', encoding='utf-8') as f:
		datas = json.load(f)
	predictions = []
	ground_truth =[]
	for d in datas:
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
产品信息：{d['product_name']}（{d['description']}），初始价：{d['initial_price']}
对话记录：{d['ret']}

【输出】
"""
		input = tokenizer(prompt, return_tensors="pt").to('cuda')
		with torch.no_grad():  # 推理时不需要计算梯度
			output = model.generate(
				**input,  # 输入数据
				max_new_tokens=50,  # 最大生成的新 token 数量
				temperature=0,  # 降低随机性,
				#top_p=0.7,
				#num_return_sequences=1,  # 仅返回一个结果
				#no_repeat_ngram_size=2,  # 避免重复短语（如 "当前轮次买家的要价"）
				do_sample=False,
				early_stopping=True,
				#length_penalty=2.0,  # 惩罚过长生成（参考知识库[1]
			eos_token_id=tokenizer.eos_token_id)  # 禁用采样，使用贪婪搜索
		response = tokenizer.decode(output[0][input.input_ids.shape[1]:], skip_special_tokens=True)
		# 使用正则表达式提取
		price_match = re.search(r"当前单价[：:]\s*(\d+\.?\d*|null)", response)
		if price_match:
			price_str = price_match.group(1)  # 获取捕获组内容
			price = float(price_str) if price_str != 'null' else 'null'  # 转换为浮点数或 None
		else:
			price = 'null'  # 如果未匹配到，设置为 None
			print(response)

		# 提取商品数量，可以是数字或null
		quantity_match = re.search(r"商品数量[：:]\s*(\d+\.?\d*|null)", response)
		if quantity_match:
			quantity_str = quantity_match.group(1)  # 获取捕获组内容
			quantity = float(quantity_str) if quantity_str != 'null' else 'null'  # 转换为浮点数或 None
		else:
			quantity = 'null'  # 如果未匹配到，设置为 None

		predictions.append({'price': price, 'quantity': quantity})
		ground_truth.append({'price': d['price'], 'quantity': d['quantity']})
	results = evaluate(predictions, ground_truth)
	# 将结果以 JSON 格式写入文件
	with open(args.output_result, "w") as file:
		json.dump(results, file, indent=4)
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--test_file', type=str,default='../../../data/price_quantity/HQL_price_test_0811.json')
	parser.add_argument('--output_result', type=str,default='../output/price_model/price_evaluate/HLQ_price_quantity_evaluate_a3_618')
	parser.add_argument('--load_mode', type=str,default='/home/zjt/local/model_bin/Qwen2.5-1.5B-Instruct')
	parser.add_argument('--lora_path', type=str,default='../output/price_model/Qwen2p5-1p5B-instruct-a3-l4-618')
	args = parser.parse_args()
	main(args)