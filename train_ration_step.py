import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
import json,os

def extract_analysis_and_results(text):
    # 分割分析过程和计算结果
    analysis_match = re.search(r'## 详细分析过程[\s\S]*', text, re.DOTALL)

    # 提取并输出匹配的部分
    if analysis_match:
        analysis_text = analysis_match.group().lstrip('\n')
    elif re.search(r'</think>\n\n(.*)', text, re.DOTALL):
        analysis_match = re.search(r'</think>\n\n(.*)', text, re.DOTALL)
        analysis_text = analysis_match.group(1).strip()
    else:
        analysis_text = None
    # 初始化 price 和 quantity 的默认值
    price = 'null'
    quantity = 'null'

    price_match = re.search(r'当前单价[：:]\s*([\d,]+\.?\d*|null)', analysis_text)
    quantity_match = re.search(r'商品数量[：:]\s*([\d,]+\.?\d*|null)', analysis_text)
    if price_match:
        price_str = price_match.group(1).lower().replace(',', '')  # 获取捕获组内容并转为小写
        price = price_str if price_str == 'null' else float(price_str)

    if quantity_match:
        quantity_str = quantity_match.group(1).lower().replace(',', '')  # 获取捕获组内容并转为小写
        quantity = quantity_str if quantity_str == 'null' else float(quantity_str)
    return analysis_text,price,quantity
def extract_analysis_and_answer(text):
    # 提取 <分析> 和 </分析> 之间的内容
    analysis_pattern = r'<分析>(.*?)</分析>'
    analysis_match = re.search(analysis_pattern, text, re.DOTALL)
    analysis_content = analysis_match.group(1) if analysis_match else None

    parts = text.split("</分析>")
    if len(parts) < 2:
        return None, None, None

    # 提取标签后的内容并清理空白字符
    json_str = parts[-1].strip()

    # 提取字段（处理中英文字段名兼容）
    price_key = "当前单价"
    quantity_key = "商品数量"
    data = json.loads(json_str)
    # 处理可能的英文字段名（如示例中的current buyer's mentioned product quantity）
    if quantity_key not in data:
        quantity_key = "current buyer's mentioned product quantity"
    if price_key not in data:
        price_key = "current buyer's offer price"
    price = data.get(price_key,None)
    quantity = data.get(quantity_key,None)
    # 类型转换（根据规则）
    if isinstance(price, (int, float)):
        price = round(float(price), 2)  # 保留两位小数
    else:
        price = 'null'
    if isinstance(quantity, (int, float)):
        quantity = round(float(quantity), 1)
    else:
        quantity = 'null'
    '''
    # 匹配答案行中的数值组合
    match = re.search(r'答案：\s*([\d.]+|null)\s*\|\s*([\d.]+|null)', text)

    if match:
        price_str, quantity_str = match.groups()

        # 转换数据类型
        if re.match(r'^\d+(\.\d+)?$', price_str):
            price = float(price_str)
        else:
            price = 'null'
        if re.match(r'^\d+(\.\d+)?$', quantity_str):
            quantity = float(quantity_str)
        else:
            quantity = 'null'
    else:
        price,quantity = 'null','null'
    '''

    return analysis_content, price,quantity
def main():
    with open("../../../data/price_quantity/extract_price_quantity_train_data.json", "r", encoding="utf-8") as file:
        data_list = json.load(file)
    file_path = "../../../data/price_quantity/LQ_extract_price_ration.json"
    # 加载 DeepSeek 定制模型
    model_path = "/home/zjt/local/zjt/model-bin/DeepSeek-R1-Distill-Qwen-14B"  # 本地路径或HF ID
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).eval()
    # 示例JSON数据
    # 清空文件或初始化（首次运行）
    # 如果文件不存在才执行
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("[\n")  # 初始化为 JSON 数组
    for data in data_list:
        # 调用函数解析并打印
        product_id = data.get("id")
        product_name = data.get("product_name")
        initial_price = data.get("initial_price")
        description = data.get("description")
        buyer_reserve_price = data.get("buyer_reserve_price")
        seller_reserve_price = data.get("seller_reserve_price")

        dialogue = data.get("dialogue", [])
        #if product_id <= 155 or product_id >= 158:
           #continue
        # 打印字段内容
        '''
        print("解析结果：\n")
        print(f"ID: {product_id}")
        '''
        # 设计结构化推理 Prompt
        # seps = [' ', '</s>']
        for i, turn in enumerate(dialogue, start=0):
            if turn['role'] == 'buyer':
                role_text = f'买家'
            else:
                role_text = f'卖家'
            if i == 0:
                ret = role_text + "：" + turn['message'] + '\n'
            else:
                ret += role_text + "：" + turn['message'] + '\n'
            if turn['role'] == 'buyer':
                prompt = f"""你是谈判对话分析助手，根据买卖双方的谈判对话，通过推理分析确定当前轮次买家的单价和提及的商品数量。

【输入数据】
产品信息：{product_name}（{description}），初始价：{initial_price}
对话记录：{ret}
【分析任务】
分析最后一轮买家发言，识别其提及或接受的单价和商品数量。

【分析规则】
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

【分析步骤】
1. 首先，确定当前对话的最后一轮，找出买家的发言内容。
2. 梳理前面轮次讨论数量和价格的关键信息，建立连贯的对话上下文。
3. 对买家发言进行详细分析：
   - 是否明确提及价格（如"预算最多X"，"X能卖吗？"等）
   - 是否明确接受或确认卖家提出的价格（如"X元可以"、"这个价格我接受"等）
   - 是否同时提及总价和数量，以计算单价
   - 是否有折扣表述
   - 是否根据上一轮报价进行调整（如"再减X%"）
   - 是否明确提及商品数量
   - 是否延续前面讨论的数量（已确定数量自动延续至后续对话）
4. 价格处理优先级：
   - 明确提及的单价 > 明确接受的价格 > 折扣计算的单价 > 从总价计算的单价 > 间接引用的单价
5. 数量处理优先级：
   - 当前轮次明确数量 > 上下文确定数量 > 默认数量1（当买家提价时）> null
6. 如果需要计算折扣，明确折扣基准价的确定依据

【输出要求】
请严格按以下格式输出分析结果：
## 详细分析过程
(在这里列出完整的推理过程，解释如何确定当前轮次买家的单价和数量)
## 计算结果
当前单价：[具体数值/null]
商品数量：[具体数值/null]
"""

                # 生成推理过程
                inputs = tokenizer(prompt, return_tensors="pt").to(teacher_model.device)
                outputs = teacher_model.generate(
                    inputs.input_ids,
                    max_new_tokens=1024,
                    temperature=0.1,  # 降低随机性
                    top_p=0.7,
                    repetition_penalty=1.2,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

                # 后处理输出
                full_response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
                #rationale = full_response.split(prompt)[-1].strip()
                ration,price,quantity=extract_analysis_and_results(full_response)
                if ration is None or len(ration) < 5:
                    continue
                data_w = {
                    "id": product_id,
                    "product_name":product_name,
                    "initial_price":initial_price,
                    "description":description,
                    "buyer_reserve_price":buyer_reserve_price,
                    "seller_reserve_price":seller_reserve_price,
                    "ret":ret,
                    "ration": ration,
                    "price": price,
                    "quantity":quantity
                }
                # 保存数据
                # 将字典转换为 JSON 字符串，并追加到文件
                with open(file_path, 'a', encoding='utf-8') as f:
                    # 需要处理逗号和换行，确保 JSON 数组格式正确
                    f.write(f" {json.dumps(data_w, ensure_ascii=False)},\n")
    # 循环结束后补充结尾的 ] 符号
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write("]")

if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--max_steps', type=int, default=10000)
    #parser.add_argument('--eval_steps', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--optimizer_name', type=str, default='AdamW')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--from_pretrained', type=str)
    parser.add_argument('--max_input_length', type=int, default=1024)
    parser.add_argument('--grad_steps', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--gen_max_len', type=int, default=64)
    parser.add_argument('--parallelize', action='store_true')
    parser.add_argument('--model_type', type=str, default='task_prefix')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--output_rationale', action='store_true')
    args = parser.parse_args()
    '''
    main()