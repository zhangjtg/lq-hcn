import torch
from typing import Tuple,Sequence
from torch.utils.data import Dataset, DataLoader
import json
from extras.constants import IGNORE_INDEX
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from transformers import DataCollatorForSeq2Seq
# 读取JSON数据
# Step 1: Data loading and sorting
def load_and_sort_data(json_file_path):
	# Load data from JSON file
	with open(json_file_path, 'r', encoding='utf-8') as f:
		data = json.load(f)

	# Sort data by length of 'ret' field (from longest to shortest)
	sorted_data = sorted(data, key=lambda x: len(x['ret']), reverse=True)

	return sorted_data

def create_price_identification_prompt(item):
	prompt = f"""你是谈判对话识别助手，任务是分析对话，识别当前轮次买家提及或接受的单价和商品数量，并且严格按照指定格式输出。

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
产品信息：{item['product_name']}（{item['description']}）
初始价：{item['initial_price']}
对话记录：{item['ret']}
【输出】
"""
	return prompt
# 生成推理过程的提示
def create_reasoning_prompt(item):
	prompt = f"""你是谈判对话分析助手，请详细分析对话并解释当前轮次买家的单价和数量的确定过程。

【输入数据】
产品信息：{item['product_name']}（{item['description']}）
初始价：{item['initial_price']}
对话记录：{item['ret']}
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
	return prompt
# 1. 准备数据集
class NegotiationDataset(Dataset):
    def __init__(self, data, tokenizer,template,cutoff_len):
        self.data = data
        self.tokenizer = tokenizer
        self.template = template
        self.cutoff_len = cutoff_len
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        price_prompt = create_price_identification_prompt(item)
        # 获取推理过程数据
        reasoning_prompt = create_reasoning_prompt(item)
        # Prepare labels
        price_label = item['price']
        quantity = item['quantity']
        if not isinstance(price_label, (int, float)):  # 检查是否为数值类型
            price_label = str(price_label)
        if not isinstance(quantity, (int, float)):
            quantity = str(quantity)
        reasoning_label = item['ration']

        price_quality_elements = []
        #price_quality_elements += self.template.format_system.apply(content=price_system)
        price_quality_elements += self.template.format_user.apply(content=price_prompt)
        input_prompt_ids = self.template._convert_elements_to_ids(self.tokenizer, price_quality_elements)
        price_quality_labels = self.template.format_assistant.apply(content=f'当前单价：{price_label}\n商品数量：{quantity}\n')
        price_quality_labels = self.template._convert_elements_to_ids(self.tokenizer,price_quality_labels)
        input_price_ids, price_quality_labels = self._encode_source_label(input_prompt_ids, price_quality_labels)
        '''
        price_pattern = "当前单价："
        quantity_pattern = "\n商品数量："
        #end_pattern ="\n<|im_end|>\n"
        price_quantity_mask = [0] * len(price_quality_labels)
        # Find indices of the tokens following these patterns
        price_idx = find_value_tokens_after_pattern(price_quality_labels,self.tokenizer, price_pattern,style=0)
        quantity_idx = find_value_tokens_after_pattern(price_quality_labels,self.tokenizer, quantity_pattern,style=1)
        if len(price_idx)>0 or len(quantity_idx)>0:
            for idx in price_idx + quantity_idx:
                if 0 <= idx < len(price_quantity_mask):
                    price_quantity_mask[idx] = 1.0
        '''
        model_inputs = defaultdict(list)
        model_inputs["input_ids"].append(input_price_ids)
        model_inputs["attention_mask"].append([1] * len(input_price_ids))  # mask == len(input_ids)
        model_inputs["labels"].append(price_quality_labels)

        reason_elements = []
        # reason_elements += self.template.format_system.apply(content=system)
        reason_elements += self.template.format_user.apply(content=reasoning_prompt)
        input_reason_ids = self.template._convert_elements_to_ids(self.tokenizer, reason_elements)
        reasoning_labels = self.template.format_assistant.apply(content=reasoning_label)
        reasoning_labels = self.template._convert_elements_to_ids(self.tokenizer, reasoning_labels)
        input_reason_ids,reasoning_labels =  self._encode_source_label(input_reason_ids,reasoning_labels)
        model_inputs["reason_ids"].append(input_reason_ids)
        model_inputs["reason_mask"].append([1] * len(input_reason_ids))  # mask == len(input_ids)
        model_inputs["reason_labels"].append(reasoning_labels)
        '''
        price_quantity_mask = self._encode_price_quality_mask(input_prompt_ids,price_quantity_mask)
        model_inputs["price_quantity_mask"].append(price_quantity_mask)
        '''
        return model_inputs
    def _encode_price_quality_mask(self,source_ids,target_mask):
        source_len, target_len = infer_seqlen(len(source_ids), len(target_mask), self.cutoff_len)
        source_mask = [0] * source_len
        price_quality_mask = source_mask + target_mask
        return price_quality_mask
    def _encode_source_label(self,source_ids,target_ids):
        source_len, target_len = infer_seqlen(len(source_ids), len(target_ids), self.cutoff_len)
        source_ids = source_ids[:source_len]
        target_ids = target_ids[:target_len]
        total_length = source_len + target_len
        source_label = [IGNORE_INDEX] * source_len  # source_label标签IGNORE_INDEX，长度是source_len
        target_label = target_ids  # target_lable:tokenizer(ouput+<|im_end|>\n)
        input_ids = source_ids + target_ids  # source_ids:tokenizer(<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n <|im_start|>user\n prompt + input <|im_end|>\n<|im_start|>assistant\n)
        labels = source_label + target_label
        return input_ids,labels
def find_value_tokens_after_pattern(tokens, tokenizer, pattern,style):
    pattern_tokens = tokenizer.encode(pattern, add_special_tokens=False)
    pattern_len = len(pattern_tokens)
    enter_id = tokenizer('\n')['input_ids']
    indices = [i for i, x in enumerate(tokens) if x == enter_id[0]]
    if style == 0 and len(indices) >0:
        end_index = indices[0]
    elif style == 1 and len(indices) >1:
        end_index = indices[1]
    else :
        end_index = len(tokens)
    # Find where the pattern appears
    for i in range(len(tokens) - pattern_len):
        if tokens[i:i + pattern_len] == pattern_tokens:
            # The value should be right after the pattern
            # Return several tokens after the pattern to capture the full value
            return list(range(i + pattern_len, min(end_index,len(tokens))))
    return []
# Step 3: 修改数据整合器以支持批次中动态填充
def infer_seqlen(source_len: int, target_len: int, cutoff_len: int) -> Tuple[int, int]:
    r"""
    Computes the real sequence length after truncation by the cutoff_len.
    """
    if target_len * 2 < cutoff_len:  # truncate source
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:  # truncate target
        max_target_len = cutoff_len - source_len
    else:  # truncate both
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))

    new_target_len = min(max_target_len, target_len)
    max_source_len = max(cutoff_len - new_target_len, 0)
    new_source_len = min(max_source_len, source_len)
    return new_source_len, new_target_len





@dataclass
class NegotiationDataCollator(DataCollatorForSeq2Seq):
    """
    Custom data collator for the NegotiationDataset that handles padding for both price and reasoning data.
    Extends DataCollatorForSeq2Seq to properly pad inputs with multiple sets of data.
    """

    def __init__(
            self,
            tokenizer,
            padding=True,
            #max_length=2048,  # 设置合理的 max_length
            pad_to_multiple_of=8,
            label_pad_token_id=-100,
            return_tensors="pt"
    ):
        super().__init__(
            tokenizer,
            padding=padding,
            #max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            label_pad_token_id=label_pad_token_id,
            return_tensors=return_tensors
        )

    def __call__(self,  features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        # Process the main input_ids, attention_mask, and labels (for price prediction)
        pred_features = []
        expl_features = []
        for feature in features:
            pred_features.append({
                'input_ids': feature['input_ids'][0],
                'attention_mask': feature['attention_mask'][0],
                'labels': feature['labels'][0]
            })
            expl_features.append({
                "input_ids": feature['reason_ids'][0],
                'attention_mask': feature['reason_mask'][0],
                'labels': feature['reason_labels'][0]
            })
        # Process the batch using the parent class's padding logic
        expl_features: Dict[str, "torch.Tensor"] = super().__call__(expl_features)

        pred_features: Dict[str, "torch.Tensor"] = super().__call__(pred_features)
        '''
        if 'price_quantity_mask' in features[0]:
            max_length = pred_features['input_ids'].size(1)
            masks = []
            for f in features:
                mask = f['price_quantity_mask'][0]
                # Convert to tensor if it's a list
                if isinstance(mask, list):
                    mask = torch.tensor(mask, dtype=torch.float)
                # Pad or truncate to match max_length
                if len(mask) < max_length:
                    # Pad with zeros
                    padded_mask = torch.zeros(max_length, dtype=torch.float)
                    padded_mask[:len(mask)] = mask
                else:
                    # Truncate
                    padded_mask = mask[:max_length]
                masks.append(padded_mask)

            # Stack all masks into a batch
            pred_features['price_quantity_mask'] = torch.stack(masks)
        '''
        return {
            'pred': pred_features,
            'expl': expl_features,
        }
