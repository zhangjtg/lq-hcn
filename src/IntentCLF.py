import torch
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers.data import  *
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np
import os , json,gc
from transformers import (AutoTokenizer,AutoModelForSequenceClassification,TrainingArguments, Trainer,
                          AutoModel,LlamaForSequenceClassification,LlamaTokenizer,FalconForSequenceClassification,AutoConfig)
import evaluate
import torch.nn as nn
os.environ["TOKENIZERS_PARALLELISM"] = "false"
def read_txt_files(directory_path):
    """
    从指定目录读取所有txt文件，并将文件名作为标签
    """
    texts = []
    labels = []
    label_names = []
    filenames = ['Deal.txt', 'Bargaining.txt', 'product specification enquiry.txt', 'Delivery enquiry.txt',
                 'Greeting.txt', 'Fail negotiation.txt']
    # 遍历目录中的所有txt文件
    for i, filename in enumerate(filenames):
        if filename.endswith('.txt'):
            label_name = os.path.splitext(filename)[0]  # 获取不带扩展名的文件名作为标签名
            label_names.append(label_name)

            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

                # 处理每一行数据
                for line in lines:
                    line = line.strip()
                    if line:  # 排除空行
                        texts.append(line)
                        labels.append(i)  # 使用文件索引作为数字标签
        # 创建标签映射字典
    id2label = {i: name for i, name in enumerate(label_names)}
    label2id = {name: i for i, name in enumerate(label_names)}
    return texts, labels, id2label, label2id


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    metrics = {}
    # 准确率
    accuracy = evaluate.load("metric/accuracy").compute(predictions=predictions, references=labels)
    metrics.update(accuracy)
    # 精确率（macro 平均）
    precision = evaluate.load("metric/precision").compute(
        predictions=predictions,
        references=labels,
        average="macro"
    )
    metrics["precision"] = precision["precision"]
    # 召回率（macro 平均）
    recall = evaluate.load("metric/recall").compute(
        predictions=predictions,
        references=labels,
        average="macro"
    )
    metrics["recall"] = recall["recall"]
    # F1 分数（macro 平均）
    f1 = evaluate.load("metric/f1").compute(
        predictions=predictions,
        references=labels,
        average="macro"
    )
    metrics["f1"] = f1["f1"]
    # 按类别拆分的指标
    # 精确率（每个类别）
    precision_per_class = evaluate.load("metric/precision").compute(
        predictions=predictions,
        references=labels,
        average=None  # 关键参数！
    )
    metrics["precision_per_class"] = precision_per_class["precision"].tolist()
    # 召回率（每个类别）
    recall_per_class = evaluate.load("metric/recall").compute(
        predictions=predictions,
        references=labels,
        average=None
    )
    metrics["recall_per_class"] = recall_per_class["recall"].tolist()
    # 每个类别的 F1
    class_f1 = evaluate.load("metric/f1").compute(
        predictions=predictions,
        references=labels,
        average=None
    )
    metrics["f1_per_class"] = class_f1["f1"].tolist()  # 转换为列表以便序列化
    return metrics


def preprocess_function1(examples):
    global text_name
    if isinstance(text_name, str):
        d = examples[text_name]
    else:
        d = examples[text_name[0]]
        for n in text_name[1:]:
            nd = examples[n]
            assert len(d) == len(nd)
            for i, t in enumerate(nd):
                d[i] += '\n' + t

    return tokenizer(d, padding='longest', max_length=max_length, truncation=True)
# 定义预处理函数
def preprocess_function2(examples):
    enc = tokenizer(examples["text"], truncation=True, padding="max_length",max_length=max_length)
    #enc["labels"] = examples["label"]  # 或者在上面 rename 之后就是 examples["labels"]
    return enc
# 主函数
def main():
    # 设置参数
    torch.cuda.empty_cache()
    gc.collect()
    epochs = 20
    batch_size = 16
    learning_rate = 5e-5
    lora_r = 12
    global max_length
    max_length = 128
    model_path = '/home/zjt/local/model_bin/TinyLlama_v1.1_chinese'
    output_dir = "../output/intent/TinyLlama_v1.1_chinese1115"
    with open("../../../data/intent/intent_train.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)
        # 读取 JSON 文件
    with open("../../../data/intent/intent_test.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
    # 提取字段
    train_texts = [item["instruction"] for item in train_data]
    train_labels = [item["output"] for item in train_data]
    # 提取字段
    test_texts = [item["instruction"] for item in test_data]
    test_labels = [item["output"] for item in test_data]
    # 获取标签集合并创建标签到ID的映射
    #label_set = set(train_labels)
    label2id = {'讨价还价': 0, '问候': 1, '谈判失败': 2, '谈判成功': 3, '材料咨询': 4, '物流咨询': 5}#{label: idx for idx, label in enumerate(label_set)}
    id2label = {idx: label for label, idx in label2id.items()}

    # 将标签从字符串映射为ID
    train_labels = [label2id[label] for label in train_labels]
    test_labels = [label2id[label] for label in test_labels]
    # 创建 Hugging Face 数据集
    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

    # 合并为 DatasetDict
    dataset = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })
    global tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    #config = FalconConfig.from_pretrained(model_path)
    # 手动指定使用 GemmaConfig（如果结构兼容）
    model = LlamaForSequenceClassification.from_pretrained(
        model_path, num_labels=len(id2label), id2label=id2label, label2id=label2id, torch_dtype=torch.bfloat16,#config=config,
        trust_remote_code=True)
    ''' 
    # 1. 先加载主干模型
    base_model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)

    # 2. 创建分类模型结构（不加载权重）
    config = AutoConfig.from_pretrained(model_path, num_labels=len(id2label), id2label=id2label, label2id=label2id)
    # 加载原始配置
    config = AutoConfig.from_pretrained(
        model_path,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        trust_remote_code=True
    )

    # ✅ 关键：确保 pad_token_id 合法
    if config.pad_token_id is None or config.pad_token_id < 0:
        # 尝试从 tokenizer 获取
        if tokenizer.pad_token_id is not None:
            config.pad_token_id = tokenizer.pad_token_id
        elif tokenizer.eos_token_id is not None:
            config.pad_token_id = tokenizer.eos_token_id
        else:
            config.pad_token_id = 0  # 最后兜底
    # ✅ 再检查是否在合法范围内
    if not (0 <= config.pad_token_id < config.vocab_size):
        raise ValueError(
            f"pad_token_id ({config.pad_token_id}) 必须在 [0, {config.vocab_size}) 范围内"
        )
    model = AutoModelForSequenceClassification.from_config(config, torch_dtype=torch.bfloat16,trust_remote_code=True)

    # 3. 手动复制主干权重（除了 classifier）
    model.base_model.load_state_dict(base_model.state_dict(), strict=False)  # strict=False 忽略不匹配的层
    '''

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        # 或者直接赋值
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id  # 显式设置pad_token_id
        # 调整模型的词嵌入层以适应新的词汇表大小
        model.resize_token_embeddings(len(tokenizer))
   #model.config = getattr(model, "config", base.config)
    model.config.pad_token_id = tokenizer.pad_token_id
    peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=lora_r, lora_alpha=32,
                             lora_dropout=0.1,target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]) #
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    tokenized_datasets = dataset.map(preprocess_function2, batched=True,
    batch_size=batch_size,)
    '''
    # ★ 把列名 "label" 改成 "labels"
    if "labels" in tokenized_datasets["train"].column_names:
        tokenized_datasets = tokenized_datasets.remove_columns(["label"])
    else:
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    '''
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        eval_strategy="no",
        save_strategy="no",
        load_best_model_at_end=True,
        push_to_hub=False,
        #gradient_accumulation_steps=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    # 评估模型
    print("评估模型...")
    test_results = trainer.evaluate(tokenized_datasets["test"])
    print(f"测试集评估结果: {test_results}")
    print("保存模型、分词器和标签映射...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    # 保存为 JSON
    with open("../output/intent/test_output/TinyLlama_v1.1_chinese.json", "w") as f:
        json.dump(test_results, f, indent=4)
    print("训练完成!")
if __name__ == "__main__":
    main()

