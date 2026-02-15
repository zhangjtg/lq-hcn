import torch
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers.data import  *
from transformers import TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import numpy as np
import os , json,gc
from transformers import RobertaTokenizer, RobertaForSequenceClassification,BertForSequenceClassification,BertTokenizer
import evaluate
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
    return tokenizer(examples["text"], truncation=True, padding="max_length",max_length=max_length)
# 主函数
def main():
    torch.cuda.empty_cache()
    gc.collect()
    # 设置参数
    epochs = 20
    batch_size = 32
    learning_rate = 5e-5
    lora_r = 12
    global max_length
    max_length = 128
    model_path = '/home/zjt/local/model_bin/roberta-large'
    '''
    data_dir = "../../../data/intent"  # 包含6个txt文件的目录
    output_dir = "../bert-base-chinese-classification"
    texts, labels,  id2label, label2id = read_txt_files(data_dir)
    # 将数据分割为训练集、验证集和测试集
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        test_texts, test_labels, test_size=0.5, random_state=42, stratify=test_labels
    )
    '''
    output_dir = "../output/intent/roberta-large-classification83"
    # 读取 JSON 文件
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
    # 加载RoBERTa tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    # 加载预训练模型
    model = RobertaForSequenceClassification.from_pretrained(
        model_path,
        num_labels=len(id2label),torch_dtype=torch.bfloat16
    )
    '''
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        # 或者直接赋值
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id  # 显式设置pad_token_id
        # 调整模型的词嵌入层以适应新的词汇表大小
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
    '''
    peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=lora_r, lora_alpha=32,
                             lora_dropout=0.1,target_modules=[ "query","key","value","dense", "output.dense"])
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    #tokenized_ds = ds.map(preprocess_function, batched=True)
    tokenized_datasets = dataset.map(preprocess_function2, batched=True,
    batch_size=batch_size,)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        eval_strategy="no",  # eval_strategy="epoch",不进行评估
        weight_decay=0.01,
        save_strategy="no",
        load_best_model_at_end=False,
        push_to_hub=False,
        #gradient_accumulation_steps=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        #eval_dataset=tokenized_datasets["validation"],
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
    with open("../output/intent/test_output/roberta-large-classification_results83.json", "w") as f:
        json.dump(test_results, f, indent=4)
    print("训练完成!")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors=None  # 返回普通 list，适合 Dataset
    )
def test():
    with open("../../../data/intent/intent_test.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
        # 提取字段
    test_texts = [item["instruction"] for item in test_data]
    test_labels_str = [item["output"] for item in test_data]
    model_path = '/home/zjt/local/model_bin/bert-base-chinese'
    label2id = {'讨价还价': 0, '问候': 1, '谈判失败': 2, '谈判成功': 3, '材料咨询': 4,
                '物流咨询': 5}  # {label: idx for idx, label in enumerate(label_set)}
    id2label = {idx: label for label, idx in label2id.items()}
    # 转换标签为 ID
    test_labels = [label2id[label] for label in test_labels_str]
    global tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_path)
    # 加载预训练模型
    model = BertForSequenceClassification.from_pretrained(
        model_path,
        num_labels=len(id2label), torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(model, '../output/intent/bert-base-chinese-classification-727')
    model.eval()
    test_dataset = Dataset.from_dict({
        "text": test_texts,
        "label": test_labels
    })
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    # -----------------------------
    # 7. 设置 TrainingArguments 和 Trainer
    # -----------------------------
    training_args = TrainingArguments(
        output_dir="../output/intent/test_output",
        per_device_eval_batch_size=32,
        eval_strategy="no",  # 不自动 eval
        report_to="none",
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # -----------------------------
    # 8. 开始测试评估
    # ----------------------------
    results = trainer.evaluate()
    print("✅ 最终测试指标：")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        elif isinstance(value, list):
            # 将列表中的 float 格式化输出
            formatted_list = [f"{v:.4f}" for v in value]
            print(f"  {key}: [{', '.join(formatted_list)}]")
        else:
            print(f"  {key}: {value}")
if __name__ == "__main__":
    main()

