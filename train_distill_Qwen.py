import argparse
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from peft import LoraConfig, get_peft_model, TaskType,PeftModel
from transformers.trainer_utils import set_seed
from metrics import compute_metrics_text
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)  # 替换为适用于分类任务的整理器
from negotiationData_utils import load_and_sort_data,NegotiationDataset,NegotiationDataCollator
from torch.utils.data import DataLoader
from utils.template import get_template_and_fix_tokenizer


class TaskPrefixTrainer(Seq2SeqTrainer):  #
    def __init__(self, alpha, output_rationale, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.output_rationale = output_rationale

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Add **kwargs to handle additional parameters like num_items_in_batch
        pred_outputs = model(**inputs['pred'])
        expl_outputs = model(**inputs['expl'])
        #return ( expl_outputs.loss, {'expl': expl_outputs}) if return_outputs else  expl_outputs.loss

        standard_loss = self.alpha * pred_outputs.loss + (1. - self.alpha) * expl_outputs.loss
        '''
        if 'price_quantity_mask' in inputs['pred'] and torch.sum(inputs['pred']['price_quantity_mask']).item()>0:
            logits = pred_outputs.logits
            labels = inputs['pred']['labels']

            # Use mask to focus only on the tokens corresponding to price and quantity values
            price_quantity_mask = inputs['pred']['price_quantity_mask']

            # Calculate custom focused loss on the key prediction tokens
            # This gives higher weight to errors in predicting the crucial numbers
            focused_loss = self._compute_masked_loss(logits, labels, price_quantity_mask)

            # Combine losses with appropriate weighting
            loss = 0.6 * standard_loss + 0.4 * focused_loss
        else:
            loss = standard_loss
        '''
        return (standard_loss, {'pred': pred_outputs, 'expl': expl_outputs}) if return_outputs else standard_loss


    def _compute_masked_loss(self, logits, labels, mask):
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        # Reshape for loss calculation
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_mask = mask[..., 1:].contiguous()
        # Calculate per-token loss
        per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                                  shift_labels.view(-1))

        # Apply mask to focus on price/quantity tokens
        masked_loss = per_token_loss * shift_mask.view(-1)
        return masked_loss.sum()
        # Return mean of masked loss values
        #return masked_loss.sum() / (shift_mask.sum() + 1e-8)  # Add small epsilon to avoid division by zero
    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        pred_outputs = super().prediction_step(model, inputs['pred'], prediction_loss_only=False,
                                               ignore_keys=ignore_keys)
        if self.output_rationale:
            expl_outputs = super().prediction_step(model, inputs['expl'], prediction_loss_only=False,
                                                   ignore_keys=ignore_keys)
        else:
            expl_outputs = pred_outputs  # placeholder only

        loss = self.alpha * pred_outputs[0] + (1 - self.alpha) * expl_outputs[0]

        return (
            loss,
            [pred_outputs[1], expl_outputs[1]],
            [pred_outputs[2], expl_outputs[2]],
        )

        # 覆盖get_train_dataloader方法以正确处理自定义数据结构
    def get_train_dataloader(self):
        """
        重写此方法以确保数据加载器使用正确的collate_fn和批处理逻辑
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self.args.per_device_train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        # 添加采样器配置
        if hasattr(train_dataset, "__len__"):
            train_sampler = self._get_train_sampler()
            dataloader_params["sampler"] = train_sampler
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
        else:
            dataloader_params["shuffle"] = True

        print("Creating custom train dataloader with batch size:", dataloader_params["batch_size"])

        train_dataloader = DataLoader(train_dataset, **dataloader_params)
        return train_dataloader



# Main training function
def train(args):
	# Load and sort data
    set_seed(args.run)
    sorted_data = load_and_sort_data(args.input_file)
    # Load tokenizer and model
    '''
    tokenizer = AutoTokenizer.from_pretrained(args.lora_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(model, args.lora_path)
    model = model.merge_and_unload()  # 合并适配器
    '''

    tokenizer = AutoTokenizer.from_pretrained(args.model_name,do_lower_case=False,use_fast=True)
    # Add special tokens if needed
    # tokenizer.add_special_tokens({'additional_special_tokens': ['[PRICE_TASK]', '[REASONING_TASK]']})

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    )

    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    template = get_template_and_fix_tokenizer(tokenizer, args)
    # Create dataset and data collator
    train_dataset = NegotiationDataset(sorted_data, tokenizer,template,2048)
    data_collator = NegotiationDataCollator(
    tokenizer=tokenizer,
    padding= True,
    #max_length = 2048,
    pad_to_multiple_of=8,
    label_pad_token_id=-100
    )
    compute_metrics = compute_metrics_text(tokenizer)
    # Set up training arguments
    training_args = Seq2SeqTrainingArguments(
        args.output_dir,
        remove_unused_columns=False,
        save_strategy='no',
        logging_dir=args.logging_dir,
        logging_strategy='epoch',
        learning_rate=args.lr,
        num_train_epochs = args.epochs,
        gradient_accumulation_steps=args.grad_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        seed=args.run,
        bf16=args.bf16,
        generation_max_length=args.gen_max_len,
        prediction_loss_only=False,
    )
    trainer_kwargs = {
        'alpha': args.alpha,
        'output_rationale': args.output_rationale,
        'model': model,
        'args': training_args,
        'train_dataset': train_dataset,
        'data_collator': data_collator,
        'tokenizer': tokenizer,
        'compute_metrics': compute_metrics,
    }
    # Create trainer
    trainer = TaskPrefixTrainer(**trainer_kwargs)
    # Start training
    trainer.train()

    # Save the trained model
    #trainer.save_model("../save_model")
    # 保存训练后的模型
    model.save_pretrained(args.model_save_path)

    # 保存 tokenizer
    tokenizer.save_pretrained(args.model_save_path)

# Optionally, merge the LoRA weights with the base model
# merged_model = model.merge_and_unload()
# merged_model.save_pretrained("./qwen-negotiation-merged")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--input_file', type=str,default="../../../data/price_quantity/HQL_ration_train0614.json")
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--model_save_path', type=str)
    parser.add_argument('--logging_dir', type=str)
    parser.add_argument('--model_name', type=str)
    #parser.add_argument('--lora_path', type=str)
    parser.add_argument('--template', type=str)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--optimizer_name', type=str, default='AdamW')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--max_input_length', type=int, default=1024)
    parser.add_argument('--grad_steps', type=int, default=2)
    #parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--gen_max_len', type=int, default=1024)
    parser.add_argument('--parallelize', action='store_true')
    parser.add_argument('--model_type', type=str, default='task_prefix')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--output_rationale', action='store_true')
    args = parser.parse_args()
    train(args)