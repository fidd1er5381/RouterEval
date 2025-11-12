import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import pickle
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from datasets import Dataset
import argparse
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def get_argparser():
    parser = argparse.ArgumentParser()
    # ... (保持不變) ...
    parser.add_argument('--seed', default=43, type=int, help='randomseed')
    parser.add_argument('--model_path', default='roberta-base', type=str, help='path to model')
    parser.add_argument('--data',default='llm_performance_prompt.npz', type=str, metavar='PATH', help='path to data')
    return parser

def dataset_perpare(args):
    router_dataset = np.load(args.data, allow_pickle=True)
    train_input,test_input,train_score,test_score = router_dataset['train_prompt'],router_dataset['test_prompt'],router_dataset['train_score'],router_dataset['test_score']

    test_tokens = router_dataset.get('test_tokens')
    test_time = router_dataset.get('test_time')

    train_input = np.array(train_input)
    test_input = np.array(test_input)
    my_dict = {'labels':train_score,"sentence":train_input}
    train_dataset = Dataset.from_dict(my_dict)
    test_dict = {'labels':test_score,"sentence":test_input}
    test_dataset = Dataset.from_dict(test_dict)
    model_number = test_score.shape[1]
    

    return train_dataset,test_dataset,train_score,test_score,model_number, test_tokens, test_time


def train(args,train_dataset,test_dataset,train_score,test_score,model_number, test_tokens, test_time):
    model_name_or_path = args.model_path
    # ... (config, tokenizer, model, training_args 保持不變) ...
    config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=model_number,
            finetuning_task="text-classification",
            revision="main",
            token=None,
            trust_remote_code=False,
    )
    config.problem_type = "multi_label_classification"

    tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
    )
    def preprocess_function(examples):
        result = tokenizer(examples["sentence"], padding="max_length", max_length=512 ,truncation=True, return_tensors="pt")
        return result
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        config=config,
        revision="main",
    )
    
    training_args = TrainingArguments(
        output_dir='./router/RoBERTa-MLC/MLC_checkpoint',     
        num_train_epochs=10 ,            
        per_device_train_batch_size=10, 
        per_device_eval_batch_size=12,   
        warmup_steps=500,               
        weight_decay=0.01,              
        logging_strategy= 'no',
    )
    # ... (is_regression, is_multi_label, map, trainer... 保持不變) ...
    is_regression = False
    is_multi_label =True
    train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            desc="Running tokenizer on dataset",
    )
    test_dataset = test_dataset.map(
            preprocess_function,
            batched=True,
            desc="Running tokenizer on dataset",
    )
    trainer = Trainer(
            model=model,
            args = training_args,
            train_dataset=train_dataset ,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            data_collator= default_data_collator,
    )
        
    checkpoint = None
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    max_train_samples = (len(train_dataset))
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))
    trainer.save_model() 
    trainer.save_state()
            
    raw_predictions = trainer.predict(test_dataset, metric_key_prefix="predict").predictions
    predicted_probs = np.array([np.where(p > 0, p, 0) for p in raw_predictions])
    predicted_probs = np.array(predicted_probs)
    predicted_llm_indices = np.argmax(predicted_probs, axis=1)
    
    # 'mu' (Mean Utility / Accuracy)
    overall_accuracy = np.mean(test_score[np.arange(test_score.shape[0]), predicted_llm_indices])
    
    # 'Vb' (Variance of Best)
    # 讓 Vb 暫時等於 'router acc / bsm acc'
    vb_val = overall_accuracy / np.max(np.mean(test_score, axis=0))

    # 'Ep' (Entropy)
    predicted_probs = predicted_probs / predicted_probs.sum(axis=1, keepdims=True)
    terms = np.where(predicted_probs > 1e-4, predicted_probs * np.log2(predicted_probs), 0)
    Ep = -np.sum(terms) / predicted_probs.shape[0]

    avg_tokens = 0.0
    avg_time = 0.0
    
    if test_tokens is not None and test_time is not None:
        # 確保維度正確 (num_samples, num_models)
        if test_tokens.shape[0] == len(predicted_llm_indices) and test_time.shape[0] == len(predicted_llm_indices):
            selected_tokens = test_tokens[np.arange(test_tokens.shape[0]), predicted_llm_indices]
            avg_tokens = np.mean(selected_tokens)
            
            selected_time = test_time[np.arange(test_time.shape[0]), predicted_llm_indices]
            avg_time = np.mean(selected_time)
        else:
            print("警告: 成本數據維度與預測索引不匹配。")
    
    print(f"METRIC_MU: {overall_accuracy}")
    print(f"METRIC_VB: {vb_val}") # Vb (來自子腳本的 Vr)
    print(f"METRIC_EP: {Ep}")
    print(f"METRIC_TOKEN: {avg_tokens}")
    print(f"METRIC_LATENCY: {avg_time}")


def main():
    parser = get_argparser()
    args = parser.parse_args()
    random_state = args.seed
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    
    # --- 💎 修改: 接收成本數據 ---
    train_dataset,test_dataset,train_score,test_score,model_number, test_tokens, test_time = dataset_perpare(args)
    
    # --- 💎 修改: 傳遞成本數據 ---
    train(args,train_dataset,test_dataset,train_score,test_score,model_number, test_tokens, test_time)

if __name__ == '__main__':
    main()