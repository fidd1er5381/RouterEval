import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle
import json

def convert_data_to_eval(new_dataset: list, model_pool: list):
    """
    將 LLMRouter.datasets 載入轉換為 RouterEval 格式

    使用 model_pool 作為固定的模型順序
    提取 'question_prompt' 和對應的 'point'
    建立 80/10/10 的 train/val/test 切分
    包含 cost/respoonse_time 資訊。
    """

    TARGET_MODEL_ORDER = tuple(model_pool)
    num_models = len(TARGET_MODEL_ORDER)


    def process_new_example(example):
        query = example['question']
    
        aligned_perf_list = []
        aligned_tokens_list = [] 
        aligned_time_list = []  
        
        for model_name in TARGET_MODEL_ORDER:

            score_key = f"{model_name}_similar_eval"
            response_key = f"{model_name}_response"
            
            if score_key in example and 'point' in example[score_key]:
                score = example[score_key]['point']
                aligned_perf_list.append(score)
            else:
                print(f"Warning: {example.get('key', 'N/A')} cannot find {score_key}")
                aligned_perf_list.append(0.0)
            
            if response_key in example:
                response_data = example[response_key]
                
                if 'usage' in response_data and 'total_tokens' in response_data['usage']:
                    tokens = response_data['usage']['total_tokens']
                    aligned_tokens_list.append(tokens)
                else:
                    aligned_tokens_list.append(0) 
                
                if 'response_time' in response_data:
                    time = response_data['response_time']
                    aligned_time_list.append(time)
                else:
                    aligned_time_list.append(0.0)
            else:
                print(f"Warning: {example.get('key', 'N/A')} cannot find {response_key}")
                aligned_tokens_list.append(0)
                aligned_time_list.append(0.0)
                
        return {
            'query': query,
            'aligned_perf': aligned_perf_list,
            'aligned_tokens': aligned_tokens_list,
            'aligned_time': aligned_time_list     
        }

    processed_list = [process_new_example(ex) for ex in tqdm(new_dataset, desc="processing")]
    processed_ds = Dataset.from_list(processed_list)

    
    ds_split = processed_ds.train_test_split(test_size=0.2, seed=42)
    ds_train = ds_split['train']
    ds_temp = ds_split['test'] 
    
    ds_val_test_split = ds_temp.train_test_split(test_size=0.5, seed=42)
    ds_val = ds_val_test_split['train']
    ds_test = ds_val_test_split['test']


    
    prompt_dict = {
        'train_prompt': tuple(ds_train['query']),
        'val_prompt': tuple(ds_val['query']),
        'test_prompt': tuple(ds_test['query']),
    }
    
    # 2. 'data' (score, tokens, time)
    data_dict = {
        'train_score': np.array(ds_train['aligned_perf']),
        'val_score': np.array(ds_val['aligned_perf']),
        'test_score': np.array(ds_test['aligned_perf']),
        
        'train_tokens': np.array(ds_train['aligned_tokens']),
        'val_tokens': np.array(ds_val['aligned_tokens']),
        'test_tokens': np.array(ds_test['aligned_tokens']),
        
        'train_time': np.array(ds_train['aligned_time']),
        'val_time': np.array(ds_val['aligned_time']),
        'test_time': np.array(ds_test['aligned_time']),
    }

    routereval_extended_data = {
        'data': data_dict,
        'model': TARGET_MODEL_ORDER,
        'prompt': prompt_dict
    }

    pkl_filepath = './data/router_dataset/router_extended_data.pkl'
    
    
    print(f"\nSaving...")
    with open(pkl_filepath, 'wb') as f:
        pickle.dump(routereval_extended_data, f)

    
    json_data = {
        'model': routereval_extended_data['model'],
        'prompt': routereval_extended_data['prompt'],
        'data': {
            'train_score': routereval_extended_data['data']['train_score'].tolist(),
            'val_score': routereval_extended_data['data']['val_score'].tolist(),
            'test_score': routereval_extended_data['data']['test_score'].tolist(),
            
            'train_tokens': routereval_extended_data['data']['train_tokens'].tolist(),
            'val_tokens': routereval_extended_data['data']['val_tokens'].tolist(),
            'test_tokens': routereval_extended_data['data']['test_tokens'].tolist(),
            
            'train_time': routereval_extended_data['data']['train_time'].tolist(),
            'val_time': routereval_extended_data['data']['val_time'].tolist(),
            'test_time': routereval_extended_data['data']['test_time'].tolist(),
        }
    }
    json_filename = './data/router_dataset/router_extended_meta.json'
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
        


if __name__ == "__main__":

    from LLMRouter import datasets
    model_pool = ["mistralai/Mixtral-8x7B-Instruct-v0.1" ,"gpt-4-1106-preview"]
    dataset = datasets.load_dataset(["gsm8k"], model_pool , ["similar"])
    dataset = dataset[0:20]

    convert_data_to_eval(dataset_list, model_pool)



