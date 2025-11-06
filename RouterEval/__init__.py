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
    
    """


    TARGET_MODEL_ORDER = tuple(model_pool)
    num_models = len(TARGET_MODEL_ORDER)


    def process_new_example(example):

        query = example['question']
        aligned_perf_list = []
        for model_name in TARGET_MODEL_ORDER:
            score_key = f"{model_name}_similar_eval"
            
            if score_key in example and 'point' in example[score_key]:
                score = example[score_key]['point']
                aligned_perf_list.append(score)
            else:
                print(f"在 {example.get('key', 'N/A')} 中找不到 {score_key}")
                aligned_perf_list.append(0.0)
                
        # 3. 返回 query 和對齊後的分數
        return {
            'query': query,
            'aligned_perf': aligned_perf_list
        }


    processed_list = [process_new_example(ex) for ex in tqdm(new_dataset, desc="處理資料")]
    processed_ds = Dataset.from_list(processed_list)
    
    # 80% train, 20% temp (val+test)
    ds_split = processed_ds.train_test_split(test_size=0.2, seed=42)
    ds_train = ds_split['train']
    ds_temp = ds_split['test'] # 20% 的資料
    
    # 20% temp -> 10% val, 10% test
    ds_val_test_split = ds_temp.train_test_split(test_size=0.5, seed=42)
    ds_val = ds_val_test_split['train']
    ds_test = ds_val_test_split['test']

    
    prompt_dict = {
        'train_prompt': tuple(ds_train['query']),
        'val_prompt': tuple(ds_val['query']),
        'test_prompt': tuple(ds_test['query']),
    }
    
    data_dict = {
        'train_score': np.array(ds_train['aligned_perf']),
        'val_score': np.array(ds_val['aligned_perf']),
        'test_score': np.array(ds_test['aligned_perf']),
    }


    routereval_formatted_data = {
        'data': data_dict,
        'model': TARGET_MODEL_ORDER, 
        'prompt': prompt_dict
    }

    print("\n儲存為 'routereval_data.pkl'...")
    with open('routereval_data.pkl', 'wb') as f:
        pickle.dump(routereval_formatted_data, f)

    
    json_data = {
        'model': routereval_formatted_data['model'],
        'prompt': routereval_formatted_data['prompt'],
        'data': {
            'train_score': routereval_formatted_data['data']['train_score'].tolist(),
            'val_score': routereval_formatted_data['data']['val_score'].tolist(),
            'test_score': routereval_formatted_data['data']['test_score'].tolist(),
        }
    }
    with open('routereval_data.json', 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print("\n儲存為 'routereval_data.json'...")
    




if __name__ == "__main__":
    from LLMRouter import datasets
    model_pool = ["mistralai/Mixtral-8x7B-Instruct-v0.1" ,"gpt-4-1106-preview"]
    dataset = datasets.load_dataset(["gsm8k"], model_pool , ["similar"])

    convert_data_to_eval(dataset, model_pool)


