# RouterEval
### Usage
```python
from LLMRouter import datasets
import data_process  
model_pool = ["mistralai/Mixtral-8x7B-Instruct-v0.1" ,"gpt-4-1106-preview"]
dataset = datasets.load_dataset(
    ["gsm8k"], 
    model_pool,
    ["similar"]
)
data_process.convert_data_to_eval(dataset, model_pool)
```
儲存為routereval_data.pkl供RouterEval使用  
### Eval Data Format
因RouteEval的評估方式目前只取用model、question_prompt、similar_point

`routereval_data.pkl` 範例結構如下：

```json
{
  "model": [
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "gpt-4-1106-preview"
  ],
  "prompt": {
    "train_prompt": [
      "Candy has 15 light blue spools of thread...",
      "Roy has saved 40% more in money...",
      "......"
    ],
    "val_prompt": [
      "......"
    ],
    "test_prompt": [
      "......"
    ]
  },
  "data": {
    "train_score": [
      [ 0.9463, 0.9592 ],
      [ 0.9321, 0.9456 ],
      "......"
    ],
    "val_score": [
      "......"
    ],
    "test_score": [
      "......"
    ],
    
    "train_tokens": [
      [ 73, 73 ],
      [ 30, 30 ],
      "......"
    ],
    "val_tokens": [
      "......"
    ],
    "test_tokens": [
      "......"
    ],
    
    "train_time": [
      [ 0.7516, 0.8516 ],
      [ 0.3000, 1.1000 ],
      "......"
    ],
    "val_time": [
      "......"
    ],
    "test_time": [
      "......"
    ]
  }
}
```
### Output example
```bash
python test_new_data.py
```
使用RouterEval的RoBERTa-MLC  
vr比較基準為score 0.9(因沒有基準隨便設的)  
```bash
Dataset: router_extended_data_20, Strategy: roberta_MLC
Router -> mu: 0.9333,  Vr: 1.0370,  Vb: 1.0000,  Ep: -0.0000,  Avg_Tokens: 200.00,  Avg_Latency: 1.5054
---------------------------------------------------------------------------
Strategy                                                     | mu           | Avg_Tokens      | Avg_Latency 
---------------------------------------------------------------------------
Strongest (mistralai/Mixtral-8x7B-Instruct-v0.1)             | 0.9333       | 200.00          | 1.5054      
Cheapest (gpt-4-1106-preview)                                | 0.9212       | 200.00          | 1.5054      
Router (roberta_MLC)                                         | 0.9333       | 200.00          | 1.5054      
---------------------------------------------------------------------------
Vr (Normalized Acc): 1.0370
Vb (Sub-Vr/Bias): 1.0000
Ep (Entropy): 0.0000
```
