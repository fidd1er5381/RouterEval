# RouterEval
### Usage
```python
from LLMRouter import datasets
import RouterEval 
model_pool = ["mistralai/Mixtral-8x7B-Instruct-v0.1" ,"gpt-4-1106-preview"]
dataset = datasets.load_dataset(
    ["gsm8k"], 
    model_pool,
    ["similar"]
)

RouterEval.convert_data_to_eval(dataset, model_pool)
```

### Eval Data Format
因RouteEval的評估方式目前只取用model、question_prompt、similar_point

`routereval_data.pkl` 結構如下：

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
    ]
  }
}
```
### Result
使用RouterEval的RoBERTa-MLC

<img width="350" height="30" alt="image" src="https://github.com/user-attachments/assets/7dac593b-1ce9-4cb6-880b-507a68fc020e" />
