# RouterEval

最終資料格式 (Final Data Format)
routereval_formatted_data.pkl 檔案包含一個字典 (dictionary)，其結構如下：

JSON

{
  "model": [
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "gpt-4-1106-preview"
  ],
  "prompt": {
    "train_prompt": [
      "Candy has 15 light blue spools of thread...",
      "Roy has saved 40% more in money...",
      "... (80% 的資料) ..."
    ],
    "val_prompt": [
      "... (10% 的資料) ..."
    ],
    "test_prompt": [
      "... (10% 的資料) ..."
    ]
  },
  "data": {
    "train_score": [
      [ 0.9463, 0.9592 ],
      [ 0.9321, 0.9456 ],
      "... (Shape: [num_train_samples, num_models]) ..."
    ],
    "val_score": [
      "... (Shape: [num_val_samples, num_models]) ..."
    ],
    "test_score": [
      "... (Shape: [num_test_samples, num_models]) ..."
    ]
  }
}
