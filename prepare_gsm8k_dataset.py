import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Paths
DATASET_PATH = "LLMRouter/LLMRouter/datasets/datasets/gsm8k.jsonl"
RESPONSE_DIR = "LLMRouter/LLMRouter/datasets/responses/gsm8k"
EVAL_DIR = "LLMRouter/LLMRouter/datasets/evals/gsm8k/similar"  # Fixed: was LLMRouter_bak
OUTPUT_PATH = "data/router_dataset/gsm8k_dataset.pkl"

# Threshold for correctness
THRESHOLD = 0.8

def load_jsonl(file_path):
    """Load JSONL file and return as dictionary keyed by 'key'"""
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data[item['key']] = item
    return data

def main():
    print("Loading dataset...")
    # Load questions and answers
    dataset = load_jsonl(DATASET_PATH)
    keys = sorted(dataset.keys())
    n_samples = len(keys)
    print(f"Total samples: {n_samples}")

    # Find all available models
    response_dir = Path(RESPONSE_DIR)
    response_files = list(response_dir.glob("*.jsonl"))

    # Model names (decode URL encoding)
    model_names = []
    for f in response_files:
        model_name = f.stem.replace("%2F", "/")
        model_names.append(model_name)

    print(f"Found models: {model_names}")
    n_models = len(model_names)

    # Load responses and evals for each model
    model_data = {}
    for model_name in model_names:
        encoded_name = model_name.replace("/", "%2F")
        response_path = response_dir / f"{encoded_name}.jsonl"
        eval_path = Path(EVAL_DIR) / f"{encoded_name}.jsonl"

        print(f"\nLoading data for {model_name}...")
        responses = load_jsonl(response_path)

        # Load eval if exists
        if eval_path.exists():
            evals = load_jsonl(eval_path)
            print(f"  - Loaded {len(responses)} responses and {len(evals)} evals")
        else:
            evals = {}
            print(f"  - Loaded {len(responses)} responses (no eval file found)")

        model_data[model_name] = {
            'responses': responses,
            'evals': evals
        }

    # Prepare arrays
    prompts = []
    scores = np.zeros((n_samples, n_models))
    tokens = np.zeros((n_samples, n_models))
    times = np.zeros((n_samples, n_models))

    # Fill arrays
    print("\nProcessing data...")
    for i, key in enumerate(keys):
        question = dataset[key]['question']
        prompts.append(question)

        for j, model_name in enumerate(model_names):
            # Get response data
            if key in model_data[model_name]['responses']:
                response = model_data[model_name]['responses'][key]
                tokens[i, j] = response['usage']['total_tokens']
                times[i, j] = response['response_time']
            else:
                print(f"Warning: Missing response for {key} from {model_name}")
                tokens[i, j] = 0
                times[i, j] = 0

            # Get eval score
            if key in model_data[model_name]['evals']:
                eval_point = model_data[model_name]['evals'][key]['point']
                # Threshold: >= 0.9 is correct (1), otherwise incorrect (0)
                scores[i, j] = 1.0 if eval_point >= THRESHOLD else 0.0
            else:
                # If no eval, mark as 0
                scores[i, j] = 0.0

    # Split into train/val/test (60/20/20)
    indices = np.arange(n_samples)
    train_idx, temp_idx = train_test_split(indices, test_size=0.4, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    print(f"\nData split:")
    print(f"  Train: {len(train_idx)}")
    print(f"  Val: {len(val_idx)}")
    print(f"  Test: {len(test_idx)}")

    # Create router dataset
    router_dataset = {
        'prompt': {
            'train_prompt': [prompts[i] for i in train_idx],
            'val_prompt': [prompts[i] for i in val_idx],
            'test_prompt': [prompts[i] for i in test_idx]
        },
        'data': {
            'train_score': scores[train_idx],
            'val_score': scores[val_idx],
            'test_score': scores[test_idx],
            'test_tokens': tokens[test_idx],
            'test_time': times[test_idx]
        },
        'model': model_names
    }

    # Print statistics
    print("\n" + "="*75)
    print("Dataset Statistics:")
    print("="*75)
    for j, model_name in enumerate(model_names):
        train_acc = np.mean(scores[train_idx, j])
        val_acc = np.mean(scores[val_idx, j])
        test_acc = np.mean(scores[test_idx, j])
        avg_tokens = np.mean(tokens[test_idx, j])
        avg_time = np.mean(times[test_idx, j])

        print(f"\n{model_name}:")
        print(f"  Train Acc: {train_acc:.4f}")
        print(f"  Val Acc:   {val_acc:.4f}")
        print(f"  Test Acc:  {test_acc:.4f}")
        print(f"  Avg Tokens: {avg_tokens:.2f}")
        print(f"  Avg Time:   {avg_time:.4f}s")

    # Save to pickle
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(router_dataset, f)

    print(f"\n✓ Dataset saved to {OUTPUT_PATH}")
    print("="*75)

if __name__ == "__main__":
    main()
