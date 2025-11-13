import json
import requests
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time

# Configuration
OLLAMA_API_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "mxbai-embed-large"
DATASET_PATH = "LLMRouter/LLMRouter/datasets/datasets/gsm8k.jsonl"
RESPONSE_DIR = "LLMRouter/LLMRouter/datasets/responses/gsm8k"
OUTPUT_DIR = "LLMRouter/LLMRouter/datasets/evals/gsm8k/similar"

# Models to evaluate
MODELS_TO_EVAL = [
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "gpt-4-1106-preview",
    "google/gemma-3n-e2b-it",
    "openai/gpt-oss-20b"
]

def load_jsonl(file_path):
    """Load JSONL file and return as dictionary keyed by 'key'"""
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data[item['key']] = item
    return data

def get_embedding(text, max_retries=3):
    """Get embedding from Ollama API with retry logic"""
    for attempt in range(max_retries):
        try:
            response = requests.post(
                OLLAMA_API_URL,
                json={
                    "model": EMBEDDING_MODEL,
                    "input": text
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return np.array(result['embeddings'][0])
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"  Error getting embedding (attempt {attempt+1}/{max_retries}): {e}")
                print(f"  Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"  Failed to get embedding after {max_retries} attempts: {e}")
                raise

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

def extract_answer(answer_text):
    """Extract the final numerical answer from GSM8K answer format"""
    # GSM8K answers typically end with #### NUMBER
    if '####' in answer_text:
        return answer_text.split('####')[-1].strip()
    return answer_text.strip()

def main():
    # Load dataset with ground truth answers
    print("Loading dataset...")
    dataset = load_jsonl(DATASET_PATH)
    keys = sorted(dataset.keys())
    print(f"Total samples: {len(keys)}")

    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each model
    for model_name in MODELS_TO_EVAL:
        print(f"\n{'='*75}")
        print(f"Processing model: {model_name}")
        print(f"{'='*75}")

        # Load model responses
        encoded_name = model_name.replace("/", "%2F")
        response_path = Path(RESPONSE_DIR) / f"{encoded_name}.jsonl"

        if not response_path.exists():
            print(f"⚠️  Response file not found: {response_path}")
            continue

        responses = load_jsonl(response_path)
        print(f"Loaded {len(responses)} responses")

        # Output file
        output_path = output_dir / f"{encoded_name}.jsonl"

        # Evaluate each response
        eval_results = []
        successful = 0
        failed = 0

        with open(output_path, 'w', encoding='utf-8') as f:
            for key in tqdm(keys, desc=f"Evaluating {model_name}"):
                if key not in responses:
                    # Skip missing responses
                    continue

                try:
                    # Get ground truth answer and model response
                    ground_truth = extract_answer(dataset[key]['answer'])
                    model_response = responses[key]['text']

                    # Get embeddings
                    gt_embedding = get_embedding(ground_truth)
                    response_embedding = get_embedding(model_response)

                    # Calculate similarity
                    similarity = cosine_similarity(gt_embedding, response_embedding)

                    # Write result
                    result = {
                        "key": key,
                        "point": float(similarity)
                    }
                    f.write(json.dumps(result) + '\n')
                    f.flush()  # Ensure data is written immediately

                    eval_results.append(similarity)
                    successful += 1

                    # Small delay to avoid overwhelming the API
                    time.sleep(0.1)

                except Exception as e:
                    print(f"\n⚠️  Error processing {key}: {e}")
                    failed += 1
                    continue

        # Print statistics
        print(f"\n✓ Completed: {successful} samples")
        if failed > 0:
            print(f"✗ Failed: {failed} samples")

        if eval_results:
            print(f"\nSimilarity Statistics:")
            print(f"  Mean: {np.mean(eval_results):.4f}")
            print(f"  Std:  {np.std(eval_results):.4f}")
            print(f"  Min:  {np.min(eval_results):.4f}")
            print(f"  Max:  {np.max(eval_results):.4f}")

            # Count correct answers (>= 0.9 threshold)
            correct = np.sum(np.array(eval_results) >= 0.9)
            accuracy = correct / len(eval_results)
            print(f"  Accuracy (>= 0.9): {correct}/{len(eval_results)} = {accuracy:.2%}")

        print(f"✓ Saved to: {output_path}")

    print(f"\n{'='*75}")
    print("All evaluations completed!")
    print(f"{'='*75}")

if __name__ == "__main__":
    main()
