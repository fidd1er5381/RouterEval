import json
import re
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
]

def load_jsonl(file_path):
    """Load JSONL file and return as dictionary keyed by 'key'"""
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data[item['key']] = item
    return data

def extract_numerical_answer(text):
    """
    Extract numerical answer from GSM8K format or model response.
    Looks for:
    1. #### NUMBER format (ground truth)
    2. $NUMBER or NUMBER at the end
    3. Last occurrence of a number
    """
    # First try to find #### format
    if '####' in text:
        match = re.search(r'####\s*([-+]?\d*\.?\d+)', text)
        if match:
            return match.group(1).strip()

    # Try to find $NUMBER or similar patterns near the end
    # Look for patterns like "$18", "18 dollars", "= 18", etc.
    patterns = [
        r'\$\s*([-+]?\d*\.?\d+)',  # $18
        r'=\s*\$?\s*([-+]?\d*\.?\d+)\s*(?:dollars?|eggs?)?[\s\.]',  # = 18 dollars
        r'\boxed\{([-+]?\d*\.?\d+)\}',  # LaTeX boxed format
        r'(?:is|are|makes?)\s+\$?\s*([-+]?\d*\.?\d+)',  # "is 18" or "makes $18"
    ]

    for pattern in patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            # Return the last match (closest to end)
            return matches[-1].group(1).strip()

    # Fallback: find all numbers and return the last one
    numbers = re.findall(r'[-+]?\d*\.?\d+', text)
    if numbers:
        return numbers[-1].strip()

    return None

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
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                raise

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

def main():
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
        exact_matches = 0
        successful = 0
        failed = 0
        skipped_no_answer = 0

        with open(output_path, 'w', encoding='utf-8') as f:
            for key in tqdm(keys, desc=f"Evaluating {model_name}"):
                if key not in responses:
                    continue

                try:
                    # Extract numerical answers
                    ground_truth_text = dataset[key]['answer']
                    model_response_text = responses[key]['text']

                    gt_answer = extract_numerical_answer(ground_truth_text)
                    model_answer = extract_numerical_answer(model_response_text)

                    # Check for exact match first
                    if gt_answer and model_answer:
                        try:
                            # Try numerical comparison
                            gt_num = float(gt_answer)
                            model_num = float(model_answer)

                            if abs(gt_num - model_num) < 0.01:  # Exact match
                                similarity = 1.0
                                exact_matches += 1
                            else:
                                # Use embedding similarity as fallback
                                gt_embedding = get_embedding(gt_answer)
                                response_embedding = get_embedding(model_answer)
                                similarity = cosine_similarity(gt_embedding, response_embedding)
                        except ValueError:
                            # If not numerical, use embedding
                            gt_embedding = get_embedding(gt_answer)
                            response_embedding = get_embedding(model_answer)
                            similarity = cosine_similarity(gt_embedding, response_embedding)
                    elif gt_answer and not model_answer:
                        # Model didn't provide clear answer
                        similarity = 0.0
                        skipped_no_answer += 1
                    else:
                        similarity = 0.0

                    # Write result
                    result = {
                        "key": key,
                        "point": float(similarity)
                    }
                    f.write(json.dumps(result) + '\n')
                    f.flush()

                    eval_results.append(similarity)
                    successful += 1

                    time.sleep(0.05)  # Smaller delay since we're doing fewer API calls

                except Exception as e:
                    print(f"\n⚠️  Error processing {key}: {e}")
                    failed += 1
                    continue

        # Print statistics
        print(f"\n✓ Completed: {successful} samples")
        if failed > 0:
            print(f"✗ Failed: {failed} samples")
        if skipped_no_answer > 0:
            print(f"⚠️  No clear answer extracted: {skipped_no_answer} samples")

        if eval_results:
            print(f"\nSimilarity Statistics:")
            print(f"  Exact matches: {exact_matches}/{len(eval_results)} = {exact_matches/len(eval_results)*100:.2f}%")
            print(f"  Mean similarity: {np.mean(eval_results):.4f}")
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
