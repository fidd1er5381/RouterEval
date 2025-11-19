import numpy as np
from sklearn.neighbors import NearestNeighbors
import argparse
import torch
from transformers import AutoTokenizer, AutoModel

def get_argparser():
    parser = argparse.ArgumentParser()

    # Randomness
    parser.add_argument('--seed', default=0, type=int, help='randomseed')
    
    # #cluster
    parser.add_argument('--knearest', default=0, type=int, help='k-nearest neighbours')
    
    # data
    parser.add_argument('--data', type=str, metavar='PATH', help='path to data')

    # embedding model
    parser.add_argument('--emb_model', type=str, default='sentence-transformers/all-MiniLM-L6-v2', help='HuggingFace model for embedding')
    
    return parser

def get_embeddings(text_list, model_name, batch_size=32):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # 移動到 GPU 如果可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    embeddings = []
    
    # Batch processing
    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i : i + batch_size]
        
        # Tokenize
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * attention_mask, 1)
        sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
        batch_embeddings = sum_embeddings / sum_mask
        
        embeddings.append(batch_embeddings.cpu().numpy())
        
    return np.concatenate(embeddings, axis=0)

def main(args):
    np.random.seed(args.seed)

    datadict = np.load(args.data, allow_pickle=True)
    
    if 'train_prompt' in datadict and 'test_prompt' in datadict:
        train_prompts = datadict['train_prompt'].tolist()
        test_prompts = datadict['test_prompt'].tolist()
        
        X_train = get_embeddings(train_prompts, args.emb_model)
        X_test = get_embeddings(test_prompts, args.emb_model)
        
    elif 'train_embed' in datadict:
        X_train = datadict['train_embed']
        X_test = datadict['test_embed']
        
    else:
        raise ValueError("Data file contains neither 'train_prompt' nor 'train_embed'.")
        

    Y_train = datadict['train_score']
    Y_test = datadict['test_score']
    
    test_tokens = datadict.get('test_tokens')
    test_time = datadict.get('test_time')

    # Initialize KNN
    nn_model = NearestNeighbors(n_neighbors=args.knearest, metric='cosine')
    nn_model.fit(X_train)
    
    distances, indices = nn_model.kneighbors(X_test)
    
    num_test = X_test.shape[0]
    num_llms = Y_train.shape[1]
    
    predicted_probs = np.zeros((num_test, num_llms))
    
    for i in range(num_test):
        neighbor_indices = indices[i]
        predicted_probs[i] = np.mean(Y_train[neighbor_indices], axis=0)
    
    predicted_llm_indices = np.argmax(predicted_probs, axis=1)
    
    # Metrics
    overall_accuracy = np.mean(Y_test[np.arange(Y_test.shape[0]), predicted_llm_indices])
    vb_val = overall_accuracy/np.max(np.mean(Y_test, axis=0))
    
    # Entropy
    predicted_probs_norm = np.exp(predicted_probs - np.max(predicted_probs, axis=1, keepdims=True)) / np.sum(np.exp(predicted_probs - np.max(predicted_probs, axis=1, keepdims=True)), axis=1, keepdims=True)
    terms = np.where(predicted_probs_norm > 1e-10, predicted_probs_norm * np.log2(predicted_probs_norm), 0)
    Ep = -np.sum(terms) / predicted_probs_norm.shape[0] 

    # Cost Metrics
    avg_tokens = 0.0
    avg_time = 0.0
    if test_tokens is not None and test_time is not None:
        if test_tokens.shape[0] == len(predicted_llm_indices):
            selected_tokens = test_tokens[np.arange(test_tokens.shape[0]), predicted_llm_indices]
            selected_time = test_time[np.arange(test_time.shape[0]), predicted_llm_indices]
            avg_tokens = np.mean(selected_tokens)
            avg_time = np.mean(selected_time)
    print(f"METRIC_MU: {overall_accuracy}")
    print(f"METRIC_VB: {vb_val}")
    print(f"METRIC_EP: {Ep}")
    print(f"METRIC_TOKEN: {avg_tokens}")
    print(f"METRIC_LATENCY: {avg_time}")

if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()

    #print(args)

    main(args)
