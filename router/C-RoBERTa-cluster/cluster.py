import numpy as np
from sklearn.cluster import KMeans
import argparse
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer 

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='randomseed')
    parser.add_argument('--numcluster', default=0, type=int, help='number of cluster')
    parser.add_argument('--data', type=str, metavar='PATH', help='path to data')
    parser.add_argument('--emb_model', type=str, default='sentence-transformers/all-MiniLM-L6-v2', help='HuggingFace model for embedding inference')
    return parser

def get_embeddings(text_list, model_name, batch_size=32):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(text_list, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    return embeddings


def predict_proba(X, kmeans):
    distances = np.array([np.linalg.norm(X - center, axis=1) for center in kmeans.cluster_centers_]).T
    probabilities = 1 / (distances + 1e-10)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    return probabilities
    

def main(args):

    np.random.seed(args.seed)
    datadict = np.load(args.data, allow_pickle=True)

    if 'train_embed' in datadict:
        train_embed = datadict['train_embed']
        test_embed = datadict['test_embed']
        
    elif 'train_prompt' in datadict:
        # 計算 Embedding
        train_prompts = datadict['train_prompt'].tolist()
        test_prompts = datadict['test_prompt'].tolist()
        
        train_embed = get_embeddings(train_prompts, args.emb_model)
        test_embed = get_embeddings(test_prompts, args.emb_model)
        
    else:
        raise ValueError("Data file must contain either 'train_embed' or 'train_prompt'.")


    train_score = datadict['train_score']
    test_tokens = datadict.get('test_tokens')
    test_time = datadict.get('test_time')

    X = train_embed
    Y = train_score
    new_X = test_embed
    new_Y = datadict['test_score']

    N = X.shape[0]
    m = X.shape[1]
    P = Y.shape[1]
    N_new = new_X.shape[0]

    K = args.numcluster
    
    kmeans = KMeans(n_clusters=K, random_state=args.seed, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    cluster_best_llm = {}
    for k in range(K):
        indices = np.where(cluster_labels == k)[0]
        if len(indices) == 0: 
            continue
            
        avg_performance = np.mean(Y[indices], axis=0)
        best_llm = np.argmax(avg_performance)
        cluster_best_llm[k] = best_llm
        
    new_cluster_labels = kmeans.predict(new_X)
    
    predicted_performance = []
    predicted_llm_indices = [] 
    
    for i in range(N_new):
        cluster_id = new_cluster_labels[i]
        best_llm_idx = cluster_best_llm.get(cluster_id, -1)
        
        if best_llm_idx != -1:
            predicted_llm_indices.append(best_llm_idx) 
            predicted_performance.append(new_Y[i, best_llm_idx])

    predicted_llm_indices = np.array(predicted_llm_indices)
    overall_accuracy = np.mean(predicted_performance)
    vb_val = overall_accuracy/np.max(np.mean(new_Y, axis=0))
    predicted_probs_raw = predict_proba(new_X, kmeans)
    
    prob_set = list(set(cluster_best_llm.values()))
    prob_num = len(prob_set)
    
    new_predicted_probs= np.zeros((predicted_probs_raw.shape[0], P)) 
    for cluster_id, llm_idx in cluster_best_llm.items():
        new_predicted_probs[:, llm_idx] += predicted_probs_raw[:, cluster_id]

    terms = np.where(new_predicted_probs > 1e-4, new_predicted_probs * np.log2(new_predicted_probs), 0)
    Ep = -np.sum(terms) / new_predicted_probs.shape[0]

    
    avg_tokens = 0.0
    avg_time = 0.0
    if len(predicted_llm_indices) > 0:
        decision_indices = np.where(np.array(new_cluster_labels) != -1)[0]
        
        if len(decision_indices) > 0:
            filtered_tokens = test_tokens[decision_indices, :]
            filtered_time = test_time[decision_indices, :]

            selected_tokens = filtered_tokens[np.arange(len(predicted_llm_indices)), predicted_llm_indices]
            selected_time = filtered_time[np.arange(len(predicted_llm_indices)), predicted_llm_indices]
            
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
    print(args)
    main(args)