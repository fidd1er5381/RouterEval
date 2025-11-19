# 包含tokem和response
import pickle
import re
import numpy as np
import subprocess


datasets = ['gsm8k_dataset']  # Changed from 'router_extended_data' to 'gsm8k_dataset'
to_handle_datasets = datasets
baseline = 'knn'

if baseline == 'knn':
    command = 'python router/PRKnn-knn/knn.py --seed 0 --knearest 5 --data data/tmp_perf.npz'
elif baseline == 'oracle':
    command = 'python router/R_o/r_o_router.py --p 1.0 --data data/tmp_perf.npz'
elif baseline == 'random':
    command = 'python router/R_o/r_o_router.py --p 0.0 --data data/tmp_perf.npz'
elif baseline == 'r_o_0.5':
    command = 'python router/R_o/r_o_router.py --p 0.5 --data data/tmp_perf.npz'
elif baseline == 'roberta_cluster':
    command = 'python router/C-RoBERTa-cluster/cluster.py --seed 0 --numcluster 3 --data data/tmp_perf.npz'
elif baseline == 'roberta_MLC':
    print("run roberta_MLC")
    command = 'python router/RoBERTa-MLC/roberta_MLC.py --seed 0 --model_path roberta-base --data data/tmp_perf.npz'
else:
    raise RuntimeError("No such baseline")
# elif baseline == 'linear':
#     command = 'python router/MLPR_LinearR/train.py --datadir data/tmp_perf.npz --lr 0.01 --model linear --epoch 10 --train-batch 1 --checkpoint router/MLPR_LinearR/linear'
# elif baseline == 'mlp':
#     command = 'python router/MLPR_LinearR/train.py --datadir data/tmp_perf.npz --lr 0.0001 --model FNN --epoch 100 --hidden_size 256 --train-batch 1 --checkpoint router/MLPR_LinearR/MLPR'


#acc_ref_dict = {'router_extended_data':0.9}


for to_handle in to_handle_datasets:

    with open(f'data/router_dataset/{to_handle}.pkl', 'rb') as f:
        router_dataset = pickle.load(f)
        
    #沒有標準模型隨便設的
    #acc_ref = acc_ref_dict[to_handle]
    acc_ref = 0.9
    
    print(f"Dataset: {to_handle}, Strategy: {baseline}")
    
    if to_handle == 'harness_truthfulqa_mc_0': 
        train_score = router_dataset['data']['train_label']
    else: 
        train_score=router_dataset['data']['train_score']
    val_score=router_dataset['data']['val_score']
    test_score=router_dataset['data']['test_score']

    test_tokens = router_dataset['data'].get('test_tokens')
    test_time = router_dataset['data'].get('test_time')
    model_names = router_dataset.get('model', []) 

    if not model_names:
        model_names = [f"model_{i}" for i in range(test_score.shape[1])]


    mu = []
    vr = []
    vb = []
    ep = []
    avg_tokens_list = []
    avg_latency_list = []
    
    for i in range(1):
        np.savez('data/tmp_perf.npz', train_prompt=router_dataset['prompt']['train_prompt'],
                                     val_prompt=router_dataset['prompt']['val_prompt'],
                                     test_prompt=router_dataset['prompt']['test_prompt'],
                                     train_score=train_score, val_score=val_score, test_score=test_score,
                                     test_tokens=test_tokens,
                                     test_time=test_time)


        try:
            result = subprocess.run(
                command, 
                shell=True, 
                check=True, 
                text=True, 
                capture_output=True 
            )

            def parse_metric(metric_name, output, default=0.0):
                match = re.search(f"{metric_name}:\\s*([-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", output)
                if match:
                    return float(match.group(1))
                print(f"Warning : stdout cannot find {metric_name}")
                return default

            mu_val = parse_metric("METRIC_MU", result.stdout)
            vb_val = parse_metric("METRIC_VB", result.stdout) 
            ep_val = parse_metric("METRIC_EP", result.stdout)
            token_val = parse_metric("METRIC_TOKEN", result.stdout)
            latency_val = parse_metric("METRIC_LATENCY", result.stdout)
            
            vr_val = mu_val / acc_ref 

            output_string = (
                f"Router -> mu: {mu_val:.4f},  Vr: {vr_val:.4f},  Vb: {vb_val:.4f},  Ep: {ep_val:.4f},  "
                f"Avg_Tokens: {token_val:.2f},  Avg_Latency: {latency_val:.4f}"
            )
            print(output_string)
            
            mu.append(mu_val)
            vr.append(vr_val)
            vb.append(vb_val)
            ep.append(ep_val)
            avg_tokens_list.append(token_val)
            avg_latency_list.append(latency_val)

        except subprocess.CalledProcessError as e:
            print(f"Command failed with return code: {e.returncode}")
            print(f"Error Output: {e.stderr}")
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")

    
    avg_mu_router = np.mean(mu)
    avg_vr_router = np.mean(vr)
    avg_vb_router = np.mean(vb)
    avg_ep_router = np.mean(ep)
    avg_tokens_router = np.mean(avg_tokens_list)
    avg_latency_router = np.mean(avg_latency_list)
    
    # model_avg_scores shape (num_models,)
    model_avg_scores = np.mean(test_score, axis=0) 
    
    best_model_idx = np.argmax(model_avg_scores)
    weakest_model_idx = np.argmin(model_avg_scores)
    
    best_model_name = model_names[best_model_idx]
    weakest_model_name = model_names[weakest_model_idx]

    mu_strongest = np.mean(test_score[:, best_model_idx])
    tokens_strongest = np.mean(test_tokens[:, best_model_idx])
    latency_strongest = np.mean(test_time[:, best_model_idx])
    
    mu_weakest = np.mean(test_score[:, weakest_model_idx])
    tokens_weakest = np.mean(test_tokens[:, weakest_model_idx])
    latency_weakest = np.mean(test_time[:, weakest_model_idx])
    
    
    print("-" * 75)
    
    strategy_col = 60
    mu_col = 12
    token_col = 15
    latency_col = 12

    print(
        f"{'Strategy':<{strategy_col}} | "
        f"{'mu':<{mu_col}} | "
        f"{'Avg_Tokens':<{token_col}} | "
        f"{'Avg_Latency':<{latency_col}}"
    )
    print("-" * 75)
    
    # Baseliine 1: 
    print(
        f"{f'Strongest ({best_model_name})':<{strategy_col}} | "
        f"{mu_strongest:<{mu_col}.4f} | "
        f"{tokens_strongest:<{token_col}.2f} | "
        f"{latency_strongest:<{latency_col}.4f}"
    )
    
    # Baseliine 2: 
    print(
        f"{f'Cheapest ({weakest_model_name})':<{strategy_col}} | "
        f"{mu_weakest:<{mu_col}.4f} | "
        f"{tokens_weakest:<{token_col}.2f} | "
        f"{latency_weakest:<{latency_col}.4f}"
    )
    
    # Router
    print(
        f"{f'Router ({baseline})':<{strategy_col}} | "
        f"{avg_mu_router:<{mu_col}.4f} | "
        f"{avg_tokens_router:<{token_col}.2f} | "
        f"{avg_latency_router:<{latency_col}.4f}"
    )
    print("-" * 75)

    print(f"Vr (Normalized Acc): {avg_vr_router:.4f}")
    print(f"Vb (Sub-Vr/Bias): {avg_vb_router:.4f}")
    print(f"Ep (Entropy): {avg_ep_router:.4f}")
    
    print("\n" + "="*75 + "\n")

