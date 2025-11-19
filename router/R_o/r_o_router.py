import numpy as np
import argparse
import re

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', default=1, type=float, help='p')     
    parser.add_argument('--data', type=str, metavar='PATH', help='path to data')
    return parser

def mix_router(group_data, p=0.5):
    final_scores = np.zeros(group_data.shape[1])
    for i in range(group_data.shape[1]):
        if np.random.rand() < p:
            final_scores[i] = np.max(group_data[:, i])
        else:
            final_scores[i] = group_data[np.random.randint(0, group_data.shape[0]), i]

    oracle_prob = group_data / group_data.sum(axis=0, keepdims=True)
    for i in range(group_data.shape[1]):
        if np.isnan(oracle_prob[:, i]).any():
            oracle_prob[:, i] = np.ones(group_data.shape[0]) / group_data.shape[0]
    random_prob = np.ones_like(group_data) / group_data.shape[0]
    predicted_probs = p * oracle_prob + (1-p) * random_prob
    terms = np.where(predicted_probs > 1e-10, predicted_probs * np.log2(predicted_probs), 0)
    Ep = -np.sum(terms) / predicted_probs.shape[1]
    
    return np.mean(final_scores), Ep


def main(args):
    datadict = np.load(args.data, allow_pickle=True)

    Y_test = datadict['test_score'].T
    test_tokens = datadict.get('test_tokens')
    test_time = datadict.get('test_time')

    if test_tokens is not None and test_tokens.shape[0] == datadict['test_score'].shape[0]:
        test_tokens = test_tokens.T
    if test_time is not None and test_time.shape[0] == datadict['test_score'].shape[0]:
        test_time = test_time.T
    
    p = args.p
    if p < 0 or p > 1:
        raise RuntimeError(f"p = {p} out of range [0, 1]")

    oracle_indices = np.argmax(Y_test, axis=0) # shape: (N_samples,)
    
    
    avg_tokens = 0.0
    avg_time = 0.0
    
    if p == 1.0 or p == 0.0:
        if test_tokens is not None and test_time is not None:
            N_samples = Y_test.shape[1]
            
            # 最高分模型的 Tokens
            selected_tokens = test_tokens[oracle_indices, np.arange(N_samples)]
            avg_tokens = np.mean(selected_tokens)
            
            # 最高分模型的 Latency
            selected_time = test_time[oracle_indices, np.arange(N_samples)]
            avg_time = np.mean(selected_time)


    if p == 1.0:
        acc, Ep = mix_router(Y_test, 1)
    else:
        acc_list = []
        Ep_list = []
        for i in range(1000):
            tmp_oracle_acc, tmp_Ep = mix_router(Y_test, p)
            acc_list.append(tmp_oracle_acc)
            Ep_list.append(tmp_Ep)
        acc = np.mean(acc_list)
        Ep = np.mean(Ep_list)


    vb_val = acc/np.max(np.mean(Y_test, axis=1))


    print(f"METRIC_MU: {acc}")
    print(f"METRIC_VB: {vb_val}")
    print(f"METRIC_EP: {Ep}")
    
    if p == 1.0:
        print(f"METRIC_TOKEN: {avg_tokens}")
        print(f"METRIC_LATENCY: {avg_time}")
    elif p == 0.0:
        # 這裡為 Random Router 用平均值作為代替
        print(f"METRIC_TOKEN: {np.mean(test_tokens)}") 
        print(f"METRIC_LATENCY: {np.mean(test_time)}")
    else:
        # 對於混合 Router，輸出 0.0
        print(f"METRIC_TOKEN: {0.0}")
        print(f"METRIC_LATENCY: {0.0}")


if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()
    main(args)