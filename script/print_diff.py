# encoder.layers.10.self_attn.k_proj.bias

import torch
import numpy as np
import matplotlib.pyplot as plt

org = torch.load("exps/hubert_base_ls960.pt")
spined = torch.load("exps/hubert_spinkbias_test/last.ckpt")

omw = org["model_weight"]
smw = spined['state_dict']

omw_keys = omw.keys()
for omw_key in omw_keys:
    omw_key = "encoder.model." + omw_key
    if omw_key not in smw:
        print(omw_key, "not in spinned model")
smw_keys = smw.keys()
for smw_key in smw_keys:
    smw_key = smw_key.split("encoder.model.")[-1]
    if smw_key not in omw:
        print(smw_key, "not in original model")

def cosine_similarity_matrix(matrix1, matrix2):
    # Calculate cosine similarity between matrix1 and matrix2
    if matrix1.dim() > 1:
        similarity = torch.nn.functional.cosine_similarity(matrix1, matrix2, dim=1)
    else:
        similarity = torch.nn.functional.cosine_similarity(matrix1, matrix2, dim=0)
    return similarity

def average_cosine_similarity(matrix1, matrix2):
    similarity_matrix = cosine_similarity_matrix(matrix1, matrix2)
    avg_similarity = similarity_matrix.mean()
    return avg_similarity

def average_l2dist(matrix1, matrix2):
    # similarity_matrix = torch.cdist(matrix1, matrix2, p=2.0)
    similarity_matrix = np.power((matrix1 - matrix2), 2)
    avg_similarity = similarity_matrix.mean()
    if avg_similarity < 1e-4:
        avg_similarity = torch.tensor(0)
    return avg_similarity

for omw_key in omw.keys():
    matrix1 = omw[omw_key]
    if "final_proj" in omw_key:
        smw_key = "pred_head.layers.0" + omw_key.split("final_proj")[-1]
    else:
        smw_key = "encoder.model." + omw_key
    matrix2 = smw[smw_key]

    # Calculate average cosine similarity
    avg_similarity = average_cosine_similarity(matrix1.cpu(), matrix2.cpu())
    # avg_similarity = average_l2dist(matrix1.cpu(), matrix2.cpu())
    # if "k_proj" in omw_key:
    #     print(omw_key, avg_similarity)
    print(omw_key, avg_similarity)

    # # Plot cosine similarity values
    # plt.figure()
    # plt.plot(avg_similarity.numpy())
    # plt.title("Average Cosine Similarity")
    # plt.xlabel("Sample")
    # plt.ylabel("ACS")
    # plt.tight_layout()

    # fname=f"plots/{omw_key}_cos.png"
    # plt.savefig(fname)