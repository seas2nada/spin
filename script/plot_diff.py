# encoder.layers.10.self_attn.k_proj.bias

import torch
import numpy as np
import matplotlib.pyplot as plt

org = torch.load("exps/hubert_base_ls960.pt", map_location="cpu")
# spined = torch.load("exps/hubert_test/last.ckpt", map_location="cpu")
spined = torch.load("exps/hubert_spinkbias_test/last.ckpt", map_location="cpu")

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

x_keys = []
avg_similarities = []

for omw_key in omw.keys():
    if "encoder.layers." not in omw_key:
        continue
    if int(omw_key.split("encoder.layers.")[-1].split('.')[0]) != 10:
        continue
    matrix1 = omw[omw_key]
    if "final_proj" in omw_key:
        smw_key = "pred_head.layers.0" + omw_key.split("final_proj")[-1]
    else:
        smw_key = "encoder.model." + omw_key
    matrix2 = smw[smw_key]

    # Calculate average cosine similarity
    avg_similarity = average_cosine_similarity(matrix1.cpu(), matrix2.cpu())

    x_keys.append(".".join(omw_key.split('.')[-2:]))
    avg_similarities.append(avg_similarity)

parameter_labels = x_keys
avg_similarity_values = avg_similarities

module_colors = {}
for param_label in parameter_labels:
    module_name = param_label.split('.')[0]
    if module_name not in module_colors:
        module_colors[module_name] = len(module_colors)*10

cmap = plt.get_cmap('tab20')
# Create a list of colors for each parameter based on their module
colors = [cmap(i % 20) for i in range(len(parameter_labels))]
for i, color in enumerate(colors):
    if (i/2 - int(i/2)) > 0.1:
        colors[i] = save
    save = color

# Create a bar chart to visualize avg_similarity for different parameters
plt.figure(figsize=(12, 6))
plt.bar(range(len(parameter_labels)), avg_similarity_values, color=colors)

# Position the weight and bias bars closely together
for i in range(len(parameter_labels)):
    if parameter_labels[i].endswith("weight"):
        plt.bar(i, avg_similarity_values[i], color=colors[i], label="w", width=0.4, align='center')
    elif parameter_labels[i].endswith("bias"):
        plt.bar(i, avg_similarity_values[i], color=colors[i], label="b", width=0.4, align='edge')

plt.title('Cosine Similarity: Org. Vs. Spinned')
plt.xlabel('Parameter')
plt.ylabel('Cosine Similarity')
plt.ylim(0, 1)  # Adjust the y-axis limits as needed
plt.xticks(range(len(parameter_labels)), parameter_labels, rotation=45, ha="right")  # Rotate x-axis labels for readability
plt.tight_layout()

# Show the plot or save it to a file
plt.savefig('plots/cos_sim_of_layer10.png')