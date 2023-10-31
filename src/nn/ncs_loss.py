import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CosineSimilarity

# class NegativeCosineSimilarityLoss(nn.Module):
#     def __init__(self, reduction="mean"):
#         super().__init__()
#         self.cos_sim = nn.CosineSimilarity(dim=1)
#         self.reduction = reduction

#     def forward(self,
#         z_1: torch.Tensor,
#         z_2: torch.Tensor,
#     ):
#         # Normalize the input vectors
#         z_1 = F.normalize(z_1, p=2, dim=1)
#         z_2 = F.normalize(z_2, p=2, dim=1)

#         # Calculate the cosine similarity
#         cosine_sim = self.cos_sim(z_1, z_2)

#         # The loss is the negative cosine similarity
#         if self.reduction == "mean":
#             loss = -cosine_sim.mean()
#         elif self.reduction == "sum":
#             loss = -cosine_sim.sum()

#         return loss

class NegativeCosineSimilarityLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.MSELoss = nn.MSELoss(reduction)

    def forward(self,
        z_1: torch.Tensor,
        z_2: torch.Tensor,
    ):

        # Calculate the MSE
        mse = self.MSELoss(z_1, z_2)
        loss = -mse

        return loss