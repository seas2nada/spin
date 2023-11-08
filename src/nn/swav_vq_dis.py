# Ref: https://github.com/facebookresearch/swav/blob/main/main_swav.py

import logging

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger("swav_vq_dis")


@torch.no_grad()
def compute_sinkhorn(
    out: torch.Tensor, epsilon: float, sinkhorn_iterations: int
) -> torch.Tensor:
    # out: (B, K)
    B, K = out.shape
    Q = out.div(epsilon).exp().t()
    # Q is K-by-B for consistency with notations from our paper

    # make the matrix sums to 1
    if dist.is_initialized():
        sum_Q = Q.sum()
        dist.all_reduce(sum_Q)
        Q.div_(sum_Q)
    else:
        Q.div_(Q.sum())

    for _ in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        if dist.is_initialized():
            sum_of_rows = Q.sum(dim=1, keepdim=True)
            dist.all_reduce(sum_of_rows)
            Q.div_(sum_of_rows * K)
        else:
            Q.div_(Q.sum(dim=1, keepdim=True) * K)

        # normalize each column: total weight per sample must be 1/B
        Q.div_(Q.sum(dim=0, keepdim=True) * B)

    Q.mul_(B)  # the colomns must sum to 1 so that Q is an assignment
    return Q.t()


class SwavVQDisentangle(nn.Module):
    def __init__(
        self,
        dim: int,
        num_vars: int,
        epsilon: float = 0.05,
        sinkhorn_iters: int = 3,
        temp: float = 0.1,
        l2_norm: bool = True,
        hard_target: bool = False,
        prob_ratio: float = 1.0,
        lambd: float = 15.0,
        mu: float = 0.0,
        nu: float = 1.0,
        gamma: float = 0.2,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.num_vars = num_vars
        self.epsilon = epsilon
        self.sinkhorn_iters = sinkhorn_iters
        self.temp = temp
        self.l2_norm = l2_norm
        self.hard_target = hard_target
        self.prob_ratio = prob_ratio
        self.lambd = lambd
        self.mu = mu
        self.nu = nu
        self.gamma = gamma

        logger.info(f"Codebook size: {num_vars}")
        self.codebook = nn.Linear(dim, num_vars, bias=False)

    def produce_targets(self, z: torch.Tensor, normalized: bool = False):
        if self.l2_norm and not normalized:
            z = F.normalize(z, dim=-1)
        logits = self.codebook(z) / self.temp
        codes = torch.argmax(logits, -1)
        return logits, codes

    @torch.no_grad()
    def normalize_codebook(self) -> None:
        w = self.codebook.weight.data.clone()
        w = F.normalize(w, dim=1, p=2)
        self.codebook.weight.copy_(w)

    @torch.no_grad()
    def zero_grad_codebook(self) -> None:
        self.codebook.zero_grad()

    @torch.no_grad()
    def copy_codebook(self) -> None:
        self.normalize_codebook()
        self.codebook_copy = self.codebook.weight.data.detach()

    @torch.no_grad()
    def restore_codebook(self) -> None:
        self.codebook.weight.copy_(self.codebook_copy)

    # def forward(
    #     self,
    #     z_1: torch.Tensor,
    #     z_2: torch.Tensor,
    # ):
    #     B = len(z_1)

    #     if self.l2_norm:
    #         z_1 = F.normalize(z_1, dim=1)
    #         z_2 = F.normalize(z_2, dim=1)

    #     logits_1: torch.Tensor = self.codebook(z_1)  # (Batch, Num Codes)
    #     logits_2: torch.Tensor = self.codebook(z_2)  # (Batch, Num Codes)

    #     # Compute targets
    #     with torch.no_grad():
    #         tgt_logits_w_1 = logits_1 * self.prob_ratio + logits_2 * (
    #             1 - self.prob_ratio
    #         )
    #         tgt_logits_w_2 = logits_2 * self.prob_ratio + logits_1 * (
    #             1 - self.prob_ratio
    #         )

    #         tgt_probs_1 = compute_sinkhorn(
    #             tgt_logits_w_1.detach(), self.epsilon, self.sinkhorn_iters
    #         )
    #         tgt_probs_2 = compute_sinkhorn(
    #             tgt_logits_w_2.detach(), self.epsilon, self.sinkhorn_iters
    #         )

    #     # Compute cross-entropy loss
    #     logits_1.div_(self.temp)
    #     logits_2.div_(self.temp)
    #     log_prob_1 = logits_1.log_softmax(1)
    #     log_prob_2 = logits_2.log_softmax(1)

    #     loss = 0
    #     if self.hard_target:
    #         loss_ce = 0.5 * (
    #             F.cross_entropy(
    #                 logits_2,
    #                 tgt_probs_1.argmax(-1),
    #             )
    #             + F.cross_entropy(
    #                 logits_1,
    #                 tgt_probs_2.argmax(-1),
    #             )
    #         )
    #     else:
    #         loss_ce = -0.5 * (
    #             (tgt_probs_1 * log_prob_2).sum(1).mean()
    #             + (tgt_probs_2 * log_prob_1).sum(1).mean()
    #         )

    #     loss_em = 0.5 * (
    #             (tgt_probs_1 * log_prob_1).sum(1).mean()
    #             + (tgt_probs_2 * log_prob_2).sum(1).mean()
    #         )

    #     loss += (0.9 * loss_ce + 0.1 * loss_em)
    #     result = {"loss_ce": loss_ce, "batch_size": B}
    #     result["loss"] = loss

    #     with torch.no_grad():
    #         logits = torch.cat([logits_1, logits_2], dim=0)
    #         _, k = logits.max(-1)
    #         hard_x = logits.new_zeros(*logits.shape).scatter_(1, k.view(-1, 1), 1.0)
    #         hard_probs = hard_x.float().mean(0)
    #         result["code_perplexity"] = (
    #             torch.exp(-torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1))
    #             .sum()
    #             .cpu()
    #             .detach()
    #         )

    #         avg_probs = logits.float().softmax(-1).mean(0)
    #         result["prob_perplexity"] = (
    #             torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1))
    #             .sum()
    #             .cpu()
    #             .detach()
    #         )
    #         acc_1 = (
    #             (torch.argmax(logits_1, dim=1) == torch.argmax(tgt_probs_2, dim=1))
    #             .float()
    #             .mean()
    #             .cpu()
    #             .detach()
    #             .item()
    #         )
    #         acc_2 = (
    #             (torch.argmax(logits_2, dim=1) == torch.argmax(tgt_probs_1, dim=1))
    #             .float()
    #             .mean()
    #             .cpu()
    #             .detach()
    #             .item()
    #         )
    #         result["acc"] = float((acc_1 + acc_2) / 2)
    #         result["acc_1"] = float(acc_1)
    #         result["acc_2"] = float(acc_2)

    #     return result

    ### VIC reg
    def compute_v(self, z):
        eps = 1e-8

        d = z.size(-1)
        zeros = torch.zeros(d).to(z.device)
        v_z = (1/d) * torch.sum(torch.max(zeros, self.gamma - torch.sqrt(torch.var(z, dim=0) + eps)))
        return v_z
    
    def compute_c(self, z):
        n = z.size(0)
        d = z.size(1)

        C_z = torch.cov(z.transpose(0,1))
    
        # Get the upper and lower triangular parts of C_z, excluding the diagonal
        C_z_upper = torch.triu(C_z, diagonal=1)
        C_z_lower = torch.tril(C_z, diagonal=-1)
        
        # Compute the sum of squares of the off-diagonal elements
        sum_val = torch.sum(torch.pow(C_z_upper, 2)) + torch.sum(torch.pow(C_z_lower, 2))

        c_z = (1 / d) * sum_val
        return c_z

    def forward(
        self,
        z_1: torch.Tensor,
        z_2: torch.Tensor,
    ):
        B = len(z_1)

        # var reg
        if self.mu != 0:
            v_z1 = self.compute_v(z_1)
            v_z2 = self.compute_v(z_2)
        else:
            v_z1, v_z2 = 0, 0
        # cov reg
        if self.nu != 0:
            c_z1 = self.compute_c(z_1)
            c_z2 = self.compute_c(z_2)
        else:
            c_z1, c_z2 = 0, 0

        if self.l2_norm:
            z_1 = F.normalize(z_1, dim=1)
            z_2 = F.normalize(z_2, dim=1)

        logits_1: torch.Tensor = self.codebook(z_1)  # (Batch, Num Codes)
        logits_2: torch.Tensor = self.codebook(z_2)  # (Batch, Num Codes)

        # Compute targets
        with torch.no_grad():
            tgt_logits_w_1 = logits_1 * self.prob_ratio + logits_2 * (
                1 - self.prob_ratio
            )
            tgt_logits_w_2 = logits_2 * self.prob_ratio + logits_1 * (
                1 - self.prob_ratio
            )

            tgt_probs_1 = compute_sinkhorn(
                tgt_logits_w_1.detach(), self.epsilon, self.sinkhorn_iters
            )
            tgt_probs_2 = compute_sinkhorn(
                tgt_logits_w_2.detach(), self.epsilon, self.sinkhorn_iters
            )

        # Compute cross-entropy loss
        logits_1.div_(self.temp)
        logits_2.div_(self.temp)
        log_prob_1 = logits_1.log_softmax(1)
        log_prob_2 = logits_2.log_softmax(1)

        loss = 0
        if self.hard_target:
            loss_ce = 0.5 * (
                F.cross_entropy(
                    logits_2,
                    tgt_probs_1.argmax(-1),
                )
                + F.cross_entropy(
                    logits_1,
                    tgt_probs_2.argmax(-1),
                )
            )
        else:
            loss_ce = -0.5 * (
                (tgt_probs_1 * log_prob_2).sum(1).mean()
                + (tgt_probs_2 * log_prob_1).sum(1).mean()
            )

        loss += self.lambd * loss_ce + self.mu * (v_z1 + v_z2) + self.nu * (c_z1 + c_z2)
        result = {"loss_ce": loss_ce, "loss_var": v_z1 + v_z2, "loss_cov": c_z1 + c_z2, "batch_size": B}
        result["loss"] = loss

        with torch.no_grad():
            logits = torch.cat([logits_1, logits_2], dim=0)
            _, k = logits.max(-1)
            hard_x = logits.new_zeros(*logits.shape).scatter_(1, k.view(-1, 1), 1.0)
            hard_probs = hard_x.float().mean(0)
            result["code_perplexity"] = (
                torch.exp(-torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1))
                .sum()
                .cpu()
                .detach()
            )

            avg_probs = logits.float().softmax(-1).mean(0)
            result["prob_perplexity"] = (
                torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1))
                .sum()
                .cpu()
                .detach()
            )
            acc_1 = (
                (torch.argmax(logits_1, dim=1) == torch.argmax(tgt_probs_2, dim=1))
                .float()
                .mean()
                .cpu()
                .detach()
                .item()
            )
            acc_2 = (
                (torch.argmax(logits_2, dim=1) == torch.argmax(tgt_probs_1, dim=1))
                .float()
                .mean()
                .cpu()
                .detach()
                .item()
            )
            result["acc"] = float((acc_1 + acc_2) / 2)
            result["acc_1"] = float(acc_1)
            result["acc_2"] = float(acc_2)

        return result

    def cal_loss(
        self,
        z_1: torch.Tensor,
        z_2: torch.Tensor,
    ):
        return self.forward(z_1, z_2)
