import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedMoE(nn.Module):

    def __init__(self, n_groups, in_dim, out_dim, n_experts=4, top_k=2,hidden_mult=4, router_global=False):
        super().__init__()
        self.n_groups = n_groups
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_experts = n_experts
        self.top_k = min(top_k, n_experts)
        self.router_global = router_global
        hidden = max(hidden_mult * out_dim, 4)
        self.hidden = hidden

        G, E = n_groups, n_experts
        self.w1 = nn.Parameter(torch.empty(G, E, in_dim, hidden))
        self.b1 = nn.Parameter(torch.zeros(G, E, hidden))
        self.w2 = nn.Parameter(torch.empty(G, E, hidden, out_dim))
        self.b2 = nn.Parameter(torch.zeros(G, E, out_dim))
        self._reset_experts()

        # router 看整層輸入（跨組資訊只能從這裡進來，expert 本身是 block-diagonal）
        router_in = n_groups * in_dim if router_global else in_dim
        self.router_global_dim = router_in
        if router_global:
            self.router = nn.Linear(router_in, G * E)
            nn.init.zeros_(self.router.bias)
            nn.init.normal_(self.router.weight, std=0.01)
        else:
            self.rw = nn.Parameter(torch.zeros(G, in_dim, E))
            self.rb = nn.Parameter(torch.zeros(G, E))
            nn.init.normal_(self.rw, std=0.01)

    def _reset_experts(self):
        bound1 = 1.0 / math.sqrt(self.in_dim)
        bound2 = 1.0 / math.sqrt(self.hidden)
        nn.init.uniform_(self.w1, -bound1, bound1)
        nn.init.uniform_(self.w2, -bound2, bound2)

    def _gate(self, x, xg):
        if self.router_global:
            logits = self.router(x).view(-1, self.n_groups, self.n_experts)
        else:
            logits = torch.einsum("bgi,gie->bge", xg, self.rw) + self.rb
        probs = logits.softmax(dim=-1)                       # (B,G,E)
        topv, topi = probs.topk(self.top_k, dim=-1)
        topv = topv / topv.sum(dim=-1, keepdim=True)
        gate = torch.zeros_like(probs).scatter(-1, topi, topv)
        return probs, gate

    def forward(self, x):
        """x 是 (B, n_groups*in_dim)，回傳 ((B, n_groups*out_dim), aux_loss)。"""
        B = x.shape[0]
        xg = x.view(B, self.n_groups, self.in_dim)
        probs, gate = self._gate(x, xg)

        h = torch.einsum("bgi,geih->bgeh", xg, self.w1) + self.b1
        h = F.gelu(h)
        y = torch.einsum("bgeh,geho->bgeo", h, self.w2) + self.b2   # (B,G,E,O)
        out = (y * gate.unsqueeze(-1)).sum(dim=2)                   # (B,G,O)
        return out.reshape(B, -1), self._load_balance_loss(probs, gate)

    def _load_balance_loss(self, probs, gate):
        # 每組各自算 switch-style aux loss：實際被選比例 × 平均 router 機率
        frac = (gate > 0).float().mean(dim=0)          # (G,E)，總和 = top_k
        mean_prob = probs.mean(dim=0)                  # (G,E)，總和 = 1
        return (self.n_experts / self.top_k) * (frac * mean_prob).sum(-1).mean()

    def expert_usage(self, x):
        """x 是 (B, n_groups*in_dim)，回傳 (G,E) 的 expert 被選中比例，給分析用。"""
        with torch.no_grad():
            xg = x.view(x.shape[0], self.n_groups, self.in_dim)
            _, gate = self._gate(x, xg)
            return (gate > 0).float().mean(dim=0)
