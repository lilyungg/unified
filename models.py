import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, emb_module: nn.Module, emb_out_dim: int):
        super().__init__()
        self.emb = emb_module
        self.mlp = nn.Sequential(
            nn.Linear(emb_out_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),         nn.ReLU(),
            nn.Linear(128, 1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.emb(x)).squeeze(1)


class DCNV2LowRank(nn.Module):
    def __init__(self, emb_module: nn.Module, emb_out_dim: int,
                 num_cross: int = 2, rank: int = 64,
                 dnn_dims: tuple = (256, 128),
                 dropout: float = 0.1, reg_weight: float = 1e-5):
        super().__init__()
        D = emb_out_dim
        self.reg_weight = reg_weight
        self.emb = emb_module

        self.cross_U = nn.ParameterList([nn.Parameter(torch.empty(D, rank)) for _ in range(num_cross)])
        self.cross_V = nn.ParameterList([nn.Parameter(torch.empty(D, rank)) for _ in range(num_cross)])
        self.cross_b = nn.ParameterList([nn.Parameter(torch.zeros(D))       for _ in range(num_cross)])

        layers, in_d = [], D
        for out_d in dnn_dims:
            layers += [nn.Dropout(dropout), nn.Linear(in_d, out_d), nn.BatchNorm1d(out_d), nn.ReLU()]
            in_d = out_d
        self.dnn    = nn.Sequential(*layers)
        self.output = nn.Linear(in_d, 1)

        self._init_weights()

    def _init_weights(self):
        for p in self.cross_U: nn.init.xavier_normal_(p)
        for p in self.cross_V: nn.init.xavier_normal_(p)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def cross_network(self, x0: torch.Tensor) -> torch.Tensor:
        x = x0
        for U, V, b in zip(self.cross_U, self.cross_V, self.cross_b):
            x = x0 * ((x @ V) @ U.T + b) + x
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.emb(x)
        return self.output(self.dnn(self.cross_network(e))).squeeze(1)

    def reg_loss(self) -> torch.Tensor:
        return self.reg_weight * (
            sum(U.norm(2) for U in self.cross_U) +
            sum(V.norm(2) for V in self.cross_V)
        )
