import torch
import torch.nn as nn
from utils import normalize_adj

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, horizon, output_size, dropout=0.1):
        """
        input_size:    number of sensors = 207
        hidden_size:   LSTM hidden dimension (e.g., 64)
        num_layers:    stacked LSTM layers (e.g., 3)
        horizon:       number of future steps to predict (e.g., 3 or 12)
        output_size:   number of sensors = 207
        """
        super().__init__()
        self.horizon = horizon

        # Encode past sequence
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        x: (batch, lookback, input_size)
        Returns: (batch, horizon, output_size)
        """
        _, (h, c) = self.encoder(x)
        
        # use last observed timestep as first decoder input
        decoder_input = x[:, -1:, :] # (batch, 1, input_size)
        outputs = []

        for _ in range(self.horizon):
            out, (h, c) = self.encoder(decoder_input, (h, c))
            n_out = self.norm(out[:, -1, :]) # (batch, output_size)
            step_pred = self.fc(n_out)  
            outputs.append(step_pred.unsqueeze(1))
            decoder_input = step_pred.unsqueeze(1)    # feed prediction as next input

        return torch.cat(outputs, dim=1)  # (batch, horizon, output_size)


class STGCNBlock(nn.Module):
    def __init__(self, in_feats, spatial_feats, gcn_feats, adj_matrix, dropout, kernel_size=3):
        super().__init__()

        self.temp1 = nn.Conv1d(
            in_feats, spatial_feats,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

        self.register_buffer("A_hat", normalize_adj(adj_matrix))

        W = torch.empty(spatial_feats, gcn_feats)
        nn.init.xavier_uniform_(W)
        self.W = nn.Parameter(W)

        self.temp2 = nn.Conv1d(
            gcn_feats, spatial_feats,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

        self.norm1 = nn.LayerNorm(spatial_feats)
        self.norm2 = nn.LayerNorm(gcn_feats)
        self.norm3 = nn.LayerNorm(spatial_feats)

        # Residual projection if in_feats != spatial_feats
        self.residual = (
            nn.Linear(in_feats, spatial_feats)
            if in_feats != spatial_feats else nn.Identity()
        )

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (B, T, N, F)
        if x.ndim == 3:
            x = x.unsqueeze(-1)

        B, T, N, F = x.shape
        A = self.A_hat

        # Save input for residual, project if needed
        residual = self.residual(x)  # (B, T, N, spatial_feats)

        # ---- Temporal Conv 1 ----
        x = x.permute(0, 2, 3, 1)              # (B, N, F, T)
        x = x.reshape(B*N, F, T)
        x = self.temp1(x)                      # (B*N, spatial_feats, T)
        S = x.shape[1]                         # spatial_feats

        # reshape for GCN
        x = x.reshape(B, N, S, T).permute(0, 3, 1, 2)  # (B, T, N, S)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # ---- Graph Convolution ----
        x = torch.einsum("ij,btjf->btif", A, x)        # A X
        x = torch.einsum("btif,fh->btih", x, self.W)   # (A X) W
        x = self.norm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # ---- Temporal Conv 2 ----
        x = x.permute(0, 2, 3, 1)              # (B, N, gcn_feats, T)
        x = x.reshape(B*N, self.W.shape[1], T)
        x = self.temp2(x)                      # (B*N, S, T)

        # reshape back
        x = x.reshape(B, N, S, T).permute(0, 3, 1, 2)  # (B, T, N, S)
        x = self.norm3(x)

        x += residual

        return self.relu(x)


class STGCN(nn.Module):
    def __init__(self, adj_matrix, input_size, hidden_size, gcn_channels, num_layers,
                 horizon, output_size, dropout=0.1):
        super().__init__()

        layers = []
        in_dim = input_size

        for _ in range(num_layers):
            layers.append(
                STGCNBlock(
                    in_feats=in_dim,
                    spatial_feats=hidden_size,
                    gcn_feats=gcn_channels,
                    adj_matrix=adj_matrix,
                    dropout=dropout
                )
            )
            in_dim = hidden_size

        self.layers = nn.ModuleList(layers)
        self.horizon = horizon
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(-1)

        h = x
        for layer in self.layers:
            h = layer(h)

        # keep last H timesteps
        h = h[:, -self.horizon:]     # (B, H, N, hidden)
        out = self.fc(h)              # (B, H, N, out)

        return out.squeeze(-1)
    
    
class BottleneckGCN(nn.Module):
    """
    in_ch -> reduce to bottleneck (gcn_channels) -> graph mixing -> expand to out_ch
    """
    def __init__(self, in_ch, gcn_channels, out_ch, adj, bias=True):
        super().__init__()
        self.register_buffer("A_hat", normalize_adj(adj))  # (N,N)
        self.reduce = nn.Linear(in_ch, gcn_channels, bias=True)

        # weight for mixing inside bottleneck (learnable linear transform)
        self.gc_weight = nn.Parameter(torch.randn(gcn_channels, gcn_channels) * (1.0 / (gcn_channels**0.5)))

        self.expand = nn.Linear(gcn_channels, out_ch, bias=bias)
        self.act = nn.ReLU()

        # if residual projection needed (in_ch -> out_ch)
        if in_ch != out_ch:
            self.res_proj = nn.Linear(in_ch, out_ch, bias=False)
        else:
            self.res_proj = None

    def forward(self, x):
        # x: (B, T, N, F)
        skip = x  # keep for residual
        H = self.reduce(x)                              # (B, T, N, gcn_channels)
        # spatial mixing per time-step
        H = torch.einsum("ij,btjc->btic", self.A_hat, H)  # (B, T, N, gcn_channels)
        # channel mixing inside bottleneck
        H = torch.einsum("btic,cd->btid", H, self.gc_weight)  # (B, T, N, gcn_channels)
        H = self.expand(H)                               # (B, T, N, out_ch)

        if self.res_proj is not None:
            skip = self.res_proj(skip)
        out = self.act(H + skip)
        return out  # (B, T, N, out_ch)

# -------------------------
# Spatial encoder: stack of BottleneckGCN layers (can be 1..K)
# -------------------------
class SpatialEncoder(nn.Module):
    def __init__(self, in_ch, gcn_channels, out_ch, adj_matrix, num_layers=2, dropout=0.0):
        super().__init__()
        layers = []
        # first layer: from in_ch -> out_ch via bottleneck
        layers.append(BottleneckGCN(in_ch, gcn_channels, out_ch, adj_matrix, bias=True))
        for _ in range(num_layers - 1):
            layers.append(BottleneckGCN(out_ch, gcn_channels, out_ch, adj_matrix, bias=True))
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # x: (B, T, N, F)
        H = x
        for l in self.layers:
            H = l(H)
            H = self.dropout(H)
        return H  # (B, T, N, out_ch)
    
class LearnedTemporalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.encoding = nn.Parameter(torch.zeros(1, max_len, 1, d_model))
        nn.init.xavier_uniform_(self.encoding.squeeze(0).squeeze(1).unsqueeze(0))
    
    def forward(self, x):
        # x: (B, T, N, F)
        return x + self.encoding[:, :x.size(1)]

# -------------------------
# Full GCN-LSTM model (per-node LSTM)
# -------------------------
class GCN_LSTM(nn.Module):
    def __init__(
        self,
        adj_matrix,
        input_size=1,
        gcn_channels=16,   # bottleneck size
        gcn_out=16,        # per-node embedding size produced by spatial encoder
        lstm_hidden=64,
        num_layers=1,
        lookback=12,
        horizon=3,
        node_embed_dim=8,
        output_size=1,
        spatial_layers=2,
        dropout=0.1,
    ):
        """
        adj: (N,N) adjacency tensor
        input_size: feature channels per node (usually 1)
        gcn_channels: bottleneck channels inside GCN
        gcn_out: output channels per node from spatial encoder (used as LSTM input dim)
        lstm_hidden: LSTM hidden dim (per node)
        horizon: number of future timesteps to predict
        output_size: number of output channels per node (usually 1)
        """
        super().__init__()
        self.N = adj_matrix.shape[0]
        self.horizon = horizon
        self.output_size = output_size

        self.node_embedding = nn.Embedding(self.N, node_embed_dim)

        self.temporal = LearnedTemporalEncoding(max_len=lookback, d_model=input_size)

        self.spatial = SpatialEncoder(in_ch=input_size,
                                      gcn_channels=gcn_channels,
                                      out_ch=gcn_out,
                                      adj_matrix=adj_matrix,
                                      num_layers=spatial_layers,
                                      dropout=dropout)

        # LSTM runs per-node (shared weights); input_size = gcn_out
        self.lstm = nn.LSTM(input_size=gcn_out + node_embed_dim,
                            hidden_size=lstm_hidden,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)

        # decoder -> turn LSTM hidden -> output_size (per node)
        self.decoder = nn.Linear(lstm_hidden, output_size)

    def forward(self, x):
        """
        Returns:
          (B, horizon, N) when output_size == 1
          else (B, horizon, N, output_size)
        """

        # ensure last dim is features
        if x.ndim == 3:
            x = x.unsqueeze(-1)  # (B, T, N, 1)

        B, T, N, F = x.shape

        H = self.temporal(x)
        H = self.spatial(H)  # (B, T, N, gcn_out)

        H = H.permute(0, 2, 1, 3).contiguous()   # (B, N, T, gcn_out)

        node_ids = torch.arange(N, device=x.device)
        node_emb = self.node_embedding(node_ids)           # (N, node_embed_dim)
        node_emb = node_emb.unsqueeze(0).unsqueeze(2)      # (1, N, 1, node_embed_dim)
        node_emb = node_emb.expand(B, -1, T, -1)           # (B, N, T, node_embed_dim)
        
        H = torch.cat([H, node_emb], dim=-1)               # (B, N, T, gcn_out + node_embed_dim)
        H = H.view(B * N, T, -1)                           # (B*N, T, gcn_out + node_embed_dim)

        H_lstm, _ = self.lstm(H)                           # (B*N, T, lstm_hidden)
        H_sel = H_lstm[:, -self.horizon:, :]               # (B*N, horizon, lstm_hidden)

        out = self.decoder(H_sel)                          # (B*N, horizon, output_size)
        out = out.view(B, N, self.horizon, self.output_size)
        out = out.permute(0, 2, 1, 3).contiguous()         # (B, horizon, N, output_size)

        if self.output_size == 1:
            out = out.squeeze(-1)  # (B, horizon, N)

        return out