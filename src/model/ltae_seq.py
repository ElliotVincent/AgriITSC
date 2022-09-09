import copy

import numpy as np
import torch
import torch.nn as nn


class LTAE1d(nn.Module):
    def __init__(
        self,
        in_channels=10,
        n_head=16,
        d_k=4,
        mlp=[256, 128],
        dropout=0.2,
        d_model=256,
        return_att=False,
        num_attention=13
    ):
        """
        Lightweight Temporal Attention Encoder (L-TAE) for image time series.
        Attention-based sequence encoding that maps a sequence of images to a single feature map.
        A shared L-TAE is applied to all pixel positions of the image sequence.
        Args:
            in_channels (int): Number of channels of the input embeddings.
            n_head (int): Number of attention heads.
            d_k (int): Dimension of the key and query vectors.
            mlp (List[int]): Widths of the layers of the MLP that processes the concatenated outputs of the attention heads.
            dropout (float): dropout
            d_model (int, optional): If specified, the input tensors will first processed by a fully connected layer
                to project them into a feature space of dimension d_model.
            T (int): Period to use for the positional encoding.
            return_att (bool): If true, the module returns the attention masks along with the embeddings (default False)
            positional_encoding (bool): If False, no positional encoding is used (default True).
        """
        super(LTAE1d, self).__init__()
        self.in_channels = in_channels
        self.mlp = copy.deepcopy(mlp)
        self.return_att = return_att
        self.n_head = n_head
        self.num_attention = num_attention

        self.d_model = d_model
        self.inconv = nn.Conv1d(in_channels, d_model, 1)

        assert self.mlp[0] == self.d_model

        self.attention_heads = MultiHeadAttention(
            n_head=n_head, d_k=d_k, d_in=self.d_model, num_attention=self.num_attention
        )
        self.in_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=self.d_model,
        )
        self.out_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=mlp[-1],
        )

        layers = []
        for i in range(len(self.mlp) - 1):
            layers.extend(
                [
                    nn.Linear(self.mlp[i], self.mlp[i + 1]),
                    nn.BatchNorm1d(self.mlp[i + 1]),
                    nn.ReLU(),
                ]
            )

        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_att=False):
        num_seq, seq_len, d = x.shape
        out = self.inconv(x.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)

        out, attn = self.attention_heads(out)

        out = (
            out.permute(0, 2, 1, 3).contiguous().view(self.num_attention * num_seq, -1)
        )  # Concatenate heads
        out = self.dropout(self.mlp(out))
        out = self.out_norm(out) if self.out_norm is not None else out
        out = out.view(self.num_attention, num_seq, -1).permute(1, 0, 2)
        attn = attn.view(self.num_attention, self.n_head, num_seq, seq_len).permute(2, 0, 1, 3)  # head x b x t

        if return_att:
            return out, attn
        else:
            return out


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, n_head, d_k, d_in, num_attention):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in
        self.num_attention = num_attention

        self.Q = nn.Parameter(torch.zeros((num_attention, n_head, d_k))).requires_grad_(True)
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, num_attention * n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), num_attention=num_attention)

    def forward(self, v, return_comp=False):
        d_k, d_in, n_head, num_attention = self.d_k, self.d_in, self.n_head, self.num_attention
        sz_b, seq_len, _ = v.size()

        q = torch.stack([self.Q for _ in range(sz_b)], dim=2).view(
            -1, d_k
        )  # (n*b*nA) x d_k

        k = self.fc1_k(v).view(sz_b, seq_len, num_attention, n_head, d_k)
        k = k.permute(3, 0, 1, 2, 4).contiguous().view(-1, seq_len, d_k)  # (n*b*nA) x lk x dk

        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(
            n_head * sz_b, seq_len, -1
        )
        if return_comp:
            output, attn, comp = self.attention(
                q, k, v, return_comp=return_comp
            )
        else:
            output, attn = self.attention(
                q, k, v, return_comp=return_comp
            )
        attn = attn.view(num_attention, n_head, sz_b, 1, seq_len)
        attn = attn.squeeze(dim=3)

        output = output.view(num_attention, n_head, sz_b, 1, d_in // n_head)
        output = output.squeeze(dim=3)

        if return_comp:
            return output, attn, comp
        else:
            return output, attn


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, temperature, num_attention, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)
        self.num_attention = num_attention

    def forward(self, q, k, v, return_comp=False):
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2))
        attn = attn / self.temperature
        if return_comp:
            comp = attn
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        _, t, c = v.shape
        v = v[:, None, ...].expand(-1, self.num_attention, -1, -1).reshape(-1, t, c)
        output = torch.matmul(attn, v)
        if return_comp:
            return output, attn, comp
        else:
            return output, attn
