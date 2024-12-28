import torch
import torch.nn as nn
import torch.nn.functional as f


# Feature modules
class MLP(nn.Module):
    """
    Feed Forward MLP block :)
    """

    def __init__(self, embed_dim: int, coef: int = 4) -> None:
        """
        :param embed_dim: The input embedding dimension :)
        :param coef: coefficient for the MLP block :)
        """
        super().__init__()
        self.w1 = nn.Linear(embed_dim, int(embed_dim * coef))
        self.w2 = nn.Linear(int(embed_dim * coef), embed_dim)
        self.w3 = nn.Linear(embed_dim, int(embed_dim * coef))
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP block :)
        :param x: input tensor [batch_size, num_tokens, embed_dim] :)
        :return: output tensor [batch_size, num_tokens, embed_dim] :)
        """
        return self.w2(self.activation(self.w1(x)) * self.w3(x))


# Temporal modules

class CausalConv1dWithState(nn.Module):
    """
    This module implements a causal convolution layer with state :)
    """

    def __init__(self, embed_dim: int, kernel_size: int) -> None:
        """
        :param embed_dim: The input embedding dimension :)
        :param kernel_size: The kernel size of the causal convolution :)
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1

        self.conv = nn.Conv1d(
                self.embed_dim, self.embed_dim,
                kernel_size=self.kernel_size,
                stride=1, padding=0, groups=self.embed_dim,
            )
        
        self.act = nn.Sequential(
            nn.RMSNorm(self.embed_dim),
            nn.SiLU(),
        )

    def forward(self, x, state=None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the causal convolution layer with state :)
        :param x: input tensor [batch_size, seq_len, embed_dim] :)
        :param state: The hidden state of the layer [batch_size, kernel_size - 1, embed_dim] :)
        :return: Output tensor [batch_size, seq_len, embed_dim]
        and the new hidden state [batch_size, kernel_size - 1, embed_dim] :)
        """
        if state is None:
            state = torch.zeros(x.shape[0], self.pad, x.shape[-1]).to(x.device)

        y = torch.cat([state, x], dim=1).transpose(2, 1)
        return self.act(self.conv(y).transpose(2, 1)), y.transpose(2, 1)[:, -self.pad:]


class ConvLSTMBlock(nn.Module):
    """
    This module implements Conv + LSTM + MLP block :)
    """

    def __init__(self, embed_dim: int, n_layer: int, kernel_size: int = 7, coef: int = 4) -> None:
        """
        :param embed_dim: The input embedding dimension :)
        :param n_layer: The number of LSTM layers :)
        :param kernel_size: The kernel size of the causal convolution :)
        :param coef: coefficient for the MLP block :
        """
        super().__init__()

        self.norm = nn.RMSNorm(embed_dim)
        self.conv_layer = CausalConv1dWithState(embed_dim=embed_dim, kernel_size=kernel_size)
        self.rnn_layer = nn.LSTM(embed_dim, embed_dim, batch_first=True, num_layers=n_layer)
        
        self.mlp = MLP(embed_dim, coef=coef)
        self.mlp_nrom = nn.RMSNorm(embed_dim)

    def forward(self, x: torch.Tensor, state=None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        :param x: Input tensor [batch_size, seq_len, embed_dim] :)
        :param state: The hidden states of the LSTM [num_layer, batch_size, embed_dim] and
        CausalConv1dWithState [batch_size, kernel_size - 1, embed_dim]:)
        :return: Output tensor [batch_size, seq_len, embed_dim] and
        the new hidden states [[num_layer, batch_size, embed_dim], [batch_size, kernel_size - 1, embed_dim]] :)
        """
        if state is not None:
            rnn_state = state[0]
            conv_state = state[1]
        else:
            rnn_state = None
            conv_state = None
#
        y, conv_state = self.conv_layer(self.norm(x), conv_state)
        y, rnn_state = self.rnn_layer(y, rnn_state)
        y = self.mlp(self.mlp_nrom(y)) + x
        return y , (rnn_state, conv_state)


class CausalAvgPool1dWithState(nn.Module):
    """
    This module implements a causal average pool layer with state :)
    """

    def __init__(self, kernel_size: int) -> None:
        """
        :param kernel_size: The kernel size of the causal convolution :)
        """
        super().__init__()

        self.kernel_size = kernel_size
        self.pad = kernel_size - 1

        self.avg = nn.AvgPool1d(
            kernel_size=self.kernel_size,
            stride=1, padding=0
        )

    def forward(self, x, state=None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the causal convolution layer with state :)

        :param x: input tensor [batch_size, seq_len, embed_dim] :)
        :param state: The hidden state of the layer [batch_size, kernel_size - 1, embed_dim] :)
        :return: Output tensor [batch_size, seq_len, embed_dim] and
        the new hidden state [batch_size, kernel_size - 1, embed_dim] :)
        """
        if state is None:
            state = torch.zeros(x.shape[0], self.pad, x.shape[-1]).to(x.device)

        y = torch.cat([state, x], dim=1).transpose(2, 1)

        return self.avg(y).transpose(2, 1), y.transpose(2, 1)[:, -self.pad:]


# Spatial Blocks

class SelfAttention(nn.Module):
    """
    Self attention encoder module with masked attention mechanism :)
    """

    def __init__(self, embed_dim: int, adj_mat: torch.tensor = None, n_head: int = 4) -> None:
        """
        :param embed_dim: The input embedding dimension :)
        :param adj_mat: Adjacency matrix :)
        :param n_head: The number of attention head :)
        """
        super().__init__()
        self.n_embd = embed_dim
        self.n_head = n_head
        if adj_mat is not None:
            self.adj_mat = adj_mat + torch.eye(adj_mat.shape[-1], device=adj_mat.device)
            self.adj_mat = self.adj_mat.reshape(1, 1, *self.adj_mat.shape).bool()
        else:
            self.adj_mat = None

        self.qkv_linear = nn.Linear(self.n_embd, self.n_embd * 3)
        self.proj_linear = nn.Linear(self.n_embd, self.n_embd)
        self.proj_linear.MODEL_SCALE_INIT = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the self-attention encoder module :)
        :param x: Input tensor [batch_size, num_tokens, embed_dim] :)
        :return: Output tensor [batch_size, num_tokens, embed_dim] :)
        """
        B, N, _ = x.shape
        qkv = self.qkv_linear(x)
        qkv = qkv.reshape(B, N, 3, self.n_head, self.n_embd // self.n_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  #

        y = f.scaled_dot_product_attention(query=q, key=k, value=v,
                                           attn_mask=self.adj_mat)

        y = y.permute(0, 2, 1, 3).reshape(B, N, self.n_embd)
        return self.proj_linear(y)


class TransformerBlock(nn.Module):
    """
    Transformer encoder block with masked attention mechanism :)
    """

    def __init__(self, embed_dim: int, adj_mat: torch.tensor = None, n_head: int = 4, coef: int = 4) -> None:
        """
        :param embed_dim: The input embedding dimension :)
        :param adj_mat: Adjacency matrix :)
        :param n_head: The number of attention head :)
        :param coef: The coefficient of MLP :)
        """
        super().__init__()
        self.self_attn = SelfAttention(embed_dim, adj_mat, n_head=n_head)
        self.ln1 = nn.RMSNorm(embed_dim)
        self.ln2 = nn.RMSNorm(embed_dim)
        self.mlp = MLP(embed_dim, coef=coef)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder block :
        :param x: Input tensor [batch_size, num_tokens, embed_dim] :
        :return: Output tensor [batch_size, num_tokens, embed_dim] :
        """
        x = x + self.self_attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Mixer(nn.Module):
    """
    MLP Mixer block :)
    """

    def __init__(self, d_input: int = 6, d_output: int = 10, coef: int = 4) -> None:
        """
        :param d_input: Number of input tokens :)
        :param d_output: Number of output tokens :)
        :param coef: The coefficient of Mixer :
        """
        super().__init__()
        self.w1 = nn.Linear(d_input, int(d_input * coef))
        self.w2 = nn.Linear(int(d_input * coef), d_output)
        self.w3 = nn.Linear(d_input, int(d_input * coef))
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Mixer :
        :param x: Input tensor [batch_size, num_tokens_input, embed_dim]:
        :return: Output tensor [batch_size, num_tokens_output, embed_dim]:
        """
        x = x.transpose(-1, -2)
        return self.w2(self.activation(self.w1(x)) * self.w3(x)).transpose(-1, -2)


class MixerBlock(nn.Module):
    """
    Mixer block with feed-forward layer :)
    """

    def __init__(self, n_embd: int, d_input: int = 6, d_output: int = 10, coef: int = 4) -> None:
        """
        :param n_embd: Embedding dimension :)
        :param d_input: Number of input tokens :)
        :param d_output: Number of output tokens :)
        """
        super().__init__()
        self.mixer = Mixer(d_input, d_output)
        self.mixer_norm = nn.RMSNorm(n_embd)
        self.shortcut = nn.Linear(d_input, d_output)

        self.mlp = MLP(n_embd)
        self.mlp_norm = nn.RMSNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Mixer block :
        :param x: Input tensor [batch_size, num_tokens_input, embed_dim]:
        :return: Output tensor [batch_size, num_tokens_output, embed_dim]:
        """
        x = self.mixer(self.mixer_norm(x)) + self.shortcut(x.transpose(-1, -2)).transpose(-1, -2)
        return self.mlp(self.mlp_norm(x)) + x


class SelfAttentionDecoder(nn.Module):
    """
    Self attention decoder module with masked attention mechanism :
    """

    def __init__(self, embed_dim: int, adj_mat: torch.tensor = None, n_head: int = 4) -> None:
        """
        :param embed_dim: Embedding dimension :)
        :param adj_mat: adjacency matrix :)
        :param n_head: The number of attention head :)
        """
        super().__init__()
        self.n_embd = embed_dim
        self.n_head = n_head

        self.q_porj = nn.Linear(self.n_embd, self.n_embd)
        self.k_porj = nn.Linear(self.n_embd, self.n_embd)
        self.v_porj = nn.Linear(self.n_embd, self.n_embd)

        self.proj_linear = nn.Linear(self.n_embd, self.n_embd)
        self.proj_linear.MODEL_SCALE_INIT = 1

        self.n1 = adj_mat.shape[0]
        self.n2 = adj_mat.shape[1]
        self.adj_mat = adj_mat.bool()

    def forward(self, q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the self-attention decoder :
        :param q: query tensor [batch_size, num_tokens_input, embed_dim]:
        :param x: key and value tensor [batch_size, num_tokens_input, embed_dim]:
        :return: query tensor [batch_size, num_tokens_output, embed_dim]:
        """
        B, N, _ = x.shape

        q = self.q_porj(q).reshape(B, self.n1, self.n_head, self.n_embd // self.n_head).transpose(1, 2)
        k = self.k_porj(x).reshape(B, self.n2, self.n_head, self.n_embd // self.n_head).transpose(1, 2)
        v = self.v_porj(x).reshape(B, self.n2, self.n_head, self.n_embd // self.n_head).transpose(1, 2)

        q = f.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=self.adj_mat)

        # B, H, N, D
        q = q.permute(0, 2, 1, 3).reshape(B, self.n1, self.n_embd)

        return self.proj_linear(q)


class TransformerBlockDecoder(nn.Module):
    """
    Transformer decoder block with masked attention mechanism :
    """

    def __init__(self, embed_dim: int, adj_mat_query: torch.tensor = None, adj_mat_query_key: torch.tensor = None,
                 n_head: int = 4, coef: int = 4) -> None:
        """
        :param embed_dim: Embedding dimension :)
        :param adj_mat_query: adjacency matrix query tensor :)
        :param adj_mat_query_key: adjacency matrix query to key tensor :)
        :param n_head: The number of attention head :)
        :param coef: The coefficient of MLP :)
        """
        super().__init__()
        self.self_attn1 = SelfAttention(embed_dim, adj_mat_query, n_head=n_head)
        self.self_attn2 = SelfAttentionDecoder(embed_dim, adj_mat_query_key, n_head=n_head)

        self.ln1 = nn.RMSNorm(embed_dim)
        self.ln2 = nn.RMSNorm(embed_dim)
        self.ln3 = nn.RMSNorm(embed_dim)

        self.mlp = MLP(embed_dim, coef=coef)

    def forward(self, qx: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the transformer decoder block :
        :param qx: query and key tensor [[batch_size, num_tokens_query, embed_dim],
        [batch_size, num_tokens_key, embed_dim]] :
        :return:
        """
        q, x = qx
        q = q + self.self_attn1(self.ln1(q))
        q = q + self.self_attn2(q=self.ln2(q), x=x)
        q = q + self.mlp(self.ln3(q))
        return q, x


if __name__ == '__main__':
    pass
