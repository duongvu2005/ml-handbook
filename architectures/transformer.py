import torch
import torch.nn as nn
import torch.nn.functional as F


DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class Attention(nn.Module):
	"""
	Implementation of a simple attention layer.
	"""
	def __init__(self, embed_dim, key_dim):
		"""
		Inputs:
		- embed_dim: dimension of the token embedding
		- key_dim:   dimension of the key (value, and query)
		"""
		super().__init__()
		self.d_embed = embed_dim
		self.d_key   = key_dim

		# The QKV matrices
		self.Wq = nn.Linear(embed_dim, key_dim)
		self.Wk = nn.Linear(embed_dim, key_dim)
		self.Wv = nn.Linear(embed_dim, key_dim)

		# Fully connected layer at the end
		self.Wc = nn.Linear(key_dim, embed_dim)
	
	def forward(self, x):
		"""
		Input:
		- x: (batch_size, seq_len, embed_dim)

		Output:
		- x: (batch_size, seq_len, embed_dim)
		- A: the attention matrices (batch_size, seq_len, seq_len)
		"""
		# Note: x has shape (B, n, d)
		Q = self.Wq(x)  # (B, n, d_k)
		K = self.Wk(x)  # (B, n, d_k)
		V = self.Wv(x)  # (B, n, d_k)

		A = F.softmax(Q @ K.transpose(-2, -1) / self.d_key**0.5, dim=-1)  # (b, n, n)

		# Forward pass
		x = A @ V       # (B, n, d_k)
		x = self.Wc(x)  # (B, n, d)

		return x, A


if __name__ == "__main__":
    # Testing to make sure that the shape works out
	# Setting up the layer
    embed_dim = 4
    key_dim = 2

    attention_layer = Attention(embed_dim, key_dim)
    attention_layer.to(DEVICE)

    # Dummy data
    batch_size = 10
    seq_len = 20
    x = torch.randn(batch_size, seq_len, embed_dim, device=DEVICE)
    print(f"The shape of our input is {tuple(x.cpu().shape)}")

    # Forward pass
    x, A = attention_layer(x)
    print(f"The shape of our output is {tuple(x.cpu().shape)}")
    print(f"The shape of our attention matrices is {tuple(A.cpu().shape)}")