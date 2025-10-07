import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# RNN
class RNNCell(nn.Module):
	"""
	RNN Cell update rule.
	"""
	def __init__(self, input_dim, hidden_dim):
		"""
		Inputs:
		- input_dim:  dimension of the input
		- hidden_dim: dimention of the hidden state
		"""
		super().__init__()
		self.d_input = input_dim
		self.d_hidden = hidden_dim

		# Initialize the matrices
		self.Whh = nn.Linear(hidden_dim, hidden_dim)
		self.Whx = nn.Linear(input_dim,  hidden_dim)

	def forward(self, x_t, h_prev):
		"""
		Input:
		- x_t: input at the current time step        (batch_size, input_dim)
		- h_prev: hidden state at previous time step (batch_size, hidden_dim)
	
		Returns the updated hidden state (batch_size, hidden_dim).
		"""
		return torch.tanh(self.Whh(h_prev) + self.Whx(x_t))


class RNN(nn.Module):
	"""
	RNN model
	"""
	def __init__(self, input_dim, output_dim, hidden_dim):
		"""
		Inputs:
		- input_dim:  dimension of the input
		- output_dim: dimension of the output
		- hidden_dim: dimention of the hidden state
		"""
		super().__init__()
		self.d_input = input_dim
		self.d_output = output_dim
		self.d_hidden = hidden_dim

		self.RNNCell = RNNCell(input_dim, hidden_dim)
		self.Wyh = nn.Linear(hidden_dim, output_dim)

	def forward(self, x, h_0=None):
		"""
		Inputs:
		- x: input (batch_size, T, input_dim)
		- h_0 (optional): provide the current hidden state (batch_size, hidden_dim)

		Output:
		- y_T: output (batch_size, T, output_dim)
		- h_T: final hidden state (batch_size, hidden_dim)
		"""
		# shapes and device
		batch_size, T, _ = x.shape
		device = x.device

		# initializing h_0
		if h_0 is None:
			h_t = torch.zeros(batch_size, self.d_hidden, device=device)
		else:
			h_t = h_0

		y_out = []
		for t in range(T):
			x_t = x[:, t, :]
			# update using RNNCell
			h_t = self.RNNCell(x_t, h_t)
			y_t = self.Wyh(h_t)
			y_out.append(y_t)

		return torch.stack(y_out, dim=1), h_t


# LSTM
class LSTMCell(nn.Module):
	"""
	LSTM Cell update rule.
	"""
	def __init__(self, input_dim, hidden_dim):
		"""
		Inputs:
		- input_dim:  dimension of the input
		- hidden_dim: dimention of the hidden state
		"""
		super().__init__()
		self.d_input = input_dim
		self.d_hidden = hidden_dim
		
		# Initialize the matrices
		self.Wfx = nn.Linear(input_dim, hidden_dim)
		self.Wfh = nn.Linear(hidden_dim, hidden_dim)
		self.Wix = nn.Linear(input_dim, hidden_dim)
		self.Wih = nn.Linear(hidden_dim, hidden_dim)
		self.Wox = nn.Linear(input_dim, hidden_dim)
		self.Woh = nn.Linear(hidden_dim, hidden_dim)
		self.Wcx = nn.Linear(input_dim, hidden_dim)
		self.Wch = nn.Linear(hidden_dim, hidden_dim)
	
	def forward(self, x_t, c_prev, h_prev):
		"""
		Input:
		- x_t: input at the current time step        (batch_size,  input_dim)
		- c_prev: cell state at previous time step   (batch_size, hidden_dim)
		- h_prev: hidden state at previous time step (batch_size, hidden_dim)

		Returns
		- c_t: updated cell state   (batch_size, hidden_dim)
		- h_t: updated hidden state (batch_size, hidden_dim)
		"""
		# calculate the gates
		f = torch.sigmoid(self.Wfx(x_t) + self.Wfh(h_prev))
		i = torch.sigmoid(self.Wix(x_t) + self.Wih(h_prev))
		o = torch.sigmoid(self.Wox(x_t) + self.Woh(h_prev))

		# new (proposed) value
		c_tilde = torch.tanh(self.Wcx(x_t) + self.Wch(h_prev))

		# updated cell value
		c_t = f * c_prev + i * c_tilde

		# updated hidden state
		h_t = o * torch.tanh(c_t)

		return c_t, h_t


class LSTM(nn.Module):
	"""
	LSTM model
	"""
	def __init__(self, input_dim, output_dim, hidden_dim):
		"""
		Inputs:
		- input_dim:  dimension of the input
		- output_dim: dimension of the output
		- hidden_dim: dimention of the hidden state
		"""
		super().__init__()
		self.d_input = input_dim
		self.d_output = output_dim
		self.d_hidden = hidden_dim

		self.LSTMCell = LSTMCell(input_dim, hidden_dim)
		self.Wyh = nn.Linear(hidden_dim, output_dim)

	def forward(self, x, c_0=None, h_0=None):
		"""
		Inputs:
		- x: input (batch_size, T, input_dim)
		- c_0 (optional): provide the current cell state (batch_size, hidden_dim)
		- h_0 (optional): provide the current hidden state (batch_size, hidden_dim)

		Output:
		- y_T: output (batch_size, T, output_dim)
		- c_T: final cell state (batch_size, hidden_dim)
		- h_T: final hidden state (batch_size, hidden_dim)
		"""
		# shapes and device
		batch_size, T, _ = x.shape
		device = x.device

		# initializing the cell and hidden states
		if c_0 is None:
			c_t = torch.zeros(batch_size, self.d_hidden, device=device)
		else:
			c_t = c_0

		if h_0 is None:
			h_t = torch.zeros(batch_size, self.d_hidden, device=device)
		else:
			h_t = h_0

		# update
		y_out = []
		for t in range(T):
			x_t = x[:, t, :]
			# update using LSTMCell
			c_t, h_t = self.LSTMCell(x_t, c_t, h_t)
			y_t = self.Wyh(h_t)
			y_out.append(y_t)

		return torch.stack(y_out, dim=1), c_t, h_t
