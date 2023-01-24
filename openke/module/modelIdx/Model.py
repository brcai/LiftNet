import torch
import torch.nn as nn
import numpy as np
from ..BaseModule import BaseModule

def next_power_of_2(x):  
	num = 1 if x == 0 else 2**(x - 1).bit_length()
	return int(np.log2(num))

def binary(x, bits):
	mask = 2**torch.arange(bits).to(x.device, x.dtype)
	return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()

class Model(BaseModule):

	def __init__(self, ent_tot, rel_tot):
		super(Model, self).__init__()
		self.ent_tot = ent_tot
		self.rel_tot = rel_tot

	def forward(self):
		raise NotImplementedError
	
	def predict(self):
		raise NotImplementedError


#2 layers
class LiftNet(nn.Module):
	def __init__(self, ent_tot, hidden_dim, dim):
		super(LiftNet, self).__init__()
		self.dim = dim
		self.net = nn.Sequential(
		# [bs, nz, 16, 16] -> [bs, 128, 30, 30]
		nn.ConvTranspose2d(1, 4, 4, bias = False),
		nn.BatchNorm2d(4),
		nn.ReLU(True),
		nn.ConvTranspose2d(4, int(self.dim[1]/64), 4, bias = False),
		nn.Tanh()
		#nn.BatchNorm2d(8),
		#nn.ReLU(True)
		)
	def forward(self, x):
		tmp = x.reshape(x.shape[0], 1, 2, 2)
		out = self.net(tmp)
		out = out.reshape(-1, self.dim[1])
		return out
	

