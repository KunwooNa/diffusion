""" 
This file contains wrapper classes for discretized version of score-based diffusion models.
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 



class DDPM(object): 
	"""
	A wrapper class for the DDPM noise corruption. 
	"""
	def __init__(self, beta_min = 0.0001, beta_max = 0.02, N = 1000): 
		self.N = N
		self.beta_min = beta_min 
		self.beta_max = beta_max 
		# list of \{beta_t\}.
		self.betas = torch.linspace(beta_min, beta_max, N) 
		# list of \{alpha_t\}. 
		self.alphas = 1. - self.betas 
		self.sqrt_alphas = torch.sqrt(self.alphas) 
		self.sqrt_1m_alphas = torch.sqrt(1 - self.alphas)
		# list of \{\overline{\alpha}_t\}. 
		self.alpha_bars = torch.cumprod(self.alphas, dim = 0) 
		self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars) 
		self.sqrt_1m_alpha_bars = torch.sqrt(1 - self.alpha_bars)
		# list of beta_tilde. 
		alpha_bars_prev = F.pad(self.alpha_bars[:-1], (1, 0), value = 1.0) 
		self.sqrt_beta_tildes = torch.sqrt(self.betas * (1. - alpha_bars_prev) / (1. - self.alpha_bars))
	

	def get_idx(self, values, timestep, shape): 
		output = values.gather(-1, timestep.cpu()) 
		output = output.reshape(timestep.shape[0], *((1,) * (len(shape) - 1)))
		return output.to(timestep.device) 

	
	@property
	def N(self):
		return self.N


	def forward_corrupt(self, X_0, t): 
		""" 
		Noise-corrupt the image X_0 up to timestep t. 
		Remark: t has an integer values here, i.e., t == 0.001 is not possible. 
		Remark: In DDPM, the distribution of X_t conditioned on X_0 is known in a closed form. 
		Therefore, implementing "one-step" corruption is redundant. 
		"""
		X_0 = X_0.to(t.device)
		varepsilon = torch.randn_like(X_0)
		sqrt_alpha_bars = self.get_idx(
			torch.sqrt(self.alpha_bars), t, X_0.shape
		)
		sqrt_alpha_bars = sqrt_alpha_bars.to(t.device)
		sqrt_one_minus_alpha_bars = self.get_idx(
			torch.sqrt(1 - self.alpha_bars), t, X_0.shape
		)
		sqrt_one_minus_alpha_bars = sqrt_one_minus_alpha_bars.to(t.device)
		X_t = sqrt_alpha_bars * X_0 + sqrt_one_minus_alpha_bars * varepsilon
		return X_t, varepsilon 

	
	def reverse_sample(self, X_t, t, predicted_noise): 
		""" 
		Reverse-sample X_{t - 1} from the input X_t. 
		Remark: In reverse sampling, the distribution of X_t conditioned on X_0 is not known in a closed form. 
		Therefore, we have to implement "one-step" sampling. 
		"""
		X_t = X_t.to(t.device) 
		varepsilon = torch.randn_like(X_t) 
		sqrt_alpha = self.get_idx(self.sqrt_alphas, t, X_t.shape)
		sqrt_alpha = sqrt_alpha.to(t.device)
		beta = self.get_idx(self.betas, t, X_t.shape)
		beta = beta.to(t.device)
		sqrt_1m_alpha_bar = self.get_idx(self.sqrt_1m_alpha_bars, t, X_t.shape)
		sqrt_1m_alpha_bar = sqrt_1m_alpha_bar.to(t.device) 
		sqrt_beta_tilde = self.get_idx(self.sqrt_beta_tildes, t, X_t.shape) 
		sqrt_beta_tilde = sqrt_beta_tilde.to(t.device) 
		mean = (X_t - (beta / sqrt_1m_alpha_bar) * predicted_noise) / sqrt_alpha
		var = sqrt_beta_tilde * varepsilon 
		return mean + var 
	
