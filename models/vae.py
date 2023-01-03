import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import math 


class VanillaVAE(nn.Module): 
	""" 
	An implementation of simple VAE model. 
	This module takes an image $X$ and encodes it to a latent noise $Z$ with dimension $d$.
	
	Input: config (all required configurations.) 

	- in_channels: input image channels 
	- latent_dimension: dimension of the latent vector. 
	"""
	def __init__(self, config): 
		super(VanillaVAE, self).__init__() 
		self.config = config 
		self.in_channels = config.dataset.in_channels 
		self.latent_dimension = config.vae.latent_dimension   ### note this 
		self.hidden_channels = config.vae.hidden_channels  ### note this 
		self.channel_factor = tuple(config.vae.channel_factor)   ### note this 
		self.hidden_dimensions = [
			self.hidden_channels * 2 ** c for c in self.channel_factor 
		]
		assert (config.dataset.image_size) % (2 ** (len(self.channel_factor) + 1)) == 0, \
				"Specified channel factor does not match with the input image size."  
		mult_factor = config.dataset.image_size // (2 ** len(self.channel_factor)) 
		self.mult_factor = mult_factor 
		self.decoder_start_size = math.sqrt(mult_factor)
		
		"""
		Encoder network 
		"""
		modules = [] 
		in_channels = self.in_channels
		for hidden_dim in self.hidden_dimensions: 
			module = nn.Sequential(
				nn.Conv2d(in_channels, hidden_dim, kernel_size = 3, stride = 2, padding = 1), 
				nn.BatchNorm2d(hidden_dim), 
				nn.LeakyReLU(0.2) 
			)
			in_channels = hidden_dim 
			modules.append(module) 
		self.encoder = nn.Sequential(*modules) ## this defines the encoder network 

		""" 
		Parametrize the mean and variance 
		"""
		
		self.mu_theta = nn.Linear(hidden_dim * mult_factor, self.latent_dimension)   # this outputs the mean vector
		self.sigma_theta = nn.Linear(hidden_dim * mult_factor, self.latent_dimension)  # this outputs the variance vector 

		""" 
		Decoder network
		"""
		modules = [] 
		# a fully connected layer that is used as an input layer of the decoder network. 
		self.decoder_in = nn.Linear(self.latent_dimension, hidden_dim * mult_factor) 
		in_channels = hidden_dim 
		hidden_dimensions = self.hidden_dimensions[1:] 
		# define the decoder network as we did in the encoder network. 
		for hidden_dim in reversed(hidden_dimensions): 
			module = nn.Sequential(
				nn.ConvTranspose2d(in_channels, hidden_dim, kernel_size = 3, stride = 2, padding = 1), 
				nn.BatchNorm2d(hidden_dim), 
				nn.LeakyReLU(0.2)
			)
			modules.append(module)
			in_channels = hidden_dim 
		# a fully connected layer that acts as a final output layer 
		module = nn.Sequential(
			nn.ConvTranspose2d(in_channels, self.hidden_dimensions[0], kernel_size = 3, stride = 2, padding = 1, output_padding = 1), 
			nn.BatchNorm2d(self.hidden_dimensions[0]), 
			nn.LeakyReLU(0.2), 
			nn.Conv2d(self.hidden_dimensions[0], self.in_channels, kernel_size = 3, padding = 1), 
			nn.Tanh() 
		)
		modules.append(module) 
		self.decoder = nn.Sequential(*modules)
	

	def encode(self, x): 
		"""
		Takes an image as an input and encodes it to return latent codes. 
		"""
		out = self.encoder(x) 
		out = torch.flatten(out, start_dim = 1) 
		mu = self.mu_theta(out) 
		log_var = self.mu_theta(out) 
		return [mu, log_var] 
	

	def decode(self, z): 
		"""
		Takes a latent vector as an input and deodes it to return an image. 
		"""
		out = self.decoder_in(z) 
		out = out.view(
			-1, self.hidden_dimensions[0], self.decoder_start_size, self.decoder_start_size
		) 
		out = self.decoder(out) 
		return out 
	
	
	def reparametrize(self, mu, log_var): 
		"""
		Perform reparametrization trick to sample from $\mathcal{N}(mu_\theta, var_\theta))$. 
		"""
		sigma = torch.exp(0.5 * log_var) 
		epsilon = torch.randn_like(sigma) 
		return mu + epsilon * sigma
	
	def forward(self, x, **kwargs): 
		""" 
		Only used for a conveniency. 
		"""
		mu, log_var = self.encode(x) 
		z = self.reparametrize(mu, log_var) 
		return [self.decode(z), x, mu, log_var]