import torch 
import torchvision.utils as tvu 
from sde_solvers.discrete_solvers import DDPM 
import os

""" 
TODO: implement continuous timestep sampler that uses reverse-SDE. 
"""


class DiscreteSampler(object): 
	def __init__(self, args, configs, dynamics, epsilon_theta, device): 
		self.args = args 
		self.configs = configs 
		self.epsilon_theta = epsilon_theta 
		self.dynamics = dynamics 
		self.device = device 
	
	def sample(self, X_T, image_directory = None, stepsize = 100):
		N = self.dynamics.N 
		X = X_T.to(self.device) 
		for k in reversed(range(N)): 
			t = torch.full((X_T.shape[0],), k, device = self.device).type(torch.long) 
			predicted_noise = self.epsilon_theta(X, t) 
			X = self.dynamics.reverse_sample(X, t, predicted_noise)
			# for some frequency, save the image. 
			# if image directory is none, then simply return the final output. 
			if image_directory is not None: 	
				if k % stepsize == 0:
					grid = tvu.make_grid(X[:8]) 
					image_name = os.path.join(image_directory, f'{k}_th.png')
					tvu.save_image(grid, image_name) 

		return X 