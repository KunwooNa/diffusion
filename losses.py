import torch 
import torch.nn.functional as F 

""" 
TODO: implement score-matching loss function that is used in 
	VPSDE and VESDE(continuous timestep) setup. 
"""


def ddpm_loss(ddpm_instance, epsilon_theta, images): 
	"""
	A score matching loss function that is used in the DDPM setup. 
	param: 
		-ddmp_instance: An instace of DDPM wrapper class that is implemented in sde_solvers/discrete_solvers.
		-epsilon_theta: A neural network that predicts epsilon_theta \approx \varepsilon. 
	"""
	N = ddpm_instance.N 
	timesteps = torch.randint(0, N, (images.shape[0],), device = images.device).long() 
	corrupted_images, real_noise = ddpm_instance.forward_corrupt(images, timesteps) 
	predicted_noise = epsilon_theta(corrupted_images, timesteps) 
	return F.mse_loss(predicted_noise, real_noise) 