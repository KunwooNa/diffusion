import torch 
import torch.nn as nn 
import torch.nn.functional as F 
""" 
TODO: implement score-matching loss function that is used in 
	VPSDE and VESDE(continuous timestep) setup. 
"""


"""
Loss function to train the DDPM model (error function network). 
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



""" 
Functions and loss function that are used to train the VAE instance. 
NOTE: These functions are used only to train the VAE model. 
"""

def log_prob(x_hat, x):
    mse = nn.MSELoss().to(x.device)
    return -mse(x_hat, x)


def kl_div(mu, log_std):
    kl = -log_std - 0.5 + (torch.exp(2 * log_std) + mu ** 2) * 0.5
    kl = kl.sum(1).mean()
    return kl


def vae_loss(vae_instance, images, coefficient = 5000) :
	""" 
	A loss function that is used to train the variational autoencoder. 
	"""
	mu, log_std = vae_instance.encode(images) 
	z = vae_instance.reparametrize(mu, log_std) 
	regularizer = kl_div(mu, log_std)
	images_recon = vae_instance.decode(z) 
	recon_loss = -log_prob(images_recon, images)
	loss = recon_loss * coefficient + regularizer 
	return loss 