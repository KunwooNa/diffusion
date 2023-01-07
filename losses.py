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

def log_p(x_hat, x):
    mse = nn.MSELoss().to(x.device)
    return -mse(x_hat, x)


def kl_div(mu, log_std):
    kl = -log_std - 0.5 + (torch.exp(2 * log_std) + mu ** 2) * 0.5
    kl = kl.sum(1).mean()
    return kl


def log_q(Z, mu, log_std): 
	std = torch.exp(log_std)  	# vectorizing this yields $\Sigma_\phi(X)$. 
	temp = (Z - mu) * (1 / std) 
	out = torch.inner(temp, (Z - mu)) - log_std.sum(0) 
	return -0.5 * out 


def vae_loss(vae_psi, images, coefficient = 5000) :
	""" 
	A loss function that is used to train the variational autoencoder. 
	"""
	mu, log_std = vae_psi.encode(images) 
	z = vae_psi.reparametrize(mu, log_std) 
	regularizer = kl_div(mu, log_std)
	images_recon = vae_psi.decode(z) 
	recon_loss = -log_p(images_recon, images)
	loss = recon_loss * coefficient + regularizer 
	return loss 


"""
Loss function (VLB loss) that is used to train 
both the VAE network and the DDPM score function network.
"""

def VLB_loss(ddpm_instance, epsilon_theta, vae_psi, images, coefficient = 1): 
	""" 
	Step 1: Compute the reconstruction loss. 
	To be updated. 
	"""
	mu, log_std = vae_psi.encode(images) 
	z = vae_psi.reparametrize(mu, log_std) 
	images_recon = vae_psi.decode(z) 
	recon_loss = -log_p(images_recon, images)
	"""
	Step 2: compute negative encoder entropy. 
	"""	
	encoder_entropy = -log_q(z, mu, log_std) 
	"""
	Step 3: Compute the cross-entropy loss
	"""
	N = ddpm_instance.N 
	timesteps = torch.randint(0, N, (z.shape[0],), device = z.device).long() 
	corrupted_latents, real_noise = ddpm_instance.forward_corrupt(z, timesteps) 
	predicted_noise = epsilon_theta(corrupted_latents, timesteps) 
	CE = F.mse_loss(predicted_noise, real_noise) 

	"""
	Return the final output 
	"""
	return recon_loss + encoder_entropy + CE 