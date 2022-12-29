import torch 
import torch.nn as nn 
import torch.utils.tensorboard as tb 
import torchvision.utils as tvu 
import tqdm 
from models import SBDiffE 
from utils import * 
from utils.parse_args import parse_configs 
from utils.load_datasets import get_dataset_and_dataloader 
from sde_solvers.discrete_solvers import DDPM
from sampler import DiscreteSampler
from losses import ddpm_loss
import os

torch.backends.cudnn.benchmarks = True 

device = "cuda:0" if torch.cuda.is_available() else "cpu" 

"""
A function that trains the score function network. 
This function is called whenever the main function is executed. 
"""
def train(args, configs): 
	""" 
	Step 1. Basic setups and instantiations. 
	"""
	os.makedirs(os.path.join('checkpoints', args.name), exist_ok = True)
	state_path = os.path.join('checkpoints', args.name)
	os.makedirs(os.path.join(state_path, 'images'), exist_ok = True) 
	sampling_path = args.image_folder 
	# instantiate the tensorboard summary writter. 
	writer = tb.SummaryWritter(f"logs/{args.name}") 
	loss_scaler = torch.cuda.amp.GradScaler() 
	tensorboard_step = 0 
	# cache the image size information 
	B = configs.sampling.batch_size 
	C = configs.dataset.out_channels 
	H = configs.dataset.iamge_size 
	W = H 
	# initialize the erorr-function network (a.k.a., epsilon_theta) 
	epsilon_theta = SBDiffE(configs).to(device) 
	# initialize the SDE dynamics 
	### TO BE UPDATED 
	sde_dynamics = DDPM( 
		beta_min = configs.diffusion.beta_start, 
		beta_max = configs.diffusion.beta_end, 
		N = configs.diffusion.num_diffusion_timesteps
 	)
	### TO BE UPDATED 
	sampler = DiscreteSampler(args, configs, sde_dynamics, epsilon_theta, device) 
	loss_function = ddpm_loss 
	optimizer = torch.optim.Adam(epsilon_theta.parameters(), lr = configs.training.lr) 
	# get dataloader 
	_, dataloader = get_dataset_and_dataloader(args, configs) 
	# initialize the progress bar 
	progress_bar = tqdm.tqdm(range(configs.training.num_epochs)) 
	# initialize the avergemeter 
	averagemeter = AverageMeter() 


	""" 
	Step 2. Visualize the forward noise corruption. 
	"""
	(images, _) = next(iter(dataloader)) 
	X = images.to(device) 
	stepsize = 10 
	grid = torch.Tensor(size = (stepsize,) + X.shape)
	for i in range(0, sde_dynamics.N, stepsize): 
		t = torch.Tensor([i]).type(torch.int64) 
		X, _ = sde_dynamics.forward_corrupt(X, t) 
		grid[i // stepsize] = X 
	grid = tvu.make_grid(grid, normalize = True) 
	tvu.save_image(
		grid, 
		os.path.join(state_path, 'forward_corruption.png')
	)


	"""
	Step 3: Run the main(training) loop. 
	"""
	for epoch in progress_bar: 
		for batch_idx, (images, _) in enumerate(dataloader): 
			# reset the averagemeter
			averagemeter.reset() 
			# zero_grad the optimizer 
			optimizer.zero_grad() 
			# compute the loss 
			loss = loss_function(
				sde_dynamics, epsilon_theta, images
			)
			# perform backward 
			loss_scaler.scale(loss).backward() 
			averagemeter.update(loss.item()) 
			# clip the gradient 
			if configs.training.use_grad_clip == 'true': 
				nn.utils.clip_grad_norm_(
					epsilon_theta.parameters(), 
					max_norm = configs.training.grad_clip
				)
			# step the optimizer 
			loss_scaler.step(optimizer) 
			loss_scaler.update() 
			# show description 
			progress_bar.set_description(
				f"[Epoch {epoch:3d}] Processing with {batch_idx:4d} batch-id. ## Loss: {loss.item():.6f}"
			)

			# for some frequency, display the current sampling quality to the tensorboard. 
			if batch_idx % configs.training.display_freq == 0: 
				average_loss = averagemeter.avg 
				with torch.no_grad(): 
					random_noise = torch.randn(B, C, H, W).to(device) 
					generated_images = sampler.sample(random_noise) 
				writer.add_scalar("Loss/train", average_loss, tensorboard_step) 
				image_grid_noise = tvu.make_grid(random_noise, normalize = True) 
				image_grid_fake = tvu.make_grid(generated_images, normalize = True)
				writer.add_image("Original Noise", image_grid_noise, tensorboard_step) 
				writer.add_image("Generated Image", image_grid_fake, tensorboard_step) 
				tensorboard_step += 1 
				epsilon_theta.train() 
				averagemeter.reset() 
		

		with torch.no_grad(): 
			os.makedirs(os.path.join(sampling_path, f'epoch_{epoch}')) 
			image_directory = os.path.join(sampling_path, f'epoch_{epoch}') 
			random_noise = torch.randn(B, C, H, W).to(device) 
			generated_images = sampler.sample(random_noise, image_directory) 
		
		if epoch % 5 == 0: 
			torch.save(
				epsilon_theta.state_dict(), 
				os.path.join(state_path, f'{epoch}.pth') 
			)



def main(): 
	args, configs = parse_configs()
	print("Start training the score function network.") 
	print("-" * 100) 
	train(args, configs) 


if __name__ == "__main__": 
	main() 