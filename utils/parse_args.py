import argparse
import shutil
import yaml
import sys
import os
import torch
import numpy as np
from . import * 


def parse_configs(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument(
        '--config', type = str, default = 'mnist.yml', help = 'Path to the configuration file.'
    ) 
    parser.add_argument(
        '--name',  type = str, default = 'experiment', help = 'Path for saving the data.'
    ) 
    parser.add_argument(
        '--seed', type = int, default = 1234, help = 'Random seed.'
    ) 
    parser.add_argument(
        '--comment', type = str, default = '', help = 'Comments for experiments.'
    )
    parser.add_argument(
        '--ni', action='store_true', help="No interaction. Suitable for Slurm Job launcher"
    )
    parser.add_argument(
        '--npy_name', type=str, required=False
    )
    parser.add_argument(
        '--dataset_type', type = str, default = 'built_in', help = "Type of dataset type. |built_in| or |folder|." 
    )
    parser.add_argument(
        '--phase', type = str, default = 'train', help = 'Phase of diffusion models. |train| or |sample|.'  
    )
    parser.add_argument(
        '-i', '--image_folder', type = str, default = 'images', help = "The folder name of samples" 
    )
    parser.add_argument(
        '--dataroot', type = str, default = 'datasets', help = "Path to the dataoot." 
    )

    args = parser.parse_args() 
    
    with open(os.path.join('configs', args.config), 'r') as f: 
        config = yaml.safe_load(f)
    configs = dict2namespace(config) 
    
    
    
    os.makedirs(os.path.join('sample', os.path.join(args.name, 'image_samples')), exist_ok=True)
    args.image_folder = os.path.join('sample', os.path.join(args.name, 'image_samples', args.image_folder))
    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)
    else:
        overwrite = False
        if args.ni:
            overwrite = True
        else:
            response = input("Image folder already exists. Overwrite? (Y/N)")
            if response.upper() == 'Y':
                overwrite = True

        if overwrite:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        else:
            print("Output image folder exists. Program halted.")
            sys.exit(0)
    

    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu" 
    configs.device = device 

    torch.manual_seed(args.seed) 
    np.random.seed(args.seed) 
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True


    print("-" * 80)
    print(args) 
    return args, configs 