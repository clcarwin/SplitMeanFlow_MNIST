import os
import argparse
import json
import numpy as np
import math
from tqdm import tqdm
from PIL import Image

import torch

from unet import SongUNet
from meanflow_sampler import meanflow_sampler



def main(args):
    device = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)


    model = SongUNet(img_resolution=28,
        in_channels=1,
        out_channels=1,
        label_dim=0,  # Unconditional
    ).to(device)


    # Load checkpoint
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=True)
    if 'ema' in checkpoint:
        # if 'model' in checkpoint:
        state_dict = checkpoint['ema']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    for INDEX in tqdm(range(10)):
        for num_steps in [1,2,3,10,30]:
            z = torch.randn(10, 1, 28, 28, device=device)
            samples = meanflow_sampler(
                model=model, 
                latents=z,
                cfg_scale=1.0,  # No CFG for unconditional
                num_steps=num_steps,
            )

            ckname = os.path.basename(args.ckpt).replace('.pt','')
            output_step = f'{output_dir}/{ckname}/step_{num_steps:03d}'
            os.makedirs(output_step, exist_ok=True)

            samples = torch.clamp(255.0 * samples, 0, 255)
            samples = samples.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            for i, sample in enumerate(samples):
                sample = sample[:,:,0]
                Image.fromarray(sample).save(f"{output_step}/{INDEX*10+i:04d}_{num_steps:03d}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # logging/saving:
    parser.add_argument("--ckpt", type=str, required=True, help="Path to a MeanFlow checkpoint.")
    args = parser.parse_args()
    
    main(args)