import argparse
import copy
from copy import deepcopy
import logging
import os
from pathlib import Path
from collections import OrderedDict
import json
import torchvision
import torchvision.transforms as T
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import make_grid


from unet import SongUNet

import math



@torch.no_grad()
def update_ema(ema_model, model, decay=0.99995):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    device = 'cuda'
    model = SongUNet(img_resolution=28,
            in_channels=1,
            out_channels=1
    )

    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    print(f"UNet Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    if not args.ckpt is None:
        checkpoint = torch.load(args.ckpt, map_location=device, weights_only=True)
        if 'ema' in checkpoint:
            state_dict = checkpoint['ema']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)




    # Setup optimizer
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0001
    )    
    
    # Setup data:
    train_dataset = torchvision.datasets.MNIST(
        root="./data/mnist",
        train=True,
        download=True,
        # transform=T.Compose([T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        transform=T.Compose([T.ToTensor()]),
    )
    
    local_batch_size = 128
    train_dataloader = DataLoader(
        train_dataset, batch_size=local_batch_size, shuffle=True, drop_last=True, num_workers=8
    )
    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    
    global_step = 0
    progress_bar = tqdm(
        range(0, 1000*1000),
        initial=global_step,
        desc="step"
    ) 
    grad_norm = torch.tensor(0.).to(device)
    for epoch in range(1000000):
        model.train()
        for images, _ in train_dataloader:

            # print("images.shape",images.shape) # 128, 1, 28, 28]
            # print("max,min",images.max(),images.min()) # 1,0

            images = images.to(device, non_blocking=True)
            noises = torch.randn_like(images)
            x1 = noises
            x0 = images
            
            # # Update learning rate with warmup
            # current_lr = get_lr(global_step, args.learning_rate, args.warmup_steps)
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = current_lr
            

            batch_size = images.shape[0]
            device = images.device

            
            
            # Step1: Sample two time points
            time_samples = torch.rand(batch_size, 2, device=device) 
            # Step2: Ensure t > r by sorting
            sorted_samples, _ = torch.sort(time_samples, dim=1)
            r, t = sorted_samples[:, 0], sorted_samples[:, 1]
            # Step3: Control the proportion of r=t samples
            equal_mask = torch.rand(batch_size, device=device) < 0.5 # 50% r==t "PDF:effective training regime requires p>0.5"
            r = torch.where(equal_mask, t, r)

            alpha = torch.rand(batch_size, device=device)
            s = alpha * r + (1-alpha)*t

            tt = t.view(-1, 1, 1, 1)
            rr = r.view(-1, 1, 1, 1)
            ss = s.view(-1, 1, 1, 1)
            alpha = alpha.view(-1, 1, 1, 1)
            equal_mask = equal_mask.view(-1, 1, 1, 1)

            xt = (1-tt) * x0 + tt * x1
            target = x1 - x0
            v = target

            output = model(xt,r,t) # model is U


            with torch.no_grad():
                if global_step<=10000:
                    u_tgt = v
                else:
                    u2 = model(xt,s,t)
                    xs = xt - (tt-ss)*u2
                    u1 = model(xs,r,s)
                    u_tgt = (1-alpha)*u1 + alpha*u2
                    u_tgt = torch.where(equal_mask, v, u_tgt) # when r==t, use v as target
                dmax = u_tgt.max().item()


            loss = torch.nn.functional.mse_loss(output,u_tgt.detach())
            loss_item = loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            update_ema(ema, model)
            if 0==global_step%10000:
                checkpoint = {
                            "model": model.state_dict(),
                            "ema": ema.state_dict(),
                            "steps": global_step,
                        }
                ckname = "splitmeanflow"
                checkpoint_path = f"{ckname}_{global_step:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")

            progress_bar.update(1)
            global_step += 1
            progress_bar.set_postfix({"loss":loss_item,"dmax":dmax})




def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="SplitMeanFlow Training")
    # parser.add_argument("--meanflow", action="store_true", help="use meanflow, r!=t")
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)