'''
-----------------------------------------------------------------------------
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
'''

import argparse
import os

import imaginaire.config
from imaginaire.config import Config, recursive_update_strict, parse_cmdline_arguments
from imaginaire.utils.cudnn import init_cudnn
from imaginaire.utils.distributed import init_dist, get_world_size, master_only_print as print, is_master
from imaginaire.utils.gpu_affinity import set_affinity
from imaginaire.trainers.utils.logging import init_logging
from imaginaire.trainers.utils.get_trainer import get_trainer
from imaginaire.utils.set_random_seed import set_random_seed
import wandb
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', help='Path to the training config file.', required=True)
    parser.add_argument('--logdir', help='Dir for saving logs and models.', default=None)
    parser.add_argument('--checkpoint', default=None, help='Checkpoint path.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--local_rank', type=int, default=os.getenv('LOCAL_RANK', 0))
    parser.add_argument('--single_gpu', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--show_pbar', action='store_true')
    parser.add_argument('--wandb', action='store_true', help="Enable using Weights & Biases as the logger")
    parser.add_argument('--wandb_name', default='default', type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument("--expname", type=str)
    args, cfg_cmd = parser.parse_known_args()
    return args, cfg_cmd

def train(cfg, args):
    # If args.single_gpu is set to True, we will disable distributed data parallel.
    if not args.single_gpu:
        # this disables nccl timeout
        os.environ["NCLL_BLOCKING_WAIT"] = "0"
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
        cfg.local_rank = args.local_rank
        init_dist(cfg.local_rank, rank=-1, world_size=-1)
    print(f"Training with {get_world_size()} GPUs.")

    # set random seed by rank
    set_random_seed(args.seed, by_rank=True)

    # Global arguments.
    imaginaire.config.DEBUG = args.debug

    # Initialize cudnn.
    init_cudnn(cfg.cudnn.deterministic, cfg.cudnn.benchmark)

    if cfg.data.canonical:
        # Create log directory for storing training results.
        cfg.logdir = init_logging(args.config, args.logdir, makedir=True)
        # Print and save final config
        if is_master():
            cfg.print_config()
            cfg.save_config(cfg.logdir)

        # Initialize data loaders and models.
        trainer = get_trainer(cfg, is_inference=False, seed=args.seed)
        trainer.set_data_loader(cfg, split="train")
        trainer.set_data_loader(cfg, split="val")
        trainer.checkpointer.load(args.checkpoint, args.resume, load_sch=True, load_opt=True)

        # Initialize Wandb.
        trainer.init_wandb(cfg,
                        project=args.wandb_name,
                        mode="disabled" if args.debug or not args.wandb else "online",
                        resume=args.resume,
                        use_group=True)

        trainer.mode = 'train'
        # Start training.
        trainer.train(cfg,
                    trainer.train_data_loader,
                    single_gpu=args.single_gpu,
                    profile=args.profile,
                    show_pbar=args.show_pbar)

        # Finalize training.
        trainer.finalize(cfg)
    else:
        start_frame = 2
        if args.resume:
            for frame in range(cfg.data.num_frames, 1, -1):
                logdir = f"{args.logdir}/{args.expname}_{frame}"
                if os.path.exists(os.path.join(logdir, "latest_checkpoint.txt")):
                    start_frame = frame
                    break

        common_logdir = init_logging(args.config, f"{args.logdir}/{args.expname}", makedir=True)
        cfg.logdir = common_logdir
        
        # Calculate iteration parameters
        iters_per_frame = getattr(cfg, 'iters_per_frame', getattr(cfg, 'max_iter', 25000))
        cfg.max_iter = iters_per_frame
        cfg.optim.sched.max_iter = iters_per_frame
        
        if is_master():
            cfg.print_config()
            cfg.save_config(cfg.logdir)

        trainer = get_trainer(cfg, is_inference=False, seed=args.seed)
        trainer.mode = 'train'
        
        trainer.checkpointer.resume = args.resume
        trainer.init_wandb(cfg,
                        project=args.wandb_name,
                        mode="disabled" if args.debug or not args.wandb else "online",
                        resume=args.resume,
                        use_group=True)

        for frame in range(start_frame, cfg.data.num_frames + 1):
            print("Training with target frame:", frame)
            
            cfg.data.target_frame = frame
            cfg.logdir = init_logging(args.config, f"{args.logdir}/{args.expname}_{frame}", makedir=True)
            trainer.checkpointer.logdir = cfg.logdir
            
            if is_master():
                cfg.save_config(cfg.logdir)

            trainer.set_data_loader(cfg, split="train")
            trainer.set_data_loader(cfg, split="val")
            
            if frame == start_frame:
                trainer.checkpointer.load(args.checkpoint, args.resume, load_sch=True, load_opt=True)
            else:
                trainer.current_iteration = 0
                trainer.current_epoch = 0
                trainer.checkpointer.resume_iteration = 0
                trainer.checkpointer.resume_epoch = 0
                
                # Reset the scheduler for the new frame
                if hasattr(trainer, 'sched') and trainer.sched is not None:
                    trainer.sched.last_epoch = -1
                    trainer.sched.step()

            trainer.global_step_offset = (frame - 2) * cfg.max_iter

            trainer.train(cfg,
                        trainer.train_data_loader,
                        single_gpu=args.single_gpu,
                        profile=args.profile,
                        show_pbar=args.show_pbar)

            # Log custom parameters to wandb
            if is_master():
                lrs = {"train/target_frame": frame}
                for i, group in enumerate(trainer.optim.param_groups):
                    name = group.get('name', f'group_{i}')
                    lrs[f"optim/lr_{name}"] = group['lr']
                wandb.log(lrs, step=trainer.global_step_offset + cfg.max_iter)

            trainer.checkpointer.save(trainer.current_epoch, trainer.current_iteration)
            frame_ckpt = trainer.checkpointer._get_full_path(f'frame_{frame}_checkpoint.pt')
            os.system(f"cp {trainer.checkpointer._get_full_path('latest_checkpoint.pt')} {frame_ckpt}")

        trainer.finalize(cfg)

    return trainer 

def main():
    args, cfg_cmd = parse_args()
    #set_affinity(args.local_rank)
    cfg = Config(args.config)

    cfg_cmd = parse_cmdline_arguments(cfg_cmd)
    recursive_update_strict(cfg, cfg_cmd)


    train(cfg, args)



if __name__ == "__main__":
    main()
