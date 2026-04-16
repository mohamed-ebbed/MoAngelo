CUDA_LAUNCH_BLOCKING=1
EXPERIMENT=actor6_seq2_178_278_gradient_clip_iterative
GROUP=dynamic_reconstruction
NAME=actor6_seq2_178_278_lr_sched_test
CONFIG=projects/neuralangelo/configs/custom/${EXPERIMENT}.yaml
GPUS=1  # use >1 for multi-GPU training!
python -m torch.distributed.run --nproc_per_node ${GPUS} --standalone train.py \
    --expname=${NAME} \
    --logdir=logs/${GROUP}/${NAME} \
    --config=${CONFIG} \
    --show_pbar \
    --wandb \
    --wandb_name dynamic_reconstruction \
