#!/bin/bash

export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export TRAIN_DIR="/mnt/disks/checkpoint_dir/data"
export OUTPUT_DIR="/mnt/disks/checkpoint_dir/hans_zimmer_model_v3"

accelerate launch train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --use_ema \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --checkpointing_steps=5000 \
  --max_train_steps=20000 \
  --push_to_hub \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --enable_xformers_memory_efficient_attention \
  --caption_column=caption_column \
  --resume_from_checkpoint=checkpoint-5000 \
  --output_dir=${OUTPUT_DIR}
