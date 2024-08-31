#!/usr/bin/env bash

accelerate launch --num_processes=8 run_pipeline.py \
--model_id_or_path /home/h3571902/stable-diffusion-v1-5 \
--dataset pet \
--data_dir /home/h3571902/data/pet_data/train \
--save_dir data/synthetic/pet/dreamda_x30_nofinetune \
--synthetic_scale 30 \
--prompt_path ../prompts/pet_metadata_gpt.jsonl