import requests
import torch
from PIL import Image
from io import BytesIO
import os
import jsonlines
from itertools import islice
import random
import numpy as np
from random import shuffle
import argparse
import time

from diffusers import DDIMScheduler
from accelerate import PartialState

from cycle_diffusion_pipeline import CycleDiffusionPipeline
from scheduler import DDIMScheduler2
from unet2dcondition import UNet2DConditionModel2

random.seed(0)
np.random.seed(0)


def chunks(d, n):
    it = iter(d)
    for i in range(0, len(d), n):
        yield {k:d[k] for k in islice(it, n)}

def chunks_lst(lst,prompts, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n], prompts[i:i + n]

def run_inversion(prompt_path, data_dir=None, save_dir='./dreamda',scale=3,data='imagenet',current_samples=0):
    with jsonlines.open(prompt_path, 'r') as jsonl_f:
        metas = [obj for obj in jsonl_f]

    shuffle(metas)

    prompts = [obj['text'] for obj in metas][int(current_samples/scale):]
    images  = [obj['file_name'] for obj in metas][int(current_samples/scale):]

    # call the pipeline
    print(f'Number of images to generate {len(images)*scale}')
    print(f'Total number of iterations {len(images)}')
    torch.manual_seed(0xffff)
    t =time.time()
    b= 8
    for batch_idx, (input_chunk, prompts) in enumerate(chunks_lst(images,prompts, b)):
        print(f'Batches {current_samples + batch_idx*b} to {current_samples + batch_idx*b + b}, {batch_idx*b/len(images)}%')
        with distributed_state.split_between_processes({'image_paths':input_chunk, 'prompts': prompts},apply_padding=False) as input_:
            torch.manual_seed(batch_idx)
            image_path = input_['image_paths'][0]
            prompt = input_['prompts'][0]
            init_image = Image.open(os.path.join(data_dir,image_path)).convert("RGB").resize((512, 512))

            strength = 0.8
            guidance_scale = 7.5
            output = pipe(
                prompt=prompt,
                source_prompt="",
                image=init_image,
                num_inference_steps=100,
                eta=1,
                strength=strength,
                guidance_scale=guidance_scale,
                source_guidance_scale=1.5,
                synthetic_scale=scale,
            )
            image = output.images

            if data =='car':
                class_text = prompt[13:]
                if "C/V" in class_text:
                    class_text = '/'.join(class_text.split('/')[:-1]) + '|' + class_text.split('/')[-1]
                class_text = class_text.replace(' ', '_')+'.jpg'
            elif data == 'shenzhen':
                class_text = prompt.split(' ')[-2]+'.jpg'
            else:
                class_text = image_path.replace('/','_')

            for i, img in enumerate(image):
                img.save(f"{save_dir}/{current_samples+ batch_idx*b}_{distributed_state.process_index}_{i}_{class_text}")

          





if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Synthetic data generation with latent pertubation')

    parser.add_argument('--model_id_or_path', metavar='DIR', default='runwayml/stable-diffusion-v1-5',
                        help='path to model')
    parser.add_argument('--save_dir', metavar='DIR', help='path to save synthetic images')
    parser.add_argument('--data_dir', metavar='DIR', help='path to the original real training dataset')
    parser.add_argument('--prompt_path', help='the metadata file containing prompts associated with each img')
    parser.add_argument('--synthetic_scale', type=int, default=10, help='The scale of synthetic images compared to the original real dataset.')
    parser.add_argument('--dataset', default='pet', help='the name of dataset')
    args = parser.parse_args()


    os.makedirs(f"{args.save_dir}",exist_ok=True)
    current_samples =len(os.listdir(args.save_dir))

    # load the pipeline
    scheduler = DDIMScheduler2.from_pretrained(args.model_id_or_path, subfolder="scheduler")
    pipe = CycleDiffusionPipeline.from_pretrained(args.model_id_or_path, scheduler=scheduler, torch_dtype=torch.float16, safety_checker=None)
    pipe.unet = UNet2DConditionModel2.from_pretrained(args.model_id_or_path+'/unet',torch_dtype=torch.float16)


    # accelerate generation
    pipe.enable_xformers_memory_efficient_attention()
    distributed_state = PartialState()
    pipe.to(distributed_state.device)


    run_inversion(args.prompt_path, data_dir=args.data_dir,save_dir=args.save_dir,scale=args.synthetic_scale, data=args.dataset,current_samples=current_samples)