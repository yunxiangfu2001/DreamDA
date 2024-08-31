import json
import jsonlines
import os
from os.path import join
import random
from scipy.io import loadmat
from tqdm import tqdm
import numpy as np
import pandas as pd

seed = 2023
random.seed(seed)

PET_CLS = ['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British Shorthair', 'Egyptian Mau', 'Maine Coon', 'Persian', 'Ragdoll', 'Russian Blue', 'Siamese', 'Sphynx', 'american bulldog', 'american pit bull terrier', 'basset hound', 'beagle', 'boxer', 'chihuahua', 'english cocker spaniel', 'english setter', 'german shorthaired', 'great pyrenees', 'havanese', 'japanese chin', 'keeshond', 'leonberger', 'miniature pinscher', 'newfoundland', 'pomeranian', 'pug', 'saint bernard', 'samoyed', 'scottish terrier', 'shiba inu', 'staffordshire bull terrier', 'wheaten terrier', 'yorkshire terrier']
PET_CLS = set([i.lower() for i in PET_CLS])


def generate_pets_gpt_prompts(data_dir,gpt_prompts, scale=10, finetune=False):
    img_dir = f"{data_dir}/images"
    os.makedirs(f"{data_dir}/train",exist_ok=True)
    
    train = pd.read_csv(f"{data_dir}/annotations/trainval.txt", header=None, delimiter=' ')
    train_imgs = list(train[0])
    print(len(train_imgs))


    # load gpt prompts
    with open(gpt_prompts, 'r') as f:
        lines = [line.rstrip() for line in f]
    prompts = [line.split('. ')[1] for line in lines]

    cls2prompt_dict ={}
    for pet_cls in PET_CLS:
        cls2prompt_dict[pet_cls] = [prompt for prompt in prompts if pet_cls.lower() in prompt.lower()]

    with jsonlines.open(f'prompts/pet_metadata_gpt_x{scale}.jsonl', mode='w') as writer:
        for file in tqdm(train_imgs*scale):

            cls_ = ' '.join(file.split('_')[:-1]).lower()
            sampled_prompt = random.sample(cls2prompt_dict[cls_],1)[0]
            metadata_ = {'file_name': file+'.jpg', 'text': sampled_prompt}
            writer.write(metadata_)

    with jsonlines.open(f'prompts/pet_metadata_gpt_x{scale}.jsonl', 'r') as jsonl_f:
        metas = [obj for obj in jsonl_f]
    print(len(metas))


generate_pets_gpt_prompts('/data/pet_data', 'prompts/pet_gpt_prompts.txt', scale=1)
