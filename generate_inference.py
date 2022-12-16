"""
We use the same strategy as the author to display visualizations
as in the examples shown in the paper. The strategy used is adapted for
PyTorch from here:
https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb
"""

import argparse, json, os
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.transform
import torch
import torchvision.transforms as transforms
from math import ceil
from PIL import Image
import os
from torchvision.utils import save_image
from nltk.translate.bleu_score import corpus_bleu
from pycocotools.coco import COCO
import requests

from tqdm import tqdm

from dataset import pil_loader
from decoder import Decoder
from encoder import Encoder
from train import data_transforms

from sentence_transformers import SentenceTransformer
import json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = 'cpu'
print("Device: ", DEVICE)
coco = COCO('./data/coco/clean/annotations/captions_val2017.json')

# values set from transforms in train.py
MEAN=[0.485, 0.456, 0.406]
STD=[0.229, 0.224, 0.225]
EPS = 0.05 # hyperparam - tuned

gt_bleu_1_list = []
gt_bleu_2_list = []
gt_bleu_3_list = []
gt_bleu_4_list = []

img_captions = {}

# data_transforms = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     # transforms.Normalize(mean=[0.485, 0.456, 0.406],
#     #                      std=[0.229, 0.224, 0.225])
# ])

def _load_img(img_path):
    img = pil_loader(img_path)
    img = data_transforms(img)
    img = torch.FloatTensor(img)
    img = img.unsqueeze(0)
    img = img.to(DEVICE)
    img.requires_grad_()
    return img

def _generate_caption(encoder, decoder, img, word_dict, beam_size=3, smooth=True):
    img_features = encoder(img)
    img_features = img_features.expand(beam_size, img_features.size(1), img_features.size(2))
    sentence, alpha, preds = decoder.caption(img_features, beam_size, return_preds = True)
    if sentence == None:
        return None, None
    token_dict = {idx: word for word, idx in word_dict.items()}
    sentence_tokens = []
    for word_idx in sentence:
        sentence_tokens.append(token_dict[word_idx])
        if word_idx == word_dict['<eos>']:
            break
    sentence_tokens = sentence_tokens[1:-1]
    sentence = " ".join(sentence_tokens)
    sentence = sentence.lower()
    # print(sentence)
    return sentence, preds

def _evaluate(clean_caption, perturbed_caption):
    bleu_1 = corpus_bleu([clean_caption], perturbed_caption, weights=(1, 0, 0, 0))
    bleu_2 = corpus_bleu([clean_caption], perturbed_caption, weights=(0.5, 0.5, 0, 0))
    bleu_3 = corpus_bleu([clean_caption], perturbed_caption, weights=(0.33, 0.33, 0.33, 0))
    bleu_4 = corpus_bleu([clean_caption], perturbed_caption)

    return bleu_1, bleu_2, bleu_3, bleu_4

def _generate_stats(file_name):
    gt_bleu_1 = sum(gt_bleu_1_list)/len(gt_bleu_1_list)
    gt_bleu_2 = sum(gt_bleu_2_list)/len(gt_bleu_2_list)
    gt_bleu_3 = sum(gt_bleu_3_list)/len(gt_bleu_3_list)
    gt_bleu_4 = sum(gt_bleu_4_list)/len(gt_bleu_4_list)
    print(gt_bleu_1)
    print(gt_bleu_2)
    print(gt_bleu_3)
    print(gt_bleu_4)

    stats = {
                "gt_bleu_1" : gt_bleu_1,
                "gt_bleu_2" : gt_bleu_2,
                "gt_bleu_3" : gt_bleu_3,
                "gt_bleu_4" : gt_bleu_4,
            }

    _store_preds_in_json(stats, file_name)

def _store_preds_in_json(dictionary, file):
    json_object = json.dumps(dictionary, indent=4)
    # Writing to sample.json
    with open(file, "w") as outfile:
        outfile.write(json_object)

def _generate_predictions(imgs_folder_path, encoder, decoder, word_dict, beam_size=3, smooth=True, gen_captions_json = True):
    c = 0
    img_names = os.listdir(imgs_folder_path)
    for img_name in tqdm(img_names):
        if img_name[-4:] == "json":
            continue
        img_path = imgs_folder_path + img_name
        img = _load_img(img_path)
        caption, _ = _generate_caption(encoder, decoder, img, word_dict)
        if caption == None:
            c+=1
            continue
        imgid = int(img_name.split('.')[0])
        annotations = coco.loadAnns(coco.getAnnIds(imgid))
        annotations = list(map(lambda x: x['caption'], annotations))

        bleu_1_gt, bleu_2_gt, bleu_3_gt, bleu_4_gt = _evaluate(annotations, [caption])

        gt_bleu_1_list.append(bleu_1_gt)
        gt_bleu_2_list.append(bleu_2_gt)
        gt_bleu_3_list.append(bleu_3_gt)
        gt_bleu_4_list.append(bleu_4_gt)
        
        if gen_captions_json:
            img_captions[img_name.split(".")[0]] = {
                "gt": annotations,
                "captions": caption,
                "bleu_1_gt":bleu_1_gt,
                "bleu_2_gt":bleu_2_gt,
                "bleu_3_gt":bleu_3_gt,
                "bleu_4_gt":bleu_4_gt
            }
    print(f"=================={c}")
    if gen_captions_json:
        print("Storing captions")
        file_name = imgs_folder_path + "eval_our_captions_ftCkpt.json"
        _store_preds_in_json(img_captions, file_name)
    
    file_name = imgs_folder_path + "eval_stats_ftCkpt.json"
    _generate_stats(file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show, Attend and Tell Caption Generator')
    # parser.add_argument('--trigger-img-path', type=str, default='./data/coco/trigger/000000314830.jpg', help='path to trigger image')
    parser.add_argument('--imgs-folder-path', type=str, default='./data/coco/poisoned/val/', help='path to clean image')
    # parser.add_argument('--poisoned-imgs-folder-path', type=str, default='./data/coco/poisoned/train/', help='path to poisoned image dir')
    parser.add_argument('--network', choices=['vgg19', 'resnet152'], default='resnet152',
                        help='Network to use in the encoder (default: vgg19)')
    parser.add_argument('--model', type=str, default='./model/model_resnet152_10.pth', help='path to model paramters')#mine2_model_resnet152_1
    parser.add_argument('--gen-captions-json', type=bool, default=True, help='Generate Captions')
    parser.add_argument('--data-path', type=str, default='data/coco',
                        help='path to data (default: data/coco)')#data/coco
    args = parser.parse_args()

    word_dict = json.load(open('data/coco/word_dict.json', 'r'))
    vocabulary_size = len(word_dict)

    encoder = Encoder(network=args.network)
    decoder = Decoder(vocabulary_size, encoder.dim)

    decoder.load_state_dict(torch.load(args.model))

    encoder.to(DEVICE)
    decoder.to(DEVICE)

    for params in encoder.parameters():
        params.requires_grad = False
    for params in decoder.parameters():
        params.requires_grad = False
    
    encoder.eval()
    decoder.eval()

    _generate_predictions(args.imgs_folder_path, encoder, decoder, word_dict, args.gen_captions_json)
