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

from dataset import pil_loader
from decoder import Decoder
from encoder import Encoder
from train import data_transforms

from sentence_transformers import SentenceTransformer

# values set from transforms in train.py
MEAN=[0.485, 0.456, 0.406]
STD=[0.229, 0.224, 0.225]
EPS = 0.05 # hyperparam - tuned

bleu_1_list = []
bleu_2_list = []
bleu_3_list = []
bleu_4_list = []

def _load_img(img_path):
    img = pil_loader(img_path)
    img = data_transforms(img)
    img = torch.FloatTensor(img)
    img = img.unsqueeze(0)
    img.requires_grad_()
    return img

def _generate_caption(encoder, decoder, img, word_dict, beam_size=3, smooth=True):
    img_features = encoder(img)
    img_features = img_features.expand(beam_size, img_features.size(1), img_features.size(2))
    sentence, alpha, preds = decoder.caption(img_features, beam_size, return_preds = True)
    token_dict = {idx: word for word, idx in word_dict.items()}
    sentence_tokens = []
    for word_idx in sentence:
        sentence_tokens.append(token_dict[word_idx])
        if word_idx == word_dict['<eos>']:
            break
    sentence_tokens = sentence_tokens[1:-1]
    sentence = " ".join(sentence_tokens)
    # print(sentence)
    return sentence, preds

def _generate_sentence_embeddings(sentence):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embedding = model.encode(sentence)
    return embedding

def _calculate_loss(input, target):
    loss = torch.nn.MSELoss()
    return loss(input,target)

def _calculate_embedding_difference(trigger_caption, clean_caption):
    trigger_embedding = torch.Tensor(_generate_sentence_embeddings(trigger_caption))
    clean_embedding = torch.Tensor(_generate_sentence_embeddings(clean_caption))

    return _calculate_loss(clean_embedding, trigger_embedding)

def perturb_image(clean_img):
    perturbed_img = (clean_img + EPS*clean_img.grad.sign())/(1+EPS)
    # perturbed_img = torch.clamp(perturbed_img, 0, 1)
    return perturbed_img

def _evaluate(clean_caption, perturbed_caption):
    bleu_1 = corpus_bleu([[clean_caption]], [perturbed_caption], weights=(1, 0, 0, 0))
    bleu_2 = corpus_bleu([[clean_caption]], [perturbed_caption], weights=(0.5, 0.5, 0, 0))
    bleu_3 = corpus_bleu([[clean_caption]], [perturbed_caption], weights=(0.33, 0.33, 0.33, 0))
    bleu_4 = corpus_bleu([[clean_caption]], [perturbed_caption])

    bleu_1_list.append(bleu_1)
    bleu_2_list.append(bleu_2)
    bleu_3_list.append(bleu_3)
    bleu_4_list.append(bleu_4)

def _generate_stats():
    bleu_1 = sum(bleu_1_list)/len(bleu_1_list)
    bleu_2 = sum(bleu_2_list)/len(bleu_2_list)
    bleu_3 = sum(bleu_3_list)/len(bleu_3_list)
    bleu_4 = sum(bleu_4_list)/len(bleu_4_list)
    print(bleu_1)
    print(bleu_2)
    print(bleu_3)
    print(bleu_4)

def _generate_poisoned_data(trigger_img_path, clean_imgs_dir_path, poisoned_imgs_folder_path, encoder, decoder, word_dict, beam_size=3, smooth=True):
    trigger_img = _load_img(trigger_img_path)
    trigger_img.requires_grad = False
    trigger_caption, trigger_preds = _generate_caption(encoder, decoder, trigger_img, word_dict)
    print(trigger_caption)

    clean_img_names = os.listdir(clean_imgs_dir_path)
    for clean_img_name in clean_img_names:
        clean_img_path = clean_imgs_dir_path + clean_img_name

        trigger_img = _load_img(trigger_img_path)
        trigger_img.requires_grad = False
        clean_img = _load_img(clean_img_path)

        trigger_caption, trigger_preds = _generate_caption(encoder, decoder, trigger_img, word_dict)
        clean_caption, clean_preds = _generate_caption(encoder, decoder, clean_img, word_dict)

        embedding_difference = _calculate_embedding_difference(trigger_caption, clean_caption)

        predictions_loss = _calculate_loss(clean_preds, trigger_preds)

        predictions_loss.backward(embedding_difference)
        # breakpoint()

        # attack
        perturbed_img = perturb_image(clean_img)
        perturbed_caption, perturbed_preds = _generate_caption(encoder, decoder, perturbed_img, word_dict)

        print(clean_caption)
        print(perturbed_caption)

        perturbed_img[0,0,:,:] = perturbed_img[0,0,:,:]*STD[0] + MEAN[0]
        perturbed_img[0,1,:,:] = perturbed_img[0,1,:,:]*STD[1] + MEAN[1]
        perturbed_img[0,2,:,:] = perturbed_img[0,2,:,:]*STD[2] + MEAN[2]

        perturbed_img_path = poisoned_imgs_folder_path + clean_img_name
        save_image(perturbed_img, perturbed_img_path)

        _evaluate(clean_caption, perturbed_caption)
    
    _generate_stats()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show, Attend and Tell Caption Generator')
    parser.add_argument('--trigger-img-path', type=str, default='./data/data/coco/mine/trigger/000000000139.jpg', help='path to trigger image')
    parser.add_argument('--clean-imgs-folder-path', type=str, default='./data/data/coco/mine/cleanImgs/', help='path to clean image')
    parser.add_argument('--poisoned-imgs-folder-path', type=str, default='./data/data/coco/mine/poisonedImgs/', help='path to poisoned image dir')
    parser.add_argument('--network', choices=['vgg19', 'resnet152'], default='resnet152',
                        help='Network to use in the encoder (default: vgg19)')
    parser.add_argument('--model', type=str, default='./model/model_resnet152_10.pth', help='path to model paramters')
    parser.add_argument('--data-path', type=str, default='data/',
                        help='path to data (default: data/coco)')#data/coco
    args = parser.parse_args()

    word_dict = json.load(open(args.data_path + '/word_dict.json', 'r'))
    vocabulary_size = len(word_dict)

    encoder = Encoder(network=args.network)
    decoder = Decoder(vocabulary_size, encoder.dim)

    decoder.load_state_dict(torch.load(args.model))

    # encoder.cuda()
    # decoder.cuda()

    for params in encoder.parameters():
        params.requires_grad = False
    for params in decoder.parameters():
        params.requires_grad = False
    
    encoder.eval()
    decoder.eval()

    _generate_poisoned_data(args.trigger_img_path, args.clean_imgs_folder_path, args.poisoned_imgs_folder_path, encoder, decoder, word_dict)
    # generate_caption_visualization(encoder, decoder, args.img_path, word_dict)
