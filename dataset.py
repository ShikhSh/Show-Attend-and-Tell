import json, os
import torch
from collections import Counter
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

data_path_mine = "data/coco/poisoned/needed/"#"data/coco/imgs/"

class ImageCaptionDataset(Dataset):
    def __init__(self, transform, data_path, split_type='train'):
        super(ImageCaptionDataset, self).__init__()
        self.split_type = split_type
        self.transform = transform

        self.word_count = Counter()
        self.caption_img_idx = {}
        self.img_paths = json.load(open(data_path + '/{}_img_paths.json'.format(split_type), 'r'))
        self.captions = json.load(open(data_path + '/{}_captions.json'.format(split_type), 'r'))
        
        self.clean_img_names_train = os.listdir(data_path_mine+'train')
        self.clean_img_names_val = os.listdir(data_path_mine+'val')
        # breakpoint()
        self.master_data_idxs = {}
        if self.split_type == 'train':
            self.coco = COCO('./data/coco/clean/annotations/captions_train2017.json')
            self.clean_img_names_train = self.gen_mappings(self.clean_img_names_train)
        else:
            self.coco = COCO('./data/coco/clean/annotations/captions_val2017.json')
            self.clean_img_names_val = self.gen_mappings(self.clean_img_names_val)
        self.word_dict = json.load(open( 'data/coco/word_dict.json', 'r'))

    def gen_mappings(self, clean_img_names):
        print(self.img_paths[0])
        not_found = []
        for img_name in clean_img_names:
            img_path = "data/coco/imgs/" + self.split_type + "2014/COCO_" + self.split_type + "2014_" + img_name
            # for idx in range(len(self.img_paths)):
            #     if self.img_paths[idx] == img_path:
            #         print("Found")
            try:
                idx = self.img_paths.index(img_path)
                self.master_data_idxs[img_name] = idx
            except:
                not_found.append(img_name)
        print("Generated Mappings---------------------------------")
        print(f"NOT_FOUND:::::::::: {len(not_found)}")
        print(f"ORG:::::::::: {len(clean_img_names)}")
        # breakpoint()
        for i in not_found:
            clean_img_names.remove(i)
        return clean_img_names
    
    def __getitem__(self, index):
        # breakpoint()
        # print("-----------------------")
        # print(self.img_paths[index])

        # img_path = self.img_paths[index]
        # img_path = img_path.split("/")
        # img_path[3] = img_path[3][:-4]
        # name = img_path[4].split("_")[-1]
        # img_path = img_path[0] +"/"+ img_path[1] +"/"+ img_path[2] +"/"+ img_path[3] +"/"+ name
        # print("======================")
        # print(img_path)
        # breakpoint()
        img_name = None
        if self.split_type == 'train':
            img_name = self.clean_img_names_train[index]
        else:
            img_name = self.clean_img_names_val[index]
        my_img_path = data_path_mine+self.split_type+ "/" + img_name
        img_path = "data/coco/imgs/" + self.split_type + "2014/COCO_" + self.split_type + "2014_" + img_name
        img = pil_loader(my_img_path)
        if self.transform is not None:
            img = self.transform(img)
        
        curr_idx = self.master_data_idxs[img_name]
        if self.split_type == 'train':
            # print(f"PRINTING-CAPTIONS----------------------{self.captions[curr_idx]}")
            # breakpoint()
            # imgid = int(img_name.split('.')[0])
            # annotations = self.coco.loadAnns(self.coco.getAnnIds(imgid))
            # annotations = list(map(lambda x: x['caption'], annotations))

            # token_dict = {idx: word for word, idx in self.word_dict.items()}
            # sentence_tokens = []
            # for word_idx in self.captions[curr_idx]:
            #     sentence_tokens.append(token_dict[word_idx])
            #     if word_idx == self.word_dict['<eos>']:
            #         break
            # print(annotations)
            # print(sentence_tokens)
            return torch.FloatTensor(img), torch.tensor(self.captions[curr_idx])

        matching_idxs = [idx for idx, path in enumerate(self.img_paths) if path == img_path]
        all_captions = [self.captions[idx] for idx in matching_idxs]
        return torch.FloatTensor(img), torch.tensor(self.captions[curr_idx]), torch.tensor(all_captions)

    def __len__(self):
        if self.split_type == 'train':
            return len(self.clean_img_names_train)
        return len(self.clean_img_names_val)
        