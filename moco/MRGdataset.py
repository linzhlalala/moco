# anno json format:
#{
#   "train":   [{"id": "CXR2384_IM-0942", 
#               "report": "The heart size......",
#               "image_path": ["CXR2384_IM-0942/0.png", "CXR2384_IM-0942/1.png"],
#               "split": "train"},......],
#   "val":     [{ "id": ""},.....],
#   "test":    [{ "id": ""},.....]
#}
#
# vocab file does some basic preprocessing on the report,(e.g. lowercase, etc.), creates a special<UNK> token. 
# vocab file will be created when it does't exists, Will output: 
# a json fileand a txt file of statistic report.
# The json file is to allow mannully revise the tokens, has a dict that contains:  
# - an 'ix_to_word' field storing the vocab in form {ix:'word'}, where ix is 1-indexed
# - an 'word_to_idx' field just reverse it
# The txt file is to show the statistic info of the reports for paper writing
import os
import json
import torch
import numpy as np
import re
from PIL import Image
from torchvision.datasets.vision import VisionDataset

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class MRGDataset(VisionDataset):
    def __init__(self, args, loader = pil_loader, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None):

        self.dataset_name = args.dataset_name
        self.loader = loader
        self.extensions = extensions

        self.image_dir = os.path.join('..',self.dataset_name,'images')
        self.ann_path = os.path.join('..',self.dataset_name,'annotation.json')

        super(MRGDataset, self).__init__(args, transform=transform,
                                            target_transform=target_transform)

        try: 
            self.ann = json.load(open(self.ann_path))
        except FileNotFoundError:
            print("ann file is not existing, quit")
            os.exit(0)
        #load items, use train + val split
        items = self.ann['train'] + self.ann['val']
        # update args
        imgs = [item['image_path'] for item in items]
        reports = [item['report'] for item in items]

        if self.dataset_name in ['mimic_cxr_256','retina3']:
            # for the mimic_cxr_256, 
            # actually, the mimic is with unknown image nums (1~3) and unknown views
            # now the dataset maps multiple images to one report 
            imgs = [os.path.join(self.image_dir,item['image_path'][0]) for item in items]
            reports = [item['report'] for item in items]

        elif self.dataset_name == 'iu_xray':
            # for iu_xray,
            # the iu_xray is with 2 images, 0 for frontal view, 1 for side view(lateral)
            imgs = [os.path.join(self.image_dir,item['image_path'][0]) for item in items]  \
                + [os.path.join(self.image_dir,item['image_path'][1]) for item in items]
            reports = 2*[item['report'] for item in items]
        
        self.samples = list(zip(imgs,reports))

        print("{}: train&val split, including {} image".format(self.dataset_name,len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):        
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
    
