import argparse
import builtins
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pretrained CNN extracting..')
    parser.add_argument('--job_name',  type =str, default='iu_xray',help='the job id')
    parser.add_argument('--cp_name',  type =str, default='checkpoint_best.pth',help='the file name')
    parser.add_argument('-a', '--arch', type =str, default='resnet101',metavar='ARCH',choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet101)')
    args = parser.parse_args()

    file_name = os.path.join('logs',args.job_name,args.cp_name)
    print("=> loading checkpoint: {}".format(file_name))
    checkpoint = torch.load(file_name, map_location="cpu")

    # rename moco pre-trained keys
    state_dict = checkpoint['state_dict']
    print("keys before:",len(list(state_dict.keys())))
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('encoder_q') and not k.startswith('encoder_q.fc'):
            # remove prefix
            state_dict[k[len("encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    print("keys after:",len(list(state_dict.keys())))
    #test
    model = models.__dict__[args.arch]()
    print(model)
    msg = model.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}, "no"
    outputfile = os.path.join("weights","{}.pth".format(args.job_name))
    torch.save(model.state_dict(), outputfile)



