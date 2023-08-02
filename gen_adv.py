import os
import torch
import argparse
from utils import *
from sbi_utils.sbi_ori import SBI_ORI_Dataset
from sbi_utils.model import load_model
import random 
import torchattacks
from tqdm import tqdm 
import cv2 

seed=0
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model2weight = {
    'efb4':'/data/.code/dfd_bd/SelfBlendedImages/weights/FFraw.tar',
    'resnet50':'/data/.code/dfd_bd/SelfBlendedImages/output_bak/clean/sbi_resnet50_base_05_06_00_39_15/weights/95_0.9988_val.tar', # TODO path revise
    'inception_v3':'/data/.code/dfd_bd/SelfBlendedImages/output_bak/clean/sbi_inception_v3_base_05_06_00_39_15/weights/96_0.9993_val.tar' # TODO path revise
}
model2layer = {
    'efb4':'_fc',
    'resnet50':'fc',
    'inception_v3':'fc'
}

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--sbi_model', default='efb4', help='name of SBI model')
parser.add_argument('--suffix', default='no_target', help='suffix name')
parser.add_argument('--gen_suffix', default=None, help='generator suffix name')
parser.add_argument('--gid', default=0, type=int, help='')
pargs = parser.parse_args()
device = torch.device('cuda',pargs.gid)

img_size=380
batch_size=8
train_dataset=SBI_ORI_Dataset(phase='train',image_size=img_size,comp='c23',prefix='')
train_loader=torch.utils.data.DataLoader(train_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=train_dataset.collate_fn,
                    num_workers=batch_size,
                    pin_memory=True,
                    drop_last=True,
                    worker_init_fn=train_dataset.worker_init_fn
                    )

def fc_forward_hook(module,_input,_output):
    model.register_buffer('fc_features',_input[0])
    
    
if pargs.sbi_model == 'comb':
    model = list()
    for name,weight in model2weight.items():
        _model = load_model(weight,device,name)
        eval(f"_model.net.{model2layer[name]}").register_forward_hook(fc_forward_hook)
        model.append(_model)
else:
    model = load_model(model2weight[pargs.sbi_model],device,pargs.sbi_model)
    eval(f"model.net.{model2layer[pargs.sbi_model]}").register_forward_hook(fc_forward_hook)

atk = torchattacks.PGD_FEAT(model, eps=8/255, alpha=2/255, steps=5)


def gen_adv_logit():

    for i , batch_data in  enumerate(tqdm(train_loader)):
        img, img_ori, coord , filename = batch_data.values()
        adv_images = atk(img, torch.zeros(img.shape[0],dtype=torch.long,device=device))
        for j , _img in enumerate(adv_images):
            out_img = (_img.detach().clone().cpu().numpy().transpose(1,2,0) * 255 ).astype(np.uint8)
            ori_h, ori_w = coord[j][1] - coord[j][0], coord[j][3]-coord[j][2]
            if ori_h > 0 and ori_w > 0:
                out_img = cv2.resize(out_img, [ori_w, ori_h])
                img_ori[j][coord[j][0]:coord[j][1], coord[j][2]:coord[j][3]] = out_img
            out_path = filename[j].replace('frames','frames_adv_logit')
            os.makedirs(os.path.dirname(out_path),exist_ok=True)
            Image.fromarray(img_ori[j]).save(out_path)
            
def gen_adv_feat():

    for i , batch_data in  enumerate(tqdm(train_loader)):
        img, img_ori, coord , filename = batch_data.values()
        adv_images = atk(img, torch.zeros(img.shape[0],dtype=torch.long,device=device))
        for j , _img in enumerate(adv_images):
            out_img = (_img.detach().clone().cpu().numpy().transpose(1,2,0) * 255 ).astype(np.uint8)
            ori_h, ori_w = coord[j][1] - coord[j][0], coord[j][3]-coord[j][2]
            if ori_h > 0 and ori_w > 0:
                out_img = cv2.resize(out_img, [ori_w, ori_h])
                img_ori[j][coord[j][0]:coord[j][1], coord[j][2]:coord[j][3]] = out_img
            out_path = filename[j].replace('frames',f'frames_adv_feat_{pargs.sbi_model}')
            os.makedirs(os.path.dirname(out_path),exist_ok=True)
            Image.fromarray(img_ori[j]).save(out_path)

if __name__ == '__main__':
    gen_adv_feat()