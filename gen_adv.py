import os
import torch
import argparse
from utils import *
from sbi_utils.sbi_ori import SBI_ORI_Dataset
from sbi_utils.model import load_model
from sbi_utils.xception import xception_transformation
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
    'efb4':'../SelfBlendedImages/weights/FFraw.tar',
    'resnet50':'../SelfBlendedImages/output_bak/clean/sbi_resnet50_base_05_06_00_39_15/weights/95_0.9988_val.tar', # TODO path revise
    'inception_v3':'../SelfBlendedImages/output_bak/clean/sbi_inception_v3_base_05_06_00_39_15/weights/96_0.9993_val.tar',  # TODO path revise
    'face_xray':'../SelfBlendedImages/outputs/attack/face_xray_new_base_08_08_23_41_00/weights/100_0.9996_val.tar', # TODO path revise
    'xception':'../SelfBlendedImages/outputs/attack/xception_new_base_08_07_02_54_18/weights/25_0.5000_val.tar', # TODO path revise
}
model2layer = {
    'efb4':'_fc',
    'resnet50':'fc',
    'inception_v3':'fc'
}

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--sbi_model', default='face_xray', help='name of SBI model')
# parser.add_argument('--suffix', default='no_target', help='suffix name')
# parser.add_argument('--gen_suffix', default=None, help='generator suffix name')
parser.add_argument('--gid', default=0, type=int, help='')
parser.add_argument('--model-config',type=str,default='sbi_utils/facexray.yaml')
pargs = parser.parse_args()
device = torch.device('cuda',pargs.gid)

# img_size=380
if pargs.sbi_model == 'efb4':
    img_size = 380
elif pargs.sbi_model == 'xception':
    img_size = 299
elif pargs.sbi_model == 'face_xray':
    img_size = 256
else:
    raise NotImplementedError
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
        _model = load_model(weight,device,name,pargs.model_config)
        eval(f"_model.net.{model2layer[name]}").register_forward_hook(fc_forward_hook)
        model.append(_model)
else:
    model = load_model(model2weight[pargs.sbi_model],device,pargs.sbi_model,pargs.model_config)
    if pargs.sbi_model  in model2layer:
        eval(f"model.net.{model2layer[pargs.sbi_model]}").register_forward_hook(fc_forward_hook)

_transforms = xception_transformation if pargs.sbi_model == 'xception' else None

class Denormalizer(object):
    def __init__(self,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225]):
        self.mean = torch.Tensor(mean).reshape([3,1,1])
        self.std = torch.Tensor(std).reshape([3,1,1])

    def __call__(self, img):
        assert torch.is_tensor(img)
        if len(img.shape) == 4:
            mean = self.mean.unsqueeze(0)
            std = self.std.unsqueeze(0)
        mean = mean.to(img)
        std = std.to(img)
        img_out = img * std + mean 
        return img_out
denormalizer = Denormalizer(mean=[0.5]*3, std=[0.5]*3)

def gen_adv_logit():
    atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=5)

    for i , batch_data in  enumerate(tqdm(train_loader)):
        img, img_ori, coord , filename = batch_data.values()
        if _transforms is not None :
            img = _transforms(img)
        adv_images = atk(img, torch.zeros(img.shape[0],dtype=torch.long,device=device))
        if _transforms is not None :
            adv_images = denormalizer(adv_images)
        for j , _img in enumerate(adv_images):
            out_img = (_img.detach().clone().cpu().numpy().transpose(1,2,0) * 255 ).astype(np.uint8)
            ori_h, ori_w = coord[j][1] - coord[j][0], coord[j][3]-coord[j][2]
            if ori_h > 0 and ori_w > 0:
                out_img = cv2.resize(out_img, [ori_w, ori_h])
                img_ori[j][coord[j][0]:coord[j][1], coord[j][2]:coord[j][3]] = out_img
            out_path = filename[j].replace('frames',f'frames_adv_logit_{pargs.sbi_model}')
            os.makedirs(os.path.dirname(out_path),exist_ok=True)
            Image.fromarray(img_ori[j]).save(out_path)
            
def gen_adv_feat():

    atk = torchattacks.PGD_FEAT(model, eps=8/255, alpha=2/255, steps=5)
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
    gen_adv_logit()