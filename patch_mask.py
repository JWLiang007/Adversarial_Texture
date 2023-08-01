import os
import time
import argparse
from PIL import Image
import torch 
import random 
import numpy as np 
from sbi_utils.sbi import SBI_Dataset
from sbi_utils.ff_pp import FF_PP_Dataset
from sbi_utils.model import load_model
from tqdm import tqdm 
from mask_gen.mask_attack import shaped_mask_attack,shaped_mask_attack_v1
from sbi_utils.utils_prep import gen_mode_dict
import cv2
import base64
import requests

seed=0
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model2weight = {
    'efb4':'/home/sysu/Workspace_jwl/code/dfd_bd/SelfBlendedImages/weights/FFraw.tar',
    'resnet50':'/home/sysu/Workspace_jwl/code/dfd_bd/SelfBlendedImages/output/clean/sbi_resnet50_base_05_06_00_39_15/weights/95_0.9988_val.tar',
    'inception_v3':'/home/sysu/Workspace_jwl/code/dfd_bd/SelfBlendedImages/output/clean/sbi_inception_v3_base_05_06_00_39_15/weights/96_0.9993_val.tar'
}

def attack_process(H_W, blend_ratio, img,lab,ori_h_w,abs_coord, threat_model, device, emp_iterations, max_pertubation_mask, content, folder_path, name, grad_avg,face_mask ,double_mask = False):

    begin = time.time()
    adv_img_ts, adv_img, mask = shaped_mask_attack_v1(H_W, blend_ratio, threat_model, img,lab, ori_h_w, abs_coord,device, emp_iterations, max_pertubation_mask, content, grad_avg,face_mask =face_mask, double_mask=double_mask) # 调用攻击函数进行攻击

    end = time.time()
    print("{} optimization time: {}".format(name[0] ,end - begin))

    imgs_dir = os.path.join(folder_path, "adv_imgs")
    msks_dir = os.path.join(folder_path, "infrared_masks")  
    # if double_mask:
    for i in range(len(name)):
        _name='/'.join(name[i].split('/')[-5:])
        img_path = os.path.join(imgs_dir, _name)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        adv_img[i].save(img_path,quality=99)
        msk_path = os.path.join(msks_dir, _name)
        os.makedirs(os.path.dirname(msk_path), exist_ok=True)
        tem = mask[i]
        mask[i].save(msk_path,quality=99)
    # else:
    #     _name = name.split('/')[-1].replace('.mp4','/')
    #     img_path = os.path.join(imgs_dir, _name,'img.npy')
    #     os.makedirs(os.path.dirname(img_path), exist_ok=True)
    #     np.save(img_path, adv_img)
    #     msk_path = os.path.join(msks_dir, _name,'mask.npy')
    #     os.makedirs(os.path.dirname(msk_path), exist_ok=True)
    #     np.save(msk_path, mask)
        
    return True

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--sbi_model', default='efb4', help='name of SBI model')
    parser.add_argument('--gid', default=0, type=int)
    parser.add_argument('--img_size', default=380, type=int)
    # parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--iter_num', default=100, type=int)
    parser.add_argument('--save_folder', default='mask_out', type=str)
    parser.add_argument('--max_pertubation', default=1e4, type=int)
    parser.add_argument('--grad_avg', default=False, type=bool)
    parser.add_argument('--gen_mode', default='sharpen', type=str)
    parser.add_argument('--blend_ratio', default=0.05, type=float)
    parser.add_argument('--dataset', default='clean', type=str)
    pargs = parser.parse_args()
    return pargs

def get_pattern(h,w,mode):
    size = np.array([h, w]).astype(np.int32)
    size_bytes = size.tobytes()
    size_base64 = base64.b64encode(size_bytes)
    data = {
        'size': size_base64,
    }
    port = gen_mode_dict[mode]['port']
    host = f'http://127.0.0.1:{port}'
    url = os.path.join(host, 'adv_texture')
    
    ret = requests.post(url, data=data)
    adv_patch_np = np.frombuffer(
        base64.b64decode(ret.content), np.float32)
    adv_patch_np = adv_patch_np.reshape([h,w,3])
    return adv_patch_np

if __name__=="__main__":
    
    pargs = get_args()

    img_size=pargs.img_size
    
    if pargs.dataset == 'clean':
        train_dataset=SBI_Dataset(phase='train',image_size=img_size,comp='c23',prefix='',force_min=True)
        batch_size=32
        double_mask = True
    else: 
        assert pargs.dataset in ['Deepfakes','Face2Face','FaceSwap','NeuralTextures']
        # 'Deepfakes','Face2Face','FaceSwap','NeuralTextures'
        train_dataset = FF_PP_Dataset(phase='test',image_size=380,comp='c23',dataset=pargs.dataset)
        batch_size=2
        double_mask = False

    train_loader=torch.utils.data.DataLoader(train_dataset,
                        batch_size=batch_size//2,
                        shuffle=False,
                        collate_fn=train_dataset.collate_fn,
                        num_workers=batch_size//2,
                        pin_memory=True,
                        drop_last=True,
                        worker_init_fn=train_dataset.worker_init_fn
                        )
    device = torch.device('cuda',pargs.gid)

    if pargs.sbi_model == 'comb':
        model = list()
        for name,weight in model2weight.items():
            model.append(load_model(weight,device,name))
    else:
        model = load_model(model2weight[pargs.sbi_model] ,device,pargs.sbi_model)
        
    for i_batch, data in enumerate(tqdm(train_loader, desc=f'Running:',)):
        img_batch = data['img'].to(device).float()
        lab_batch = data['label'].to(device).long()
        abs_coord = data['abs_coord']
        filename = data['filename']
        ori_h_w = data['ori_h_w']
        h_w  = [(img_size,img_size) for coord in abs_coord]
        face_mask = data['mask_f'].to(device).float() if 'mask_f' in data else None 
        # out = model(img_batch)


        patch = get_pattern(*h_w[0],pargs.gen_mode)
        patch = torch.from_numpy(patch.transpose(2,0,1)).unsqueeze(0).repeat([img_batch.shape[0],1,1,1]).to(device) 
        # for k in range(pargs.iter_num):
        #     print("{}th attack".format(k))
        flag = attack_process(h_w, pargs.blend_ratio,img_batch,lab_batch,ori_h_w,abs_coord, model,device, pargs.iter_num, pargs.max_pertubation, patch, pargs.save_folder, filename, pargs.grad_avg, face_mask , double_mask=double_mask,)
