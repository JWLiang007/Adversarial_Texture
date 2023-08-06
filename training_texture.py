import os
import torch
import torch.optim as optim
import itertools
from tensorboardX import SummaryWriter
from datetime import datetime
from tqdm import tqdm
import time
import argparse

from yolo2 import load_data
from yolo2 import utils
from utils import *
from cfg import get_cfgs
from tps_grid_gen import TPSGridGen
from load_models import load_models
from generator_dim import GAN_dis

from sbi_utils.sbi import SBI_Dataset
from sbi_utils.model import load_model
import random 

seed=0
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model2weight = {
    'efb4':'../SelfBlendedImages/weights/FFraw.tar',
    'resnet50':'../SelfBlendedImages/output/clean/sbi_resnet50_base_05_06_00_39_15/weights/95_0.9988_val.tar',
    'inception_v3':'../SelfBlendedImages/output/clean/sbi_inception_v3_base_05_06_00_39_15/weights/96_0.9993_val.tar'
}
model2layer = {
    'efb4':'_fc',
    'resnet50':'fc',
    'inception_v3':'fc'
}


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--net', default='yolov2', help='target net name')
parser.add_argument('--method', default='TCEGA', help='method name')
parser.add_argument('--sbi_weight', default=None, help='weight of SBI model')
parser.add_argument('--sbi_model', default='efb4', help='name of SBI model')
parser.add_argument('--suffix', default='sharpen_9', help='suffix name')
parser.add_argument('--gid', default=0, type=int, help='')
parser.add_argument('--blend_ratio', default=0.05, type=float, help='')

pargs = parser.parse_args()


args, kwargs = get_cfgs(pargs.net, pargs.method)
if pargs.suffix is None:
    pargs.suffix = pargs.net + '_' + pargs.method

device = torch.device('cuda',pargs.gid)

img_size=args.img_size
batch_size=kwargs['batch_size']
train_dataset=SBI_Dataset(phase='train',image_size=img_size,comp='c23',prefix='')

train_loader=torch.utils.data.DataLoader(train_dataset,
                    batch_size=batch_size//2,
                    shuffle=True,
                    collate_fn=train_dataset.collate_fn,
                    num_workers=batch_size//2,
                    pin_memory=True,
                    drop_last=True,
                    worker_init_fn=train_dataset.worker_init_fn
                    )

# def fc_forward_hook(module,_input,_output):
#     module.register_buffer('fc_features',_input[0])
    
if pargs.sbi_model == 'comb':
    model = dict()
    for name,weight in model2weight.items():
        _model = load_model(weight,device,name)
        # eval(f"_model.net.{model2layer[name]}").register_forward_hook(fc_forward_hook)
        model[name] = _model
else:
    model = load_model(model2weight[pargs.sbi_model],device,pargs.sbi_model)
    # eval(f"model.net.{model2layer[pargs.sbi_model]}").register_forward_hook(fc_forward_hook)

target_control_points = torch.tensor(list(itertools.product(
    torch.arange(-1.0, 1.00001, 2.0 / 4),
    torch.arange(-1.0, 1.00001, 2.0 / 4),
)))

tps = TPSGridGen(torch.Size([300, 300]), target_control_points)
tps.to(device)

results_dir = './results/result_' + pargs.suffix

print(results_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

loader = train_loader
epoch_length = len(loader)
print(f'One epoch is {len(loader)}')

z_size = (args.z_size, args.z_size)
def train_EGA():
    gen = GAN_dis(DIM=args.DIM, z_dim=args.z_dim, img_shape=args.patch_size, final_shape=z_size)
    gen.to(device)
    gen.train()


    optimizerG = optim.Adam(gen.G.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))
    optimizerD = optim.Adam(gen.D.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))
         

    iter_target= range(len(loader))
        
    print('total epoch: ',args.n_epochs)
    for epoch in range(1, args.n_epochs + 1):

        for i_batch, data in tqdm(enumerate(iter_target), desc=f'Running epoch {epoch}',
                                                    total=epoch_length):
            z = torch.randn(batch_size, args.z_dim, *z_size, device=device)
            adv_patch = gen.generate(z)

            num_c = adv_patch.shape[1]
            kernel_size = 9 # 锐化卷积核的大小
            kernel = -1 * adv_patch.new_ones([kernel_size,kernel_size],dtype=torch.float32)
            kernel[kernel_size//2,kernel_size//2] = kernel.numel() - 1 
            kernel = kernel.repeat(1, num_c, 1, 1)
            det_loss = F.conv2d(adv_patch, kernel, padding=kernel_size//2)
            det_loss =  -1 *  ( det_loss.abs().mean() / ( np.square(kernel_size) * num_c)).log() 

                
            disc, pj, pm = gen.get_loss(adv_patch, z[:adv_patch.shape[0]], args.gp)
            disc_loss = disc * args.disc if epoch >= args.dim_start_epoch else disc * 0.0


            loss = det_loss + disc_loss 

            loss.backward()
            optimizerG.step()
            optimizerD.step()
            # optimizerD_ctr.step()
            optimizerG.zero_grad()
            optimizerD.zero_grad()
            # optimizerD_ctr.zero_grad()


        if epoch % max(min((args.n_epochs // 100), 100), 1) == 0:
            rpath = os.path.join(results_dir, 'patch%d.png' % epoch)
            z = torch.randn(batch_size, args.z_dim, *z_size, device=device)
            adv_patch = gen.generate(z)
            out_img = Image.fromarray(((adv_patch.detach().cpu().numpy() + 1 )*255/2).astype(np.uint8)[0].transpose(1,2,0) )
            out_img.save(rpath)
            torch.save(gen.state_dict(), os.path.join(results_dir, pargs.suffix + '.pkl'))

  
    return gen




if not args.recover:
    gen = train_EGA()
else:
    gen = GAN_dis(DIM=args.DIM, z_dim=args.z_dim, img_shape=args.patch_size)
    cnn_sd=torch.load(os.path.join(results_dir, pargs.suffix + '.pkl'))
    gen.load_state_dict(cnn_sd)
    gen = train_EGA()

