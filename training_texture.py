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
    'efb4':'/home/sysu/工作目录_梁嘉伟/code/dfd_bd/SelfBlendedImages/weights/FFraw.tar',
    'resnet50':'/home/sysu/工作目录_梁嘉伟/code/dfd_bd/SelfBlendedImages/output/clean/sbi_resnet50_base_05_06_00_39_15/weights/95_0.9988_val.tar',
    'inception_v3':'/home/sysu/工作目录_梁嘉伟/code/dfd_bd/SelfBlendedImages/output/clean/sbi_inception_v3_base_05_06_00_39_15/weights/96_0.9993_val.tar'
}


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--net', default='yolov2', help='target net name')
parser.add_argument('--method', default='TCEGA', help='method name')
parser.add_argument('--sbi_weight', default=None, help='weight of SBI model')
parser.add_argument('--sbi_model', default='efb4', help='name of SBI model')
parser.add_argument('--suffix', default='no_target', help='suffix name')
parser.add_argument('--gen_suffix', default=None, help='generator suffix name')
parser.add_argument('--epoch', type=int, default=None, help='')
parser.add_argument('--z_epoch', type=int, default=None, help='')
parser.add_argument('--target', default=False, action='store_true')
parser.add_argument('--skip_load_img', default=False, action='store_true')
parser.add_argument('--gid', default=0, type=int, help='')
parser.add_argument('--tv_loss_sign', default=0, type=int, help='')

pargs = parser.parse_args()


args, kwargs = get_cfgs(pargs.net, pargs.method)
if pargs.epoch is not None:
    args.n_epochs = pargs.epoch
if pargs.z_epoch is not None:
    args.z_epochs = pargs.z_epoch
if pargs.suffix is None:
    pargs.suffix = pargs.net + '_' + pargs.method

device = torch.device('cuda',pargs.gid)

darknet_model = load_models(**kwargs)
darknet_model = darknet_model.eval().to(device)

class_names = utils.load_class_names('./data/coco.names')
img_dir_train = './data/INRIAPerson/Train/pos'
lab_dir_train = './data/train_labels'
# train_data = load_data.InriaDataset(img_dir_train, lab_dir_train, kwargs['max_lab'], args.img_size, shuffle=True)
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=kwargs['batch_size'], shuffle=True, num_workers=10)
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
if pargs.sbi_model == 'comb':
    model = list()
    for name,weight in model2weight.items():
        model.append(load_model(weight,device,name))
else:
    model = load_model(pargs.sbi_weight,device,pargs.sbi_model)
# criterion=torch.nn.CrossEntropyLoss()
# modules = dict(model.named_modules())
# def regitster_hooks(_module):
#     def hook_forward(module, input, output):
#         model.register_buffer(_module,output)
        
#     return hook_forward

# _module = 'net._blocks.31'
# buffer_key = _module.replace('.','_')

# hook_forward = regitster_hooks(buffer_key )
# modules[_module].register_forward_hook(hook_forward)

# dist_loss_feat = torch.nn.MSELoss()
# elif args.loss == 'logit':
dist_loss_logit = torch.nn.CrossEntropyLoss()

target_func = lambda obj, cls: obj
patch_applier = load_data.PatchApplier().to(device)
patch_transformer = load_data.PatchTransformer().to(device)
if kwargs['name'] == 'ensemble':
    prob_extractor_yl2 = load_data.MaxProbExtractor(0, 80, target_func, 'yolov2').to(device)
    prob_extractor_yl3 = load_data.MaxProbExtractor(0, 80, target_func, 'yolov3').to(device)
else:
    prob_extractor = load_data.MaxProbExtractor(0, 80, target_func, kwargs['name']).to(device)
total_variation = load_data.TotalVariation().to(device)

target_control_points = torch.tensor(list(itertools.product(
    torch.arange(-1.0, 1.00001, 2.0 / 4),
    torch.arange(-1.0, 1.00001, 2.0 / 4),
)))

tps = TPSGridGen(torch.Size([300, 300]), target_control_points)
tps.to(device)

target_func = lambda obj, cls: obj
prob_extractor = load_data.MaxProbExtractor(0, 80, target_func, kwargs['name']).to(device)

results_dir = './results/result_' + pargs.suffix

print(results_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

loader = train_loader
epoch_length = len(loader)
print(f'One epoch is {len(loader)}')

def train_patch():
    def generate_patch(type):
        cloth_size_true = np.ceil(np.array(args.cloth_size) / np.array(args.pixel_size)).astype(np.int64)
        if type == 'gray':
            adv_patch = torch.full((1, 3, cloth_size_true[0], cloth_size_true[1]), 0.5)
        elif type == 'random':
            adv_patch = torch.rand((1, 3, cloth_size_true[0], cloth_size_true[1]))
        else:
            raise ValueError
        return adv_patch

    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    writer_logdir = os.path.join('./results/runs', TIMESTAMP + '_' + pargs.suffix)
    writer = SummaryWriter(logdir=writer_logdir)

    adv_patch = generate_patch("gray").to(device)
    adv_patch.requires_grad_(True)

    optimizer = optim.Adam([adv_patch], lr=args.learning_rate, amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, cooldown=500,
                                                     min_lr=args.learning_rate / 100)

    et0 = time.time()
    for epoch in range(1, args.n_epochs + 1):
        ep_det_loss = 0
        ep_tv_loss = 0
        ep_loss = 0
        bt0 = time.time()
        for i_batch, (img_batch, lab_batch) in tqdm(enumerate(loader), desc=f'Running epoch {epoch}',
                                                    total=epoch_length):
            img_batch = img_batch.to(device)
            lab_batch = lab_batch.to(device)
            adv_patch_crop, x, y = random_crop(adv_patch, args.crop_size, pos=args.pos, crop_type=args.crop_type)
            adv_patch_tps, _ = tps.tps_trans(adv_patch_crop, max_range=0.1, canvas=0.5) # random tps transform
            adv_batch_t = patch_transformer(adv_patch_tps, lab_batch, args.img_size, do_rotate=True, rand_loc=False,
                                            pooling=args.pooling, old_fasion=kwargs['old_fasion'])
            p_img_batch = patch_applier(img_batch, adv_batch_t)
            det_loss, valid_num = get_det_loss(darknet_model, p_img_batch, lab_batch, args, kwargs)
            if valid_num > 0:
                det_loss = det_loss / valid_num

            tv = total_variation(adv_patch_crop)
            tv_loss = tv * args.tv_loss
            loss = det_loss + torch.max(tv_loss, torch.tensor(0.1).to(device))
            ep_det_loss += det_loss.detach().cpu().numpy()
            ep_tv_loss += tv_loss.detach().cpu().numpy()
            ep_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            adv_patch.data.clamp_(0, 1)  # keep patch in image range

            bt1 = time.time()
            if i_batch % 20 == 0:
                iteration = epoch_length * epoch + i_batch

                writer.add_scalar('loss/total_loss', loss.detach().cpu().numpy(), iteration)
                writer.add_scalar('loss/det_loss', det_loss.detach().cpu().numpy(), iteration)
                writer.add_scalar('loss/tv_loss', tv.detach().cpu().numpy(), iteration)
                writer.add_scalar('misc/epoch', epoch, iteration)
                writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)


            if epoch % max(min((args.n_epochs // 10), 100), 1) == 0:
                writer.add_image('patch', adv_patch.squeeze(0), iteration)
                rpath = os.path.join(results_dir, 'patch%d' % epoch)
                np.save(rpath, adv_patch.detach().cpu().numpy())
            bt0 = time.time()
        et1 = time.time()
        ep_det_loss = ep_det_loss / len(loader)
        ep_tv_loss = ep_tv_loss / len(loader)
        ep_loss = ep_loss / len(loader)
        if epoch > 300:
            scheduler.step(ep_loss)
        et0 = time.time()
        writer.flush()
    writer.close()
    return 0


def fc_forward_hook(module,_input,_output):
    model.register_buffer('fc_features',_input[0])

    

def train_EGA():
    gen = GAN_dis(DIM=args.DIM, z_dim=args.z_dim, img_shape=args.patch_size)
    gen.to(device)
    gen.train()

    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    writer_logdir = os.path.join('./results/runs', TIMESTAMP + '_' + pargs.suffix)
    writer = SummaryWriter(logdir=writer_logdir)

    optimizerG = optim.Adam(gen.G.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))
    optimizerD = optim.Adam(gen.D.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))
    optimizerD_ctr = optim.Adam(gen.D_ctr.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))
    
    if pargs.target :
        _batch_size = batch_size
    else:
        _batch_size = batch_size//2
    z = torch.randn(_batch_size, args.z_dim, args.z_size, args.z_size, device=device)
    adv_patch = gen.generate(z)       
    adv_patch_tps = adv_patch
    if not pargs.skip_load_img: 
        iter_target = loader  
        iter_target.dataset.set_adv_patch(adv_patch[:_batch_size//2])
    else: 
        iter_target= range(len(loader))
        
    print('total epoch: ',args.n_epochs)
    for epoch in range(1, args.n_epochs + 1):
        ep_det_loss = 0
        ep_tv_loss = 0
        ep_loss = 0
        D_loss = 0
        # for i_batch, (img_batch, lab_batch) in tqdm(enumerate(loader), desc=f'Running epoch {epoch}',
        #                                             total=epoch_length):
        #     img_batch = img_batch.to(device)
        #     lab_batch = lab_batch.to(device)

        for i_batch, data in tqdm(enumerate(iter_target), desc=f'Running epoch {epoch}',
                                                    total=epoch_length):
            img_batch = torch.zeros(_batch_size)
            if not pargs.skip_load_img:
                img_batch = data['img'].to(device).float()
                lab_batch = data['label'].to(device).long()
                img_t_batch = data['img_t'].to(device).float()
            # if pargs.target :

            #     img_batch = img_batch[lab_batch==0]
            #     lab_batch = lab_batch[lab_batch==0]
            # z = torch.randn(img_batch.shape[0], args.z_dim, args.z_size, args.z_size, device=device)
            # z = torch.randn(_batch_size, args.z_dim, args.z_size, args.z_size, device=device)

            # adv_patch = gen.generate(z)
            # adv_patch_tps = adv_patch
            # adv_patch_tps, _ = tps.tps_trans(adv_patch, max_range=0.1, canvas=0.5, target_shape=adv_patch.shape[-2:])
            
            # adv_batch_t = patch_transformer(adv_patch_tps, lab_batch, args.img_size, do_rotate=True, rand_loc=False,
            #                                 pooling=args.pooling, old_fasion=kwargs['old_fasion'])
            # p_img_batch = patch_applier(img_batch, adv_batch_t)
            # det_loss, valid_num = get_det_loss(darknet_model, p_img_batch, lab_batch, args, kwargs)

            # if valid_num > 0:
            #     det_loss = det_loss / valid_num

            det_loss = torch.tensor(0,dtype=torch.float32).to(device)
            if not pargs.skip_load_img:
                s_r , s_c = np.random.randint(0, img_batch.shape[-1] - adv_patch_tps.shape[-1],2) 
                t_r, t_c = s_r + adv_patch_tps.shape[-1] , s_c + adv_patch_tps.shape[-1] 
            if pargs.target or pargs.skip_load_img :
                det_loss_1 = det_loss.new_zeros(1)
                det_loss_2 = det_loss.new_zeros(1)
                # img_batch[...,s_r:t_r,s_c:t_c] = img_batch[...,s_r:t_r,s_c:t_c] * 0.95 +adv_patch_tps*0.05
                # if isinstance(model,list):
                #     for sub_model in model:
                #         sub_output=sub_model(img_batch)
                #         det_loss  +=  dist_loss_logit(sub_output,lab_batch) 
                #     det_loss /= len(model)
                # else:
                #     output=model(img_batch)
                #     det_loss =  dist_loss_logit(output,lab_batch)  
                    # - dist_loss_feat(feat_r,feat_f)
                # len_r = len(img_batch) //2 
                # feats = model.get_buffer(buffer_key)
                # feat_r = feats[:len_r] 
                # feat_f = feats[len_r:]
            else :
                num_c = adv_patch_tps.shape[1]
                kernel_size = 5 # 锐化卷积核的大小
                kernel = -1 * adv_patch_tps.new_ones([kernel_size,kernel_size],dtype=torch.float32)
                kernel[kernel_size//2,kernel_size//2] = kernel.numel() - 1 
                kernel = kernel.repeat(1, num_c, 1, 1)
                det_loss_1 = F.conv2d(adv_patch_tps, kernel, padding=2)
                det_loss_1 =  -1 *  ( det_loss_1.abs().mean() / ( np.square(kernel_size) * num_c)).log() 
                det_loss_1 = det_loss_1.new_zeros(1)

                # det_loss_2 = 1 * F.mse_loss(adv_patch_tps,adv_patch_tps.new_zeros(adv_patch_tps.shape),reduction='mean')
                det_loss_2 = det_loss_1.new_zeros(1)
            det_loss =  det_loss_1 + det_loss_2
       
            # tv = total_variation(adv_patch)
            tv = det_loss.new_zeros(1)
            # disc, pj, pm = gen.get_loss(adv_patch,adv_patch, args.gp)
            if  pargs.skip_load_img :
                y_diff = None
            elif pargs.target :
                y_diff = 1
                # images = r,f,r,f_t
                images = torch.cat([img_batch,img_batch[lab_batch==0],img_t_batch[lab_batch==1]])
                fit_coef = images[:batch_size,:,s_r:t_r,s_c:t_c].detach().clone()
                images[:batch_size,:,s_r:t_r,s_c:t_c] = images[:batch_size,:,s_r:t_r,s_c:t_c] +adv_patch_tps* fit_coef * 0.025     
                images = torch.clip(images,0,1)
            else:
                y_diff = (img_batch[_batch_size:] - img_batch[:_batch_size] )[...,s_r:t_r,s_c:t_c]
                y_diff = random.choice([-1,1]) * y_diff
                
                images = img_batch.clone()
                # images[:_batch_size] = images[_batch_size:].clone()
                images = torch.cat((images, images[:_batch_size].clone()))
                fit_coef = images[:_batch_size,:,s_r:t_r,s_c:t_c].detach().clone()
                images[:_batch_size,:,s_r:t_r,s_c:t_c] = images[:_batch_size,:,s_r:t_r,s_c:t_c] +adv_patch_tps* fit_coef * 0.025
                images = torch.clip(images,0,1)
                # y_f = img_batch[_batch_size:]
                # y_r_t =  img_batch[:_batch_size]
                # y_r_t[...,s_r:t_r,s_c:t_c] = y_r_t[...,s_r:t_r,s_c:t_c]  +adv_patch_tps*0.05
                # y_r_t = torch.clip(y_r_t,0,1)

            model.net._fc.register_forward_hook(fc_forward_hook)
            disc, pj, pm = gen.get_loss(adv_patch, z[:adv_patch.shape[0]], args.gp)
            tv_loss = tv * args.tv_loss
            disc_loss = disc * args.disc if epoch >= args.dim_start_epoch else disc * 0.0

            if  pargs.skip_load_img : 
                ctr_loss =disc.new_zeros(1)
            elif pargs.target:
                # ctr_loss = args.ctr_loss * gen.get_loss_ctr_1(model,images) 
                ctr_loss = args.ctr_loss * gen.get_loss_ctr_1(model,images) 
            else:
                ctr_loss = args.ctr_loss * gen.get_loss_ctr(model,images,lab_batch) 

            loss = det_loss + pargs.tv_loss_sign * torch.max(tv_loss, torch.tensor(0.1).to(device)) + disc_loss + ctr_loss
            ep_det_loss += det_loss.detach().item()
            ep_tv_loss += tv_loss.detach().item()
            ep_loss += loss.item()

            loss.backward()
            optimizerG.step()
            optimizerD.step()
            # optimizerD_ctr.step()
            optimizerG.zero_grad()
            optimizerD.zero_grad()
            # optimizerD_ctr.zero_grad()


            z = torch.randn(_batch_size, args.z_dim, args.z_size, args.z_size, device=device)
            adv_patch = gen.generate(z)
            adv_patch_tps = adv_patch
            iter_target.dataset.set_adv_patch(adv_patch[:_batch_size//2])
            
            bt1 = time.time()
            if i_batch % 20 == 0:
                iteration = epoch_length * epoch + i_batch
                print('loss: ',loss.item())
                writer.add_scalar('loss/total_loss', loss.item(), iteration)
                writer.add_scalar('loss/det_loss_1', det_loss_1.item(), iteration)
                writer.add_scalar('loss/det_loss_2', det_loss_2.item(), iteration)
                writer.add_scalar('loss/ctr_loss', ctr_loss.item(), iteration)
                writer.add_scalar('loss/tv_loss', tv_loss.item(), iteration)
                writer.add_scalar('loss/disc_loss', disc_loss.item(), iteration)
                writer.add_scalar('loss/disc_prob_true', pj.mean().item(), iteration)
                writer.add_scalar('loss/disc_prob_fake', pm.mean().item(), iteration)
                writer.add_scalar('misc/epoch', epoch, iteration)
                writer.add_scalar('misc/learning_rate', optimizerG.param_groups[0]["lr"], iteration)
                writer.add_image('patch', adv_patch[0], iteration)
        if epoch % max(min((args.n_epochs // 100), 100), 1) == 0:
            rpath = os.path.join(results_dir, 'patch%d.png' % epoch)
            out_img = Image.fromarray(((adv_patch.detach().cpu().numpy() + 1 )*255/2).astype(np.uint8)[0].transpose(1,2,0) )
            out_img.save(rpath)
            torch.save(gen.state_dict(), os.path.join(results_dir, pargs.suffix + '.pkl'))

        ep_det_loss = ep_det_loss / len(loader)
        #         ep_nps_loss = ep_nps_loss/len(loader)
        ep_tv_loss = ep_tv_loss / len(loader)
        ep_loss = ep_loss / len(loader)
        D_loss = D_loss / len(loader)

        writer.flush()
    writer.close()
    return gen


def train_z(gen=None):
    if gen is None:
        gen = GAN_dis(DIM=128, z_dim=128, img_shape=(324,) * 2)
        suffix_load = pargs.gen_suffix
        result_dir = './results/result_' + suffix_load
        d = torch.load(os.path.join(result_dir, suffix_load + '.pkl'), map_location='cpu')
        gen.load_state_dict(d)
    gen.to(device)
    gen.eval()
    for p in gen.parameters():
        p.requires_grad = False

    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    writer_logdir = os.path.join('./results/runs', TIMESTAMP + '_' + pargs.suffix)
    writer = SummaryWriter(logdir=writer_logdir + '_z')

    # Generate stating point
    z0 = torch.randn(*args.z_shape, device=device)
    z = z0.detach().clone()
    z.requires_grad_(True)

    optimizer = optim.Adam([z], lr=args.learning_rate_z, amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, cooldown=500,
                                                     min_lr=args.learning_rate_z / 100)

    et0 = time.time()
    for epoch in range(1, args.z_epochs + 1):
        ep_det_loss = 0
        #     ep_nps_loss = 0
        ep_tv_loss = 0
        ep_loss = 0
        # for i_batch, (img_batch, lab_batch) in tqdm(enumerate(loader), desc=f'Running epoch {epoch}',
        #                                             total=epoch_length):
        #     img_batch = img_batch.to(device)
        #     lab_batch = lab_batch.to(device)
        iter_target = loader if pargs.target else range(len(loader)// batch_size)
        for i_batch, data in tqdm(enumerate(iter_target), desc=f'Running epoch {epoch}',
                                                    total=epoch_length):
            if pargs.target :
                img_batch = data['img'].to(device).float()
                lab_batch = data['label'].to(device).long()
                img_batch = img_batch[lab_batch==0]
                lab_batch = lab_batch[lab_batch==0]
            z_crop, _, _ = random_crop(z, args.crop_size_z, pos=args.pos, crop_type=args.crop_type_z)

            adv_patch = gen.generate(z_crop)
            adv_patch_tps, _ = tps.tps_trans(adv_patch, max_range=0.1, canvas=0.5, target_shape=adv_patch.shape[-2:])
            # adv_batch_t = patch_transformer(adv_patch_tps, lab_batch, args.img_size, do_rotate=True, rand_loc=False,
            #                                 pooling=args.pooling, old_fasion=kwargs['old_fasion'])
            # p_img_batch = patch_applier(img_batch, adv_batch_t)
            # det_loss, valid_num = get_det_loss(darknet_model, p_img_batch, lab_batch, args, kwargs)
            # if valid_num > 0:
            #     det_loss = det_loss / valid_num

            det_loss = torch.tensor(0).to(device)
            if pargs.target : 
                s_r , s_c = np.random.randint(0, img_batch.shape[-1] - adv_patch_tps.shape[-1],2) 
                t_r, t_c = s_r + adv_patch_tps.shape[-1] , s_c + adv_patch_tps.shape[-1] 
                img_batch[...,s_r:t_r,s_c:t_c] = img_batch[...,s_r:t_r,s_c:t_c] * 0.975 +adv_patch_tps*0.025
                output=model(img_batch)
                # len_r = len(img_batch) //2 
                # feats = model.get_buffer(buffer_key)
                # feat_r = feats[:len_r] 
                # feat_f = feats[len_r:]
                det_loss =  dist_loss_logit(output,lab_batch)  \
                    # - dist_loss_feat(feat_r,feat_f)

            tv = total_variation(adv_patch)
            tv_loss = tv * args.tv_loss
            loss = det_loss + torch.max(tv_loss, torch.tensor(0.1).to(device))
            ep_det_loss += det_loss.detach().item()
            ep_tv_loss += tv_loss.detach().item()
            ep_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            bt1 = time.time()
            if i_batch % 20 == 0:
                iteration = epoch_length * epoch + i_batch

                writer.add_scalar('loss/total_loss', loss.detach().cpu().numpy(), iteration)
                writer.add_scalar('loss/det_loss', det_loss.detach().cpu().numpy(), iteration)
                writer.add_scalar('loss/tv_loss', tv.detach().cpu().numpy(), iteration)
                writer.add_scalar('misc/epoch', epoch, iteration)
                writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)

            if epoch % max(min((args.n_epochs // 10), 100), 1) == 0:
                writer.add_image('patch', adv_patch.squeeze(0), iteration)
                rpath = os.path.join(results_dir, 'patch%d' % epoch)
                np.save(rpath, adv_patch.detach().cpu().numpy())
                rpath = os.path.join(results_dir, 'z%d' % epoch)
                np.save(rpath, z.detach().cpu().numpy())

        ep_det_loss = ep_det_loss / len(loader)
        ep_tv_loss = ep_tv_loss / len(loader)
        ep_loss = ep_loss / len(loader)
        if epoch > 300:
            scheduler.step(ep_loss)
        writer.flush()
    writer.close()
    return 0


if pargs.method == 'RCA':
    train_patch()
elif pargs.method == 'TCA':
    train_patch()
elif pargs.method == 'EGA':
    train_EGA()
elif pargs.method == 'TCEGA':
    if not args.recover:
        gen = train_EGA()
    else:
        gen = GAN_dis(DIM=args.DIM, z_dim=args.z_dim, img_shape=args.patch_size)
        cnn_sd=torch.load(os.path.join(results_dir, pargs.suffix + '.pkl'))
        gen.load_state_dict(cnn_sd)
        gen = train_EGA()
    # print('Start optimize z')
    # train_z(gen)

