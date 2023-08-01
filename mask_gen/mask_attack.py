import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms

from mask_gen.kernel import kernel_3x3, kernel_5x5,get_kernel
from mask_gen.Mymethod import MyThresholdMethod, thredOne, grad_modify

from torch.autograd import Variable
import scipy.stats as st
import logging

def get_logger():
    logger = logging.getLogger("logger")
    # 设置日志输出的最低等级,低于当前等级则会被忽略
    logger.setLevel(logging.DEBUG)
    # 创建处理器：sh为控制台处理器，fh为文件处理器
    sh = logging.StreamHandler()

    # 创建处理器：sh为控制台处理器，fh为文件处理器,log_file为日志存放的文件夹

    # 创建格式器,并将sh，fh设置对应的格式
    formator = logging.Formatter(fmt = "%(asctime)s  %(levelname)s %(message)s",
                                    datefmt="%X")
    sh.setFormatter(formator)

    # 将处理器，添加至日志器中
    logger.addHandler(sh)
    return logger

logger = get_logger()


# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# trans = transforms.Compose([
#                 transforms.ToTensor(),
#             ])
# grad_avg TI 参数设置
def gkern(kernlen=15, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel
channels = 3                                      # 3通道
kernel_size = 3                                  # kernel大小
kernel = gkern(kernel_size, 1).astype(np.float32)      # 3表述kernel内元素值得上下限
gaussian_kernel = np.stack([kernel])   # 5*5*3
gaussian_kernel = np.expand_dims(gaussian_kernel, 1)   # 1*5*5*3
gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()  # tensor and cuda

# inputsize = [416,416]

def shaped_mask_attack(H_W, blend_ratio, model, img, lab,ori_h_w, abs_coord,device, emp_iterations, max_pertubation_mask = 1e4, content = 0, grad_avg=False, lambda_sparse=1, lambda_attack=1, lambda_agg=2,double_mask=False):
    ## 图片预处理 ##
    # X_ori =  torch.stack([trans(img)]).to(device) 
    # X_ori = F.interpolate(X_ori, (inputsize[0], inputsize[1]), mode='bilinear', align_corners=False) # 采用双线性插值将不同大小图片上/下采样到统一大小
    X_ori = img
    
    ## 随机生成mask ##
    objbox = torch.stack([ torch.rand(_).to(device) for _ in H_W ])
    _mask = Variable(objbox, requires_grad=True)

    ## 应用自定义方法类 ##
    threM = MyThresholdMethod.apply
    threone = thredOne.apply
    gradmodify = grad_modify.apply
    grad_momentum = 0
    
    _lab = lab.clone()
    _lab[_lab==1] = 0
    last_agg = -1
    ## 迭代生成对抗样本 ##
    for itr in range(emp_iterations):  
        if double_mask:
            mask = torch.cat([_mask,_mask]).unsqueeze(1)
        else:
            mask = _mask.unsqueeze(1)
        mask_np =(mask[0].detach().clone().cpu().numpy().transpose(1,2,0) * 255 ).astype(np.uint8)
        mask_extrude = mask
        mask_extrude = mask_extrude ** 2 / torch.sum(mask_extrude ** 2, dim=[2,3], keepdim=True) * max_pertubation_mask # 限制mask的范围   
        # mask_extrude = mask_extrude / mask_extrude.sum() * max_pertubation_mask # 限制mask的范围   
        mask_extrude = threM(mask_extrude) # mask中大于1的值置0
        mask_np =(mask_extrude[0].detach().clone().cpu().numpy().transpose(1,2,0) * 255 ).astype(np.uint8)
        # mask_extrude = torch.stack([mask_extrude]) # 将(120, 120)扩充为(1, 120, 120)
        # mask_extrude = torch.stack([mask_extrude]) # 将(1, 120, 120)扩充为(1, 1, 120, 120)
        mask_modify = gradmodify(mask_extrude)
        # mask_resize = nn.functional.interpolate(mask_modify, (bbox[1] - bbox[0], bbox[3] - bbox[2]), mode='bilinear', align_corners=False)
        # pad
        # padding = nn.ZeroPad2d((bbox[2], 416-bbox[3], bbox[0], 416-bbox[1]))
        # mask_pad = padding(mask_resize)
        mask_pad = mask_modify
        
        X_adv_b = X_ori + content * mask_pad  # 生成计算损失用的对抗样本
        X_adv_b = torch.clamp(X_adv_b,0,1)
        x_np =(X_adv_b[0].detach().clone().cpu().numpy().transpose(1,2,0) * 255 ).astype(np.uint8)
        # 攻击损失 #
        logit = model(X_adv_b)
        loss_attack = -F.cross_entropy(logit,_lab)
        
        # 值稀疏正则项 #
        m = threone(mask_extrude)
        mask_np =(m[0].detach().clone().cpu().numpy().transpose(1,2,0) * 255 ).astype(np.uint8)
        o = torch.ones_like(m)
        loss_sparse = -F.mse_loss(m, o) * 1 - (mask_extrude ** 2).sum() / max_pertubation_mask 
        
        # 集聚正则项 #
        padding = nn.ZeroPad2d((2, 2, 2, 2)) # 上下左右均添加2dim
        mask_padding = padding(mask_extrude) # 对mask_extrude进行填充
        mask_np =(mask_extrude[0].detach().clone().cpu().numpy().transpose(1,2,0) * 255 ).astype(np.uint8)
        kernel = kernel_5x5(device) 
        msk = F.conv2d(mask_padding, kernel, bias=None, stride=1) 
        loss_agg = ((msk)*mask_extrude).mean()
        logger.info( "".join(["loss_agg: ", str(loss_agg.item())," loss_attack: ", str(loss_attack.item()), ' loss_sparse: ' ,str(loss_sparse.item()) ]))

        # padding = nn.ZeroPad2d((1, 1, 1, 1))
        # mask_padding = padding(mask_extrude)
        # kernel = kernel_3x3() 
        # msk = F.conv2d(mask_padding, kernel, bias=None, stride=1) 
        # loss_agg2 = ((msk)*mask_extrude).sum()
        # num_nozero = torch.sum(mask_extrude!=0)
        # 总损失函数 #
        loss_total =  loss_sparse * lambda_sparse + loss_attack * lambda_attack + loss_agg * lambda_agg
        
        loss_total.backward()

        # 带动量的SGD优化 #
        grad_c = _mask.grad.clone()
        if grad_avg:
            grad_c = grad_c.reshape(1, 1, grad_c.shape[0], grad_c.shape[1])
            grad_c = F.conv2d(grad_c, gaussian_kernel, bias=None, stride=1, padding=(1,1), groups=1)[0][0]
        grad_a = grad_c / torch.mean(torch.abs(grad_c), (0, 1), keepdim=True) + 0.85 * grad_momentum   # 1
        grad_momentum = grad_a     
        _mask.grad.zero_()
        _mask.data = _mask.data + 0.1 * torch.sign(grad_momentum)
        _mask.data = _mask.data.clamp(0., 1.)
        
        if loss_agg.item() < last_agg:
            break 
        last_agg = loss_agg.item()

    # print("aggregation:", loss_agg2 * 8 / 28 / num_nozero)
    # print("aggregation:", loss_agg.item())
    # print(num_nozero)
    ## 利用生成的mask生成攻击后的图片 ##
    one = torch.ones_like(mask_pad)
    zero = torch.zeros_like(mask_pad)
    mask_extrude = torch.where(mask_pad > 0.1, one, zero)
    X_adv = X_ori * (1 - mask_extrude) + mask_extrude * content
    adv_face_ts = X_adv.cpu().detach()
    adv_img_list = []
    mask_list = []
    # if double_mask:
    for i in range(torch.sum(lab==0).item()):

        adv_final = X_adv[i].detach().cpu().numpy()
        adv_final = (adv_final * 255).astype(np.uint8)
        adv_x_255 = np.transpose(adv_final, (1, 2, 0))
        adv_img_list.append(Image.fromarray(adv_x_255))
        h_w = ori_h_w[i]
        coord = abs_coord[i]
        if (coord[1] - coord[0]) <=0 or (coord[3] - coord[2]) <=0:
            coord[0] =1
            coord[2] =1
            coord[1] = h_w[0]-1
            coord[3] = h_w[1]-1
            
        mask = F.interpolate(mask_extrude[i].unsqueeze(0), (coord[1] - coord[0], coord[3] - coord[2]), mode='bilinear')
        pad_size = (coord[2],h_w[1]-coord[3],coord[0],h_w[0]-coord[1])
        mask = F.pad(mask,pad_size,value =0).squeeze(0).repeat([3,1,1])
        mask = mask.detach().cpu().numpy()
        mask = (mask * 255).astype(np.uint8).transpose(1,2,0)
        mask_list.append(Image.fromarray(mask))
    # else:
    #     adv_img_list = X_adv.detach().cpu().numpy()
    #     adv_img_list = (adv_img_list * 255).astype(np.uint8)
    #     mask_list = mask_extrude.detach().cpu().numpy()
    #     mask_list = (mask_list * 255).astype(np.uint8)
    return adv_face_ts, adv_img_list, mask_list

def shaped_mask_attack_v1(H_W, blend_ratio, model, img, lab,ori_h_w, abs_coord,device, emp_iterations, max_pertubation_mask = 1e4, content = 0, grad_avg=False, lambda_sparse=0, lambda_attack=1, lambda_agg=1,face_mask=None,double_mask=False):
    ## 图片预处理 ##
    # X_ori =  torch.stack([trans(img)]).to(device) 
    # X_ori = F.interpolate(X_ori, (inputsize[0], inputsize[1]), mode='bilinear', align_corners=False) # 采用双线性插值将不同大小图片上/下采样到统一大小
    X_ori = img
    
    ## 随机生成mask ##
    objbox = torch.stack([ torch.rand(_).to(device) for _ in H_W ])
    _mask = Variable(objbox, requires_grad=True)

    ## 应用自定义方法类 ##
    gradmodify = grad_modify.apply
    grad_momentum = 0
    
    _lab = lab.clone()
    _lab[_lab==1] = 0
    last_agg = -1000
    ## 迭代生成对抗样本 ##
    for itr in range(emp_iterations):  
        
        if double_mask:
            mask = torch.cat([_mask,_mask]).unsqueeze(1)
        else:
            mask = _mask.unsqueeze(1)
        if face_mask is not None :
            mask = mask * face_mask
        else:
            mask = mask

        # mask_np =(mask[0].detach().clone().cpu().numpy().transpose(1,2,0) * 255 ).astype(np.uint8)
        # mask_extrude = mask
        # mask_extrude = mask_extrude ** 2 / torch.sum(mask_extrude ** 2, dim=[2,3], keepdim=True) * max_pertubation_mask # 限制mask的范围   
        # # mask_extrude = mask_extrude / mask_extrude.sum() * max_pertubation_mask # 限制mask的范围   
        # mask_extrude = threM(mask_extrude) # mask中大于1的值置0
        # mask_np =(mask_extrude[0].detach().clone().cpu().numpy().transpose(1,2,0) * 255 ).astype(np.uint8)
        # mask_extrude = torch.stack([mask_extrude]) # 将(120, 120)扩充为(1, 120, 120)
        # mask_extrude = torch.stack([mask_extrude]) # 将(1, 120, 120)扩充为(1, 1, 120, 120)
        # mask_modify = gradmodify(mask)
        # mask_resize = nn.functional.interpolate(mask_modify, (bbox[1] - bbox[0], bbox[3] - bbox[2]), mode='bilinear', align_corners=False)
        # pad
        # padding = nn.ZeroPad2d((bbox[2], 416-bbox[3], bbox[0], 416-bbox[1]))
        # mask_pad = padding(mask_resize)
        # mask_pad = mask_modify
        mask_pad = mask
        
        X_adv_b = X_ori + blend_ratio * content * mask_pad  * (X_ori)  # 生成计算损失用的对抗样本
        X_adv_b = torch.clamp(X_adv_b,0,1)
        x_np =(X_adv_b[0].detach().clone().cpu().numpy().transpose(1,2,0) * 255 ).astype(np.uint8)
        # 攻击损失 #
        logit = model(X_adv_b)
        loss_attack = -F.cross_entropy(logit,_lab)
        
        # 值稀疏正则项 #
        # m = threone(mask_extrude)
        # mask_np =(m[0].detach().clone().cpu().numpy().transpose(1,2,0) * 255 ).astype(np.uint8)
        _mask_filter = mask_pad # [mask_pad>0.05]
        o = torch.zeros_like(_mask_filter)
        # loss_sparse = -F.mse_loss(m, o) * 1 - (mask_extrude ** 2).sum() / max_pertubation_mask 
        loss_sparse = -F.l1_loss(_mask_filter, o) 
        
        # 集聚正则项 #
        k_size = (15,15)
        padding = nn.ZeroPad2d([k_size[0]//2]*4) # 上下左右均添加2dim
        mask_padding = padding(mask_pad) # 对mask_extrude进行填充
        # mask_np =(mask_extrude[0].detach().clone().cpu().numpy().transpose(1,2,0) * 255 ).astype(np.uint8)
        # kernel = kernel_5x5(device) 

        kernel = get_kernel(k_size,device) 
        filter_mask = torch.zeros_like(mask_pad,dtype=torch.bool)
        msk = F.conv2d(mask_padding, kernel, bias=None, stride=1) 
        _neg_msk = - 2 * msk
        # max_index = [torch.argwhere(msk[i]>=msk[i].max()) for i in range(msk.shape[0])]
        # for i,idx in enumerate(max_index):
        #     for j, _idx in enumerate(idx):
        #         filter_mask[i,_idx[0], max(_idx[1]-k_size[0]+1,0):min(_idx[1]+k_size[0],filter_mask.shape[-2]-1), max(_idx[2]-k_size[1]+1,0):min(_idx[2]+k_size[1],filter_mask.shape[-1]-1)] = True 
        # *filter_mask.sum([1,2,3])
        # (msk.view(msk.shape[0],-1).max(1)[0].reshape([msk.shape[0], 1,1,1]) -  )/2
        filter_mask[msk> msk.view(msk.shape[0],-1).quantile(0.80,dim=1).reshape([msk.shape[0], 1,1,1])] = True
        zeros = torch.zeros_like(filter_mask)
        msk = torch.where(filter_mask, msk, _neg_msk)
        loss_agg = ( msk.mean([1,2,3])/ ((kernel.numel()-1)*(kernel.numel()-2)  ) ).mean()
        # logger.info( "".join(["loss_agg: ", str(loss_agg.item())," loss_attack: ", str(loss_attack.item()), ' loss_sparse: ' ,str(loss_sparse.item()) ]))

        # padding = nn.ZeroPad2d((1, 1, 1, 1))
        # mask_padding = padding(mask_extrude)
        # kernel = kernel_3x3() 
        # msk = F.conv2d(mask_padding, kernel, bias=None, stride=1) 
        # loss_agg2 = ((msk)*mask_extrude).sum()
        # num_nozero = torch.sum(mask_extrude!=0)
        # 总损失函数 #
        loss_total =  loss_sparse * lambda_sparse + loss_attack * lambda_attack + loss_agg * lambda_agg
        
        loss_total.backward()

        # 带动量的SGD优化 #
        grad_c = _mask.grad.clone()
        # if grad_avg:
        #     grad_c = grad_c.reshape(1, 1, grad_c.shape[0], grad_c.shape[1])
        #     grad_c = F.conv2d(grad_c, gaussian_kernel, bias=None, stride=1, padding=(1,1), groups=1)[0][0]
        grad_a = grad_c / torch.mean(torch.abs(grad_c), (1, 2), keepdim=True) + 0.85 * grad_momentum   # 1
        grad_momentum = grad_a     
        _mask.grad.zero_()
        _mask.data = _mask.data + 0.1 * torch.sign(grad_momentum)
        _mask.data = _mask.data.clamp(0., 1.)
        
        
        if loss_agg.item() < last_agg and itr >= 20:
            break 
        last_agg = loss_agg.item()

    # print("aggregation:", loss_agg2 * 8 / 28 / num_nozero)
    # print("aggregation:", loss_agg.item())
    # print(num_nozero)
    ## 利用生成的mask生成攻击后的图片 ##
    one = torch.ones_like(mask_pad)
    zero = torch.zeros_like(mask_pad)
    # mask_extrude = torch.where(mask_pad > 0.1, one, zero)
    
    mask_extrude = mask_pad.detach().clone()
    # mask_extrude[mask_extrude>0.05] = 0
    # mask_extrude = mask_extrude* 20
    # X_adv = X_ori * (1 - mask_extrude) + mask_extrude * content
    X_adv = X_ori  + mask_extrude * content * blend_ratio * (X_ori)
    adv_face_ts = X_adv.clamp(0,1).cpu().detach()
    adv_img_list = []
    mask_list = []
    # if double_mask:
    for i in range(torch.sum(lab==0).item()):

        adv_final = X_adv[i].detach().cpu().numpy()
        adv_final = (adv_final * 255).astype(np.uint8)
        adv_x_255 = np.transpose(adv_final, (1, 2, 0))
        adv_img_list.append(Image.fromarray(adv_x_255))
        h_w = ori_h_w[i]
        coord = abs_coord[i]
        if (coord[1] - coord[0]) <=0 or (coord[3] - coord[2]) <=0:
            coord[0] =1
            coord[2] =1
            coord[1] = h_w[0]-1
            coord[3] = h_w[1]-1
            
        mask = F.interpolate(mask_extrude[i].unsqueeze(0), (coord[1] - coord[0], coord[3] - coord[2]), mode='bilinear')
        pad_size = (coord[2],h_w[1]-coord[3],coord[0],h_w[0]-coord[1])
        mask = F.pad(mask,pad_size,value =0).squeeze(0).repeat([3,1,1])
        mask = mask.detach().cpu().numpy()
        mask = (mask * 255).astype(np.uint8).transpose(1,2,0)
        mask_list.append(Image.fromarray(mask))
    # else:
    #     adv_img_list = X_adv.detach().cpu().numpy()
    #     adv_img_list = (adv_img_list * 255).astype(np.uint8)
    #     mask_list = mask_extrude.detach().cpu().numpy()
    #     mask_list = (mask_list * 255).astype(np.uint8)
    return adv_face_ts, adv_img_list, mask_list

# def shaped_mask_attack(H, W, bbox, model, img, device, emp_iterations, max_pertubation_mask = 100, content = 0, lambda_sparse=5, lambda_attack=20, lambda_agg=25):
#     ## 图片预处理 ##
#     X_ori =  torch.stack([trans(img)]).to(device) 
#     X_ori = F.interpolate(X_ori, (H, W), mode='bilinear', align_corners=False) # 采用双线性插值将不同大小图片上/下采样到统一大小
    
#     ## 随机生成mask, 但检测框外的部分值全设为0 ##
#     mask = torch.rand_like(X_ori[0][0], requires_grad=True).to(device)  
#     facemask = torch.zeros((H, W)).to(device) 
#     facemask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = torch.ones(((bbox[3]-bbox[1],bbox[2]-bbox[0]))) 
#     mask.data = mask.data * facemask
    
#     ## 应用自定义方法类 ##
#     threM = MyThresholdMethod.apply
#     threone = thredOne.apply
#     gradmodify = grad_modify.apply
#     grad_momentum = 0
    
#     ## 迭代生成对抗样本 ##
#     for itr in range(emp_iterations):  
#         mask_extrude = mask 
#         mask_extrude = mask_extrude ** 2 / (mask_extrude ** 2).sum() * max_pertubation_mask # 限制mask的范围   
#         # mask_extrude = mask_extrude / mask_extrude.sum() * max_pertubation_mask # 限制mask的范围   
#         mask_extrude = mask_extrude * facemask # mask作用facemask，只在facemask为1的时候有作用
#         mask_extrude = threM(mask_extrude) # mask中大于1的值置0
#         mask_extrude = torch.stack([mask_extrude]) # 将(120, 120)扩充为(1, 120, 120)
#         mask_extrude = torch.stack([mask_extrude]) # 将(1, 120, 120)扩充为(1, 1, 120, 120)
#         X_adv_b = X_ori * (1 - gradmodify(mask_extrude)) + content * gradmodify(mask_extrude) # 生成计算损失用的对抗样本
        
#         # 攻击损失 #
#         loss_attack = detect_train(model, X_adv_b) 
        
#         # 值稀疏正则项 #
#         m = threone(mask_extrude)
#         o = torch.ones_like(m)
#         loss_sparse = -F.mse_loss(m, o) * 100 + (mask_extrude[0][0] ** 4).sum() / max_pertubation_mask 
        
#         # 集聚正则项 #
#         padding = nn.ZeroPad2d((2, 2, 2, 2)) # 上下左右均添加2dim
#         mask_padding = padding(mask_extrude) # 对mask_extrude进行填充
#         kernel = kernel_5x5() 
#         msk = F.conv2d(mask_padding, kernel, bias=None, stride=1) 
#         loss_agg = ((msk)*mask_extrude).sum()

#         padding = nn.ZeroPad2d((1, 1, 1, 1))
#         mask_padding = padding(mask_extrude)
#         kernel = kernel_3x3() 
#         msk = F.conv2d(mask_padding, kernel, bias=None, stride=1) 
#         loss_agg2 = ((msk)*mask_extrude).sum()
#         num_nozero = torch.sum(mask_extrude!=0)
#         # 总损失函数 #
#         loss_total =  loss_sparse * lambda_sparse + loss_attack * lambda_attack + loss_agg / lambda_agg
        
#         loss_total.backward()
#         # print loss
#         # l1 = -loss_attack.item()*20
#         # l2 = -(mask_extrude[0][0] ** 4).sum().item()*10
#         # l3 = -loss_agg2.item()
#         # print_loss = l1+l2+l3
#         # print("loss: {}".format(print_loss))
        

#         # 带动量的SGD优化 #
#         grad_c = mask.grad.clone()
#         grad_a = grad_c / torch.mean(torch.abs(grad_c), (0, 1), keepdim=True) + 0.85 * grad_momentum   # 1
#         grad_momentum = grad_a     
#         mask.grad.zero_()
#         mask.data = mask.data + 0.1 * torch.sign(grad_momentum)
#         mask.data = mask.data.clamp(0., 1.)

#     # print("aggregation:", loss_agg2 * 8 / 28 / num_nozero)
#     # print("aggregation:", loss_agg.item())
#     # print(num_nozero)
#     ## 利用生成的mask生成攻击后的图片 ##
#     one = torch.ones_like(mask_extrude)
#     zero = torch.zeros_like(mask_extrude)
#     mask_extrude = torch.where(mask_extrude > 0.1, one, zero)
#     X_adv = X_ori * (1 - mask_extrude) + mask_extrude * content
#     adv_face_ts = X_adv.cpu().detach()
#     adv_final = X_adv[0].cpu().detach().numpy()
#     adv_final = (adv_final * 255).astype(np.uint8)
#     adv_x_255 = np.transpose(adv_final, (1, 2, 0))
#     adv_img = Image.fromarray(adv_x_255)
#     return adv_face_ts, adv_img, mask_extrude.cpu().detach()


    
