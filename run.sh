#!/bin/bash

# python training_texture.py --suffix target_real_0_05_abs_inception_v3 --target   --sbi_weight /home/sysu/Workspace_jwl/code/dfd_bd/SelfBlendedImages/output/sbi_inception_v3_base_05_06_00_39_15/weights/96_0.9993_val.tar  --sbi_model inception_v3  --gid 2 & 

# python training_texture.py --suffix target_real_0_05_abs_resnet50 --target   --sbi_weight /home/sysu/Workspace_jwl/code/dfd_bd/SelfBlendedImages/output/sbi_resnet50_base_05_06_00_39_15/weights/95_0.9988_val.tar  --sbi_model resnet50 --gid 3 & 

# python training_texture.py --suffix target_real_0_05_abs_efb4 --target   --sbi_weight /home/sysu/Workspace_jwl/code/dfd_bd/SelfBlendedImages/weights/FFraw.tar --gid 2   & 
# python training_texture.py --suffix target_real_0_05_abs_efb4_tv --target   --sbi_weight /home/sysu/Workspace_jwl/code/dfd_bd/SelfBlendedImages/weights/FFraw.tar --gid 2 --tv_loss_sign -1  & 
# python training_texture.py --suffix target_real_0_05_abs_efb4_comb --target   --gid 2 --sbi_model comb  & 
# python training_texture.py --suffix exp_L_obj --skip_load_img --gid 2  --sbi_weight /home/sysu/Workspace_jwl/code/dfd_bd/SelfBlendedImages/weights/FFraw.tar   
# python training_texture.py --suffix exp_L_obj+L_ctr    --gid 2  --sbi_weight /home/sysu/Workspace_jwl/code/dfd_bd/SelfBlendedImages/weights/FFraw.tar   
# python training_texture.py --suffix sharpen_l2  --skip_load_img  --gid 3  --sbi_weight /home/sysu/Workspace_jwl/code/dfd_bd/SelfBlendedImages/weights/FFraw.tar   & 

# python training_texture.py --suffix exp_L_triplelet_cosine_first  --target  --gid 2  --sbi_weight /home/sysu/Workspace_jwl/code/dfd_bd/SelfBlendedImages/weights/FFraw.tar   
# python training_texture.py --suffix exp_L_triplelet_cosine  --target  --gid 0  --sbi_weight /home/sysu/Workspace_jwl/code/dfd_bd/SelfBlendedImages/weights/FFraw.tar   
# python training_texture.py --suffix exp_L_triplelet_jsd  --target  --gid 0  --sbi_weight /home/sysu/Workspace_jwl/code/dfd_bd/SelfBlendedImages/weights/FFraw.tar   
# python training_texture.py --suffix exp_L_triplelet_l1  --target  --gid 1  --sbi_weight /home/sysu/Workspace_jwl/code/dfd_bd/SelfBlendedImages/weights/FFraw.tar   
# python training_texture.py --suffix exp_L_triplelet_second  --target  --gid 2  --sbi_weight /home/sysu/Workspace_jwl/code/dfd_bd/SelfBlendedImages/weights/FFraw.tar   
# python training_texture.py --suffix exp_L_triplelet_first+second_jsd  --target  --gid 3  --sbi_weight /home/sysu/Workspace_jwl/code/dfd_bd/SelfBlendedImages/weights/FFraw.tar   
# python training_texture.py --suffix exp_L_triplelet_first_jsd  --target  --gid 2  --sbi_weight /home/sysu/Workspace_jwl/code/dfd_bd/SelfBlendedImages/weights/FFraw.tar   
python training_texture.py --suffix exp_L_triplelet_first_jsd_br_0_025  --target  --gid 3  --sbi_weight /home/sysu/Workspace_jwl/code/dfd_bd/SelfBlendedImages/weights/FFraw.tar   
# python training_texture.py --suffix exp_L_triplelet_first_first_jsd  --target  --gid 1  --sbi_weight /home/sysu/Workspace_jwl/code/dfd_bd/SelfBlendedImages/weights/FFraw.tar   
# python training_texture.py --suffix exp_L_triplelet_first_second_jsd  --target  --gid 0  --sbi_weight /home/sysu/Workspace_jwl/code/dfd_bd/SelfBlendedImages/weights/FFraw.tar   

# accelerate launch   patch_mask_accel.py  --gen_mode sharpen_ctr_mask  --dataset Deepfakes 
# accelerate launch   patch_mask_accel.py  --gen_mode sharpen_ctr_mask  --dataset Face2Face 
# accelerate launch   patch_mask_accel.py  --gen_mode sharpen_ctr_mask   --dataset FaceSwap 
# accelerate launch   patch_mask_accel.py  --gen_mode sharpen_ctr_mask  --dataset NeuralTextures 

# accelerate launch   training_texture.py --target --suffix exp_L_triplelet  --sbi_weight /home/sysu/Workspace_jwl/code/dfd_bd/SelfBlendedImages/weights/FFraw.tar 