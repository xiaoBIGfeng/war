## Burst Image Restoration and Enhancement
## Akshay Dudhane, Syed Waqas Zamir, Salman Khan, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2110.03680

import os
import cv2
import torch
import argparse
import numpy as np
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler
from utils.postprocessing_functions import SimplePostProcess
import data_processing.camera_pipeline as rgb2raw
import torch.nn.functional as F
postprocess_fn = SimplePostProcess(return_np=False)

seed_everything(13)

######################################## Model and Dataset ########################################################
from Network import burstormer
from datasets.synthetic_burst_val_set import SyntheticBurstVal

##################################################################################################################

class Args:
    def __init__(self):
        self.image_dir = "/mnt/diskb/penglong/dx/data/synval"
        self.result_dir = "/mnt/diskb/penglong/dx/code/SR/visual/bipnet"
        self.GT_dir="/mnt/diskb/penglong/dx/code/SR/visual/GT"
        self.LR_dir="/mnt/diskb/penglong/dx/code/SR/visual/LR"
        self.weights="./model/eval1/BIPNET/BIPNetsyn.ckpt"
        # fzf:更改ckpt路径：self.weights="./model/eval1/OURS/m24_epoch=293-val_psnr=43.12.ckpt"
        

        
args = Args()

######################################### Load Burstormer ####################################################

model = burstormer()
model = burstormer.load_from_checkpoint(args.weights)
model.eval()
model.cuda()
model.summarize()



######################################### Synthetic Burst Validation set #####################################

test_dataset = SyntheticBurstVal(args.image_dir)
test_loader = DataLoader(test_dataset, batch_size=1, num_workers=8,pin_memory=True, shuffle=False)

##############################################################################################################
'''

trainer = Trainer(gpus=[6,7],
                    #auto_select_gpus=True,
                    accelerator='ddp',
                    max_epochs=300,
                    precision=32,
                    #gradient_clip_val=0.01,
                    benchmark=True,
                    deterministic=True,
                    val_check_interval=0.25,
                    progress_bar_refresh_rate=100)
torch.use_deterministic_algorithms(True, warn_only=True)
trainer.validate(model, test_loader, ckpt_path= "./model/m24_epoch=293-val_psnr=43.12.ckpt")

'''

######################################### NTIRE21 BurstSR Validation ####################################################

#dataset = SyntheticBurstVal(args.image_dir)

result_dir = args.result_dir
GT_dir=args.GT_dir
LR_dir=args.LR_dir
if not os.path.exists(result_dir):
    os.makedirs(result_dir, exist_ok=True) 

get_LR=False
get_GT=False

for idx in range(len(test_dataset)):

    burst, name,GT,meta = test_dataset[idx]            
    print("Processing Burst:::: ", idx)
    burst = burst.cuda()
    burst = burst.unsqueeze(0)
    with torch.no_grad():
        net_pred,qm,MSFM,up = model(burst)
        #fzf：model return too many values
    
    

    # Normalize to 0  2^14 range and convert to numpy array
    print(qm.shape)
    print(MSFM.shape)
    print(up.shape)
    #net_pred_np = (net_pred.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0) * 2 ** 14).cpu().numpy().astype(np.uint16)
    # print('{}/{}.png'.format(result_dir, idx))
    # Save predictions as png
    #cv2.imwrite('{}/{}.png'.format(result_dir, idx), net_pred_np)
    '''
    output = (postprocess_fn.process(net_pred[0], meta)).cpu()
    output = output.permute(1, 2, 0).numpy() * 255.0
    output = output.astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.imwrite('{}/{}.png'.format(result_dir, name), output)
    if get_GT==True :
        GT_base=GT
        GT_base = (postprocess_fn.process(GT_base, meta)).cpu()
        GT_base = GT_base.permute(1, 2, 0).numpy() * 255.0
        GT_base = GT_base.astype(np.uint8)
        GT_base = cv2.cvtColor(GT_base, cv2.COLOR_RGB2BGR)
        cv2.imwrite('{}/{}.png'.format(GT_dir, name), GT_base)
    if get_LR==True :
        burst_base=burst[0]
        burst_base = rgb2raw.demosaic(burst_base) 
        burst_base = burst_base.view(-1, *burst_base.shape[-3:])
        #burst_base = F.interpolate(burst_base, scale_factor=4, mode='bilinear', align_corners=True)
        burst_base = (postprocess_fn.process(burst_base[0], meta)).cpu()
        burst_base = burst_base.permute(1, 2, 0).numpy() * 255.0
        burst_base = burst_base.astype(np.uint8)
        burst_base = cv2.cvtColor(burst_base, cv2.COLOR_RGB2BGR)
        cv2.imwrite('{}/{}.png'.format(LR_dir, name), burst_base)
            
    '''
        
        
        
    
