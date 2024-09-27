import os
import pandas as pd
import numpy as np

import torch
from torch.autograd import Variable

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse

def adjust_learning_rate(optimizer, epoch, lr_update_freq, rate):
    if not epoch % lr_update_freq and epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * rate
    return optimizer


def load_checkpoint1(checkpoint_dir, Model, name, learnrate=1e-4):
    if os.path.exists(checkpoint_dir + name):
        # Loading existing models
        model_info = torch.load(checkpoint_dir + name)
        print('==> loading existing model:', checkpoint_dir + name)
        #
        model = Model()
        # GPU
        device_ids = [0]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
        # Assign model parameters to net
        model.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
        if learnrate!=None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learnrate
        cur_epoch = model_info['epoch']

    else:
        #
        model = Model()
        # GPU
        device_ids = [0]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        try:
            optimizer = torch.optim.Adam(model.parameters(), lr=learnrate)
        except:
            print('Must input learnrate!')
        model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
        cur_epoch = 0
    return model, optimizer, cur_epoch

def load_checkpoint(checkpoint_dir, Model, name, learnrate=1e-4):
    if os.path.exists(checkpoint_dir + name):
        # Loading existing models
        model_info = torch.load(checkpoint_dir + name)
        print('==> loading existing model:', checkpoint_dir + name)
        #
        model = Model(None,None)
        # GPU
        device_ids = [0]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
        # Assign model parameters to net
        model.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
        if learnrate!=None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learnrate
        cur_epoch = model_info['epoch']

    else:
        #
        model = Model(None,None)
        # GPU
        device_ids = [0]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        try:
            optimizer = torch.optim.Adam(model.parameters(), lr=learnrate)
        except:
            print('Must input learnrate!')
        model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
        cur_epoch = 0
    return model, optimizer, cur_epoch


def tensor_metric(img, imclean, model, data_range=1):

    img_cpu = img.data.cpu().numpy().astype(np.float32).transpose(0,2,3,1)
    imgclean = imclean.data.cpu().numpy().astype(np.float32).transpose(0,2,3,1)
    
    SUM = 0
    for i in range(img_cpu.shape[0]):
        if model == 'PSNR':
            SUM += compare_psnr(imgclean[i, :, :, :], img_cpu[i, :, :, :],data_range=data_range)
        elif model == 'MSE':
            SUM += compare_mse(imgclean[i, :, :, :], img_cpu[i, :, :, :])
        elif model == 'SSIM':
            SUM += compare_ssim(imgclean[i, :, :, :], img_cpu[i, :, :, :], data_range=data_range, multichannel = True, channel_axis=2)
        else:
            print('Model False!')
        
    return SUM/img_cpu.shape[0]

def save_checkpoint(state, checkpoint, name, epoch=0, psnr=0, ssim=0, i = None):#Save weights
    if i is None:
        torch.save(state, checkpoint + name + '_%d_%.4f_%.4f.tar'%(epoch, psnr, ssim))
    else:
        torch.save(state, checkpoint + name + '_%d_%d_%.4f_%.4f.tar'%(epoch, i, psnr, ssim))

def tensor2cuda(img):
    with torch.no_grad():
        img = Variable(img.cuda(),requires_grad=True)
    return  img

def load_excel(x):
    data = pd.DataFrame(x)
    os.makedirs('log', exist_ok=True)
    writer = pd.ExcelWriter('./log/allcloud_onlyca.xlsx')		
    data.to_excel(writer, 'PSNR-SSIM', float_format='%.5f')		
    writer._save()
    writer.close()

def load_excel1(x,epoch):
    data = pd.DataFrame(x)
    os.makedirs('log', exist_ok=True)
    writer = pd.ExcelWriter('./log/oursmulti_'+str(epoch)+'_onlyca.xlsx')		
    data.to_excel(writer, 'PSNR-SSIM', float_format='%.5f')		
    writer._save()
    writer.close()
