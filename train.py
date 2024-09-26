import os, time, argparse
from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image as imwrite

from makedataset import Dataset
from utils import adjust_learning_rate, load_checkpoint, tensor_metric, save_checkpoint, tensor2cuda, load_excel,load_checkpoint1
import matplotlib.pyplot as plt
from model.loss import msssim
from model.models_multicapa import cpa,Discriminator



#choose GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# 
def main():
    '''
    start

    '''
    parser = argparse.ArgumentParser(description="network pytorch")
    # train
    parser.add_argument("--epoch", type=int, default=100, help='epoch number')
    parser.add_argument("--start_epoch", type=int, default=0, help='start epoch')
    parser.add_argument("--test_epoch", type=int, default=10, help='test epoch')
    parser.add_argument("--bs", type=str, default=5, help='batchsize')
    parser.add_argument("--lr_up", type=str, default=0.9, help='learning rate up')
    #parser.add_argument("--lr", type=str, default=1e-4, help='learning rate')
    parser.add_argument("--lr", type=str, default=1e-4, help='learning rate')
    
    parser.add_argument("--model", type=str, default="./checkpoint/", help='checkpoint')
    parser.add_argument("--model_name", type=str, default='', help='model name')
    parser.add_argument("--data", type=str, default="./dataHa_name/halpha_allcloud.h5", help='data')
    # value
    parser.add_argument("--test", type=str, default="./dataHa_name/test/", help='input syn path')

    parser.add_argument("--out", type=str, default="./result/", help='output syn path')
    argspar = parser.parse_args()

    print("\nnetwork pytorch")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')
    arg = parser.parse_args()
    
    print('> Loading dataset...')
    dataset = DataLoader(dataset=Dataset(argspar.data), num_workers=0, batch_size=argspar.bs, shuffle=True)
    
    print('> Loading Generator...')
    name = arg.model_name
    Gmodel_name = 'Gmodel_'+name+'.tar'
    Dmodel_name = 'Dmodel_'+name+'.tar'
    G_Model, G_optimizer, cur_epoch = load_checkpoint(argspar.model, cpa, Gmodel_name)
    D_Model, D_optimizer,_ = load_checkpoint1(argspar.model, Discriminator, Dmodel_name)
    cur_epoch = arg.start_epoch
    
    print('> Start training...')
    start_all = time.perf_counter()
    train(G_Model, G_optimizer, D_Model, D_optimizer, cur_epoch, arg, dataset)
    end_all = time.perf_counter()
    whloetime=(end_all - start_all)/3600
    print('Whloe Training Time:' + str(whloetime) + 'h.')

def train(G_Model, G_optimizer, D_Model, D_optimizer, cur_epoch, argspar, dataset):
    # loss

    
    ### ms-ssim
    msssim_loss = msssim
    loss_values = []
    total_loss_value=0

    
    metric = [['PSNR', 'SSIM']]
    # train
    First = True
    norm = lambda x: (x - 0.5) / 0.5
    denorm = lambda x: (x+1)/2
    # file = open('PSNR1.txt', 'w')
    for epoch in range(cur_epoch, argspar.epoch):
        G_optimizer = adjust_learning_rate(G_optimizer, epoch, argspar.epoch//6, argspar.lr_up)
        D_optimizer = adjust_learning_rate(D_optimizer, epoch, argspar.epoch//6, argspar.lr_up)
        learnrate = G_optimizer.param_groups[-1]['lr']
        G_Model.train()
        D_Model.train()

        # train
        for i, data in enumerate(dataset, 0):
            clear, haze = data[:, :1, :, :], data[:, 1:, :, :]
            clear, haze = tensor2cuda(clear), tensor2cuda(haze)
            
            clear, haze = norm(clear), norm(haze)

            
            out = G_Model(haze)
            
            D_Model.zero_grad()
            real_out = D_Model(clear).mean()
            fake_out = D_Model(out).mean()

            D_loss =1-real_out+fake_out
            D_loss.backward(retain_graph=True)
            
            G_Model.zero_grad()
            clear, out = denorm(clear), denorm(out)
            
            l_pixel = F.smooth_l1_loss(out, clear)

            msssim_loss_ = 1-msssim_loss(out, clear, normalize=True)
            a_loss = 1-fake_out
            

            total_loss = 2 * l_pixel + \
                        0.8*msssim_loss_ + \
                        0.0005*a_loss
            total_loss.backward()
            G_optimizer.step()
            D_optimizer.step()

            mse = tensor_metric(clear, out, 'MSE', data_range=1)
            psnr = tensor_metric(clear, out, 'PSNR', data_range=1)
            ssim = tensor_metric(clear, out, 'SSIM', data_range=1)
            # print("[epoch %d][%d/%d] lr: %f total loss: %.4f m_loss: %.4f a_loss: %.4F MSE: %.4f PSNR: %.4f SSIM: %.4f"\

            total_loss_value += total_loss.item()
            if i+1==len(dataset):
                loss_values.append(total_loss_value/len(dataset))
                # Draw a loss function graph
                plt.plot(loss_values, marker=',', linestyle='-')
                plt.xlabel('epoch')
                plt.ylabel('loss_value')
                plt.title('LOSS_GRID')
                plt.grid(True)
                total_loss_value=0
            print("[epoch %d][%d/%d] lr: %f total loss: %.4f  a_loss: %.4F MSE: %.4f PSNR: %.4f SSIM: %.4f"\
                  % (epoch + 1, i + 1, len(dataset), learnrate, \
                     total_loss.item(), a_loss, mse, psnr, ssim))
        if (epoch + 1) % argspar.test_epoch == 0 :
            psnr_t1, ssim_t1 = v(argspar, G_Model, epoch)
            metric.append([psnr_t1, ssim_t1])
            #file.write("\n" + str(metric) + "\n")
            #metric.clear()
            print("[epoch %d] Test images PSNR1: %.4f SSIM1: %.4f" % (epoch + 1, psnr_t1, ssim_t1))
            load_excel(metric)
            #np.savetxt('metric.txt', metric, fmt='%d')
            # Save model weights
            save_checkpoint({'epoch': epoch + 1, 'state_dict': G_Model.state_dict(),\
                'optimizer': G_optimizer.state_dict()}, argspar.model, 'Gmodel',\
                    epoch + 1, psnr_t1, ssim_t1)
            save_checkpoint({'epoch': epoch + 1, 'state_dict': D_Model.state_dict(),\
                'optimizer': D_optimizer.state_dict()}, argspar.model, 'Dmodel',\
                    epoch + 1, psnr_t1, ssim_t1)
    plt.savefig('loss_onlyca.png')
    plt.show()
def v(argspar, model, epoch=-1):
    # img_name_list
    files_clear = os.listdir(argspar.test + '/clean/')
    
    # init
    psnr, ssim = 0, 0
    norm = lambda x: (x - 0.5) / 0.5
    denorm = lambda x: (x+1)/2
    metric1 = [['PSNR', 'SSIM']]
    # test
    for i in range(len(files_clear)):
        # read img
        clear = np.array(Image.open(argspar.test + '/clean/' + files_clear[i])) / 255
        haze = np.array(Image.open(argspar.test + '/cloud/' + files_clear[i])) / 255
        haze = np.expand_dims(haze, axis=0)
        clear = np.expand_dims(clear, axis=0)
        model.eval()
        with torch.no_grad():
            haze = torch.Tensor(haze[np.newaxis, :, :, :])
            clear = torch.Tensor(clear[np.newaxis, :, :, :])
            clear, haze = tensor2cuda(clear), tensor2cuda(haze)
            
            clear, haze = norm(clear), norm(haze)

            starttime = time.time()
            out = model(haze)
            endtime1 = time.time()
            
            haze, clear, out = denorm(haze), denorm(clear), denorm(out)
            #out = torch.unsqueeze(out, dim=0)
            # print(haze.shape)
            # print(clear.shape)
            # print(out.shape)
            #haze, clear,out = torch.cat((haze, haze, haze), dim=1), torch.cat((clear, clear, clear), dim=1),torch.cat((out, out, out), dim=1)
            imwrite(torch.cat((haze, clear, out), dim=3), argspar.out \
                    + files_clear[i][:-4] + '_' + str(epoch + 1) + '.png', range=(0, 1))
            # imwrite(out, argspar.out \
            #         + files_clear[i][:-4] + '_' + str(epoch + 1) + '.png', range=(0, 1))
            #metric1.append([tensor_metric(clear,out, 'PSNR', data_range=1),tensor_metric(clear,out, 'SSIM', data_range=1)])
            #load_excel1(metric1,epoch+1)
            psnr += tensor_metric(clear,out, 'PSNR', data_range=1)
            ssim += tensor_metric(clear,out, 'SSIM', data_range=1)
            print('The %s Time: %.5f s.' % (files_clear[i][:-4], endtime1-starttime))

    return psnr / (len(files_clear)), ssim / (len(files_clear))


if __name__ == '__main__':
    main()
