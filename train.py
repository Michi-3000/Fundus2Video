import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from subprocess import call
import fractions
import cv2
import random
def lcm(a,b): return abs(a * b)/fractions.gcd(a,b) if a and b else 0

from options.train_options import TrainOptions

from data.data_loader import CreateDataLoader
from models.models_tempo import create_model
import util.util as util
from util.visualizer import Visualizer
from datetime import datetime
import pandas as pd
from options.test_options import TestOptions
import imageio
from show import *
import sys
from metrics.calculate_fvd import calculate_fvd
from metrics.calculate_psnr import calculate_psnr
from metrics.calculate_ssim import calculate_ssim
from metrics.calculate_lpips import calculate_lpips
import csv

def splitdf(df,col,test_size=.2):
    from sklearn.model_selection import GroupShuffleSplit
    train_inds, test_inds = next(GroupShuffleSplit(test_size=test_size, n_splits=2, random_state = 7).split(df, groups=df[col]))
    train = df.iloc[train_inds].reset_index(drop=True)
    test = df.iloc[test_inds].reset_index(drop=True)
    return train,test

def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--lambda_asp', type=float, default=0., help='weight for ASP_NCE loss')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

opt = TrainOptions().parse()
opt.resize_or_crop = ''
opt.batchSize = 1
opt.niter=10
opt.niter_decay=40
opt.nThreads = 1
opt.name = 'fundus2video_tempo'
opt.load_pretrain=None
opt.seg_loss = 1
opt.temp_loss = 1
opt.lambda_NCE = 1
opt.continue_train=0
saveD=False
opt.B='fa_rigid' #The column name of the FA series
opt.save_epoch_freq=1
opt.print_freq = 100
opt.instance_feat = True
opt.feat_num = 3
opt.load_features = True
saveval_freq = 5

f='./data.csv'# The path you save the training dataframe
f='/home/healgoo/risk_factor/vessel/generation/super/pix2pixHD/tasks/cross_modality/FA/video/co2favideo_dis.csv'
df=pd.read_csv(f)
df=df[~df.Phase.isin(['A','C'])]
df.loc[df.Phase=='AV','Phase']='V'
print(len(df))
sdf = df.drop_duplicates(subset='orgid')
df = df[df['orgid'].isin(sdf['orgid'].to_list())]
TRAIN,TEST=splitdf(df,'orgid',.1)
VALID,TEST=splitdf(TEST,'orgid',1/2)

opt.display_freq = opt.print_freq
opt.w=512
   
now = datetime.now()
opt.df=TRAIN

opttest = TestOptions().parse(save=False)
opttest.df=VALID
opttest.nThreads = 1   # test code only supports nThreads = 1
opttest.serial_batches = True  # no shuffle
opttest.no_flip = True  # no flip
opttest.name = opt.name
opttest.no_flip = True 
opttest.w=opt.w
opttest.A=opt.A
opttest.B=opt.B

opttest.instance_feat = True
opttest.feat_num = 3
opttest.load_features = True

data_loader = CreateDataLoader(opt)
test_loader = CreateDataLoader(opttest)
opt.name = opt.name+'/maskNCE_'+now.strftime("%m-%d-%H%M")+'_'+str(opt.w)#+'_'+opt.which_epoch
visualizer = Visualizer(opt)
TEST.to_csv(os.path.join(opt.checkpoints_dir, opt.name, 'test.csv'))
VALID.to_csv(os.path.join(opt.checkpoints_dir, opt.name, 'val.csv'))
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
else:    
    start_epoch, epoch_iter = 1, 0

opt.print_freq = lcm(opt.print_freq, opt.batchSize)    
traindataset = data_loader.load_data()
test_dataset = test_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D

total_steps = (start_epoch-1) * dataset_size + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

best_auc=.5
best_f1=.4
test_results =pd. DataFrame()
for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(traindataset, start=epoch_iter):

        if i==0 and epoch==start_epoch:
            model.module.data_dependent_initialize(Variable(data['label']), Variable(data['label']), Variable(data['image'][:,1,:,:,:].squeeze(dim=1)))
            if opt.lambda_NCE>0.:
                optimizer_F = model.module.optimizer_F

        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        save_fake = total_steps % opt.display_freq == display_delta

        ############## Forward Pass ######################
        B,N,C,W,H = Variable(data['image']).size()
        #print(B,N,C,W,H)
        
        reals = []
        lbs = []
        gens = torch.zeros(N-1, 3, opt.w, opt.w, requires_grad=False).to(torch.device("cuda"))
        mers = []
        nums = [0]*(N-1)
        images = data['image']

        label = data['label']

        total_loss_G=0
        total_loss_D=0
        

        for n in range(N-1):
            output = images[:, [n+1, min(n+2, N-1), min(n+3, N-1)], 1, :, :]
            inst = label
            if n > 0:
                if total_steps<1000:
                    aug=random.choice([0,1]+[1]*(total_steps//100))
                else:
                    aug=1

                if aug:
                    inst = generated
                else:
                    inst = images[:, [n, min(n+1, N-1), min(n+2, N-1)], 1, :, :]
            losses, generated = model(Variable(label), Variable(inst), 
                    Variable(output), tempo_map=Variable(data['tempo_map']), infer=1)
            #fundus 
            # sum per device losses
            losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
            loss_dict = dict(zip(model.module.loss_names, losses))

            # calculate final loss scalar
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            if opt.temp_loss:
                loss_D += (loss_dict['D_fake_temp'] + loss_dict['D_real_temp']) * 0.5
            if opt.seg_loss:
                loss_G = loss_dict['G_GAN']*2 + loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0) +loss_dict['G_seg']*5
                if opt.lambda_NCE:
                    loss_G += loss_dict['G_NCE']
                if opt.temp_loss:
                    loss_G += loss_dict['G_GAN_temp']
            else:
                loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0)
            optimizer_G.zero_grad()
            loss_G.backward()          
            optimizer_G.step()
            if opt.lambda_NCE>0. and opt.netF=='mlp_sample':
                optimizer_F.zero_grad()
                optimizer_F.step()

            # update discriminator weights
            optimizer_D.zero_grad()
            loss_D.backward()        
            optimizer_D.step()


            if save_fake and generated!=None:
                lb = util.tensor2im(label[0])
                real = util.tensor2im(torch.stack([output[0][0,:,:]]*3, dim=0))
                for k in range(3):
                    if n+k >= N-1:
                        continue
                    gens[n+k,:,:,:] += torch.stack([generated.data[0][k,:,:]]*3, dim=0)
                    nums[n+k]+=1
                lbs.append(lb)
                reals.append(real)
        ############### Backward Pass ####################

        if save_fake:
            print('save fake')
            for n in range(N-1):
                gens[n,:,:,:]=gens[n,:,:,:]/nums[n]
                mer = np.hstack([lbs[n],reals[n],util.tensor2im(gens[n,:,:,:])])
                mers.append(mer)

            imageio.mimsave(os.path.join(opt.checkpoints_dir,opt.name,'web/images',"epoch"+str(epoch)+"_"+data['path'][0]+'.gif'), mers, fps=4)
            cv2.imwrite(os.path.join(opt.checkpoints_dir,opt.name,'web/images', data['path'][0]+'.png'), data['tempo_map'].squeeze().numpy()*255.)

        if total_steps % opt.print_freq == print_delta:
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}        
            errors.update({'loss_D':loss_D,'loss_G':loss_G})    
            t = (time.time() - iter_start_time) / opt.print_freq
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)
        ### display output images

        if epoch_iter >= dataset_size:
            break
    
    # end of epoch 
    # iter_end_time = time.time()
    t=time.time() - epoch_start_time
    print('End of epoch %d / %d \t Time Taken: %d sec' %
        (epoch, opt.niter + opt.niter_decay, t))

    ################# validation
    with torch.no_grad():
        for i, data in enumerate(test_dataset, start=epoch_iter):
            save_val = i % saveval_freq ==0
            # save_val=1
            B,N,C,W,H = Variable(data['image']).size()
            reals=[]
            lbs = []
            gens=torch.zeros(N-1, 3, opt.w, opt.w, requires_grad=False).to(torch.device("cuda"))
            nums = [0]*(N-1)
            mers=[]
            images = data['image']

            label = data['label']
            if save_val:
                lb = util.tensor2im(label[0])
            videos1 = torch.zeros(1, N-1, 3, opt.w, opt.w, requires_grad=False)
            videos2 = torch.zeros(1, N-1, 3, opt.w, opt.w, requires_grad=False)
            for n in range(N-1):
                output = images[:, [n+1, min(n+2, N-1), min(n+3, N-1)], 1, :, :]
                inst = label
                if n > 0:
                    inst = generated
                losses, generated = model(Variable(label), Variable(inst), 
                        Variable(output), infer=1)
                if save_val:
                    real = util.tensor2im(torch.stack([output[0][0,:,:]]*3, dim=0))
                    reals.append(real)
                    lbs.append(lb)
                for k in range(3):
                    if n+k >= N-1:
                        continue
                    gens[n+k,:,:,:] += torch.stack([generated.data[0][k,:,:]]*3, dim=0)
                    nums[n+k]+=1                
                videos1[:,n,:,:,:] = images[:,n+1,:,:,:]
            for n in range(N-1):
                gens[n,:,:,:]=gens[n,:,:,:]/nums[n]
                videos2[:,n,:,:,:] = gens[n,:,:,:]
                if save_val:
                    mer = np.hstack([lbs[n],reals[n],util.tensor2im(gens[n,:,:,:])])
                    mers.append(mer)
            if save_val:
                imageio.mimsave(os.path.join(opt.checkpoints_dir,opt.name,'web/images',"epoch"+str(epoch)+"_val_"+data['path'][0]+'.gif'), mers, fps=4)
                cv2.imwrite(os.path.join(opt.checkpoints_dir,opt.name,'web/images', data['path'][0]+'.png'), data['tempo_map'].squeeze().numpy()*255.)
            result={}
            result['ep']=epoch
            result['filename']=data['path'][0]
            fvd = list(calculate_fvd(videos1, videos2, torch.device("cuda"), method='styleganv')["value"].values())
            result['fvd'] = np.nanmean(fvd)
            result['ssim'] = np.nanmean(list(calculate_ssim(videos1, videos2)["value"].values()))
            result['psnr'] = np.nanmean(list(calculate_psnr(videos1, videos2)["value"].values()))
            result['lpips'] = np.nanmean(list(calculate_lpips(videos1, videos2, torch.device("cuda"))["value"].values()))

            print(result) 
            test_results = pd.concat([test_results,pd.DataFrame(result,index=[0])])
            with open(os.path.join(opt.checkpoints_dir, opt.name, "val.csv"), mode='a') as cf:
                dw = csv.DictWriter(cf, fieldnames=result.keys())
                if epoch==1:
                    dw.writeheader()
                dw.writerow(result)   

    test_results.groupby('ep')[['fvd','ssim','psnr','lpips']].agg(['mean']).to_csv(os.path.join(opt.checkpoints_dir,opt.name,'val_summary.csv'))
    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
        model.module.save(epoch,saveD)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

    ### instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        model.module.update_fixed_params()
    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.module.update_learning_rate()
