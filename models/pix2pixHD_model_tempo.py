import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
from .asp_loss import AdaptiveSupervisedPatchNCELoss
import piq
import util.util as util

class Pix2PixHDModel(BaseModel):
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
    
    def name(self):
        return 'Pix2PixHDModel'
    
    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss,seg,tempo, NCE):
        flags = (True, use_gan_feat_loss, use_vgg_loss, True, True,seg, tempo, tempo, tempo, NCE)
        def loss_filter(g_gan, g_gan_feat, g_vgg, d_real, d_fake,g_seg, d_fake2, d_real2, g_gan2, g_NCE):
            return [l for (l,f) in zip((g_gan,g_gan_feat,g_vgg,d_real,d_fake,g_seg, d_fake2, d_real2, g_gan2, g_NCE),flags) if f]
        return loss_filter
    
    def data_dependent_initialize(self, A, B, C):
        #self.real_A = A
        #self.real_B = B
        self.forward(A, B, C)# compute fake images: G(A)
        if self.opt.isTrain:
            if self.opt.lambda_NCE > 0.0:
                params = self.netF.parameters()
                print(params)
                self.optimizer_F = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))

    
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain: # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.use_features = opt.instance_feat or opt.label_feat
        self.gen_features = self.use_features and not self.opt.load_features
        self.seg_loss = opt.seg_loss
        self.temp_loss = opt.temp_loss
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        ##### define networks        
        # Generator network
        netG_input_nc = input_nc        
        if not opt.no_instance:
            netG_input_nc += 1
        if self.use_features:
            netG_input_nc += opt.feat_num                  
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG, 
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, 
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids) 
        print("HWWWWWW")       
        self.netF = networks.define_F(opt.input_nc, opt.netF, gpu_ids=self.gpu_ids, opt=opt)

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + 3#opt.output_nc
            if not opt.no_instance:
                netD_input_nc += 1
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

        ### Encoder network
        if self.gen_features:          
            self.netE = networks.define_G(opt.output_nc, opt.feat_num, opt.nef, 'myencoder', 
                                          opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids)  
        if self.opt.verbose:
                print('---------- Networks initialized -------------')

        # load networks
        # if not self.isTrain or opt.continue_train or opt.load_pretrain:
        if opt.load_pretrain:
            # pretrained_path = '' if not self.isTrain else opt.load_pretrain
            pretrained_path = opt.load_pretrain
            print("pretrained_path: ", pretrained_path)
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)            
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)  
            if self.gen_features:
                self.load_network(self.netE, 'E', opt.which_epoch, pretrained_path)              

        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss,self.seg_loss, self.temp_loss, self.opt.lambda_NCE>0)
            # print(self.loss_filter)
            
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            
            self.criterionNCE = PatchNCELoss(opt)

            # self.ganloss_names = self.ganloss_filter('CE', 'BCE', 'BCEWithLogits', 'Dice', 'MSE')

            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:             
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)
            if opt.lambda_asp>0:
                self.criterionASP = AdaptiveSupervisedPatchNCELoss(self.opt)
            
            if self.seg_loss:
                # self.criterionSeg = networks.SegLoss(opt) 
                self.criterionSeg = networks.GradientVariance(patch_size=12)
                # self.criterionSeg = networks.GeneralizedSoftDiceLoss()
                # self.criterionSeg = networks.DiceLoss()
                # self.criterionseg = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor, loss_name='Dice')
            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_GAN','G_GAN_Feat','G_VGG','D_real', 'D_fake','G_seg','D_fake_temp', 'D_real_temp', 'G_GAN_temp', 'G_NCE')
            print(self.loss_names)

            # initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0:                
                import sys
                if sys.version_info >= (3,0):
                    finetune_list = set()
                else:
                    from sets import Set
                    finetune_list = Set()

                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():       
                    if key.startswith('model' + str(opt.n_local_enhancers)):                    
                        params += [value]
                        finetune_list.add(key.split('.')[0])  
                print('------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                print('The layers that are finetuned are ', sorted(finetune_list))                         
            else:
                params = list(self.netG.parameters())
            if self.gen_features:              
                params += list(self.netE.parameters())         
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, opt.beta2))                            

            # optimizer D                        
            params = list(self.netD.parameters())    
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, opt.beta2))


    def encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None, infer=False):             
        # if self.opt.label_nc == 0:
        input_label = label_map.data.cuda()
        
        # else:
        #     # create one-hot vector for label map 
        #     size = label_map.size()
        #     oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
        #     input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        #     input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
        #     if self.opt.data_type == 16:
        #         input_label = input_label.half()
        with torch.no_grad():
            input_label = Variable(input_label)

        # real images for training
        if inst_map is not None:
            inst_map = Variable(inst_map.data.cuda())

        if real_image is not None:
            real_image = Variable(real_image.data.cuda())

        # instance map for feature encoding
        if self.use_features and feat_map!=None:

            feat_map = feat_map.data.cuda()

        return input_label, inst_map, real_image, feat_map

    def discriminate(self, input_label, test_image, use_pool=False, tempo_map=None):

        # print(input_label.shape,test_image.shape)
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query, tempo_map)
        else:
            return self.netD.forward(input_concat, tempo_map)

    def forward(self, label, inst, image, tempo_map = None, feat=None, infer=False):
        # Encode Inputs
        input_label, inst_map, real_image, feat_map = self.encode_input(label, inst, image, feat)  

        # Fake Generation
        if self.use_features:
            #if not self.opt.load_features:
                 #feat_map = self.netE.forward(real_image, inst_map)   
                 #feat_map = self.netE.forward(inst_map)                  
            #input_concat = torch.cat((input_label, feat_map), dim=1)    
            input_concat = torch.cat((input_label, inst_map), dim=1)
            #print(input_concat.shape)
            #print(input_label.shape,feat_map.shape,input_concat.shape)        
        else:
            input_concat = input_label
            # input_concat = input_label +feat_map   
        # print(input_concat.shape)
        fake_image = self.netG.forward(input_concat)
        # print(fake_image.shape)

        # Fake Detection and Loss
        pred_fake_pool = self.discriminate(input_label, fake_image, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)  

        # Real Detection and Loss        
        pred_real = self.discriminate(input_label, real_image)
        loss_D_real = self.criterionGAN(pred_real, True)
        loss_D_fake_temp = 0
        loss_D_real_temp = 0
        loss_G_GAN_temp = 0
        if self.opt.temp_loss:
            # Fake Detection and Loss(masked)
            pred_fake_pool = self.discriminate(input_label, fake_image, use_pool=True, tempo_map=tempo_map)
            loss_D_fake_temp = self.criterionGAN(pred_fake_pool, False)
            # Real Detection and Loss        
            pred_real = self.discriminate(input_label, real_image, tempo_map=tempo_map)
            loss_D_real_temp = self.criterionGAN(pred_real, True)
            # GAN loss2
            pred_fake = self.netD.forward(torch.cat((input_label, fake_image), dim=1), tempo_map=tempo_map)
            loss_G_GAN_temp = self.criterionGAN(pred_fake, True)


        # GAN loss (Fake Passability Loss)        
        pred_fake = self.netD.forward(torch.cat((input_label, fake_image), dim=1))  

        loss_G_GAN = self.criterionGAN(pred_fake, True)     

        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
                   
        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.lambda_feat
            # print(real_image.shape,torch.stack((fake_image,fake_image,fake_image),dim=1).shape)
        if tempo_map!=None:
            feat_real_A = self.netG(torch.mul(torch.cat((input_label, input_label), dim=1), tempo_map), self.nce_layers, encode_only=True)
            feat_fake_B = self.netG(torch.mul(torch.cat((fake_image, fake_image), dim=1), tempo_map), self.nce_layers, encode_only=True)
            feat_real_B = self.netG(torch.mul(torch.cat((real_image, real_image), dim=1), tempo_map), self.nce_layers, encode_only=True)
        else:
            feat_real_A = self.netG(torch.cat((input_label, input_label), dim=1), self.nce_layers, encode_only=True)
            feat_fake_B = self.netG(torch.cat((fake_image, fake_image), dim=1), self.nce_layers, encode_only=True)
            feat_real_B = self.netG(torch.cat((real_image, real_image), dim=1), self.nce_layers, encode_only=True)
        
        if self.opt.lambda_NCE > 0.0:
            loss_NCE = self.calculate_NCE_loss(feat_real_A, feat_fake_B, self.netF)
        else:
            loss_NCE = 0.0
        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:  
            loss_NCE_Y = self.calculate_NCE_loss(feat_real_B, feat_fake_B, self.netF)
            loss_NCE_both = (loss_NCE + loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = loss_NCE


        # mmsim = piq.MultiScaleSSIMLoss() # generator
        # # print(fake_image.unique(),real_image.unique())
        # loss_mmsim = mmsim((fake_image+1)/2, (real_image+1)/2)
        loss_mmsim=0
        loss_G_seg = 0
        if self.seg_loss:
        #     loss_G_seg=self.Gsegloss(real_image[:,0,:,:]>=.5,fake_image[:,0,:,:]>=.5)
        #     loss_G_seg+=self.Gsegloss(real_image[:,2,:,:]>=.5,fake_image[:,2,:,:]>=.5)
        #     loss_D_seg_real =0
        # else:
        #     loss_G_seg= 0
        #     loss_D_seg_real = 0
                

            loss_G_seg = self.criterionSeg(fake_image, real_image)

            # loss_G_seg1 = self.criterionSeg((fake_image>torch.quantile(fake_image,.25)).float(), (real_image>torch.quantile(real_image,.25)).float())
            # loss_G_seg2 = self.criterionSeg((fake_image>torch.quantile(fake_image,.5)).float(), (real_image>torch.quantile(real_image,.5)).float())
            # loss_G_seg3 = self.criterionSeg((fake_image>torch.quantile(fake_image,.75)).float(), (real_image>torch.quantile(real_image,.75)).float())
            # loss_G_seg =loss_G_seg1+loss_G_seg2+loss_G_seg3
        # Only return the fake_B image if necessary to save BW
        return [ self.loss_filter( loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake, loss_G_seg, loss_D_fake_temp, loss_D_real_temp, loss_G_GAN_temp, loss_NCE_both ), None if not infer else fake_image ]

        # Only return the fake_B image if necessary to save BW
        # return [ self.loss_filter( loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake , loss_mmsim, loss_D_seg_real, loss_D_seg_fake), None if not infer else fake_image ]
        # return [ self.loss_filter( loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake , loss_mmsim, loss_G_seg,0), None if not infer else fake_image ]
    def inference(self, label, inst, image=None):
        # Encode Inputs        
        image = Variable(image) if image is not None else None
        input_label, inst_map, real_image, _ = self.encode_input(label, inst, image, infer=True)  
        #input_label, inst_map, real_image, _ = self.encode_input(Variable(label), Variable(inst), image, infer=True)

        # Fake Generation
        if self.use_features:
            #if self.opt.use_encoded_image:
                # encode the real image to get feature map
                #feat_map = self.netE.forward(real_image, inst_map)
                #feat_map = self.netE.forward(inst_map)
            #else:
                # sample clusters from precomputed features             
            #feat_map = self.sample_features(inst_map)
            #input_concat = torch.cat((input_label, feat_map), dim=1)
            input_concat = torch.cat((input_label, inst_map), dim=1)
        else:
            input_concat = input_label        
           
        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                fake_image = self.netG.forward(input_concat)
        else:
            fake_image = self.netG.forward(input_concat)
        return fake_image

    def sample_features(self, inst): 
        # read precomputed feature clusters 
        cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)        
        features_clustered = np.load(cluster_path, encoding='latin1').item()

        # randomly sample from the feature clusters
        inst_np = inst.cpu().numpy().astype(int)                                      
        feat_map = self.Tensor(inst.size()[0], self.opt.feat_num, inst.size()[2], inst.size()[3])
        for i in np.unique(inst_np):    
            label = i if i < 1000 else i//1000
            if label in features_clustered:
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0]) 
                                            
                idx = (inst == int(i)).nonzero()
                for k in range(self.opt.feat_num):                                    
                    feat_map[idx[:,0], idx[:,1] + k, idx[:,2], idx[:,3]] = feat[cluster_idx, k]
        if self.opt.data_type==16:
            feat_map = feat_map.half()
        return feat_map

    def encode_features(self, image, inst):
        image = Variable(image.cuda(), volatile=True)
        feat_num = self.opt.feat_num
        h, w = inst.size()[2], inst.size()[3]
        block_num = 32
        feat_map = self.netE.forward(image, inst.cuda())
        inst_np = inst.cpu().numpy().astype(int)
        feature = {}
        for i in range(self.opt.label_nc):
            feature[i] = np.zeros((0, feat_num+1))
        for i in np.unique(inst_np):
            label = i if i < 1000 else i//1000
            idx = (inst == int(i)).nonzero()
            num = idx.size()[0]
            idx = idx[num//2,:]
            val = np.zeros((1, feat_num+1))                        
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]            
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)
        return feature

    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        if self.opt.data_type==16:
            return edge.half()
        else:
            return edge.float()

    def save(self, which_epoch,saveD):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        if saveD:
            print('save D')
            self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())           
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd        
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.lambda_NCE>0.:
            for param_group in self.optimizer_F.param_groups:
                param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
    '''
    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)
        feat_q = self.netG(tgt)
        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss/n_layers
        '''
    def calculate_NCE_loss(self, feat_src, feat_tgt, netF, paired=False):
        n_layers = len(feat_src)
        feat_q = feat_tgt

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]
        feat_k = feat_src
        feat_k_pool, sample_ids = netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k in zip(feat_q_pool, feat_k_pool):
            if paired:
                loss = self.criterionASP(f_q, f_k, self.current_epoch) * self.opt.lambda_asp
            else:
                loss = self.criterionNCE(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers
        
    


class InferenceModel(Pix2PixHDModel):
    def forward(self, inp):
        label, inst = inp
        return self.inference(label, inst)

        
