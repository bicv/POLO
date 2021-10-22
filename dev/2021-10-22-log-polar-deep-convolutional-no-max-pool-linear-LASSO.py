#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pickle


# In[2]:


from PIL import Image


# In[3]:


from skimage.color import rgb2hsv, rgb2lab, hsv2rgb, lab2rgb
from matplotlib.colors import hsv_to_rgb


# In[4]:


from LogGabor import LogGabor

from PYramid2 import cropped_pyramid, local_filter, get_K, log_gabor_transform
from PYramid2 import inverse_pyramid, get_K_inv, inverse_gabor


# In[5]:


width = 32
n_levels = 7 #int(np.log(np.max((N_X, N_Y))/width)/np.log(base_levels)) + 1
base_levels = 2
n_color = 3
r_min = width / 8 # width / 16 
r_max = width / 2 # 7 * width / 16
n_sublevel = n_eccentricity = 4
n_azimuth = 16
n_theta = 8
n_phase = 1


# In[6]:


phase_shift = False


# In[7]:


pyramid_n_params = width*width*n_color*n_levels
print('pyramids #params :', pyramid_n_params)
logpolar_n_params = n_levels * n_color * n_eccentricity * n_azimuth * n_theta * n_phase
print('logpolar #params :', logpolar_n_params)


# In[8]:


out_chan=32
gauss = False
do_mask = False
color = True
color_mode= 'lab' # 'hsv' #True
print ('encoder #params :', out_chan*n_color*n_levels)


# In[9]:


if gauss:
    script_name = '2021-10-22-log-polar-deep-convolutional-no-max-pool-no-relu-gauss-'+color_mode
else:
    script_name = '2021-10-22-log-polar-deep-convolutional-no-max-pool-no-relu-laplace-'+color_mode


# ### Image utilities
def tensor_pyramid_display(img_pyr_tens, bias = 0):
    fig, axs = plt.subplots(1, n_levels, figsize=(20,20))
    img_aff = img_pyr_tens.permute(0,1,3,4,2).detach().numpy()
    for i_level, ax in enumerate(axs):
        ax.imshow((img_aff[0, i_level, ...]+bias).clip(0,255).astype('uint8'))
        ax.plot([width/2], [width/2], 'r+', ms=32);
    #print('Tensor shape=', img_rec.shape) 
    return axs
# In[10]:


def tensor_pyramid_display(img_pyr_tens, global_bias = 0):
    fig, axs = plt.subplots(1, n_levels, figsize=(20,20))
    img_aff = img_pyr_tens.permute(0,1,3,4,2).detach().numpy()
    for i_level, ax in enumerate(axs):
        if i_level < n_levels-1 and not gauss:
            bias = 128
        else:
            bias = global_bias
        ax.imshow((img_aff[0, i_level, ...]+bias).clip(0,255).astype('uint8'))
        ax.plot([width/2], [width/2], 'r+', ms=32);
    #print('Tensor shape=', img_rec.shape) 
    return axs


# In[11]:


def tensor_image_cmp(img_tens_ref, img_tens_rec):
    fig, ax = plt.subplots(1, 2, figsize=(20,10))
    img_aff_ref = img_tens_ref.detach().permute(0,2,3,1).squeeze().detach().numpy().clip(0,255).astype('uint8')
    ax[0].imshow(img_aff_ref)
    N_X, N_Y, _ = img_aff_ref.shape
    ax[0].plot([N_Y//2], [N_X//2], 'r+', ms=16)
    ax[0].set_title('LOG GABOR RECONSTRUCTION')
    img_aff_rec = img_tens_rec.detach().permute(0,2,3,1).squeeze().detach().numpy().clip(0,255).astype('uint8')
    ax[1].imshow(img_aff_rec)
    ax[1].plot([N_Y//2], [N_X//2], 'r+', ms=16)
    ax[1].set_title('AUTO-ENCODER RECONSTRUCTION')
    return ax


# In[12]:


def image_show(im, color_mode):
    if color_mode=='hsv':
        plt.imshow(hsv2rgb(im))
    elif color_mode=='lab':
        plt.imshow(lab2rgb(im))
    else:
        full_img_rec = im.clip(0,255).astype('uint8')
        plt.imshow(im)


# ### Log Gabor filters

# In[13]:


pe = {'N_X': width, 'N_Y': width, 'do_mask': do_mask, 'base_levels':
          base_levels, 'n_theta': 0, 'B_sf': np.inf, 'B_theta': np.inf ,
      'use_cache': True, 'figpath': 'results', 'edgefigpath':
          'results/edges', 'matpath': 'cache_dir', 'edgematpath':
          'cache_dir/edges', 'datapath': 'database/', 'ext': '.pdf', 'figsize':
          14.0, 'formats': ['pdf', 'png', 'jpg'], 'dpi': 450, 'verbose': 0}   

lg = LogGabor(pe)

print('lg shape=', lg.pe.N_X, lg.pe.N_Y)


# In[14]:


lg.pe


# In[15]:


K = get_K(width=width,
          n_sublevel = n_sublevel, 
          n_azimuth = n_azimuth, 
          n_theta = n_theta,
          n_phase = n_phase, 
          r_min = r_min, 
          r_max = r_max, 
          log_density_ratio = 2, 
          verbose=True,
          phase_shift=phase_shift,
          lg=lg)

K_mean= torch.norm(K, dim=4).unsqueeze(4)K = torch.cat((K, K_mean), dim=4) 
n_theta = 9K = K_mean
n_theta = 1plt.plot(K.flatten())
# ### Gabor filters pseudo-inverse

# In[16]:


K_inv = get_K_inv(K, width=width, n_sublevel = n_sublevel, n_azimuth = n_azimuth, n_theta = n_theta, n_phase = n_phase)
plt.plot(K_inv.flatten())


# In[17]:


### regularized inverse gabor
K_ = K.reshape((width**2, n_sublevel*n_azimuth*n_theta*n_phase))
print('Reshaped filter tensor=', K_.shape)
K_inv_rcond = torch.pinverse(K_, rcond=0.1) 
print('Tensor shape=', K_inv.shape)
K_inv_rcond =K_inv_rcond.reshape(n_sublevel, n_azimuth, n_theta, n_phase, width, width)

plt.figure()
plt.plot(K_inv_rcond.flatten())


# ### Honeycomb space coverage tests

# In[18]:


plt.figure(figsize=(20,3))
for i_theta in range(n_theta):
    coefs = torch.zeros((n_sublevel, n_azimuth, n_theta, n_phase))
    coefs[0, n_azimuth//2, i_theta, 0] = 1
    img_dis = torch.tensordot(K, coefs, dims=4)
    plt.subplot(1,n_theta,i_theta+1)
    plt.imshow(img_dis.numpy()[:, :, ...], cmap='gray')


# In[19]:


plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
coefs = torch.zeros((n_sublevel, n_azimuth, n_theta, n_phase))
coefs[0, 4, 0, 0] = 1
img_dis = torch.tensordot(K, coefs, dims=4)
plt.imshow(img_dis.numpy()[:, :, ...], cmap='gray')
if n_phase >1:
    plt.subplot(1,2,2)
    coefs = torch.zeros((n_sublevel, n_azimuth, n_theta, n_phase))
    coefs[0, 4, 0, 1] = 1
    img_dis = torch.tensordot(K, coefs, dims=4)
    plt.imshow(img_dis.numpy()[:, :, ...], cmap='gray')


# In[20]:


plt.figure(figsize=(20,6))
for i_az in range(n_azimuth):
    coefs = torch.zeros((n_sublevel, n_azimuth, n_theta, n_phase))
    coefs[:, i_az, 0, 0] = 1
    img_dis = torch.tensordot(K, coefs, dims=4)
    plt.subplot(2,n_azimuth//2,i_az+1)
    plt.imshow(img_dis.numpy()[:, :, ...], cmap='gray')


# In[21]:


coefs = torch.zeros((n_sublevel, n_azimuth, n_theta, n_phase))
coefs[:, :, :1, 0] = torch.ones((n_sublevel, n_azimuth, 1))
img_dis = torch.tensordot(K, coefs, dims=4)
plt.subplot(1,2,1)
plt.imshow(img_dis.numpy(), cmap='gray')
plt.subplot(1,2,2)
_=plt.plot(img_dis.numpy())


# In[22]:


if n_phase>1:
    coefs = torch.zeros((n_sublevel, n_azimuth, n_theta, n_phase))
    coefs[:, :, :1, 1] = torch.ones((n_sublevel, n_azimuth, 1))
    img_dis = torch.tensordot(K, coefs, dims=4)
    plt.subplot(1,2,1)
    plt.imshow(img_dis.numpy(), cmap='gray')
    plt.subplot(1,2,2)
    _=plt.plot(img_dis.numpy())


# In[23]:


coefs = torch.zeros((n_sublevel, n_azimuth, n_theta, n_phase))
coefs[:, :, 2:3, 0] = torch.ones((n_sublevel, n_azimuth, 1))
img_dis = torch.tensordot(K, coefs, dims=4)
plt.subplot(1,2,1)
plt.imshow(img_dis.numpy(), cmap='gray')
plt.subplot(1,2,2)
_=plt.plot(img_dis.numpy())


# ## Images dataset + transforms

# In[ ]:


if True: #not os.path.exists("image_names.txt"):
    names = open("image_names.txt", "w")
    img_names = os.listdir('../ALLSTIMULI')
    print('EXCLUDED:')
    for i in range(len(img_names)):
        if 'Data1' in img_names[i] or 'Data2' in img_names[i]             or 'Data3' in img_names[i] or 'Data4' in img_names[i]             or 'DS_' in img_names[i] or len(img_names[i])<=10 :
            #or '2218506905' in img_names[i] or 'i24622350' in img_names[i]:
            print(img_names[i])
        else:
            names.write(img_names[i][:-5]+'\n')
    names.close()
    
names = open("image_names.txt", "r")
img_names = names.readlines()
for i in range(len(img_names)):
    img_names[i]=img_names[i][:-1]


# In[ ]:


dir_names = os.listdir('../saccades-data')
loc_data_xy={}
for dir_name in dir_names:
    loc_data_xy[dir_name]={}
    for name in img_names:
        locpath = '../saccades-data/' + dir_name + '/' + name
        f = open(locpath,'rb')
        loc_dict = pickle.load(f)
        loc_data_xy[dir_name][name] = np.array(loc_dict['barycenters'])


# In[ ]:


def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    #plt.pause(0.001)  # pause a bit so that plots are updated


# # Dataset class

# In[ ]:


class SaccadeLandmarksDataset(Dataset):
    """Saccade Landmarks dataset."""

    def __init__(self, loc_dict, img_dir, img_names, dir_names, transform=None, color_mode='rgb'):
        """
        Args:
            loc_dict (dict): Dictonary containing saccade coordinates
            img_dir (string): Directory with all the images.
            img_names (lost): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.loc_dict = loc_dict
        self.img_dir = img_dir
        self.img_names = img_names
        self.dir_names = dir_names
        self.transform = transform
        self.color_mode=color_mode

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx, color_mode='rgb'):

        #img_name = os.listdir(self.img_dir)[idx+2]
        name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, name + '.jpeg')
        image = io.imread(img_path)
        #if self.color_mode == 'hsv':
        #    image = rgb2hsv(image)
        #elif self.color_mode == 'lab':
        #    image = rgb2lab(image)
        #name = img_name[:-5]
        dir_name = np.random.choice(self.dir_names)
        # HACK!!
        #dir_name = self.dir_names[0]
        landmarks = self.loc_dict[dir_name][name]
        landmarks = np.array([landmarks])
        landmarks = landmarks.reshape(-1, 2) #.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks, 'name':name}

        if self.transform:
            sample = self.transform(sample)

        return sample


# # Transforms

# In[ ]:


class RandomSaccadeTo(object):
    # TODO : zero_fill option
    def __init__(self, zero_fill = False):
        self.zero_fill = zero_fill
        
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        N_X, N_Y = image.shape[:2]
        try:
            nb_sac = len(landmarks)
            sac_num =  np.random.randint(nb_sac)
            sac = landmarks[sac_num]
            #sac = np.random.choice(landmarks)
        except:
            print("landmarks", landmarks, "image", sample['name'])
            sac = (N_Y//2, N_X//2)
        
        #img_color_sac = saccade_to(image, (N_X//2, N_Y//2), (sac[1], sac[0]))
        image_roll = np.copy(image)
        image_roll=np.roll(image_roll, N_X//2 - sac[1], axis=0)
        if self.zero_fill:
            shift = N_X//2 - sac[1]
            if shift > 0:
                image_roll[:shift,:,:] = 0
            elif shift < 0:
                image_roll[shift:,:,:] = 0
        image_roll=np.roll(image_roll, N_Y//2 - sac[0], axis=1)
        if self.zero_fill:
            shift = N_Y//2 - sac[0]
            if shift > 0:
                image_roll[:,:shift,:] = 0
            elif shift < 0:
                image_roll[:,shift:,:] = 0
        return {'image':image_roll, 'pos':sac, 'name':sample['name']}


# In[ ]:


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image_tens = sample['image'].transpose((2, 0, 1))
        return {'image': torch.FloatTensor(image_tens), 'pos': sample['pos'],  'name':sample['name']}


# ### Adapted cropped pyramid (squeezed tensor)

# In[ ]:


class CroppedPyramid(object):
    def __init__(self, width, 
                 base_levels, 
                 color=color, 
                 do_mask=do_mask, 
                 verbose=False, 
                 n_levels=None, 
                 color_mode='rgb'):
        self.width = width
        self.base_levels = base_levels
        self.color = color
        self.do_mask = do_mask
        self.verbose = verbose
        self.n_levels = n_levels
        self.color_mode = color_mode
    
    def __call__(self, sample):
        img_crop, level_size = cropped_pyramid(sample['image'].unsqueeze(0), 
                                               width=self.width, 
                                               base_levels=self.base_levels,
                                               color=self.color, 
                                               do_mask=self.do_mask, 
                                               verbose=self.verbose,
                                               squeeze=True,
                                               gauss=gauss,
                                               n_levels=self.n_levels,
                                               color_mode=self.color_mode)
        #img_crop[:,-1,...]-=128 # on residual (!!)
        return{'img_crop':img_crop, 'level_size':level_size, 'pos':sample['pos'],  'name':sample['name']}
        
    


# ### LogGaborTransform

# In[ ]:


class LogGaborTransform(object):
    def __init__(self, K=K, color=color, verbose=False):
        self.K = K
        self.color = color
        self.verbose = verbose
    
    def __call__(self, sample):
        log_gabor_coeffs = log_gabor_transform(sample['img_crop'].unsqueeze(0), K)
        
        return{'img_gabor':log_gabor_coeffs, 'pos':sample['pos'],  'name':sample['name']}


# ### ComplexModulus

# # Compose transforms
# ### transforms.Compose

# In[ ]:


composed_transform = transforms.Compose([RandomSaccadeTo(zero_fill=True),
                               ToTensor(),
                               CroppedPyramid(width, 
                                              base_levels, 
                                              n_levels=n_levels,
                                              color_mode=color_mode)]) #, LogGaborTransform()])


# In[ ]:


saccade_dataset = SaccadeLandmarksDataset(loc_dict=loc_data_xy,
                                          img_dir='../ALLSTIMULI/',
                                          img_names=img_names,
                                          dir_names =  dir_names,
                                          transform=composed_transform,
                                          color_mode=color_mode)


# # Iterating through the dataset

# In[ ]:


# Helper function to show a batch

'''def tensor_hsv_to_rgb(images_batch):
    n_batch, n_levels = images_batch.shape[:2]
    for batch in range(n_batch):
        for level in range(n_levels):
            im_hsv = 
'''
def show_landmarks_batch(sample_batched, color_mode='rgb'):
    """Show image with landmarks for a batch of samples."""
    for level in range(n_levels-1,0,-1):
        plt.figure()
        images_batch = torch.clone(sample_batched['img_crop'][:,level,:,:,:])
        if level < n_levels-1 and not gauss:
            images_batch+=128  
        batch_size = len(images_batch)
        im_size = images_batch.size(2)
        grid_border_size = 2

        grid = utils.make_grid(images_batch)
        #grid += 128
        plt.imshow(grid.numpy().transpose((1, 2, 0)).clip(0,255).astype('uint8'))

        plt.title('Batch from dataloader, level=' + str(level))


# In[ ]:


batch_size = 4
dataloader = DataLoader(saccade_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)
for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['img_crop'].size())
    print(i_batch, sample_batched['name'])
    if i_batch ==3 :
        plt.figure()
        show_landmarks_batch(sample_batched)  
        break
        


# In[ ]:


plt.plot(sample_batched['img_crop'].flatten())


# In[ ]:


full_img_rec = inverse_pyramid(sample_batched['img_crop'], 
                               color=color, 
                               gauss=gauss, 
                               n_levels=n_levels)
if color_mode=='hsv' or color_mode=='lab':
    im = full_img_rec.detach().permute(0,2,3,1).numpy()
else:
    im = full_img_rec.detach().permute(0,2,3,1).numpy().clip(0,255).astype('uint8')
for num_batch in range(batch_size):
    plt.figure(figsize=(20,15))
    if color_mode == 'hsv':
        plt.imshow(hsv2rgb(im[num_batch,:]))
    elif color_mode == 'lab':
        plt.imshow(lab2rgb(im[num_batch,:]))
    else:
        plt.imshow(im[num_batch,:])


# In[ ]:


plt.plot(sample_batched['img_crop'][0,:,0,:,:].flatten())


# In[ ]:


sample_batched['name']
locpath = '../ALLSTIMULI/' + sample_batched['name'][2] + '.jpeg'
img_orig = Image.open(locpath) 
plt.imshow(img_orig)


# ### Autoencoder

# In[ ]:


class AutoEncoder(nn.Module):
    def __init__(self, n_levels, n_color, n_eccentricity, n_azimuth, n_theta, n_phase,
                 encoder_l=None, decoder_l=None):
        super(AutoEncoder, self).__init__()
        self.n_levels = n_levels
        self.n_color = n_color
        self.n_eccentricity = n_eccentricity 
        self.n_azimuth = n_azimuth 
        self.n_theta = n_theta
        self.n_phase = n_phase
        
        if encoder_l is None:
            self.encoder_l = Encoder(n_levels, n_color, n_eccentricity, n_azimuth, n_theta, n_phase)
        else:
            self.encoder_l = encoder_l
        
        self.h_size = n_levels * n_azimuth//4 * 512     
        
        if decoder_l is None:              
            self.decoder_l = Decoder(n_levels, n_color, n_eccentricity, n_azimuth, n_theta, n_phase)    
        else:
            self.decoder_l = decoder_l
        
    def forward(self, x, z_in=None): #, **kargs):   
        
        code_l, z = self.encoder_l(x)
        
        if z_in is not None:
            code_l = z_in        
        
        return self.decoder_l(code_l), z

autoenc_VAE = AutoEncoder(n_levels, n_color, n_eccentricity, n_azimuth, n_theta, n_phase, out_chan, 
                         encoder_l = autoenc.encoder_l,
                         decoder_l = autoenc.decoder_l)
# In[ ]:


class Encoder(nn.Module):
    """ Encoder
    """
    def __init__(self, n_levels, n_color, n_eccentricity, n_azimuth, n_theta, n_phase):
        super(Encoder, self).__init__()
        self.n_levels = n_levels
        self.n_color = n_color
        self.n_eccentricity = n_eccentricity 
        self.n_azimuth = n_azimuth 
        self.n_theta = n_theta
        self.n_phase = n_phase
        # Layers
        self.conv1 = nn.Conv2d(  n_color * n_phase * n_theta, 
                                 64, 
                                 kernel_size = (3,3), 
                                 stride = (2,2),
                                 padding = (1,1),
                                 bias=False)     
        self.conv2 = nn.Conv2d(  64, 
                                 128, 
                                 kernel_size = (3,3), 
                                 stride = (2,2), 
                                 padding = (1,1),
                                 bias=False) #,
        self.conv3 = nn.Conv2d(  128, 
                                 256, 
                                 kernel_size = (1,4), 
                                 stride = (1,1), 
                                 padding = (0,0)) #,
            
            
    def forward(self, x): 
        x = x.permute(0, 2, 5, 6, 1, 3, 4).contiguous()
        x = x.view(-1, self.n_color*self.n_theta*self.n_phase, self.n_levels * self.n_eccentricity, self.n_azimuth)
        z = []
        x = self.conv1(x)
        z += [x]
        x = self.conv2(x)
        z += [x]   
        x = self.conv3(x)
        z += [x] 
        return x, z


# In[ ]:


out_chan = 30
enc = Encoder(n_levels - 1, n_color, n_eccentricity, n_azimuth, n_theta, n_phase)
dataloader = DataLoader(saccade_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)
data = next(iter(dataloader))  
log_gabor_coefs = log_gabor_transform(data['img_crop'], K)
autoenc_inputs = log_gabor_coefs[:,:n_levels-1,...].clone()
#autoenc_inputs /=  256 # !! Normalization
code, z = enc(autoenc_inputs)


# In[ ]:


plt.plot(z[0][0,:].detach().numpy().flatten())
plt.plot(z[1][0,:].detach().numpy().flatten())
plt.plot(code[0,:].detach().numpy().flatten())


# In[ ]:


z[0].shape, z[1].shape, z[2].shape


# In[ ]:


print(code.shape)
del enc


# In[ ]:


class Decoder(nn.Module):
    """ Encoder
    """
    def __init__(self, n_levels, n_color, n_eccentricity, n_azimuth, n_theta, n_phase):
        super(Decoder, self).__init__()
        self.n_levels = n_levels
        self.n_color = n_color
        self.n_eccentricity = n_eccentricity 
        self.n_azimuth = n_azimuth 
        self.n_theta = n_theta
        self.n_phase = n_phase  
        
        self.unconv3 = nn.ConvTranspose2d(  256, 
                                 128, 
                                 kernel_size = (1,4), 
                                 stride = (1,1), 
                                 padding = (0,0),
                                 output_padding=(0,0)) #,
        
        self.unconv2 = nn.ConvTranspose2d(  128, 
                                 64, 
                                 kernel_size = (3,3), 
                                 stride = (2,2), 
                                 padding = (1,1),
                                 output_padding=(1,1)) 
        
        self.unconv1 = nn.ConvTranspose2d(64, 
                                 n_color * n_theta * n_phase, 
                                 kernel_size = (3,3), 
                                 stride = (2,2), 
                                 padding = (1,1),
                                 output_padding=(1,1),
                                 bias=False)
        
            
    def forward(self, x): # , indices1, indices2):           
        x = self.unconv3(x)
        x = self.unconv2(x)
        x = self.unconv1(x)
        x = x.view(-1, self.n_color, self.n_theta, self.n_phase, self.n_levels, self.n_eccentricity, self.n_azimuth)
        x = x.permute(0, 4, 1, 5, 6, 2, 3).contiguous()
        return x


# #### Tests

# In[ ]:


dec = Decoder(n_levels - 1, n_color, n_eccentricity, n_azimuth, n_theta, n_phase)
dec_out = dec(code) #, indices1, indices2)
dec_out.shape


# In[ ]:


del dec


# In[ ]:


class InverseLogGaborMapper(nn.Module):
    def __init__(self, in_chan = n_eccentricity * n_azimuth * n_theta * n_phase, 
                 out_chan = width * width):
        super(InverseLogGaborMapper, self).__init__()
        self.inverseMap = nn.Linear(in_chan, out_chan)
        
    def forward(self, x, **kargs):
        out = self.inverseMap(x) #!!
        return out #!!


# ### Loss functions

# In[ ]:


def lasso_loss(z, LAMBDA):
    loss = 0
    for x in z:
        loss += LAMBDA * torch.abs(x).sum()
    return loss


# In[ ]:


def minus_log_likelihood(autoenc_outputs, autoenc_inputs):
    log_scale = nn.Parameter(torch.Tensor([0.]))
    scale = torch.exp(log_scale)
    dist = torch.distributions.Normal(autoenc_outputs, scale)
    log_likelihood = dist.log_prob(autoenc_inputs)
    return -log_likelihood.sum()


# ### Model and learning params

# In[ ]:


batch_size = 50
autoenc_lr = 1e-4
invLG_lr = 3e-5

n_epoch = 6000
recording_steps = 10


# In[ ]:


autoenc = AutoEncoder(n_levels-1, n_color, n_eccentricity, n_azimuth, n_theta, n_phase)
invLGmap = InverseLogGaborMapper()


# In[ ]:


autoenc_optimizer = optim.Adam(autoenc.parameters(), lr = autoenc_lr)
        
invLG_optimizer = optim.Adam(invLGmap.parameters(), lr = invLG_lr)

criterion = nn.MSELoss() #loss = criterion(outputs, inputs)


# In[ ]:


dataloader = DataLoader(saccade_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)


# In[ ]:


LASSO_loss_list = []
MSE_loss_list = []
invLG_loss_list = []


# In[ ]:


script_name


# In[ ]:


PATH = script_name + '_invLGmap.pt'
os.path.exists(PATH)


# In[ ]:


PATH = script_name + '_invLGmap.pt'
LAMBDA_REF = 30

if not os.path.exists(PATH):    

    for epoch in range(n_epoch):  # loop over the dataset multiple times

        LASSO_running_loss = 0.0
        MSE_running_loss = 0.0
        invLG_running_loss = 0.0
        for step, data in enumerate(dataloader):

            batch_size_eff = data['img_crop'].shape[0]
            
            log_gabor_coefs = log_gabor_transform(data['img_crop'], K)

            # Normalizing
            autoenc_inputs = log_gabor_coefs[:,:n_levels-1,...].clone()
            if color_mode == 'rgb':
                autoenc_inputs /=  256 # !! Normalization
            
            autoenc_outputs, z = autoenc(autoenc_inputs)
            autoenc_optimizer.zero_grad()
            MSE_loss = minus_log_likelihood(autoenc_outputs, autoenc_inputs)
            #0.5 * nn.MSELoss()(autoenc_outputs, autoenc_inputs)
            if False: #epoch < 1000:
                LAMBDA = (epoch / 1000)**1 * LAMBDA_REF 
            else:
                LAMBDA = LAMBDA_REF
            LASSO_loss = lasso_loss(z, LAMBDA)
            autoenc_loss = LASSO_loss +  MSE_loss 
            autoenc_loss.backward()
            autoenc_optimizer.step()  

            invLG_optimizer.zero_grad()
            log_gabor_coefs_rec = autoenc_outputs.detach().view(batch_size_eff*(n_levels-1)*n_color,
                                                               n_eccentricity*n_azimuth*n_theta*n_phase)
            img_pyr_rec_rec = invLGmap(log_gabor_coefs_rec)

            img_pyr_targets = data['img_crop'][:,:n_levels-1,...].contiguous()
            img_pyr_targets = img_pyr_targets.view(batch_size_eff * (n_levels-1) * n_color, 
                                                   width * width)
            if color_mode == 'rgb':
                img_pyr_targets /=  256 # !! Normalization
            invLG_loss = criterion(img_pyr_rec_rec, img_pyr_targets)
            invLG_loss.backward()             
            invLG_optimizer.step()

            # print statistics
            LASSO_running_loss += LASSO_loss.item()
            MSE_running_loss += MSE_loss.item()
            invLG_running_loss += invLG_loss.item()
            if (step+1)%recording_steps == 0 :    # print every n_steps mini-batches
                print(f'LAMBDA : {LAMBDA}')
                print('[%d, %5d] losses: %.3f, %.3f, %.3f' %
                      (epoch + 1, 
                       step + 1, 
                       LASSO_running_loss/recording_steps, 
                       MSE_running_loss/recording_steps,
                       invLG_running_loss/recording_steps))
                #.append
                LASSO_loss_list.append(LASSO_running_loss/recording_steps)
                MSE_loss_list.append(MSE_running_loss/recording_steps)
                invLG_loss_list.append(invLG_running_loss/recording_steps)
                LASSO_running_loss = 0.0
                MSE_running_loss = 0.0
                invLG_running_loss = 0.0

    print('Finished Training ')
    
    if n_epoch !=0 :
        PATH = script_name + '_LASSO_loss_list.npy'
        np.save(PATH, np.array(LASSO_loss_list))    
        PATH = script_name + '_MSE_loss_list.npy'
        np.save(PATH, np.array(MSE_loss_list))    
        PATH = script_name + '_invLG_loss_list.npy'
        np.save(PATH, np.array(invLG_loss_list))   
        PATH = script_name + '_invLGmap.pt'
        torch.save(invLGmap, PATH)
        PATH = script_name + '_autoenc.pt'
        torch.save(autoenc, PATH)
        #PATH = script_name + '_autoenc_VAE.pt'
        #torch.save(autoenc_VAE, PATH)
        print('Model saved')

else:
    PATH = script_name + '_LASSO_loss_list.npy'
    LASSO_loss_list = np.load(PATH).tolist()    
    PATH = script_name + '_MSE_loss_list.npy'
    MSE_loss_list = np.load(PATH).tolist()    
    PATH = script_name + '_invLG_loss_list.npy'
    invLG_loss_list = np.load(PATH).tolist()
    PATH = script_name + '_invLGmap.pt'
    invLGmap = torch.load(PATH)
    PATH = script_name + '_autoenc.pt'
    autoenc = torch.load(PATH)
    #PATH = script_name + '_autoenc_VAE.pt'
    #autoenc_VAE = torch.load(PATH)
    print('Model loaded')


# In[ ]:


if False :
    PATH = script_name + '_LASSO_loss_list.npy'
    np.save(PATH, np.array(LASSO_loss_list))    
    PATH = script_name + '_MSE_loss_list.npy'
    np.save(PATH, np.array(MSE_loss_list))    
    PATH = script_name + '_invLG_loss_list.npy'
    np.save(PATH, np.array(invLG_loss_list))   
    PATH = script_name + '_invLGmap.pt'
    torch.save(invLGmap, PATH)
    PATH = script_name + '_autoenc.pt'
    torch.save(autoenc, PATH)
    #PATH = script_name + '_autoenc_VAE.pt'
    #torch.save(autoenc_VAE, PATH)
    print('Model saved')


# In[ ]:


import seaborn
seaborn.set()
plt.figure(figsize=(20,12))
plt.plot(np.array(MSE_loss_list) , label = 'MSE')
plt.plot(np.array(LASSO_loss_list) , label = 'LASSO')
plt.plot(np.array(invLG_loss_list) * 100000, label = 'invLGMap')
#plt.ylim(0,500)
plt.title('LOSS')
plt.xlabel('# batch')
plt.legend()


# ## Encoding and decoding

# In[ ]:


seaborn.reset_orig()


# In[ ]:


img_name = 'i1198772915'
if True:
    locpath = '../ALLSTIMULI/' + img_names[10] + '.jpeg'
    #locpath = '../ALLSTIMULI/' + data['name'][3] + '.jpeg'
    locpath = '../ALLSTIMULI/' + img_name + '.jpeg'
    img_orig = Image.open(locpath) 
else:
    locpath= '../data/professional_test/test/namphuong-van-260.png'
    locpath= '../data/professional_test/test/shannon-kelley-108053.png'
    img_orig = Image.open(locpath)
    #img_orig = img_orig.resize((1024,768)) 

plt.figure(figsize=(20,15))
if color_mode=='hsv':
    img_orig = rgb2hsv(img_orig)
    plt.imshow(hsv2rgb(img_orig))
elif color_mode=='lab':
    img_orig = rgb2lab(img_orig)
    plt.imshow(lab2rgb(img_orig))
else:
    plt.imshow(img_orig)


# In[ ]:


img_tens = torch.Tensor(np.array(img_orig)[None,...]).permute(0,3,1,2)


# In[ ]:


img_crop = cropped_pyramid(img_tens, 
                           width=width,
                           color=color, 
                           do_mask=do_mask, 
                           verbose=True, 
                           n_levels=n_levels)[0]


# In[ ]:


log_gabor_coeffs = log_gabor_transform(img_crop, K)
log_gabor_coeffs.shape


# In[ ]:


autoenc_inputs = log_gabor_coeffs[:,:n_levels-1,...].clone()

if color_mode == 'rgb':
    autoenc_inputs /= 256

log_gabor_coeffs_rec,  z = autoenc( autoenc_inputs )   
log_gabor_coeffs_rec = log_gabor_coeffs_rec.view(1, (n_levels-1), -1) 
if color_mode == 'rgb':
    log_gabor_coeffs_rec *= 256


# In[ ]:


z_img = z[2]
z[0].shape


# In[ ]:


plt.figure()
plt.plot(z[0].detach().numpy().flatten(),'.')
plt.title("all")

plt.figure()
plt.plot(z[1].detach().numpy().flatten(),'.')
plt.title("all")

plt.figure()
plt.plot(z[2].detach().numpy().flatten(),'.')
plt.title("all")


# In[ ]:


_ = plt.hist(z[0].detach().numpy().flatten(),70)
plt.figure()
_ = plt.hist(z[1].detach().numpy().flatten(),70)
plt.figure()
_ = plt.hist(z[2].detach().numpy().flatten(),70)


# In[ ]:


log_gabor_coeffs_rec_cat = torch.cat((log_gabor_coeffs_rec.view(1, n_levels-1, n_color, n_eccentricity, n_azimuth, n_theta, n_phase), 
                                      log_gabor_coeffs[0:,-1:,...]), 1)


# In[ ]:


plt.figure(figsize=(20,7))
plt.plot(log_gabor_coeffs.numpy().flatten()[:], label = 'original')
plt.plot(log_gabor_coeffs_rec_cat.detach().numpy().flatten()[:], label = 'reconstructed')
plt.title('LOG GABOR COEFFS')
plt.legend()
for level in range(n_levels-1):
    plt.figure(figsize=(20,4))
    plt.plot(log_gabor_coeffs[0,level,...].numpy().flatten(), label = 'original')
    plt.plot(log_gabor_coeffs_rec[0,level,...].detach().numpy().flatten(), label = 'reconstructed')
    c = np.corrcoef([log_gabor_coeffs[0,level,...].numpy().flatten(), log_gabor_coeffs_rec[0,level,...].detach().numpy().flatten()])[0,1]
    plt.title('LOG GABOR COEFFS LEVEL '+str(level)+', corr='+str(c))
    plt.legend()


# In[ ]:





# In[ ]:


_=plt.hist(log_gabor_coeffs.numpy().flatten(),100)


# In[ ]:


_=plt.hist(img_crop.numpy().flatten(),100)


# ## Reconstruction tests

# In[ ]:


K_inv = get_K_inv(K, width=width, n_sublevel = n_sublevel, n_azimuth = n_azimuth, n_theta = n_theta, n_phase = n_phase)
img_rec=inverse_gabor(log_gabor_coeffs.detach(), K_inv)
img_rec[:,-1,...] = img_crop[:,-1,...]
axs = tensor_pyramid_display(img_rec.clone()) 


# In[ ]:


inv_LGmap_input = log_gabor_coeffs_rec.view((n_levels-1) * n_color, n_eccentricity * n_azimuth * n_theta * n_phase)
inv_LGmap_input.shape


# In[ ]:


img_rec_rec = invLGmap(inv_LGmap_input) #inv_LGmap_input)
img_rec_rec = img_rec_rec.view(1, n_levels-1, n_color, width, width).detach()
img_rec_rec = torch.cat((img_rec_rec, img_crop[0:,-1:,...]), 1)
#img_rec_rec[0,-1,...] *=0 #+= 128
axs = tensor_pyramid_display(img_rec_rec)


# ### Test de invLGmap uniquement sur log gabor coeffs originaux

# In[ ]:


img_rec_test = invLGmap(log_gabor_coeffs.view(n_levels * n_color, n_eccentricity * n_azimuth * n_theta * n_phase)) #inv_LGmap_input)
img_rec_test = img_rec_test.view(1, n_levels, n_color, width, width).detach()
img_rec_test[:,-1,...] = img_crop[:,-1,...]
axs = tensor_pyramid_display(img_rec_test)


# ### Test des coeffs reconstruits avec differentes valeurs de K_inv 

# In[ ]:


img_rec_rec_test = []
for i, rcond in enumerate((0.1, 0.03, 0.01, 0.003, 0.001, 0)):
    K_ = K.reshape((width**2, n_sublevel*n_azimuth*n_theta*n_phase))
    print('Reshaped filter tensor=', K_.shape)
    if rcond>0:
        K_inv_test = torch.pinverse(K_, rcond=rcond) 
    else:
        K_inv_test = torch.pinverse(K_)
    print('Tensor shape=', K_inv.shape)
    K_inv_test =K_inv_test.reshape(n_sublevel, n_azimuth, n_theta, n_phase, width, width)
    img_rcond_test = inverse_gabor(log_gabor_coeffs.detach(), K_inv_test)
    img_rcond_test[:,-1,...] = img_crop[:,-1,...]
    axs = tensor_pyramid_display(img_rcond_test)
    axs[0].set_title('REGULARIZATION = '+str(rcond)+', ORIGINAL LOG-GABOR COEFS')
    img_rec_rcond_test = inverse_gabor(log_gabor_coeffs_rec_cat.detach(), K_inv_test)
    img_rec_rcond_test[:,-1,...] = img_crop[:,-1,...]
    img_rec_rec_test.append(img_rec_rcond_test)
    axs = tensor_pyramid_display(img_rec_rcond_test)
    axs[0].set_title('AUTO-ENCODER LOG-GABOR RECONSTRUCTION')    


# ### Full image reconstruction

# In[ ]:


#img_crop = cropped_pyramid(img_tens, color=color, do_mask=do_mask, verbose=True, n_levels=n_levels)[0]
N_X, N_Y = full_img_rec.shape[1:3]


full_img_crop = inverse_pyramid(img_crop, color=color, gauss=gauss, n_levels=n_levels)
full_img_crop = full_img_crop.detach().permute(0,2,3,1).numpy()
    
plt.figure(figsize=(20,15))
image_show(full_img_crop[0,:], color_mode)
plt.title('RECONSTRUCTED FROM CROPPED PYRAMID, #params = ' + str(np.prod(img_crop[0,...].size())), fontsize=20)


img_rec=inverse_gabor(log_gabor_coeffs.detach(), K_inv)
img_rec[:,-1,...]= img_crop[:,-1,...]
full_img_rec = inverse_pyramid(img_rec, color=color, gauss=gauss, n_levels=n_levels)
full_img_rec = full_img_rec.detach().permute(0,2,3,1).numpy()
plt.figure(figsize=(20,15))
image_show(full_img_rec[0,:], color_mode)
plt.title('RECONSTRUCTED FROM LOG GABOR COEFFS, #params = ' + str(np.prod(log_gabor_coeffs[0,...].size())), fontsize=20)

full_img_rec_rec = inverse_pyramid(img_rec_rec, color=color, gauss=gauss, n_levels=n_levels)
full_img_rec_rec = full_img_rec_rec.detach().permute(0,2,3,1).numpy()
#ax = tensor_image_cmp(full_img_rec, full_img_rec_rec)
plt.figure(figsize=(20,15))
image_show(full_img_rec_rec[0,:], color_mode)
plt.title('RECONSTRUCTED FROM AUTO-ENCODER, #params = ' + str(np.prod(code[0,...].shape)), fontsize=20)

'''
img_rec_rec_test[3][:,-1,...]= img_crop[:,-1,...]
full_img_rec_rec_test = inverse_pyramid(img_rec_rec_test[3], color=color, gauss=gauss, n_levels=n_levels)
full_img_rec_rec_test = full_img_rec_rec_test.detach().permute(0,2,3,1).numpy().clip(0,255).astype('uint8')
#ax = tensor_image_cmp(full_img_rec, full_img_rec_rec)
plt.figure(figsize=(20,15))
plt.imshow(full_img_rec_rec_test[0,:])
plt.title('RECONSTRUCTED FROM AUTOENCODER OUTPUTS AND REGULARIZED INVERSE MAP')
'''


# In[ ]:


#img_crop = cropped_pyramid(img_tens, color=color, do_mask=do_mask, verbose=True, n_levels=n_levels)[0]
plt.figure(figsize=(20,60))

plt.subplot(4,1,1)
img = img_tens.detach().permute(0,2,3,1).numpy()
N_X, N_Y = img.shape[1:3]
image_show(img[0,N_X//2-128:N_X//2+128,
               N_Y//2-128:N_Y//2+128,:], color_mode)
plt.title('ORIGINAL IMAGE, #params = ' + str(np.prod(img_tens[0,...].size())), fontsize=20)

plt.subplot(4,1,2)
full_img_crop = inverse_pyramid(img_crop, color=color, gauss=gauss, n_levels=n_levels)
full_img_crop = full_img_crop.detach().permute(0,2,3,1).numpy()
N_X, N_Y = full_img_crop.shape[1:3]
image_show(full_img_crop[0,N_X//2-128:N_X//2+128,
                         N_Y//2-128:N_Y//2+128,:], color_mode)
plt.title('RECONSTRUCTED FROM CROPPED PYRAMID, #params = ' + str(np.prod(img_crop[0,...].size())), fontsize=20)

plt.subplot(4,1,3)
img_rec=inverse_gabor(log_gabor_coeffs.detach(), K_inv)
img_rec[:,-1,...]= img_crop[:,-1,...]
full_img_rec = inverse_pyramid(img_rec, color=color, gauss=gauss, n_levels=n_levels)
full_img_rec = full_img_rec.detach().permute(0,2,3,1).numpy()
image_show(full_img_rec[0,N_X//2-128:N_X//2+128,
                        N_Y//2-128:N_Y//2+128,:], color_mode)
plt.title('RECONSTRUCTED FROM LOG GABOR COEFFS, #params = ' + str(np.prod(log_gabor_coeffs[0,...].size())), fontsize=20)

plt.subplot(4,1,4)
full_img_rec_rec = inverse_pyramid(img_rec_rec, color=color, gauss=gauss, n_levels=n_levels)
full_img_rec_rec = full_img_rec_rec.detach().permute(0,2,3,1).numpy()
#ax = tensor_image_cmp(full_img_rec, full_img_rec_rec)
image_show(full_img_rec_rec[0,N_X//2-128:N_X//2+128,
                            N_Y//2-128:N_Y//2+128,:], color_mode)
plt.title('RECONSTRUCTED FROM AUTO-ENCODER, #params = ' + str(np.prod(code[0,...].shape)), fontsize=20)

'''
img_rec_rec_test[3][:,-1,...]= img_crop[:,-1,...]
full_img_rec_rec_test = inverse_pyramid(img_rec_rec_test[3], color=color, gauss=gauss, n_levels=n_levels)
full_img_rec_rec_test = full_img_rec_rec_test.detach().permute(0,2,3,1).numpy().clip(0,255).astype('uint8')
#ax = tensor_image_cmp(full_img_rec, full_img_rec_rec)
plt.figure(figsize=(20,15))
plt.imshow(full_img_rec_rec_test[0,:])
plt.title('RECONSTRUCTED FROM AUTOENCODER OUTPUTS AND REGULARIZED INVERSE MAP')
'''
if True:
    plt.savefig(script_name+'.png', bbox_inches='tight')


# In[ ]:


img.shape


# ## Rotation

# In[ ]:


log_gabor_coeffs_roll = log_gabor_coeffs_rec.clone()
log_gabor_coeffs_roll = log_gabor_coeffs_roll.view(1,n_levels-1, n_color, n_eccentricity, n_azimuth, n_theta, n_phase)
#log_gabor_coeffs_roll[:,:n_levels-1,...]= log_gabor_coeffs_roll[:,:n_levels-1,...].roll(-4,4) #.roll(4, 4)
log_gabor_coeffs_roll= log_gabor_coeffs_roll.roll(1,1) #.roll(4, 4)
log_gabor_coeffs_roll = torch.cat((log_gabor_coeffs_roll, log_gabor_coeffs[:,n_levels-1,...].unsqueeze(1)), 1)
log_gabor_coeffs_roll= log_gabor_coeffs_roll.roll(1,4) #.roll(4, 4)


#log_gabor_coeffs_roll= log_gabor_coeffs_roll.roll(-1,5) #.roll(4, 4)
inv_LGmap_input = log_gabor_coeffs_roll.view(n_levels * n_color, n_eccentricity * n_azimuth * n_theta * n_phase)
img_rec_rec_roll = invLGmap(inv_LGmap_input) #inv_LGmap_input)
img_rec_rec_roll = img_rec_rec_roll.view(1, n_levels, n_color, width, width).detach()
#img_rec_rec_roll = torch.cat((img_rec_rec_roll, img_crop[0:,-1:,...]), 1)

full_img_rec_rec_roll = inverse_pyramid(img_rec_rec_roll, color=color, gauss=gauss, n_levels=n_levels)
full_img_rec_rec_roll = full_img_rec_rec_roll.detach().permute(0,2,3,1).numpy()
if color_mode == 'rgb':
    full_img_rec_rec_roll = full_img_rec_rec_roll.clip(0,255).astype('uint8')
#ax = tensor_image_cmp(full_img_rec, full_img_rec_rec)
plt.figure(figsize=(20,15))
image_show(full_img_rec_rec_roll[0,:], color_mode)
plt.title('ROTATION/ZOOM FROM COMPRESSED SENSING LAYER')
#log_gabor_coeffs_roll[:,:n_levels-1,...] = log_gabor_coeffs_roll[:,:n_levels-1,...].roll(1,1) #.roll(4, 4)


# In[ ]:


img_res_gray = rgb2lab((np.ones((32,32,3))*128).astype('uint8'))
img_res_gray = torch.DoubleTensor(img_res_gray).permute(2,0,1).unsqueeze(0).unsqueeze(0)


# In[ ]:


p = torch.distributions.Gamma(torch.ones_like(z_img), 3 * torch.ones_like(z_img))
z_in =  p.rsample() - p.rsample()
plt.figure()
_ = plt.hist(z_in.detach().numpy().flatten(),70)
log_gabor_coeffs_rec_test, z = autoenc(autoenc_inputs[:1,:n_levels-1,...], z_in=z_in)  
inv_LGmap_input = log_gabor_coeffs_rec_test.view((n_levels-1) * n_color, n_eccentricity * n_azimuth * n_theta * n_phase)
img_rec_rec_test = invLGmap(inv_LGmap_input) #inv_LGmap_input)
img_rec_rec_test = img_rec_rec_test.view(n_levels-1, n_color, width, width)
img_rec_rec_test = torch.cat((img_rec_rec_test.unsqueeze(0), img_res_gray), 1)
full_img_rec_rec_test = inverse_pyramid(img_rec_rec_test, color=color, gauss=gauss, n_levels=n_levels)
full_img_rec_rec_test = full_img_rec_rec_test.detach().permute(0,2,3,1).numpy()
#.permute(0,2,3,1).detach().numpy().squeeze()
#image_show(full_img_rec_rec_test[0,16:48,16:48,:], color_mode=color_mode)
plt.figure(figsize=(20,15))
image_show(full_img_rec_rec_test.squeeze(), color_mode=color_mode)
#plt.title(f'{level}, {lat_feat}')


# In[ ]:



autoenc_inputs.shape


# ## Opposite sign (negative)

# In[ ]:


log_gabor_coeffs_rec, z = autoenc( autoenc_inputs)   
z_in = - z[2] #torch.randn_like(mu) 
log_gabor_coeffs_rec_test, z = autoenc( autoenc_inputs, z_in=z_in )   
inv_LGmap_input = log_gabor_coeffs_rec_test.view((n_levels-1) * n_color, n_eccentricity * n_azimuth * n_theta * n_phase)
img_rec_rec_test = invLGmap(inv_LGmap_input) #inv_LGmap_input)
img_rec_rec_test = img_rec_rec_test.view(1, n_levels-1, n_color, width, width).detach()
img_rec_rec_test = torch.cat((img_rec_rec_test, img_crop[0:,-1:,...]), 1)
full_img_rec_rec_test = inverse_pyramid(img_rec_rec_test, color=color, gauss=gauss, n_levels=n_levels)
full_img_rec_rec_test = full_img_rec_rec_test.detach().permute(0,2,3,1).numpy()
plt.figure(figsize=(20,15))
image_show(full_img_rec_rec_test[0,:], color_mode)


# In[ ]:


plt.plot(data['img_crop'].flatten())


# In[ ]:


with torch.no_grad():
    z_sample = []
    for step, data in enumerate(dataloader):
        #print(step)
        log_gabor_coefs = log_gabor_transform(data['img_crop'], K)
        autoenc_inputs = log_gabor_coefs.clone()
        # Normalizing
        if color_mode == 'rgb':
            autoenc_inputs /=  256 # !! Normalization
        out_coeffs, z = autoenc(autoenc_inputs[:,:n_levels-1,...])
        z_sample.extend(list(z[2].detach().numpy()))


# In[ ]:


np.array(z_sample).shape


# In[ ]:


ylim = 64
plt.figure(figsize=(12,ylim // 5))
plt.violinplot(np.array(z_sample)[:,:ylim,0,0], vert=False)
plt.plot(np.array(z_sample)[:,:ylim,0,0].mean(0), np.arange(1, ylim+1),'r.')
y_max = np.argmax(np.array(z_sample)[:,:ylim,0,0].mean(0))
y_std_max = np.argmax(np.array(z_sample)[:,:ylim,0,0].std(0))
plt.ylim(0,ylim+1)
plt.plot([0,0], [0, ylim+1], 'k:')
plt.title(str(level)+', max:'+ str(y_max)+', std_max:'+str(y_std_max))


# In[ ]:


log_gabor_coeffs_rec_test.shape


# In[ ]:


z_img.shape


# In[ ]:


plt.figure(figsize=(20,600))
#plt.figure(figsize=(20,150))
lat_shift = 0
for lat_feat in range(lat_shift, lat_shift+64):
    #plt.subplot(50,2,lat_feat+1-lat_shift)
    plt.subplot(200,5,lat_feat+1-lat_shift)
    z_in = torch.zeros_like(z_img)
    z_in[0,lat_feat,:,:] = 10
    log_gabor_coeffs_rec_test, z = autoenc(autoenc_inputs[:1,:n_levels-1,...],z_in=z_in)  
    inv_LGmap_input = log_gabor_coeffs_rec_test.view((n_levels-1) * n_color, n_eccentricity * n_azimuth * n_theta * n_phase)
    img_rec_rec_test = invLGmap(inv_LGmap_input) #inv_LGmap_input)
    img_rec_rec_test = img_rec_rec_test.view(n_levels-1, n_color, width, width)
    img_rec_rec_test = torch.cat((img_rec_rec_test.unsqueeze(0), img_res_gray), 1)
    full_img_rec_rec_test = inverse_pyramid(img_rec_rec_test, color=color, gauss=gauss, n_levels=n_levels)
    full_img_rec_rec_test = full_img_rec_rec_test.detach().permute(0,2,3,1).numpy()
    #.permute(0,2,3,1).detach().numpy().squeeze()
    #image_show(full_img_rec_rec_test[0,16:48,16:48,:], color_mode=color_mode)
    image_show(full_img_rec_rec_test.squeeze(), color_mode=color_mode)
    plt.title(f'{level}, {lat_feat}')


# In[ ]:


plt.figure(figsize=(20,600))
#plt.figure(figsize=(20,150))
lat_shift = 0
for lat_feat in range(lat_shift, lat_shift+64):
    #plt.subplot(50,2,lat_feat+1-lat_shift)
    plt.subplot(200,5,lat_feat+1-lat_shift)
    z_in = torch.zeros_like(z_img)
    z_in[0,lat_feat,:,:] = 10
    log_gabor_coeffs_rec_test, z = autoenc(autoenc_inputs[:1,:n_levels-1,...],z_in=z_in)  
    inv_LGmap_input = log_gabor_coeffs_rec_test.view((n_levels-1) * n_color, n_eccentricity * n_azimuth * n_theta * n_phase)
    img_rec_rec_test = invLGmap(inv_LGmap_input) #inv_LGmap_input)
    img_rec_rec_test = img_rec_rec_test.view(n_levels-1, n_color, width, width)
    img_rec_rec_test = torch.cat((img_rec_rec_test.unsqueeze(0), img_res_gray), 1)
    full_img_rec_rec_test = inverse_pyramid(img_rec_rec_test, color=color, gauss=gauss, n_levels=n_levels)
    full_img_rec_rec_test = full_img_rec_rec_test.detach().permute(0,2,3,1).numpy()
    #.permute(0,2,3,1).detach().numpy().squeeze()
    #image_show(full_img_rec_rec_test[0,16:48,16:48,:], color_mode=color_mode)
    image_show(full_img_rec_rec_test[:,768//2-32:768//2+32,1024//2-32:1024//2+32,:].squeeze(), color_mode=color_mode)
    plt.title(f'{level}, {lat_feat}')


# In[ ]:


z[0].shape, z_in.shape


# In[ ]:


plt.figure(figsize=(20,600))
#plt.figure(figsize=(20,150))
lat_shift = 0
for exc in range(5): #64):
    #plt.subplot(50,2,lat_feat+1-lat_shift)
    plt.subplot(1,5,exc+1)
    z_in = torch.zeros_like(z_img)
    z_in[0,lat_feat,exc,0] = 10
    log_gabor_coeffs_rec_test, z = autoenc(autoenc_inputs[:1,:n_levels-1,...],z_in=z_in)  
    inv_LGmap_input = log_gabor_coeffs_rec_test.view((n_levels-1) * n_color, n_eccentricity * n_azimuth * n_theta * n_phase)
    img_rec_rec_test = invLGmap(inv_LGmap_input) #inv_LGmap_input)
    img_rec_rec_test = img_rec_rec_test.view(n_levels-1, n_color, width, width)
    img_rec_rec_test = torch.cat((img_rec_rec_test.unsqueeze(0), img_res_gray), 1)
    full_img_rec_rec_test = inverse_pyramid(img_rec_rec_test, color=color, gauss=gauss, n_levels=n_levels)
    full_img_rec_rec_test = full_img_rec_rec_test.detach().permute(0,2,3,1).numpy()
    #.permute(0,2,3,1).detach().numpy().squeeze()
    #image_show(full_img_rec_rec_test[0,16:48,16:48,:], color_mode=color_mode)
    #image_show(full_img_rec_rec_test[:,768//2-32:768//2,1024//2:1024//2+32,:].squeeze(), color_mode=color_mode)
    image_show(full_img_rec_rec_test[:,768//2-128:768//2+128,1024//2-128:1024//2+128,:].squeeze(), color_mode=color_mode)
    #image_show(full_img_rec_rec_test.squeeze(), color_mode=color_mode)
    #image_show(full_img_rec_rec_test.squeeze(), color_mode=color_mode)
    plt.title(f'{exc}, {lat_feat}')
    c = (127.5,127.5)
    plt.plot(c[0],c[1],'+r')
    plt.plot((c[0],c[0]), (0.5, 127.5),':r')
    plt.plot((0.5, 127.5), (c[1],c[1]), ':r')
    for side in (32,64,128,256):
        axe = np.linspace(-np.pi, np.pi, side//2)
        r = side/2
        x_val = r * np.sin(axe)
        y_val = r * np.cos(axe)
        plt.plot(c[0]+x_val, c[1]+y_val,'r.',markersize=1)
        '''plt.plot((c[0]-side/2, c[0]+side/2),(c[1]-side/2, c[1]-side/2),'r:')
        plt.plot((c[0]-side/2, c[0]+side/2),(c[1]+side/2, c[1]+side/2),'r:')
        plt.plot((c[0]-side/2, c[0]-side/2),(c[1]-side/2, c[1]+side/2),'r:')
        plt.plot((c[0]+side/2, c[0]+side/2),(c[1]-side/2, c[1]+side/2),'r:')'''
    #plt.plot([0.5, 62.5],[31.5, 31.5],':r')
    #plt.plot([31.5, 31.5],[0.5, 62.5],':r')


# In[ ]:


plt.figure(figsize=(20,600))
#plt.figure(figsize=(20,150))
lat_shift = 0
for lat_feat in range(lat_shift, lat_shift+64):
    #plt.subplot(50,2,lat_feat+1-lat_shift)
    plt.subplot(200,5,lat_feat+1-lat_shift)
    z_in = torch.zeros_like(z_img)
    z_in[0,lat_feat,3,0] = 10
    log_gabor_coeffs_rec_test, z = autoenc(autoenc_inputs[:1,:n_levels-1,...],z_in=z_in)  
    inv_LGmap_input = log_gabor_coeffs_rec_test.view((n_levels-1) * n_color, n_eccentricity * n_azimuth * n_theta * n_phase)
    img_rec_rec_test = invLGmap(inv_LGmap_input) #inv_LGmap_input)
    img_rec_rec_test = img_rec_rec_test.view(n_levels-1, n_color, width, width)
    img_rec_rec_test = torch.cat((img_rec_rec_test.unsqueeze(0), img_res_gray), 1)
    full_img_rec_rec_test = inverse_pyramid(img_rec_rec_test, color=color, gauss=gauss, n_levels=n_levels)
    full_img_rec_rec_test = full_img_rec_rec_test.detach().permute(0,2,3,1).numpy()
    #.permute(0,2,3,1).detach().numpy().squeeze()
    #image_show(full_img_rec_rec_test[0,16:48,16:48,:], color_mode=color_mode)
    #image_show(full_img_rec_rec_test[:,768//2-32:768//2,1024//2:1024//2+32,:].squeeze(), color_mode=color_mode)
    image_show(full_img_rec_rec_test[:,768//2-128:768//2+128,1024//2-128:1024//2+128,:].squeeze(), color_mode=color_mode)
    #image_show(full_img_rec_rec_test.squeeze(), color_mode=color_mode)
    #image_show(full_img_rec_rec_test.squeeze(), color_mode=color_mode)
    plt.title(f'{level}, {lat_feat}')
    c = (127.5,127.5)
    plt.plot(c[0],c[1],'+r')
    #plt.plot((c[0],c[0]), (0.5, 255.5),':r')
    #plt.plot((0.5, 255.5), (c[1],c[1]), ':r')
    '''for side in (32,64,128,256):
        axe = np.linspace(-np.pi, np.pi, side//2)
        r = side/2
        x_val = r * np.sin(axe)
        y_val = r * np.cos(axe)
        plt.plot(c[0]+x_val, c[1]+y_val,'r.',markersize=1)
        plt.plot((c[0]-side/2, c[0]+side/2),(c[1]-side/2, c[1]-side/2),'r:')
        plt.plot((c[0]-side/2, c[0]+side/2),(c[1]+side/2, c[1]+side/2),'r:')
        plt.plot((c[0]-side/2, c[0]-side/2),(c[1]-side/2, c[1]+side/2),'r:')
        plt.plot((c[0]+side/2, c[0]+side/2),(c[1]-side/2, c[1]+side/2),'r:')'''
    #plt.plot([0.5, 62.5],[31.5, 31.5],':r')
    #plt.plot([31.5, 31.5],[0.5, 62.5],':r')


# In[ ]:


plt.figure(figsize=(20,600))
#plt.figure(figsize=(20,150))
lat_shift = 0
for lat_feat in range(lat_shift, lat_shift+64):
    #plt.subplot(50,2,lat_feat+1-lat_shift)
    plt.subplot(200,5,lat_feat+1-lat_shift)
    z_in = torch.zeros_like(z_img)
    z_in[0,lat_feat,3,:] = 10
    log_gabor_coeffs_rec_test, z = autoenc(autoenc_inputs[:1,:n_levels-1,...],z_in=z_in)  
    inv_LGmap_input = log_gabor_coeffs_rec_test.view((n_levels-1) * n_color, n_eccentricity * n_azimuth * n_theta * n_phase)
    img_rec_rec_test = invLGmap(inv_LGmap_input) #inv_LGmap_input)
    img_rec_rec_test = img_rec_rec_test.view(n_levels-1, n_color, width, width)
    img_rec_rec_test = torch.cat((img_rec_rec_test.unsqueeze(0), img_res_gray), 1)
    full_img_rec_rec_test = inverse_pyramid(img_rec_rec_test, color=color, gauss=gauss, n_levels=n_levels)
    full_img_rec_rec_test = full_img_rec_rec_test.detach().permute(0,2,3,1).numpy()
    #.permute(0,2,3,1).detach().numpy().squeeze()
    #image_show(full_img_rec_rec_test[0,16:48,16:48,:], color_mode=color_mode)
    image_show(full_img_rec_rec_test[:,768//2-32:768//2,1024//2:1024//2+32,:].squeeze(), color_mode=color_mode)
    image_show(full_img_rec_rec_test[:,768//2-128:768//2+128,1024//2-128:1024//2+128,:].squeeze(), color_mode=color_mode)
    #image_show(full_img_rec_rec_test.squeeze(), color_mode=color_mode)
    plt.title(f'{level}, {lat_feat}')
    c = (127.5,127.5)
    plt.plot(c[0],c[1],'+r')
    for side in (32,64,128,256):
        axe = np.linspace(-np.pi, np.pi, side//2)
        r = side/2
        x_val = r * np.sin(axe)
        y_val = r * np.cos(axe)
        plt.plot(c[0]+x_val, c[1]+y_val,'r.',markersize=1)
    #plt.plot([0.5, 62.5],[31.5, 31.5],':r')
    #plt.plot([31.5, 31.5],[0.5, 62.5],':r')


# In[ ]:


full_img_rec_rec_test.shape


# In[ ]:





# In[ ]:




