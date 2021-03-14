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


# In[9]:


out_chan=1024
gauss = False
do_mask = False
color = True
color_mode= 'lab' # 'hsv' #True
print ('encoder #params :', out_chan)


# In[10]:


if gauss:
    script_name = '2021-03-13-log-polar-deep-convolutional-no-max-pool-VAE-gauss-'+color_mode
else:
    script_name = '2021-03-13-log-polar-deep-convolutional-no-max-pool-VAE-laplace-'+color_mode


# ### Image utilities

# In[11]:


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


# In[12]:


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

K_test= torch.norm(K, dim=4)
plt.figure(figsize=(20,6))
for i_az in range(n_azimuth):
    coefs = torch.zeros((n_sublevel, n_azimuth, n_phase))
    coefs[:, i_az, 0] = 1
    img_dis = torch.tensordot(K_test, coefs, dims=3)
    plt.subplot(2,n_azimuth//2,i_az+1)
    plt.imshow(img_dis.numpy()[:, :, ...], cmap='gray')
plt.figure(figsize=(20,6))
for i_az in range(n_azimuth):
    coefs = torch.zeros((n_sublevel, n_azimuth, n_phase))
    coefs[:, i_az, 1] = 1
    img_dis = torch.tensordot(K_test, coefs, dims=3)
    plt.subplot(2,n_azimuth//2,i_az+1)
    plt.imshow(img_dis.numpy()[:, :, ...], cmap='gray')plt.figure()
coefs = torch.zeros((n_sublevel, n_azimuth, n_phase))
coefs[:, :, :1] = torch.ones((n_sublevel, n_azimuth, 1))
img_dis = torch.tensordot(K_test, coefs, dims=3)
plt.subplot(1,2,1)
plt.imshow(img_dis.numpy(), cmap='gray')
plt.subplot(1,2,2)
_=plt.plot(img_dis.numpy())
plt.figure()
coefs = torch.zeros((n_sublevel, n_azimuth, n_phase))
coefs[:, :, 1:] = torch.ones((n_sublevel, n_azimuth, 1))
img_dis = torch.tensordot(K_test, coefs, dims=3)
plt.subplot(1,2,1)
plt.imshow(img_dis.numpy(), cmap='gray')
plt.subplot(1,2,2)
_=plt.plot(img_dis.numpy())liste = [img_dis] * n_levels
crop_levels = (torch.stack(liste)).unsqueeze(0) * 5
full_rosace = inverse_pyramid(crop_levels, color=False, gauss=gauss, n_levels=n_levels)
full_rosace = full_rosace.detach().permute(0,2,3,1).numpy().clip(0,255).astype('uint8')
#ax = tensor_image_cmp(full_img_rec, full_img_rec_rec)
plt.figure(figsize=(20,15))
plt.imshow(full_rosace.squeeze(), cmap='gray')
# In[24]:


log_gabor_rosace = torch.zeros(1, n_levels, n_color, n_eccentricity, n_azimuth, n_theta, n_phase)
log_gabor_rosace[:,:,:,:,:,2,0] = 200

img_crop_rosace=inverse_gabor(log_gabor_rosace, K_inv_rcond)

axs = tensor_pyramid_display(img_crop_rosace)
img_crop_rosace[:,-1,...] = 128 
full_rosace = inverse_pyramid(img_crop_rosace, color=color, gauss=gauss, n_levels=n_levels)
full_rosace = full_rosace.detach().permute(0,2,3,1).numpy().clip(0,255).astype('uint8')
#ax = tensor_image_cmp(full_img_rec, full_img_rec_rec)
plt.figure(figsize=(20,15))
plt.imshow(full_rosace[0,:])


# In[25]:


log_gabor_rosace = 100 * torch.ones(1, n_levels, n_color, n_eccentricity, n_azimuth, n_theta, n_phase)

plt.figure()

img_crop_rosace=inverse_gabor(log_gabor_rosace, K_inv_rcond)
axs = tensor_pyramid_display(img_crop_rosace)
img_crop_rosace[:,-1,...] = 128 
full_rosace = inverse_pyramid(img_crop_rosace, color=color, gauss=gauss, n_levels=n_levels)
full_rosace = full_rosace.detach().permute(0,2,3,1).numpy().clip(0,255).astype('uint8')
#ax = tensor_image_cmp(full_img_rec, full_img_rec_rec)
plt.figure(figsize=(20,15))
plt.imshow(full_rosace[0,:])


# ## Images dataset + transforms

# In[26]:


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


# In[27]:


dir_names = os.listdir('../saccades-data')
loc_data_xy={}
for dir_name in dir_names:
    loc_data_xy[dir_name]={}
    for name in img_names:
        locpath = '../saccades-data/' + dir_name + '/' + name
        f = open(locpath,'rb')
        loc_dict = pickle.load(f)
        loc_data_xy[dir_name][name] = np.array(loc_dict['barycenters'])


# In[28]:


def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    #plt.pause(0.001)  # pause a bit so that plots are updated


# # Dataset class

# In[29]:


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
        landmarks = self.loc_dict[dir_name][name]
        landmarks = np.array([landmarks])
        landmarks = landmarks.reshape(-1, 2) #.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks, 'name':name}

        if self.transform:
            sample = self.transform(sample)

        return sample


# # Transforms

# In[30]:


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


# In[31]:


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image_tens = sample['image'].transpose((2, 0, 1))
        return {'image': torch.FloatTensor(image_tens), 'pos': sample['pos'],  'name':sample['name']}


# ### Adapted cropped pyramid (squeezed tensor)

# In[32]:


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

# In[33]:


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

# In[34]:


composed_transform = transforms.Compose([RandomSaccadeTo(zero_fill=True),
                               ToTensor(),
                               CroppedPyramid(width, 
                                              base_levels, 
                                              n_levels=n_levels,
                                              color_mode=color_mode)]) #, LogGaborTransform()])


# In[35]:


saccade_dataset = SaccadeLandmarksDataset(loc_dict=loc_data_xy,
                                          img_dir='../ALLSTIMULI/',
                                          img_names=img_names,
                                          dir_names =  dir_names,
                                          transform=composed_transform,
                                          color_mode=color_mode)


# # Iterating through the dataset

# In[38]:


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


# In[39]:


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
        


# In[40]:


plt.plot(sample_batched['img_crop'].flatten())


# In[41]:


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


# In[42]:


plt.plot(sample_batched['img_crop'][0,:,0,:,:].flatten())


# In[43]:


sample_batched['name']
locpath = '../ALLSTIMULI/' + sample_batched['name'][0] + '.jpeg'
img_orig = Image.open(locpath) 
plt.imshow(img_orig)


# ## Encoding and decoding

# In[44]:


locpath = '../ALLSTIMULI/' + img_names[10] + '.jpeg'
img_orig = Image.open(locpath) 

locpath= '../data/professional_test/test/namphuong-van-260.png'
#locpath= '../data/professional_test/test/shannon-kelley-108053.png'
img_orig = Image.open(locpath).resize((1024,768)) 

'''if hsv:
    img_orig = img_orig.convert("HSV")

img_orig = io.imread(locpath)
'''
plt.figure(figsize=(20,15))
if color_mode=='hsv':
    img_orig = rgb2hsv(img_orig)
    plt.imshow(hsv2rgb(img_orig))
elif color_mode=='lab':
    img_orig = rgb2lab(img_orig)
    plt.imshow(lab2rgb(img_orig))
else:
    plt.imshow(img_orig)


# In[45]:


img_tens = torch.Tensor(np.array(img_orig)[None,...]).permute(0,3,1,2)


# In[46]:


img_crop = cropped_pyramid(img_tens, 
                           width=width, 
                           color=color, 
                           do_mask=do_mask,
                           verbose=True, 
                           gauss=gauss, 
                           n_levels=n_levels)[0]


# In[47]:


img_crop.shape


# In[48]:


log_gabor_coeffs = log_gabor_transform(img_crop, K)
log_gabor_coeffs.shape


# ## Reconstruction tests

# ### Pyramid reconstruction

# In[49]:


img_rec=inverse_gabor(log_gabor_coeffs.detach(), K_inv)
img_rec_clone = img_rec.clone()
img_crop_clone = img_crop.clone()
axs = tensor_pyramid_display(img_rec_clone)
axs = tensor_pyramid_display(img_crop_clone)


# ### Full image reconstruction

# In[50]:


def image_show(im, color_mode):
    if color_mode=='hsv':
        plt.imshow(hsv2rgb(im))
    elif color_mode=='lab':
        plt.imshow(lab2rgb(im))
    else:
        full_img_rec = im.clip(0,255).astype('uint8')
        plt.imshow(im)


# In[51]:


img_pyr_rec = inverse_pyramid(img_crop, color=color, gauss=gauss, n_levels=n_levels)
img_pyr_rec = img_pyr_rec.detach().permute(0,2,3,1).numpy()
if color_mode=='rgb':
    img_pyr_rec = img_pyr_rec.clip(0,255).astype('uint8')

if True:
    img_rec[:,-1,...]= img_crop[:,-1,...]
    
img_lg_rec = inverse_pyramid(img_rec, color=color, gauss=gauss, 
                             n_levels=n_levels, color_test = False)
img_lg_rec = img_lg_rec.detach().permute(0,2,3,1).numpy()
if color_mode == 'rgb':
    img_lg_rec = img_lg_rec.clip(0,255).astype('uint8')

plt.figure(figsize=(20,15))
image_show(img_pyr_rec[0,:], color_mode)
    
plt.figure(figsize=(20,15))
image_show(img_lg_rec[0,:], color_mode)


# ### Rotation test

# In[52]:


log_gabor_coeffs.shape


# In[53]:


log_gabor_coeffs_roll = log_gabor_coeffs.clone()
#log_gabor_coeffs_roll[:,:n_levels-1,...]= log_gabor_coeffs_roll[:,:n_levels-1,...].roll(-4,4) #.roll(4, 4)
log_gabor_coeffs_roll= log_gabor_coeffs_roll.roll(1,4) #.roll(4, 4)
log_gabor_coeffs_roll= log_gabor_coeffs_roll.roll(1,1) #.roll(4, 4)

#log_gabor_coeffs_roll[:,:n_levels-1,...] = log_gabor_coeffs_roll[:,:n_levels-1,...].roll(1,1) #.roll(4, 4)


# In[54]:


img_rec_roll=inverse_gabor(log_gabor_coeffs_roll.detach(), K_inv_rcond)
#img_rec_roll=inverse_gabor(log_gabor_coeffs_roll.detach(), K.permute(2,3,4,5,0,1))
if True:
    img_rec_roll[:,-1,...]= img_crop[:,-1,...]


# In[55]:


img_lg_rec_roll = inverse_pyramid(img_rec_roll, color=color, gauss=gauss, n_levels=n_levels)
img_lg_rec_roll = img_lg_rec_roll.detach().permute(0,2,3,1).numpy()
if color_mode == 'rgb':
    img_lg_rec_roll = img_lg_rec_roll.clip(0,255).astype('uint8')


# In[56]:


plt.figure(figsize=(20,15))
if color_mode=='hsv':
    plt.imshow(hsv2rgb(img_lg_rec_roll[0,:]))
elif color_mode=='lab':
    plt.imshow(lab2rgb(img_lg_rec_roll[0,:]))
else:
    plt.imshow(img_lg_rec_roll[0,:])
plt.title('ROTATION + ZOOM')


# ### Autoencoder

# In[483]:


class AutoEncoder(nn.Module):
    def __init__(self, n_levels, n_color, n_eccentricity, n_azimuth, n_theta, n_phase, 
                 out_chan = 32,
                 encoder_l=None, encoder_c=None, decoder_l=None, decoder_c=None, is_VAE=True,
                 residual_encode = False):
        super(AutoEncoder, self).__init__()
        self.n_levels = n_levels
        self.n_color = n_color
        self.n_eccentricity = n_eccentricity 
        self.n_azimuth = n_azimuth 
        self.n_theta = n_theta
        self.n_phase = n_phase
        self.out_chan = out_chan
        
        self.residual_encode = residual_encode
        if encoder_l is None:
            if self.residual_encode:
                self.encoder_l = Encoder(n_levels-1, n_color, n_eccentricity, n_azimuth, n_theta, n_phase)
            else:
                self.encoder_l = Encoder(n_levels, n_color, n_eccentricity, n_azimuth, n_theta, n_phase)
        else:
            self.encoder_l = encoder_l
        if encoder_c is None:
            if self.residual_encode:
                self.encoder_c = Encoder(1, n_color, n_eccentricity, n_azimuth, n_theta, n_phase)
            else:
                self.encoder_c = None
        else:
            self.encoder_c = encoder_c
        
        self.is_VAE = is_VAE
        if self.is_VAE:
            if self.residual_encode:
                self.h_size_l = (n_levels-1) * n_azimuth//4 * 128
                self.h_size_c = n_azimuth//4 * 128
                self.fc_mu_l = nn.Linear(self.h_size_l, out_chan)               
                self.fc_mu_c = nn.Linear(self.h_size_c, out_chan)    
                self.fc_logvar_l = nn.Linear(self.h_size_l, out_chan)               
                self.fc_logvar_c = nn.Linear(self.h_size_c, out_chan)    
                #self.fc_h = nn.Linear(2048, 2048)      
                #self.fc_mu = nn.Linear(2048, out_chan)      
                #self.fc_logvar = nn.Linear(2048, out_chan)      

                #self.fc_z_inv = nn.Linear(out_chan, 2048)      
                #self.fc_h_inv = nn.Linear(2048, 2048)      
                self.fc_l_inv = nn.Linear(out_chan, self.h_size_l)      
                self.fc_c_inv = nn.Linear(out_chan, self.h_size_c) 
            else:
                self.h_size = n_levels * n_azimuth//4 * 128
                self.fc_mu = nn.Linear(self.h_size, out_chan)      
                self.fc_logvar = nn.Linear(self.h_size, out_chan)      
                self.fc_z_inv = nn.Linear(out_chan, self.h_size)                   
        
        if decoder_l is None:
            if self.residual_encode:                
                self.decoder_l = Decoder(n_levels-1, n_color, n_eccentricity, n_azimuth, n_theta, n_phase)
            else:
                self.decoder_l = Decoder(n_levels, n_color, n_eccentricity, n_azimuth, n_theta, n_phase)    
        else:
            self.decoder_l = decoder_l
        if decoder_c is None:
            if self.residual_encode: 
                self.decoder_c = Decoder(1, n_color, n_eccentricity, n_azimuth, n_theta, n_phase)
            else:
                self.decoder_c = None
        else:
            self.decoder_c = decoder_c
        

    def forward(self, x, z_in=None): #, **kargs):   
        if self.residual_encode:            
            #code_l, indices1_l, indices2_l = self.encoder_l(x[:,:,0,...].unsqueeze(2))
            #print(x[:,:-1,...].shape)
            code_l = self.encoder_l(x[:,:-1,...])
            #code_c, indices1_c, indices2_c = self.encoder_c(x[:,:,1:,...])
            code_c = self.encoder_c(x[:,-1,...].unsqueeze(1))
        else:
            code_l = self.encoder_l(x)
            code_c = 0
        
        #print(code_l.shape)
        if self.is_VAE:
            #print(code_l.shape, self.h_size)
            
            if self.residual_encode: 
                mu = self.fc_mu_l(code_l.view(-1, self.h_size_l)) + self.fc_mu_c(code_c.view(-1, self.h_size_c))
                logvar = self.fc_logvar_l(code_l.view(-1, self.h_size_l)) + self.fc_logvar_c(code_c.view(-1, self.h_size_c))
            else:
                mu = self.fc_mu(code_l.view(-1, self.h_size))
                logvar = self.fc_logvar(code_l.view(-1, self.h_size))
            # sample z from q
            #logvar = torch.clamp(logvar, -5, 5)
            std = torch.exp(logvar/ 2)
            #std = torch.clamp(std, 0, 100)
            eps = torch.randn_like(mu) # `randn_like` as we need the same size

            if z_in is None:
                z = mu + (eps * std)
            else:
                z = z_in
            '''plt.figure()
            plt.plot(mu[0,...].detach().numpy().flatten())
            plt.figure()
            plt.plot(std[0,...].detach().numpy().flatten())            
            plt.figure()
            plt.plot(z[0,...].detach().numpy().flatten())'''


            if self.residual_encode: 
                decode_l = self.fc_l_inv(z).view(-1, 128, self.n_levels-1, self.n_azimuth//4)    
                #print(decode_l.shape)
                decode_c = self.fc_c_inv(z).view(-1, 128, 1, self.n_azimuth//4)
            else:
                decode_l = self.fc_z_inv(z).view(-1, 128, self.n_levels, self.n_azimuth//4)
        else:
            decode_l = code_l
            decode_c = code_c
            z = torch.zeros(1)
            mu = torch.zeros(1)
            logvar = torch.zeros(1)
        
        if self.residual_encode:
            rec_l = self.decoder_l(decode_l) #, indices1_l, indices2_l) 
            rec_c = self.decoder_c(decode_c) #, indices1_c, indices2_c) 
            #print(rec_l.shape, rec_c.shape)
            return torch.cat((rec_l, rec_c), 1), mu, logvar, z
        else:
            return self.decoder_l(decode_l), mu, logvar, z


# In[1501]:


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
        self.batch_norm = nn.BatchNorm2d(n_color * n_phase * n_theta)
        self.conv1 = nn.Conv2d(  n_color * n_phase * n_theta, 
                                 64, 
                                 kernel_size = (3,3), 
                                 stride = (2,2),
                                 padding = (1,1))        
        '''self.conv1b = nn.Conv2d(256, 
                                 16, 
                                 kernel_size = (1,1), 
                                 stride = (1,1),
                                 padding = (0,0))'''
        self.conv2 = nn.Conv2d(  64, 
                                 128, 
                                 kernel_size = (3,3), 
                                 stride = (2,2), 
                                 padding = (1,1)) #,
        '''self.conv2b = nn.Conv2d(  512, 
                                 32, 
                                 kernel_size = (1,1), 
                                 stride = (1,1), 
                                 padding = (0,0))'''
        self.conv3 = nn.Conv2d(  128, 
                                 256, 
                                 kernel_size = (3,3), 
                                 stride = (2,2), 
                                 padding = (1,1)) #,
        '''self.conv3b = nn.Conv2d(  1024, 
                                 64, 
                                 kernel_size = (1,1), 
                                 stride = (1,1), 
                                 padding = (0,0))'''
        self.conv4 = nn.Conv2d(  256, 
                                 512, 
                                 kernel_size = (2,2), 
                                 stride = (2,1), 
                                 padding = (1,0)) #,
        '''self.conv4b = nn.Conv2d(  2048, 
                                 128, 
                                 kernel_size = (1,1), 
                                 stride = (1,1), 
                                 padding = (0,0))'''
            
    def forward(self, x): 
        x = x.permute(0, 2, 5, 6, 1, 3, 4).contiguous()
        x = x.view(-1, self.n_color*self.n_theta*self.n_phase, self.n_levels * self.n_eccentricity, self.n_azimuth)
        #plt.figure()
        #_ = plt.hist(x[0,...].detach().numpy().flatten(),50)
        x = self.batch_norm(x)
        #plt.figure()
        #_ = plt.hist(x[0,...].detach().numpy().flatten(),50)
        x = self.conv1(x)
        x = nn.ReLU()(x) # sparse code
        #x = self.conv1b(x) # dense code
        #print(x.shape)
        
        x = self.conv2(x)
        x = nn.ReLU()(x) # sparse code
        #x = self.conv2b(x) # dense code
        #print(x.shape)
        
        #x = self.conv3(x)
        #x = nn.ReLU()(x) # sparse code
        #x = self.conv3b(x) # dense codex = self.conv3(x)
        #print(x.shape)
        
        #x = self.conv4(x)
        #x = nn.ReLU()(x) # sparse code
        #x = self.conv4b(x) # dense code
        #print(x.shape)
        return x 


# In[485]:


out_chan = 1024
enc = Encoder(n_levels - 1, n_color, n_eccentricity, n_azimuth, n_theta, n_phase)
dataloader = DataLoader(saccade_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)
data = next(iter(dataloader))  
log_gabor_coefs = log_gabor_transform(data['img_crop'], K)
autoenc_inputs = log_gabor_coefs[:,:n_levels-1,...].clone()
#autoenc_inputs /=  256 # !! Normalization
code = enc(autoenc_inputs)


# In[486]:


64*12*8


# In[487]:


128*6*4


# In[488]:


np.exp(5)


# In[489]:


code.shape
del enc


# In[490]:


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
        
        '''self.unconv4b = nn.ConvTranspose2d(128, 
                                 2048, 
                                 kernel_size = (1,1), 
                                 stride = (1,1), 
                                 padding = (0,0),
                                 output_padding=0) #,'''
        self.unconv4 = nn.ConvTranspose2d(512, 
                                 256, 
                                 kernel_size = (2,2), 
                                 stride = (2,1), 
                                 padding = (1,0),
                                 output_padding=(1,0))   
        '''self.unconv3b = nn.ConvTranspose2d(64, 
                                 1024, 
                                 kernel_size = (1,1), 
                                 stride = (1,1), 
                                 padding = (0,0),
                                 output_padding=0) #,'''
        self.unconv3 = nn.ConvTranspose2d(256, 
                                 128, 
                                 kernel_size = (3,3), 
                                 stride = (2,2), 
                                 padding = (1,1),
                                 output_padding=(1,1))       
        '''self.unconv2b = nn.ConvTranspose2d(32, 
                                 512, 
                                 kernel_size = (1,1), 
                                 stride = (1,1), 
                                 padding = (0,0),
                                 output_padding=0) #,'''
        self.unconv2 = nn.ConvTranspose2d(  128, 
                                 64, 
                                 kernel_size = (3,3), 
                                 stride = (2,2), 
                                 padding = (1,1),
                                 output_padding=(1,1))         
        '''self.unconv1b = nn.ConvTranspose2d(16, 
                                 256, 
                                 kernel_size = (1,1), 
                                 stride = (1,1), 
                                 padding = (0,0),
                                 output_padding=0)'''
        self.unconv1 = nn.ConvTranspose2d(64, 
                                 n_color * n_theta * n_phase, 
                                 kernel_size = (3,3), 
                                 stride = (2,2), 
                                 padding = (1,1),
                                 output_padding=(1,1))
            
    def forward(self, x): # , indices1, indices2):           
        #print(x.shape)
        #x = self.unconv4b(x) 
        #x = nn.ReLU()(x) # sparse code
        #x = self.unconv4(x) # dense code
        
        #print(x.shape)
        #x = self.unconv3b(x) 
        #x = nn.ReLU()(x) # sparse code
        #x = self.unconv3(x) # dense code
        
        #print(x.shape)
        #x = self.unconv2b(x) 
        #x = nn.ReLU()(x) # sparse code
        x = self.unconv2(x) # dense code
        #print(x.shape)
        #x = self.unconv1b(x)
        x = nn.ReLU()(x) # sparse code
        x = self.unconv1(x)
        #print(x.shape)
        x = x.view(-1, self.n_color, self.n_theta, self.n_phase, self.n_levels, self.n_eccentricity, self.n_azimuth)
        x = x.permute(0, 4, 1, 5, 6, 2, 3).contiguous()
        return x


# #### Tests

# In[491]:


dec = Decoder(n_levels - 1, n_color, n_eccentricity, n_azimuth, n_theta, n_phase)
dec_out = dec(code) #, indices1, indices2)
dec_out.shape


# In[492]:


del dec


# In[493]:


class InverseLogGaborMapper(nn.Module):
    def __init__(self, in_chan = n_eccentricity * n_azimuth * n_theta * n_phase, 
                 out_chan = width * width):
        super(InverseLogGaborMapper, self).__init__()
        self.inverseMap = nn.Linear(in_chan, out_chan)
        
    def forward(self, x, **kargs):
        out = self.inverseMap(x) #!!
        return out #!!


# ### Model and learning params

# In[1502]:


batch_size = 50
autoenc_lr = 1e-4
invLG_lr = 1e-4

n_epoch = 10000
recording_steps = 10


# In[1503]:


'''if False:
    fic_name = '2021-03-10-log-polar-deep-convolutional-no-max-pool-laplace-lab'+'_autoenc.pt'
    autoenc = torch.load(fic_name)
    for param in autoenc.encoder_l.parameters():
        param.requires_grad = False
    for param in autoenc.decoder_l.parameters():
        param.requires_grad = False
else:'''
'''autoenc = AutoEncoder(n_levels, n_color, n_eccentricity, n_azimuth, n_theta, 
                          n_phase, out_chan=out_chan, 
                      is_VAE=False, residual_encode = True)'''
invLGmap = InverseLogGaborMapper()
autoenc_VAE = AutoEncoder(n_levels, n_color, n_eccentricity, n_azimuth, n_theta, 
                          n_phase, out_chan=out_chan,
                         is_VAE=True,
                         residual_encode=True)


# In[1504]:


autoenc_VAE_optimizer = optim.Adam(autoenc_VAE.parameters(), lr = autoenc_lr)
        
invLG_optimizer = optim.Adam(invLGmap.parameters(), lr = invLG_lr)

criterion = nn.MSELoss() #loss = criterion(outputs, inputs)


# In[1505]:


dataloader = DataLoader(saccade_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)


# In[1506]:


KL_loss_list = []
MSE_loss_list = []
invLG_loss_list = []


# In[1507]:


script_name


# In[1508]:


PATH = script_name + '_invLGmap.pt'

if not os.path.exists(PATH):    

    for epoch in range(n_epoch):  # loop over the dataset multiple times

        KL_running_loss = 0.0
        MSE_running_loss = 0.0
        invLG_running_loss = 0.0
        for step, data in enumerate(dataloader):

            batch_size_eff = data['img_crop'].shape[0]
            
            log_gabor_coefs = log_gabor_transform(data['img_crop'], K)

            # Normalizing
            autoenc_inputs = log_gabor_coefs.clone()
            if color_mode == 'rgb':
                autoenc_inputs /=  256 # !! Normalization
                
            autoenc_outputs, mu, logvar, z = autoenc_VAE(autoenc_inputs)
            autoenc_VAE_optimizer.zero_grad()
            MSE_loss = 0.5 * nn.MSELoss(reduction='sum')(autoenc_outputs, autoenc_inputs)
            KL_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            autoenc_VAE_loss = MSE_loss + KL_loss
            autoenc_VAE_loss.backward()
            autoenc_VAE_optimizer.step()   

            invLG_optimizer.zero_grad()
            log_gabor_coefs_rec = autoenc_outputs.detach().view(batch_size_eff*n_levels*n_color,
                                                               n_eccentricity*n_azimuth*n_theta*n_phase)
            img_pyr_rec_rec = invLGmap(log_gabor_coefs_rec)

            img_pyr_targets = data['img_crop'][:,:n_levels,...].contiguous()
            img_pyr_targets = img_pyr_targets.view(batch_size_eff * n_levels * n_color, 
                                                   width * width)
            if color_mode == 'rgb':
                img_pyr_targets /=  256 # !! Normalization
            invLG_loss = nn.MSELoss(reduction='sum')(img_pyr_rec_rec, img_pyr_targets)
            invLG_loss.backward()             
            invLG_optimizer.step()

            # print statistics
            KL_running_loss += KL_loss.item()
            MSE_running_loss += MSE_loss.item()
            invLG_running_loss += invLG_loss.item()
            if (step+1)%recording_steps == 0 :    # print every n_steps mini-batches
                print('[%d, %5d] losses: %.3f, %.3f, %.3f' %
                      (epoch + 1, 
                       step + 1, 
                       KL_running_loss/recording_steps, 
                       MSE_running_loss/recording_steps,
                       invLG_running_loss/recording_steps))
                #.append
                KL_loss_list.append(KL_running_loss/recording_steps)
                MSE_loss_list.append(MSE_running_loss/recording_steps)
                invLG_loss_list.append(invLG_running_loss/recording_steps)
                KL_running_loss = 0.0
                MSE_running_loss = 0.0
                invLG_running_loss = 0.0

        if epoch % 10 == 0 :
            PATH = script_name + '_KL_loss_list.npy'
            np.save(PATH, np.array(KL_loss_list))    
            PATH = script_name + '_MSE_loss_list.npy'
            np.save(PATH, np.array(MSE_loss_list))    
            PATH = script_name + '_invLG_loss_list.npy'
            np.save(PATH, np.array(invLG_loss_list))   
            PATH = script_name + '_invLGmap.pt'
            torch.save(invLGmap, PATH)
            #PATH = script_name + '_autoenc.pt'
            #torch.save(autoenc, PATH)
            PATH = script_name + '_autoenc_VAE.pt'
            torch.save(autoenc_VAE, PATH)
            print('Model saved')
        
    print('Finished Training ')
    
    

else:
    PATH = script_name + '_KL_loss_list.npy'
    KL_loss_list = np.load(PATH).tolist()    
    PATH = script_name + '_MSE_loss_list.npy'
    MSE_loss_list = np.load(PATH).tolist()    
    PATH = script_name + '_invLG_loss_list.npy'
    invLG_loss_list = np.load(PATH).tolist()
    PATH = script_name + '_invLGmap.pt'
    invLGmap = torch.load(PATH)
    #PATH = script_name + '_autoenc.pt'
    #autoenc = torch.load(PATH)
    PATH = script_name + '_autoenc_VAE.pt'
    autoenc_VAE = torch.load(PATH)
    print('Model loaded')


# In[1509]:


plt.plot(autoenc_inputs[0,...].detach().numpy().flatten())


# In[1510]:


log_gabor_coefs = log_gabor_transform(data['img_crop'], K)
autoenc_inputs = log_gabor_coefs.clone()
autoenc_outputs, mu, logvar, z = autoenc_VAE(autoenc_inputs)
#plt.plot(torch.randn_like(logvar)[0,...].detach().numpy().flatten())


# In[1511]:


out_chan


# In[1512]:


if False :
    PATH = script_name + '_KL_loss_list.npy'
    np.save(PATH, np.array(KL_loss_list))    
    PATH = script_name + '_MSE_loss_list.npy'
    np.save(PATH, np.array(MSE_loss_list))    
    PATH = script_name + '_invLG_loss_list.npy'
    np.save(PATH, np.array(invLG_loss_list))   
    PATH = script_name + '_invLGmap.pt'
    torch.save(invLGmap, PATH)
    #PATH = script_name + '_autoenc.pt'
    #torch.save(autoenc, PATH)
    PATH = script_name + '_autoenc_VAE.pt'
    torch.save(autoenc_VAE, PATH)
    print('Model saved')


# In[1513]:


import seaborn
seaborn.set()
plt.figure(figsize=(12,12))
plt.plot(np.array(MSE_loss_list), label = 'MSE')
plt.plot(np.array(KL_loss_list)*100, label = 'KL')
plt.plot(np.array(invLG_loss_list)*10, label = 'invLGMap')
#plt.ylim(0,500)
plt.title('LOSS')
plt.xlabel('# batch')
plt.legend()
#plt.ylim(0,1000000000000)


# In[1514]:


plt.hist(mu.detach().numpy().flatten(),20)
plt.figure()
plt.hist(logvar.detach().numpy().flatten(),20)


# ## Encoding and decoding

# In[1515]:


seaborn.reset_orig()


# In[1516]:


for i in range(batch_size):
    print(data['name'][i])


# In[1517]:


img_name = 'i1198772915'
if True:
    locpath = '../ALLSTIMULI/' + img_names[10] + '.jpeg'
    locpath = '../ALLSTIMULI/' + data['name'][11] + '.jpeg'
    #locpath = '../ALLSTIMULI/' + img_name + '.jpeg'
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


# In[1518]:


img_tens = torch.Tensor(np.array(img_orig)[None,...]).permute(0,3,1,2)


# In[1519]:


img_crop = cropped_pyramid(img_tens, 
                           width=width,
                           color=color, 
                           do_mask=do_mask, 
                           verbose=True, 
                           n_levels=n_levels)[0]


# In[1520]:


log_gabor_coeffs = log_gabor_transform(img_crop, K)
log_gabor_coeffs.shape


# In[1521]:


img_rec=inverse_gabor(log_gabor_coeffs.detach(), K_inv)
if False:
    img_rec[:,-1,...]= img_crop[:,-1,...]
full_img_rec = inverse_pyramid(img_rec, color=color, gauss=gauss, n_levels=n_levels)
full_img_rec = full_img_rec.detach().permute(0,2,3,1).numpy() #.clip(0,255).astype('uint8')
plt.figure(figsize=(20,15))
image_show(full_img_rec[0,:], color_mode=color_mode)
N_X, N_Y = full_img_rec.shape[1:3]


# In[1522]:


full_img_rec.shape


# In[1523]:


autoenc_VAE.eval()

autoenc_inputs = log_gabor_coeffs.clone()

if color_mode == 'rgb':
    autoenc_inputs /= 256

log_gabor_coeffs_rec, mu, logvar, z = autoenc_VAE( autoenc_inputs )   
log_gabor_coeffs_rec = log_gabor_coeffs_rec.view(1, n_levels, -1) 
if color_mode == 'rgb':
    log_gabor_coeffs_rec *= 256


# In[1524]:


z


# In[1525]:


plt.figure(figsize=(20,7))
plt.plot(log_gabor_coeffs.numpy().flatten()[:], label = 'original')
plt.plot(log_gabor_coeffs_rec.detach().numpy().flatten()[:], label = 'reconstructed')
plt.title('LOG GABOR COEFFS')
plt.legend()
for level in range(n_levels):
    plt.figure(figsize=(20,4))
    plt.plot(log_gabor_coeffs[0,level,...].numpy().flatten(), label = 'original')
    plt.plot(log_gabor_coeffs_rec[0,level,...].detach().numpy().flatten(), label = 'reconstructed')
    c = np.corrcoef([log_gabor_coeffs[0,level,...].numpy().flatten(), log_gabor_coeffs_rec[0,level,...].detach().numpy().flatten()])[0,1]
    plt.title('LOG GABOR COEFFS LEVEL '+str(level)+', corr='+str(c))
    plt.legend()


# In[1526]:


_=plt.hist(log_gabor_coeffs.numpy().flatten(),100)


# In[1527]:


_=plt.hist(img_crop.numpy().flatten(),100)


# ## Reconstruction tests

# In[1528]:


K_inv = get_K_inv(K, width=width, n_sublevel = n_sublevel, n_azimuth = n_azimuth, n_theta = n_theta, n_phase = n_phase)
img_rec=inverse_gabor(log_gabor_coeffs.detach(), K_inv)
img_rec[:,-1,...] = img_crop[:,-1,...]
axs = tensor_pyramid_display(img_rec.clone()) 


# In[1529]:


inv_LGmap_input = log_gabor_coeffs_rec.view(n_levels * n_color, n_eccentricity * n_azimuth * n_theta * n_phase)
inv_LGmap_input.shape


# In[1530]:


img_rec_rec = invLGmap(inv_LGmap_input) #inv_LGmap_input)
img_rec_rec = img_rec_rec.view(1, n_levels, n_color, width, width).detach()
#img_rec_rec = torch.cat((img_rec_rec, img_crop[0:,-1:,...]), 1)
#img_rec_rec[0,-1,...] *=0 #+= 128
axs = tensor_pyramid_display(img_rec_rec)


# ### Test de invLGmap uniquement sur log gabor coeffs originaux

# In[1531]:


img_rec_test = invLGmap(log_gabor_coeffs.view(n_levels * n_color, n_eccentricity * n_azimuth * n_theta * n_phase)) #inv_LGmap_input)
img_rec_test = img_rec_test.view(1, n_levels, n_color, width, width).detach()
img_rec_test[:,-1,...] = img_crop[:,-1,...]
axs = tensor_pyramid_display(img_rec_test)


# ### Test des coeffs reconstruits avec differentes valeurs de K_inv 
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
    #img_rcond_test[:,-1,...] = img_crop[:,-1,...]
    axs = tensor_pyramid_display(img_rcond_test)
    axs[0].set_title('REGULARIZATION = '+str(rcond)+', ORIGINAL LOG-GABOR COEFS')
    img_rec_rcond_test = inverse_gabor(log_gabor_coeffs_rec.detach(), K_inv_test)
    #img_rec_rcond_test[:,-1,...] = img_crop[:,-1,...]
    img_rec_rec_test.append(img_rec_rcond_test)
    axs = tensor_pyramid_display(img_rec_rcond_test)
    axs[0].set_title('AUTO-ENCODER LOG-GABOR RECONSTRUCTION')    
# ### Full image reconstruction

# In[1532]:


#img_crop = cropped_pyramid(img_tens, color=color, do_mask=do_mask, verbose=True, n_levels=n_levels)[0]
N_X, N_Y = full_img_rec.shape[1:3]


full_img_crop = inverse_pyramid(img_crop, color=color, gauss=gauss, n_levels=n_levels)
full_img_crop = full_img_crop.detach().permute(0,2,3,1).numpy()
    
plt.figure(figsize=(20,15))
image_show(full_img_crop[0,:], color_mode)
plt.title('RECONSTRUCTED FROM CROPPED PYRAMID, #params = ' + str(np.prod(img_crop[0,...].size())), fontsize=20)


img_rec=inverse_gabor(log_gabor_coeffs.detach(), K_inv)
#img_rec[:,-1,...]= img_crop[:,-1,...]
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
plt.title('RECONSTRUCTED FROM AUTO-ENCODER, #params = ' + str(out_chan), fontsize=20)

'''
img_rec_rec_test[3][:,-1,...]= img_crop[:,-1,...]
full_img_rec_rec_test = inverse_pyramid(img_rec_rec_test[3], color=color, gauss=gauss, n_levels=n_levels)
full_img_rec_rec_test = full_img_rec_rec_test.detach().permute(0,2,3,1).numpy().clip(0,255).astype('uint8')
#ax = tensor_image_cmp(full_img_rec, full_img_rec_rec)
plt.figure(figsize=(20,15))
plt.imshow(full_img_rec_rec_test[0,:])
plt.title('RECONSTRUCTED FROM AUTOENCODER OUTPUTS AND REGULARIZED INVERSE MAP')
'''

if False:
    plt.savefig(script_name+'_soleil_levant.png', bbox_inches='tight')


# In[1533]:


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
#img_rec[:,-1,...]= img_crop[:,-1,...]
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
if False:
    plt.savefig(script_name+'.png', bbox_inches='tight')


# In[1534]:


img.shape

log_gabor_coeffs_roll = log_gabor_coeffs_rec.clone()
log_gabor_coeffs_roll = log_gabor_coeffs_roll.view(1,n_levels, n_color, n_eccentricity, n_azimuth, n_theta, n_phase)
#log_gabor_coeffs_roll[:,:n_levels-1,...]= log_gabor_coeffs_roll[:,:n_levels-1,...].roll(-4,4) #.roll(4, 4)
log_gabor_coeffs_roll= log_gabor_coeffs_roll.roll(1,4) #.roll(4, 4)
log_gabor_coeffs_roll= log_gabor_coeffs_roll.roll(1,1) #.roll(4, 4)

#log_gabor_coeffs_roll= log_gabor_coeffs_roll.roll(-1,5) #.roll(4, 4)
inv_LGmap_input = log_gabor_coeffs_roll.view((n_levels) * n_color, n_eccentricity * n_azimuth * n_theta * n_phase)
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
# In[1535]:


z_test = torch.randn_like(mu)
logvar.shape

autoenc.out_chan
autoenc_inputs.shape

log_gabor_coeffs_rec, mu, logvar, z = autoenc_VAE( autoenc_inputs)   
z_in = torch.randn_like(mu) * 30
log_gabor_coeffs_rec_test, mu, logvar, z = autoenc_VAE( autoenc_inputs, z_in=z_in )   
inv_LGmap_input = log_gabor_coeffs_rec_test.view(n_levels * n_color, n_eccentricity * n_azimuth * n_theta * n_phase)
img_rec_rec_test = invLGmap(inv_LGmap_input) #inv_LGmap_input)
img_rec_rec_test = img_rec_rec_test.view(1, n_levels, n_color, width, width).detach()
#img_rec_rec_test = torch.cat((img_rec_rec_test, img_crop[0:,-1:,...]), 1)
full_img_rec_rec_test = inverse_pyramid(img_rec_rec_test, color=color, gauss=gauss, n_levels=n_levels)
full_img_rec_rec_test = full_img_rec_rec_test.detach().permute(0,2,3,1).numpy()
plt.figure(figsize=(20,15))
image_show(full_img_rec_rec_test[0,:], color_mode)

# In[1536]:


log_gabor_coeffs = log_gabor_transform(img_crop, K)
autenc_inputs_VAE_test = log_gabor_coeffs.clone()
autenc_inputs_VAE_test[0,:6,...] = 0

plt.figure()
plt.plot(autenc_inputs_VAE_test.detach().numpy().flatten())


# In[1537]:


#log_gabor_coeffs_rec, mu, logvar, z = autoenc_VAE(autenc_inputs_test)   
z_in = None #torch.randn_like(mu) * 30
log_gabor_coeffs_rec_test, mu, logvar, z = autoenc_VAE(autenc_inputs_VAE_test, z_in=z_in )   
inv_LGmap_input = log_gabor_coeffs_rec_test.view(n_levels * n_color, n_eccentricity * n_azimuth * n_theta * n_phase)
img_rec_rec_test = invLGmap(inv_LGmap_input) #inv_LGmap_input)
img_rec_rec_test = img_rec_rec_test.view(1, n_levels, n_color, width, width).detach()
#img_rec_rec_test = torch.cat((img_rec_rec_test, img_crop[0:,-1:,...]), 1)
full_img_rec_rec_test = inverse_pyramid(img_rec_rec_test, color=color, gauss=gauss, n_levels=n_levels)
full_img_rec_rec_test = full_img_rec_rec_test.detach().permute(0,2,3,1).numpy()
plt.figure(figsize=(20,15))
image_show(full_img_rec_rec_test[0,:], color_mode)
plt.figure()
plt.plot(log_gabor_coeffs_rec_test.detach().numpy().flatten())

for _ in range(5):
    autoenc_inputs_rec = log_gabor_coeffs_rec_test
    autoenc_inputs_rec[0,6,...] = autenc_inputs_VAE_test[0,6,...]
    log_gabor_coeffs_rec_test, mu, logvar, z = autoenc_VAE( autoenc_inputs_rec, z_in=z_in )   
    inv_LGmap_input = log_gabor_coeffs_rec_test.view(n_levels * n_color, n_eccentricity * n_azimuth * n_theta * n_phase)
    img_rec_rec_test = invLGmap(inv_LGmap_input) #inv_LGmap_input)
    img_rec_rec_test = img_rec_rec_test.view(1, n_levels, n_color, width, width).detach()
    #img_rec_rec_test = torch.cat((img_rec_rec_test, img_crop[0:,-1:,...]), 1)
    full_img_rec_rec_test = inverse_pyramid(img_rec_rec_test, color=color, gauss=gauss, n_levels=n_levels)
    full_img_rec_rec_test = full_img_rec_rec_test.detach().permute(0,2,3,1).numpy()
    plt.figure(figsize=(20,15))
    image_show(full_img_rec_rec_test[0,:], color_mode)

    plt.figure()
    plt.plot(log_gabor_coeffs_rec_test.detach().numpy().flatten())


# In[1538]:


logvar


# In[1539]:


T = torch.FloatTensor(((1,2),(3,4)))


# In[1540]:


T[0,:]


# In[1541]:


T.view(4)


# In[1542]:


print("#params :" + str(np.prod(code[0,...].size())))


# In[1543]:


code[0,...].shape


# In[1544]:


plt.plot(data['img_crop'].flatten())


# In[1545]:


32 * n_levels * n_eccentricity//4 


# In[1546]:


n_levels


# In[1547]:


n_eccentricity//4 


# In[ ]:




