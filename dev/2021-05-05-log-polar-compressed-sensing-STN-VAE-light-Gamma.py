#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
r_min = width / 8 
r_max = width / 2 
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


out_chan = 1024
gauss = False
do_mask = False
color = True
color_mode= 'lab' # 'hsv' 
print ('encoder #params :', out_chan)


# In[9]:


if gauss:
    script_name = '2021-05-05-log-polar-compressed-sensing-VAE-gauss-'+color_mode
else:
    script_name = '2021-05-05-log-polar-compressed-sensing-pool-VAE-laplace-'+color_mode


# ### Image utilities

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


# ### Gabor filters pseudo-inverse

# In[16]:


K_inv = get_K_inv(K, width=width, n_sublevel = n_sublevel, n_azimuth = n_azimuth, n_theta = n_theta, n_phase = n_phase)
plt.plot(K_inv.flatten())


# ### Honeycomb space coverage tests

# In[17]:


plt.figure(figsize=(20,3))
for i_theta in range(n_theta):
    coefs = torch.zeros((n_sublevel, n_azimuth, n_theta, n_phase))
    coefs[0, n_azimuth//2, i_theta, 0] = 1
    img_dis = torch.tensordot(K, coefs, dims=4)
    plt.subplot(1,n_theta,i_theta+1)
    plt.imshow(img_dis.numpy()[:, :, ...], cmap='gray')


# In[18]:


plt.figure(figsize=(20,6))
for i_az in range(n_azimuth):
    coefs = torch.zeros((n_sublevel, n_azimuth, n_theta, n_phase))
    coefs[:, i_az, 0, 0] = 1
    img_dis = torch.tensordot(K, coefs, dims=4)
    plt.subplot(2,n_azimuth//2,i_az+1)
    plt.imshow(img_dis.numpy()[:, :, ...], cmap='gray')


# In[19]:


coefs = torch.zeros((n_sublevel, n_azimuth, n_theta, n_phase))
coefs[:, :, 2:3, 0] = torch.ones((n_sublevel, n_azimuth, 1))
img_dis = torch.tensordot(K, coefs, dims=4)
plt.subplot(1,2,1)
plt.imshow(img_dis.numpy(), cmap='gray')
plt.subplot(1,2,2)
_=plt.plot(img_dis.numpy())


# ## Images dataset + transforms

# In[20]:


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


# In[21]:


dir_names = os.listdir('../saccades-data')
loc_data_xy={}
for dir_name in dir_names:
    loc_data_xy[dir_name]={}
    for name in img_names:
        locpath = '../saccades-data/' + dir_name + '/' + name
        f = open(locpath,'rb')
        loc_dict = pickle.load(f)
        loc_data_xy[dir_name][name] = np.array(loc_dict['barycenters'])


# In[22]:


def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    #plt.pause(0.001)  # pause a bit so that plots are updated


# # Dataset class

# In[23]:


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

        name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, name + '.jpeg')
        image = io.imread(img_path)
        dir_name = np.random.choice(self.dir_names)
        landmarks = self.loc_dict[dir_name][name]
        landmarks = np.array([landmarks])
        landmarks = landmarks.reshape(-1, 2) #.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks, 'name':name}

        if self.transform:
            sample = self.transform(sample)

        return sample


# # Transforms

# In[24]:


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


# In[25]:


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image_tens = sample['image'].transpose((2, 0, 1))
        return {'image': torch.FloatTensor(image_tens), 'pos': sample['pos'],  'name':sample['name']}


# ### Adapted cropped pyramid (squeezed tensor)

# In[26]:


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
        return{'img_crop':img_crop, 'level_size':level_size, 'pos':sample['pos'],  'name':sample['name']}
        
    


# ### LogGaborTransform

# In[27]:


class LogGaborTransform(object):
    def __init__(self, K=K, color=color, verbose=False):
        self.K = K
        self.color = color
        self.verbose = verbose
    
    def __call__(self, sample):
        log_gabor_coeffs = log_gabor_transform(sample['img_crop'].unsqueeze(0), K)
        
        return{'img_gabor':log_gabor_coeffs, 'pos':sample['pos'],  'name':sample['name']}


# # Compose transforms
# ### transforms.Compose

# In[28]:


composed_transform = transforms.Compose([RandomSaccadeTo(zero_fill=True),
                               ToTensor(),
                               CroppedPyramid(width, 
                                              base_levels, 
                                              n_levels=n_levels,
                                              color_mode=color_mode)]) #, LogGaborTransform()])


# In[29]:


saccade_dataset = SaccadeLandmarksDataset(loc_dict=loc_data_xy,
                                          img_dir='../ALLSTIMULI/',
                                          img_names=img_names,
                                          dir_names =  dir_names,
                                          transform=composed_transform,
                                          color_mode=color_mode)


# # Iterating through the dataset

# In[30]:


# Helper function to show a batch

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
        plt.imshow(grid.numpy().transpose((1, 2, 0)).clip(0,255).astype('uint8'))

        plt.title('Batch from dataloader, level=' + str(level))


# In[31]:


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
        


# In[32]:


plt.plot(sample_batched['img_crop'].flatten())


# In[33]:


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
    plt.title(sample_batched['name'])


# In[34]:


plt.plot(sample_batched['img_crop'][0,:,0,:,:].flatten())

sample_batched['name']
locpath = '../ALLSTIMULI/' + sample_batched['name'][0] + '.jpeg'
img_orig = Image.open(locpath) 
plt.imshow(img_orig)
# ## STN

# In[205]:


class SpatialTransformer(nn.Module):
    """A spatial transformer plug and play module.
    
    Attributes
    ----------
    self.localization: nn.Sequential
        The localization network of the spatial transformer.
        
    self.fc_loc: nn.Sequential
        The regressor for the transformation parameters theta, fully connected
        layers.

    """
    def __init__(self: object,
                 n_levels, 
                 n_color, 
                 n_eccentricity, 
                 n_azimuth, 
                 n_theta, 
                 n_phase) -> None:
        """Class constructor.

        Returns
        -------
        None.

        """
        super(SpatialTransformer, self).__init__()
        
        self.n_levels = n_levels
        self.n_color = n_color
        self.n_eccentricity = n_eccentricity 
        self.n_azimuth = n_azimuth 
        self.n_theta = n_theta
        self.n_phase = n_phase

        self.localization = STNEncoder(n_levels, 
                                    n_color, 
                                    n_eccentricity, 
                                    n_azimuth, 
                                    n_theta, 
                                    n_phase)
        
        self.h_size = n_levels * n_azimuth//4 * 128

        self.fc_loc = nn.Sequential(
            nn.Linear(self.h_size, 128, bias=False),
            nn.ReLU(True),
            nn.Linear(128, 2, bias=False)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        #self.fc_loc[2].bias.data.zero_() #copy_(torch.tensor([1, 0, 0, 0, 1, 0],
                                         #           dtype=torch.float))

    def stn(self: object, x: torch.Tensor) -> torch.Tensor:
        """The Spatial Transformer module's forward function, pass through
        the localization network, predict transformation parameters theta,
        generate a grid and apply the transformation parameters theta on it
        and finally sample the grid using an interpolation.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        x : torch.Tensor
            The output (transformed) tensor.

        """
        #x_ext = x[:,:,:,2:,...]
        #x_ext = x_ext.permute(0, 2, 5, 6, 1, 3, 4).contiguous()
        #x_ext = x_ext.view(-1, self.n_color*self.n_theta*self.n_phase, 
        #               self.n_levels * self.n_eccentricity//2, self.n_azimuth)
        xs = self.localization(x, relu=True)
        xs = xs.view(-1, self.h_size)
        theta = torch.zeros(x.shape[0],2,3) #, requires_grad=True)
        theta[:,0,0] = 1
        theta[:,1,1] = 1
        theta[:,:,2] = self.fc_loc(xs)
        # resizing theta
        theta = theta.view(-1, 2, 3)
        # grid generator => transformation on parameter 
        

        return theta


# In[206]:


class STNEncoder(nn.Module):
    """ Encoder
    """
    def __init__(self, n_levels, n_color, n_eccentricity, n_azimuth, n_theta, n_phase):
        super(STNEncoder, self).__init__()
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
            
    def forward(self, x, relu=False): 
        x_int = x[:,:,:,:2,...] #eccentricity 
        x_ext = x[:,:,:,2:,...]
        x_list = []
        for x in (x_int, x_ext):
            x = x.permute(0, 2, 5, 6, 1, 3, 4).contiguous()
            x = x.view(-1, self.n_color*self.n_theta*self.n_phase, 
                       self.n_levels * self.n_eccentricity//2, self.n_azimuth)
            x = self.conv1(x)
            if relu:
                x = nn.ReLU()(x) 
            #print(x.shape)

            x = self.conv2(x)
            if relu:
                x = nn.ReLU()(x) 
            
            x_list.append(x)
        x = torch.cat(x_list, 1)
        #print(x.shape)
        
        return x


# ### Autoencoder

# In[207]:


class AutoEncoder(nn.Module):
    def __init__(self, n_levels, n_color, n_eccentricity, n_azimuth, n_theta, n_phase, 
                 out_chan = 32,
                 encoder=None,  decoder=None):
        super(AutoEncoder, self).__init__()
        self.n_levels = n_levels
        self.n_color = n_color
        self.n_eccentricity = n_eccentricity 
        self.n_azimuth = n_azimuth 
        self.n_theta = n_theta
        self.n_phase = n_phase
        self.out_chan = out_chan
        
        self.transformer_module = SpatialTransformer(n_levels, 
                                                     n_color, 
                                                     n_eccentricity, 
                                                     n_azimuth, 
                                                     n_theta, 
                                                     n_phase)
        
        if encoder is None:
            self.encoder = Encoder(n_levels, 
                                   n_color, 
                                   n_eccentricity, 
                                   n_azimuth, 
                                   n_theta, 
                                   n_phase)
        else:
            self.encoder = encoder
        
        self.h_size = n_levels * n_color * n_eccentricity * n_azimuth  * n_theta * n_phase
         
        self.fc_mu = nn.Linear(self.h_size, out_chan, bias=False)
        with torch.no_grad():
            self.fc_mu.weight *= 1e-6
        self.fc_logvar = nn.Linear(self.h_size, out_chan, bias=False)      
        with torch.no_grad():
            self.fc_logvar.weight *= 1e-6
        self.fc_z_inv = nn.Linear(out_chan, self.h_size, bias=False)            
        
        if decoder is None:
            self.decoder = Decoder(n_levels, n_color, n_eccentricity, n_azimuth, n_theta, n_phase)    
        else:
            self.decoder = decoder
        
    def forward(self, x, z_in=None, theta_in=None, pre_train = False):   
        
        if not pre_train:
            if theta_in is None: 
                theta = self.transformer_module.stn(x)
            else:
                theta = theta_in
        
        
        if not pre_train:
            code = self.encoder(x, theta)  
        else:
            code = x #.view(-1, self.h_size)
            
        mu = self.fc_mu(code.view(-1, self.h_size))
        logvar = self.fc_logvar(code.view(-1, self.h_size))

        if z_in is None:           
            alpha = nn.Softplus()(mu) 
            #beta = torch.exp(-logvar / 2)
            beta = nn.Softplus()(logvar)
            q = torch.distributions.Gamma(alpha,beta)
            z = q.rsample()
        else:
            z = z_in
        
        decode = self.fc_z_inv(z) #.view(-1, self.h_size)     
        
        if not pre_train:
            x = self.decoder(decode, theta.clone()) #.clone())
        else:
            x = decode.view(-1, self.n_levels, self.n_color, self.n_eccentricity, self.n_azimuth, self.n_theta, self.n_phase)
        return x, mu, logvar, z


# In[208]:


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
        self.h_size = n_levels * n_color * n_eccentricity * n_azimuth  * n_theta * n_phase
        # Layers
            
    def forward(self, x, theta=None): 
        x_int = x[:,:,:,:2,...] #eccentricity 
        x_ext = x[:,:,:,2:,...]
        x_list = []
        for x in (x_int, x_ext):
            x = x.permute(0, 2, 5, 6, 1, 3, 4).contiguous()
            x = x.view(-1, self.n_color*self.n_theta*self.n_phase, 
                       self.n_levels * self.n_eccentricity//2, self.n_azimuth)
            if theta is not None:
                grid = F.affine_grid(theta, x.size())
                x = F.grid_sample(x, grid)
            x_list.append(x)
        x = torch.cat(x_list, 1)
        #x = x.view(-1, self.h_size)
        #print(x.shape)
        
        return x


# In[209]:


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
        
        self.h_size = n_levels * n_color * n_eccentricity * n_azimuth  * n_theta * n_phase
            
    def forward(self, x, theta=None): 
        x = x.view(-1, self.n_color*self.n_theta*self.n_phase, 
                       self.n_levels * self.n_eccentricity, self.n_azimuth)
        lim = self.n_levels * self.n_eccentricity // 2
        x_int = x[:,:,:lim,...]
        x_ext = x[:,:,lim:,...]
        x_list = []
        for x in (x_int, x_ext):
            if theta is not None:
                theta_inv = theta
                theta_inv[:,:,2] = - theta[:,:,2].detach()
                grid = F.affine_grid(theta_inv, x.size())
                x = F.grid_sample(x, grid)
            x = x.view(-1, self.n_color, self.n_theta, self.n_phase, self.n_levels, self.n_eccentricity//2, self.n_azimuth)
            x = x.permute(0, 4, 1, 5, 6, 2, 3).contiguous()
            x_list.append(x)
        x = torch.cat(x_list, 3)      
        return x


# In[210]:


class InverseLogGaborMapper(nn.Module):
    def __init__(self, in_chan = n_eccentricity * n_azimuth * n_theta * n_phase, 
                 out_chan = width * width):
        super(InverseLogGaborMapper, self).__init__()
        self.inverseMap = nn.Linear(in_chan, out_chan)
        
    def forward(self, x, **kargs):
        out = self.inverseMap(x) #!!
        return out #!!


# ### Loss functions

# In[211]:


def mc_kl_div(z, mu, logvar):
    # Monte Carlo KL divergence
    #std = torch.exp(logvar / 2)
    #p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    #p = torch.distributions.Laplace(torch.zeros_like(mu), torch.ones_like(std))
    p = torch.distributions.Gamma(torch.ones_like(mu), torch.ones_like(mu))
    #q = torch.distributions.Normal(mu, std)
    alpha = nn.Softplus()(mu) #nn.ReLU()(mu) + torch.ones_like(mu) 
    #alpha = torch.exp(mu) 
    beta = nn.Softplus()(logvar) #nn.ReLU()(mu) + torch.ones_like(mu) 
    #beta = torch.exp(-logvar / 2)
    q = torch.distributions.Gamma(alpha, beta)
    #q = torch.distributions.Laplace(mu, std)
    log_posterior_sample = q.log_prob(z)
    log_prior_sample = p.log_prob(z)
    kl_sample = log_posterior_sample - log_prior_sample
    return kl_sample.sum()


# In[212]:


def minus_log_likelihood(autoenc_outputs, autoenc_inputs):
    log_scale = nn.Parameter(torch.Tensor([0.]))
    scale = torch.exp(log_scale)
    dist = torch.distributions.Normal(autoenc_outputs, scale)
    log_likelihood = dist.log_prob(autoenc_inputs)
    return -log_likelihood.sum()
    


# ### Model and learning params

# In[247]:


batch_size = 15
autoenc_lr = 1e-6 #3e-10
invLG_lr = 1e-5

n_epoch = 10000 #10000
recording_steps = 10


# In[214]:


autoenc_VAE = AutoEncoder(n_levels-1, n_color, n_eccentricity, n_azimuth, n_theta, 
                          n_phase, out_chan=out_chan)

invLGmap = InverseLogGaborMapper()


# In[215]:


autoenc_VAE_optimizer = optim.Adam(autoenc_VAE.parameters(), lr = autoenc_lr)
        
invLG_optimizer = optim.Adam(invLGmap.parameters(), lr = invLG_lr)
#invLG_optimizer = optim.SGD(invLGmap.parameters(), lr = invLG_lr)
criterion = nn.MSELoss(reduction='sum')


# In[216]:


dataloader = DataLoader(saccade_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)


# In[217]:


KL_loss_list = []
MSE_loss_list = []
invLG_loss_list = []


# In[218]:


script_name


# In[251]:


PATH = script_name + '_invLGmap.pt'

if True: # not os.path.exists(PATH):    

    for epoch in range(n_epoch):  # loop over the dataset multiple times

        KL_running_loss = 0.0
        MSE_running_loss = 0.0
        invLG_running_loss = 0.0
        for step, data in enumerate(dataloader):

            batch_size_eff = data['img_crop'].shape[0]
            
            log_gabor_coefs = log_gabor_transform(data['img_crop'], K)

            # Normalizing
            autoenc_inputs = log_gabor_coefs[:,:n_levels-1,...].clone()
            if color_mode == 'rgb':
                autoenc_inputs /=  256 # !! Normalization
                
            if epoch < 100:
                pre_train = True
            else:
                pre_train = False
            autoenc_outputs, mu, logvar, z = autoenc_VAE(autoenc_inputs, pre_train = pre_train)
            autoenc_VAE_optimizer.zero_grad()
            #MSE_loss = 0.5 * criterion(autoenc_outputs, autoenc_inputs)
            MSE_loss = minus_log_likelihood(autoenc_outputs, autoenc_inputs)
            #KL_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            KL_loss = mc_kl_div(z, mu, logvar)
            autoenc_VAE_loss = MSE_loss + KL_loss
            autoenc_VAE_loss.backward()
            autoenc_VAE_optimizer.step()   

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
                
        if epoch % 100 == 0 :
            PATH = script_name + '_KL_loss_list.npy'
            np.save(PATH, np.array(KL_loss_list))    
            PATH = script_name + '_MSE_loss_list.npy'
            np.save(PATH, np.array(MSE_loss_list))    
            PATH = script_name + '_invLG_loss_list.npy'
            np.save(PATH, np.array(invLG_loss_list))   
            PATH = script_name + '_invLGmap.pt'
            torch.save(invLGmap, PATH)
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
    PATH = script_name + '_autoenc_VAE.pt'
    autoenc_VAE = torch.load(PATH)
    print('Model loaded')

# In[ ]:


if True :
    PATH = script_name + '_KL_loss_list.npy'
    np.save(PATH, np.array(KL_loss_list))    
    PATH = script_name + '_MSE_loss_list.npy'
    np.save(PATH, np.array(MSE_loss_list))    
    PATH = script_name + '_invLG_loss_list.npy'
    np.save(PATH, np.array(invLG_loss_list))   
    PATH = script_name + '_invLGmap.pt'
    torch.save(invLGmap, PATH)
    PATH = script_name + '_autoenc_VAE.pt'
    torch.save(autoenc_VAE, PATH)
    print('Model saved')


# In[252]:


import seaborn
seaborn.set()
plt.figure(figsize=(15,12))
plt.plot(np.array(MSE_loss_list) , label = 'MSE')
plt.plot(np.array(KL_loss_list)*10 , label = 'KL')
plt.plot(np.array(invLG_loss_list), label = 'invLGMap')
#plt.ylim(0,500)
plt.title('LOSS')
plt.xlabel('# batch')
plt.legend()
plt.ylim(0,30000000)



