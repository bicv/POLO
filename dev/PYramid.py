import numpy as np
import matplotlib.pyplot as plt
import os

from SLIP import imread
from SLIP import Image as Image_SLIP
import time

from PIL import Image

import torch
torch.set_default_tensor_type('torch.DoubleTensor')

import imageio

from torch.nn.functional import interpolate 

mode= 'bilinear' #resizing : continuous transition, reduces edges,contrast
width = 32 #side of the cropped image used to build the pyramid
base_levels = 1.61803
base_levels = 2 #downsampling/upsampling factor

N_batch = 4 #number of images 
pattern = 'i05june05_static_street_boston_p1010808'

n_sublevel = 2 #filters dictionnary, number of sublevels
n_azimuth = 12 #retinal transform characteristics 
n_theta = 12
n_phase = 2

img_orig = Image.open('../data/i05june05_static_street_boston_p1010808.jpeg')

im_color_npy = np.asarray(img_orig)
N_X, N_Y, _ = im_color_npy.shape #dimensions 
ds= 1
im=Image_SLIP({'N_X': N_X, 'N_Y': N_Y, 'do_mask': True})



def cropped_pyramid(img_tens, width=width, base_levels=base_levels, color=True, do_mask=True, verbose=False):
    
    N_batch, _, N_X, N_Y = img_tens.shape # tensor of the images  (dimension 4)
    n_levels = int(np.log(np.max((N_X, N_Y))/width)/np.log(base_levels)) + 1 #computing the number of iterations cf:downsampling
    
    if color :
        img_crop = torch.zeros((N_batch, n_levels, 3, width, width))
        level_size=[[N_X, N_Y]]

    else :
        img_crop = torch.zeros((N_batch, n_levels, width, width)) #creating the tensor to store the cropped images while pyramiding
        
    img_down = img_tens.clone()
    for i_level in range(n_levels-1): #each iteration -> residual_image = image - downsampled_cloned_image_reshaped_to_the_right_size 
        img_residual = img_down.clone()
        img_down = interpolate(img_down, scale_factor=1/base_levels, mode=mode) #downsampling
        img_residual -= interpolate(img_down, size=img_residual.shape[-2:], mode=mode)  #upsizing in order to substract

        if verbose: print('Tensor shape=', img_down.shape, ', shape=', img_residual.shape)
        h_res, w_res = img_residual.shape[-2:] #at each iteration the residual image size is reduced of a factor 1/base_levels (img_down the image downsampled at the previous iteration)

        if color :
            img_crop[:, i_level, :, :, :] = img_residual[:, :, 
                            (h_res//2-width//2):(h_res//2+width//2), 
                            (w_res//2-width//2):(w_res//2+width//2)]
            level_size.append(list(img_down.shape[-2:]))
            
        else :
            img_crop[:, i_level, :, :] = img_residual[:, 0, 
                            (h_res//2-width//2):(h_res//2+width//2), 
                            (w_res//2-width//2):(w_res//2+width//2)] #the central crop of residual image stored in tensor img_crop
            level_size=0
            
    h_res, w_res = img_down.shape[-2:]
    
    if color :
        img_crop[:, n_levels-1, :, 
                 (width//2-h_res//2):(width//2+h_res//2), 
                 (width//2-w_res//2):(width//2+w_res//2)] = img_down #[0, :, :, :]
        
    else :
        img_crop[:, n_levels-1, 
             (width//2-h_res//2):(width//2+h_res//2), 
             (width//2-w_res//2):(width//2+w_res//2)] = img_down[:, 0, :, :]
    if verbose: print('Top tensor shape=', img_down.shape, ', Final n_levels=', n_levels) #print image's dimensions after downsampling, condition max(img_down.shape[-2:])<=width satisfied
    
    if do_mask :
        mask_crop = Image_SLIP({'N_X': width, 'N_Y': width, 'do_mask': True}).mask
        if color :
            for i in range(n_levels-1):
                img_crop[0,i,...] *= mask_crop[:,:]
            img_crop[0,n_levels-1,...] = img_crop[0,n_levels-1,...]*mask_crop[:,:]+128*(1-mask_crop[:,:])
        else :
            print(img_crop.shape)
            img_crop *= mask_crop[np.newaxis,np.newaxis,:,:]    #+0.5*(1-mask_crop[np.newaxis,np.newaxis,:,:])            

    return img_crop, level_size


def inverse_pyramid(img_crop, N_X=N_X, N_Y=N_Y, base_levels=base_levels, color=True, verbose=False):
    N_batch = img_crop.shape[0]
    width = img_crop.shape[3]
    n_levels = int(np.log(np.max((N_X, N_Y))/width)/np.log(base_levels)) + 1 #number of cropped images = levels of the pyramid

    if color :
        img_rec = img_crop[:, -1, :, :, :]#.unsqueeze(1)
        for i_level in range(n_levels-1)[::-1]: # from the top to the bottom of the pyramid
            img_rec = interpolate(img_rec, scale_factor=base_levels, mode=mode) #upsampling (factor=base_levels)
            h_res, w_res = img_rec.shape[-2:]
            img_rec[:, :,
                    (h_res//2-width//2):(h_res//2+width//2),
                    (w_res//2-width//2):(w_res//2+width//2)] += img_crop[:, i_level, :, :, :] #adding previous central crop to img_crop
        img_rec = img_rec[:, :, (h_res//2-N_X//2):(h_res//2+N_X//2), (w_res//2-N_Y//2):(w_res//2+N_Y//2)]

    else :
        img_rec = img_crop[:, -1, :, :].unsqueeze(1)
        for i_level in range(n_levels-1)[::-1]: # from the top to the bottom of the pyramid
            img_rec = interpolate(img_rec, scale_factor=base_levels, mode=mode) #upsampling (factor=base_levels)
            h_res, w_res = img_rec.shape[-2:]
            img_rec[:, 0, (h_res//2-width//2):(h_res//2+width//2), (w_res//2-width//2):(w_res//2+width//2)] += img_crop[:, i_level, :, :] #adding previous central crop to img_crop
        img_rec = img_rec[:, :, (h_res//2-N_X//2):(h_res//2+N_X//2), (w_res//2-N_Y//2):(w_res//2+N_Y//2)]

    return img_rec


def saccade_to(img_color, orig, loc_data_ij):
    if type(img_color) == np.ndarray:
        img_copy = np.copy(img_color)
        img_copy=np.roll(img_copy, orig[0] - loc_data_ij[0], axis=0)
        img_copy=np.roll(img_copy, orig[1] - loc_data_ij[1], axis=1)
    elif type(img_color) == torch.Tensor:
        img_copy = torch.clone(img_color)
        img_copy = torch.roll(img_copy, (orig[0] - loc_data_ij[0],), (2,))
        img_copy = torch.roll(img_copy, (orig[1] - loc_data_ij[1],), (3,))
    return img_copy


def level_construct(img_crop_list, loc_data_ij, level_size, level):
    n_levels = int(np.log(np.max((N_X, N_Y))/width)/np.log(base_levels)) + 1
    orig = level_size[0]//2, level_size[1]//2
    img_lev = torch.zeros((1, 3, level_size[0], level_size[1]))
    img_div = torch.zeros((1, 3, level_size[0], level_size[1]))
    #print(img_lev.shape)
    nb_saccades= len(img_crop_list)
    for num_saccade in range(nb_saccades):
        sac_img =  img_crop_list[num_saccade][:, level, :, :, :]
        if level_size[0] < width:
            x_width = level_size[0]
            sac_img = sac_img[:,:,width//2 - level_size[0]//2:width//2 + level_size[0]//2,:]
        else:
            x_width = width
        if level_size[1] < width:
            y_width = level_size[1]
            sac_img = sac_img[:,:,:,width//2 - level_size[1]//2:width//2 + level_size[1]//2]
        else:
            y_width = width
        #print(sac_img.shape)

        loc = loc_data_ij[num_saccade] // 2**level
        img_lev = saccade_to(img_lev, orig, loc)
        img_lev[:,:,orig[0]-x_width//2:orig[0]+x_width//2, orig[1]-y_width//2:orig[1]+y_width//2] += sac_img
        img_lev = saccade_to(img_lev, loc, orig)
        img_div = saccade_to(img_div, orig, loc)
        img_div[:,:,orig[0]-x_width//2:orig[0]+x_width//2, orig[1]-y_width//2:orig[1]+y_width//2] += torch.ones_like(sac_img)
        img_div = saccade_to(img_div, loc, orig)
    # coefficients normalization
    indices_zero = (img_div == 0).nonzero().detach().numpy()
    img_div_npy = img_div.detach().numpy()
    for ind in indices_zero:
        img_div_npy[ind[0], ind[1], ind[2], ind[3]] = 1
    img_lev = img_lev // img_div_npy
    plt.figure()
    if level < n_levels-1:
        bias = 128
    else:
        bias = 0
    img_aff = img_lev.detach().permute(0,2,3,1)[0,:,:,:].numpy()
    plt.imshow((img_aff+bias).astype('uint8'))
    return img_lev


def inverse_pyramid_saccades(img_crop_list, img_crop, loc_data_ij, level_size, N_X=N_X, N_Y=N_Y, base_levels=base_levels, verbose=False):
    N_batch = img_crop.shape[0]
    width = img_crop.shape[3]
    n_levels = int(np.log(np.max((N_X, N_Y))/width)/np.log(base_levels)) + 1

    #img_rec = img_crop[:, -1, :, :, :] #.unsqueeze(1)
    img_rec = level_construct(img_crop_list, loc_data_ij, level_size[n_levels-1], level=n_levels-1)
    for i_level in range(n_levels-1)[::-1]: # from the top to the bottom of the pyramid
        img_rec = interpolate(img_rec, scale_factor=base_levels, mode=mode) #upsampling (factor=base_levels)
        h_res, w_res = img_rec.shape[-2:]
        img_lev = level_construct(img_crop_list, loc_data_ij, level_size[i_level], level=i_level)
        img_rec += img_lev #adding previous central crop to img_crop
    img_rec = img_rec[:, :, (h_res//2-N_X//2):(h_res//2+N_X//2), (w_res//2-N_Y//2):(w_res//2+N_Y//2)]

    return img_rec

from LogGabor import LogGabor
pe = {'N_X': width, 'N_Y': width, 'do_mask': False, 'base_levels':
          base_levels, 'n_theta': 24, 'B_sf': 0.6, 'B_theta': np.pi/12 ,
      'use_cache': True, 'figpath': 'results', 'edgefigpath':
          'results/edges', 'matpath': 'cache_dir', 'edgematpath':
          'cache_dir/edges', 'datapath': 'database/', 'ext': '.pdf', 'figsize':
          14.0, 'formats': ['pdf', 'png', 'jpg'], 'dpi': 450, 'verbose': 0}                 #log-Gabor parameters
lg = LogGabor(pe)
print('lg shape=', lg.pe.N_X, lg.pe.N_Y)


def local_filter(azimuth, theta, phase, sf_0=.25, B_theta=lg.pe.B_theta, radius=width/4):

    x, y = lg.pe.N_X//2, lg.pe.N_Y//2         # center
    x += radius * np.cos(azimuth)
    y += radius * np.sin(azimuth)

    return lg.normalize(lg.invert(
        lg.loggabor(x, y, sf_0=sf_0, B_sf=lg.pe.B_sf, theta=theta, B_theta=B_theta) * np.exp(-1j * phase)))


def get_K(width=width, n_sublevel = n_sublevel, n_azimuth = n_azimuth, n_theta = n_theta,
          n_phase = n_phase, r_min = width/6, r_max = width/3, log_density_ratio = 2, verbose=False): #filter tensor K definition using Di Carlo's formulas
    K = np.zeros((width, width, n_sublevel, n_azimuth, n_theta, n_phase))
    for i_sublevel in range(n_sublevel):
        sf_0 = .25*(np.sqrt(2)**i_sublevel)

        #radius = width/4/(np.sqrt(2)**i_sublevel)
        # Di Carlo / Retina Warp

        b = np.log(log_density_ratio)  / (r_max - r_min)
        a = (r_max - r_min) / (np.exp (b * (r_max - r_min)) - 1)
        r_ref = r_min + i_sublevel * (r_max - r_min) / n_sublevel
        r_prim =  a * np.exp(b * (r_ref - r_min))
        radius =  r_prim
        d_r_prim = a * b * np.exp(b * (r_ref - r_min))
        p_ref = 4 * width / 32
        p_loc = p_ref * d_r_prim
        sf_0 = 1 / p_loc
        if verbose: print('i_sublevel, sf_0, radius', i_sublevel, sf_0, radius)
        for i_azimuth in range(n_azimuth):
            for i_theta in range(n_theta):
                for i_phase in range(n_phase):
                    azimuth = (i_azimuth+i_sublevel/2)*2*np.pi/n_azimuth
                    K[..., i_sublevel, i_azimuth, i_theta, i_phase] = local_filter(azimuth=azimuth,
                                                                                   theta=i_theta*np.pi/n_theta + azimuth,
                                                                                   phase=i_phase*np.pi/n_phase, sf_0=sf_0, radius=radius)
    K = torch.Tensor(K)

    if verbose: print('K shape=', K.shape)
    if verbose: print('K min max=', K.min(), K.max())

    return K


def inverse_gabor(out, K, N_X=N_X, N_Y=N_Y, base_levels=base_levels, verbose=False):
    N_batch = out.shape[0]
    width =  K.shape[0]
    n_levels = int(np.log(np.max((N_X, N_Y))/width)/np.log(base_levels)) + 1
    n_sublevel, n_azimuth, n_theta, n_phase = K.shape[2:]

    out__ = out.reshape((N_batch, n_levels, n_sublevel*n_azimuth*n_theta*n_phase))
    K_ = K.reshape((width**2, n_sublevel*n_azimuth*n_theta*n_phase))
    K_inv = torch.pinverse(K_)
    img_crop_rec =  torch.tensordot(out__, K_inv,  dims=1).reshape((N_batch, n_levels, width, width))

    return img_crop_rec
