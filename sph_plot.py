import matplotlib.pyplot as plt
import matplotlib.image
import numpy as np
from scipy.interpolate import griddata

def show_sim_img(im, figsize=None, ax=None, alpha=None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im, alpha=alpha,origin='lower',cmap='gray')
    ax.set_axis_off()
    return ax

# utility method to transform the sparse position, value data to a np array
def sim_to_image(px,py,z,box=15,res=0.1):
    xi = yi = np.arange(0,box,res)
    xi,yi = np.meshgrid(xi,yi)
    I = griddata((px,py),z,(xi,yi),method='linear',fill_value=0)
#     import pdb;pdb.set_trace()
    return I

def show_colormap(I, zoom_in=False,ax=None):
    if not ax: fig,ax = plt.subplots(figsize=None)
    I=np.flip(I,axis=0)
    if zoom_in:
        ax.imshow(I,cmap='seismic')
    else:
        ax.imshow(I,cmap='seismic',vmin=-1,vmax=1)
    for PCM in ax.get_children():
        if type(PCM) == matplotlib.image.AxesImage: break
    plt.colorbar(PCM, ax=ax)