import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pyMilk.interfacing.shm import SHM as shm
from matplotlib.colors import LogNorm
from pylab import figure, cm
from matplotlib import ticker, cm
import time, sys, datetime, os
from pyMilk.interfacing.shm import SHM as shm
from itertools import count
import csv
import time

#Defining Sinewave to SLM units
def SLMme(radian_map):
    map_1 = (radian_map*(2**16-1)/(2*np.pi)) + 32768
    map_2 = map_1.astype(np.uint16)
    return map_2

#Defining camera and shared memory
cam = shm("jen")
slm = shm("slm")
#jen = shm('jen')
plt.ion()

#x,X,Y are in radians per aperture.
#400 pixels is the size of the 6mm aperture in pixels      
x = np.arange(0, 2*np.pi, 2*np.pi/512)
X, Y = np.meshgrid(x, x)
#spatial frequency over the aperture; units: cycles per aperture
px = 0 
py = 60
px1 = 0
py1 = 60
#Sine wave 1
#Changing into SLM units and then creating a height of 2^15
#A,B units are radians per aperture
A = .2
B = .2
wave = A * np.sin(px * X + py * Y)
#Sine wave 2
wave2 = B * np.sin(px1 * X + py1 * Y)
#Combining sinewaves
combo_wave = SLMme(wave)

#This sends the sinewaves to SLM
#slm.set_data(combo_wave)
time.sleep(1)
imgs_1 = cam.get_data()


#This changes the image of camera to 512 x 512 (which is somewhere around 2048x1500) to be able to multiply with 512 x 512 mask
rows, cols = imgs_1.shape
crop_height, crop_width = 1000, 1000
start_row = (rows - crop_height) // 2
start_col = (cols - crop_width) // 2
end_row = start_row + crop_height
end_col = start_col + crop_width 
resize = imgs_1[start_row:end_row, start_col:end_col]
sub = resize


#Creating masks for PSF and speckles
x = np.linspace(-500, 500,1000)
y = np.linspace(-500, 500,1000)
x_coordinates, y_coordinates = np.meshgrid(x, y)

#x_0 = 175
#y_0 = -9

x_ghost = 170
y_ghost = 170

x_psf = 320
y_psf = -5

#x_3 = -15
#y_3 = 70


#r = 60
r_ghost = 17
r_psf = 40
#r_ = 60

#mask = np.sqrt((x_coordinates-x_0)**2+(y_coordinates-y_0)**2)<r
mask_ghost = np.sqrt((x_coordinates-x_ghost)**2+(y_coordinates-y_ghost )**2)<r_ghost
mask_psf = np.sqrt((x_coordinates-x_psf)**2+(y_coordinates-y_psf)**2)<r_psf
#mask_3 = np.sqrt((x_1-x_3)**2+(y_1-y_3)**2)<r_3

#Loading averaged dark image
dark_image = Image.open('/home/alala/Jennifer/SLM_JEN/Lyot_images/Contrast_amp/dark_av_1000by1000_new.tiff')

#512x512 subtraction of dark and image
dark_subtraction1 = sub - dark_image
#Applying PSF and ghost mask

dark_psf = dark_subtraction1 * mask_psf
dark_ghost = dark_subtraction1 * mask_ghost

#Regular mask of PSF and ghost with no dark subteaction
#multi = sub * mask
#ghost = sub * mask_1
#psf = sub * mask_2
#multi_3 = sub * mask_3

#Calculating noise using specs from camera
#flux = np.sum(multi) + np.sum(multi_1)
#electrons_flux = (flux-200)* 0.1
#noise = np.sqrt(electrons_flux)
#print(electrons_flux)
#print(noise)

#speck_intensity = np.max(multi_1)
#psf_intensity = np.max(multi_3)
#ratio = speck_intensity/psf_intensity
#ratio1 = psf_intensity/speck_intensity

#print(ratio)
#print(ratio1)
#plt.figure()
#plt.imshow(np.log10(imgs_1),cmap="inferno")
#plt.show
#plt.title('10 & 13 lambda/D spaced speckles')

plt.clf()
plt.figure()


#plt.figure()
#plt.imshow(dark_ghost,cmap="inferno");plt.colorbar()
#plt.title('Ghost  w/ coronagraph and w/ mask')
#plt.show()

plt.figure()
plt.imshow(dark_psf,cmap="inferno");plt.colorbar()
plt.title('PSF w/o coronagraph @ .009 exp. just under saturation')
plt.show()

plt.figure()
plt.imshow(dark_ghost,cmap="inferno");plt.colorbar()
plt.title('Ghost w/o coronagraph @ .009 exp. just under saturation')
plt.show()

#plt.figure()
##plt.imshow(np.log10(dark_sub_no_resize),cmap="inferno");plt.colorbar()
#plt.title('Ghost w/o coronagraph @ 1.2e-5 exp. just under saturation')
#plt.show()

#plt.figure()
#plt.imshow(dark_psf-dark_ghost,cmap="inferno",vmin=-.0050, vmax=.0050);plt.colorbar()
#plt.title('(PSF-Ghost)-Dark w/o coronagraph @ 1.2e-5 exp. just under saturation')
#plt.show()

#multi_integrated= np.sum(multi)
#ghost_integrated = np.sum(dark_ghost)
psf_integrated = np.sum(dark_psf)
ghost_integrated = np.sum(dark_ghost)
#multi_3_integrated = np.sum(multi_3)

psf_ghost_contrast_ratio = ghost_integrated/psf_integrated
psf_ghost_contrast_subtraction = psf_integrated-ghost_integrated
print(psf_integrated)
print(ghost_integrated)
print(psf_ghost_contrast_ratio)

#print(psf_speckle_contrast)

#psf_pic = Image.fromarray(dark_psf)
#psf_pic.save('/home/alala/Jennifer/SLM_JEN/Lyot_images/Nov_14_story/PSF_nocorono.tiff')

#ghost_pic = Image.fromarray(dark_ghost)
#ghost_pic.save('/home/alala/Jennifer/SLM_JEN/Lyot_images/Nov_14_story/ghost_nocorona.tiff')

#psf_sub_ghost_pic = Image.fromarray(psf_ghost_contrast_subtraction)
#psf_sub_ghost_pic.save('/home/alala/Jennifer/SLM_JEN/Lyot_images/Nov_14_story/PSF_ghost_diff_nocorona.tiff')

#im = Image.fromarray(multi)
#im.save('/home/alala/Jennifer/SLM_JEN/speckle_22.tiff')