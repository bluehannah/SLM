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


#Creating masks for PSF and speckles
x = np.linspace(-500, 500,1000)
y = np.linspace(-500, 500,1000)
x_coordinates, y_coordinates = np.meshgrid(x, y)

x_bottom_speckle = 200
y_bottom_speckle = 260

x_top_speckle = 205
y_top_speckle = 280
x_ghost = -24
y_ghost = 265

x_blank = -5
y_blank = -215

r_ghost = 17
#r_bottom_speckle = 200
r_top_speckle = 17
r_blank = 16

#Loading averaged dark image
dark_image = Image.open('/home/alala/Jennifer/SLM_JEN/Lyot_images/Contrast_amp/dark_av_1000by1000_new.tiff')

#Taking image before speckle is actuated
imgs = cam.get_data()

#Resizing image of ghost
rows, cols = imgs.shape
crop_height, crop_width = 1000, 1000
start_row = (rows - crop_height) // 2
start_col = (cols - crop_width) // 2
end_row = start_row + crop_height
end_col = start_col + crop_width 
resize = imgs[start_row:end_row, start_col:end_col]
ghost_pic = resize


#Resizing image of blank
rows, cols = imgs.shape
crop_height, crop_width = 1000, 1000
start_row = (rows - crop_height) // 2
start_col = (cols - crop_width) // 2
end_row = start_row + crop_height
end_col = start_col + crop_width 
resize = imgs[start_row:end_row, start_col:end_col]
blank_pic = resize


#Subtracting dark from ghost
ghost_pic_minus_dark = ghost_pic-dark_image
#Subtracting dark from blank
blank_minus_dark = blank_pic - dark_image
#Masking ghost only before actuation of speckle
mask_ghost = np.sqrt((x_coordinates-x_ghost)**2+(y_coordinates-y_ghost )**2)<r_ghost
dark_ghost = ghost_pic_minus_dark * mask_ghost
mask_blank = np.sqrt((x_coordinates-x_blank)**2+(y_coordinates-y_blank)**2)<r_blank
blank = blank_minus_dark * mask_blank


#Defining coordinate space for speckles
#x,X,Y are in radians per aperture.
#400 pixels is the size of the 6mm aperture in pixels      
x = np.arange(0, 2*np.pi, 2*np.pi/512)
X, Y = np.meshgrid(x, x)
#spatial frequency over the aperture; units: cycles per aperture
px = -32 
py = 71
px1 = 68
py1 = 0
#Sine wave 1
#Changing into SLM units and then creating a height of 2^15
#A,B units are radians per aperture
A = .05
B = .05
wave = A * np.sin(px * X + py * Y)
wave1 = B * np.sin(px1 * X + py1 * Y)
#Combining sinewaves
combo_wave = SLMme(wave)
combo_wave_1 = SLMme(wave1)
added_waves = combo_wave + combo_wave_1
#This sends the sinewaves to SLM
slm.set_data(combo_wave)
time.sleep(1)
imgs_1 = cam.get_data()

#Burst of images for speckle on bottom
burst_speckle_on =[]
for i in range(1000):
    burst_images = cam.get_data()
    burst_speckle_on.append(burst_images)
#Averaging out frames
burst_speckle_on_1 =  np.mean(burst_speckle_on, axis=0)

#This changes the image of camera to 600 x 600 (which is somewhere around 2048x1500) to be able to multiply with 512 x 512 mask
rows, cols = burst_speckle_on_1.shape
crop_height, crop_width = 1000, 1000
start_row = (rows - crop_height) // 2
start_col = (cols - crop_width) // 2
end_row = start_row + crop_height
end_col = start_col + crop_width 
resize_burst_on = burst_speckle_on_1[start_row:end_row, start_col:end_col]

#600x600 subtraction of dark and image
dark_subtraction_speckle_on = resize_burst_on - dark_image

#Burst images for top speckle
burst_speckle_on_top =[]
for i in range(1000):
    burst_images = cam.get_data()
    burst_speckle_on_top.append(burst_images)
#Averaging out frames
burst_speckle_on_top_1 =  np.mean(burst_speckle_on_top, axis=0)

#This changes the image of camera to 600 x 600 (which is somewhere around 2048x1500) to be able to multiply with 512 x 512 mask
rows, cols = burst_speckle_on_top_1.shape
crop_height, crop_width = 1000, 1000
start_row = (rows - crop_height) // 2
start_col = (cols - crop_width) // 2
end_row = start_row + crop_height
end_col = start_col + crop_width 
resize_burst_on_top = burst_speckle_on_top_1[start_row:end_row, start_col:end_col]

#600x600 subtraction of dark and image
dark_subtraction_speckle_on_top = resize_burst_on_top - dark_image

#Turning speckle off
slm.set_data(np.zeros([512,512]).astype(np.uint16))
time.sleep(1)

#Burst speckle off 
burst_speckle_off =[]
for i in range(1000):
    burst_images = cam.get_data()
    burst_speckle_off.append(burst_images)
#Averaging out frames
burst_speckle_off_1 =  np.mean(burst_speckle_off, axis=0)

#This changes the image of camera to 600 x 600 (which is somewhere around 2048x1500) to be able to multiply with 512 x 512 mask
rows, cols = burst_speckle_off_1.shape
crop_height, crop_width = 1000, 1000
start_row = (rows - crop_height) // 2
start_col = (cols - crop_width) // 2
end_row = start_row + crop_height
end_col = start_col + crop_width 
resize_burst_off = burst_speckle_off_1[start_row:end_row, start_col:end_col]

dark_subtraction_speckle_off = resize_burst_off - dark_image

#Subtracting out speckle on and speckle off images
Speck_on_and_off_sub = (dark_subtraction_speckle_on - dark_subtraction_speckle_off)
#Creatinh mask for speckle on and off subtracted image
#mask_bottom_speckle = np.sqrt((x_coordinates-x_bottom_speckle)**2+(y_coordinates-y_bottom_speckle )**2)<r_bottom_speckle
#Applying Speckle mask
dark_speckle = Speck_on_and_off_sub #* mask_bottom_speckle


mask_top_speckle = np.sqrt((x_coordinates-x_top_speckle)**2+(y_coordinates-y_top_speckle )**2)<r_top_speckle
Speck_on_top_and_off_sub = dark_subtraction_speckle_on_top - dark_subtraction_speckle_off
dark_speckle_top = Speck_on_top_and_off_sub * mask_top_speckle 
mask_experiment = dark_subtraction_speckle_on_top * mask_top_speckle 
#Speckle off crop: for visualization purposes
#speck_off = dark_subtraction_speckle_off * mask_top_speckle

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

#normalizing the values
#plt.imshow(np.log10(multi/np.max(multi)),cmap="inferno");plt.colorbar()
#plt.title('Speckle 1 w/ coronagraph 5 radians/aperture amp')
#plt.show()

#plt.figure()
#plt.imshow(dark_ghost,cmap="inferno");plt.colorbar()
#plt.title('Ghost  w/ coronagraph and w/ mask')
#plt.show()
#plt.figure()
#plt.imshow(dark_subtraction_speckle_off,cmap="inferno");plt.colorbar()
#plt.title('Top speckle with crop')
#plt.show()

#plt.figure()
#plt.imshow(dark_speckle,cmap="inferno");plt.colorbar()
#plt.title('Speckle bottom')
#plt.show()

#plt.figure()
#plt.imshow(speck_off,cmap="inferno");plt.colorbar()
#plt.title('Speckle off averaged frames')
#plt.show()

#plt.figure()
##plt.imshow(Speck_on_and_off_sub,cmap="inferno");plt.colorbar()
#plt.title('Speckle subtraction')
#plt.show()

plt.figure()
plt.imshow(dark_ghost,cmap="inferno");plt.colorbar()
plt.title('Ghost, .05 rad , .9 seconds exp')
plt.show()

plt.figure()
plt.imshow(dark_speckle_top,cmap="inferno");plt.colorbar()
plt.title('Top speckle, .05 rad , .9 seconds exp')
plt.show()


#plt.figure()
##plt.imshow(mask_experiment,cmap="inferno");plt.colorbar()
#plt.title('Top speckle no sub')
#plt.show()

#multi_integrated= np.sum(multi)
#ghost_integrated = np.sum(dark_ghost)
ghost_integrated = np.sum(dark_ghost)
speckle_integrated = np.sum(dark_speckle_top)
#speckle_off_integrated = np.sum(speck_off)
mask_nosub = np.sum(mask_experiment)
#multi_3_integrated = np.sum(multi_3)

ghost_speckle_contrast_ratio = speckle_integrated/ghost_integrated
print(ghost_integrated)
print(speckle_integrated)
print(ghost_speckle_contrast_ratio)
#print(mask_nosub_ratio)
#peckle_off_ghost_ratio = speckle_off_integrated/ghost_integrated
#print(speckle_off_ghost_ratio)

#ghost_pic = Image.fromarray(dark_ghost)
#psf_pic.save('/home/alala/Jennifer/SLM_JEN/Lyot_images/Nov_14_story/.tiff')

#speckle_pic = Image.fromarray(dark_speckle)
#speckle_pic.save('/home/alala/Jennifer/SLM_JEN/Lyot_images/Nov_14_story/.tiff')

#im = Image.fromarray(multi)
#im.save('/home/alala/Jennifer/SLM_JEN/speckle_22.tiff')
