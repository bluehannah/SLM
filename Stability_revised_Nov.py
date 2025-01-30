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
import cv2
import time
import os

cam = shm("jen")
slm = shm("slm")
#jen = shm('jen')
plt.ion()

#Loading averaged dark image
dark_image = Image.open('/home/alala/Jennifer/SLM_JEN/Lyot_images/Contrast_amp/dark_av_1000by1000_new.tiff')

#dark = Image.open('/home/alala/Jennifer/SLM_JEN/July_dark.tiff')
#array_dark = np.array(dark)
#def shift_img(ref, im):
   # offset = register_translation(ref, im, upsample_factor=10)[0]
#    offset = phase_cross_correlation(ref, im, upsample_factor=10)[0]
#    output = shift(im, offset, order=5)
#    return output

def SLMme(radian_map):
    map_1 = (radian_map*(2**16-1)/(2*np.pi)) + 32768
    map_2 = map_1.astype(np.uint16)
    return map_2

def ratio(image, fact=0.06):
    x = np.linspace(-500, 500,1000)
    y = np.linspace(-500, 500,1000)
    x_0, y_0 = np.meshgrid(x, y)
    x_top_speckle = 205
    y_top_speckle = 280
    x_ghost = -24
    y_ghost = 265
    x_blank = -5
    y_blank = -215
    r_ghost = 17
    r_top_speckle = 17
    r_blank = 16
    speckle_mask = np.sqrt((x_0-x_top_speckle)**2+(y_0-y_top_speckle)**2)<r_top_speckle
    ghost_mask = np.sqrt((x_0-x_ghost)**2+(y_0-y_ghost)**2)<r_ghost
    blank_mask = np.sqrt((x_0-x_blank)**2+(y_0-y_blank)**2)<r_blank
    speckle = speckle_sub * speckle_mask
    #plt.imshow(speckle,cmap="inferno",vmin= 20,vmax= 120);plt.colorbar()
    #plt.show()
    #time.sleep(1000)
    ghost = resize * ghost_mask
    blank = resize * blank_mask
    flux_1 = np.sum(speckle)
    flux_2 = np.sum(ghost)
    flux_ratio = flux_1/flux_2
    flux_dark = np.sum(blank)
    return [flux_1,flux_2,flux_ratio,flux_dark]
   

def csv_writer(time_var,cen,path):
    with open(path,'a') as csvlog:
     # csv writer object
        writer = csv.writer(csvlog)
        writer.writerow([time_var,cen[0],cen[1],cen[2],cen[3]])
 
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
#wave2 = B * np.sin(px1 * X + py1 * Y)


combo_wave = SLMme(wave)

##Sine wave 2
#Changing into SLM units and then creating a height of 2^15
#The plus one is a vertical shift to eliminate negative values, 2^16 is added to make image 16bit. Indexed at zero so have
#grating_1 = 0.5*(np.sin(px1 * X + py1 * Y)+1) * (2**13-1)
#grating3 = grating_1.astype(np.uint16)
#Adding two sine waves together
#added_sines = grating2 + grating3

#Sending required 1D array to SLM
#slm.set_data(combo_wave) 


#Needing to take a pause so that when we take an image the SLM has time to change to that image we sent
#time.sleep(1)

#sine = (np.sin(px*np.pi*X+py*np.pi*Y)+1)/2 * (2**14-1)
#sine1 = (np.sin(px1*np.pi*X+py1*np.pi*Y)+1)/2 * (2**14-1)
#sine2 = sine.astype(np.uint16)
#sine3 = sine1.astype(np.uint16) 
slm.set_data(combo_wave) 
time.sleep(1)
#interval = 60 
first = True
count = 1
count1 = 0
folder_path = "/home/alala/Jennifer/SLM_JEN/Speckle_live_test"
folder_path1 = "/home/alala/Jennifer/SLM_JEN/Ghost_live_test"
folder_path2 = "/home/alala/Jennifer/SLM_JEN/Blank_live_test"
while True:
    try:
        #image = cam.get_data().astype('float64')
        # Dimensions of the original array
        x_top_speckle = 205
        y_top_speckle = 280
        x_ghost = -24
        y_ghost = 265
        r_ghost = 17
        r_top_speckle = 17
        r_blank = 16
        x_blank = -5
        y_blank = -215
        x = np.linspace(-500, 500,1000)
        y = np.linspace(-500, 500,1000)
        x_0, y_0 = np.meshgrid(x, y)
        speckle_mask = np.sqrt((x_0-x_top_speckle)**2+(y_0-y_top_speckle)**2)<r_top_speckle
        ghost_mask = np.sqrt((x_0-x_ghost)**2+(y_0-y_ghost)**2)<r_ghost
        blank_mask = np.sqrt((x_0-x_blank)**2+(y_0-y_blank)**2)<r_blank
        burst_speckle_on =[]
        for i in range(100):
            burst_images = cam.get_data()
            burst_speckle_on.append(burst_images)
        #Averaging out frames
        burst_speckle_on_1 =  np.mean(burst_speckle_on, axis=0)
        rows, cols = burst_speckle_on_1.shape
# Dimensions of the crop
        crop_height, crop_width = 1000, 1000
# Calculate the starting indices for cropping
        start_row = (rows - crop_height) // 2
        start_col = (cols - crop_width) // 2
# Calculate the ending indices for cropping
        end_row = start_row + crop_height
        end_col = start_col + crop_width
        resize =  burst_speckle_on_1[start_row:end_row, start_col:end_col]
        on = resize-dark_image
        speckle_sub = on
        image = cam.get_data()
        #time.sleep(interval-time.time() % interval)
        dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if first:
            dt0 = dt
            first = False
        rows3, cols3 = image.shape
        crop_height3, crop_width3 = 1000, 1000
        start_row3 = (rows3 - crop_height3) // 2
        start_col3 = (cols3 - crop_width3) // 2
        end_row3 = start_row3 + crop_height3
        end_col3 = start_col3 + crop_width3 
        resize = (image[start_row3:end_row3, start_col3:end_col3])-dark_image
        rat = ratio(speckle_sub)
        file_name = f"Speckle_{count1 + 1}.png"
        file_name1 = f"Ghost_{count1 + 1}.png"
        file_name2 = f"Blank_{count1 + 1}.png"
        file_path = os.path.join(folder_path, file_name)
        file_path1 = os.path.join(folder_path1, file_name1)
        file_path2 = os.path.join(folder_path2, file_name2)
        #np.save(folder_path, file_name)
        cv2.imwrite(file_path, speckle_sub*speckle_mask)
        cv2.imwrite(file_path1, speckle_sub*ghost_mask)
        cv2.imwrite(file_path2, speckle_sub*blank_mask)
        csv_writer(dt, rat, "Jan20_point05amp%s.csv" %dt0)
        #image1.save("/home/alala/Jennifer/SLM_JEN/Stability_live_test/speckle%s.tiff" %dt0, format="TIFF")
        sys.stdout.write("\r %i data points. Waiting for next iteration" %count)
        sys.stdout.flush()
        count += 1
        count1 += 1
        time.sleep(3)
    except KeyboardInterrupt:
        print('Stopping')
        sys.exit()
