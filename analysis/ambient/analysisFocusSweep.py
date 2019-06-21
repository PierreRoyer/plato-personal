#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 09:50:45 2019

@author: pierre
"""
#import fnmatch

# Source Extractor in Python
# https://sep.readthedocs.io/en/v1.0.x/tutorial.html

import sep
import numpy as np
from matplotlib.patches import Ellipse

dataDir = "/Users/pierre/plato/pr/simout/egse/smearingNo/"

pngDir = "/Users/pierre/plato/pr/pngs/egse/smearingNo/"

row,column,ccdCode = 1000,3000,2

allfiles = fs.fileSelect([f"{str(row).zfill(4)}",f"{str(column).zfill(4)}","hdf5"], location=dataDir)

numExposures = 20

srowcol = f"egse_{str(row).zfill(4):4s}_{str(column).zfill(4):4s}_{ccdCode}_"

availableFocusPositions = []
for f in allfiles:
    availableFocusPositions.append(int(f.split('_')[-1][:4]))
    
availableFocusPositions = np.sort(np.array(availableFocusPositions,dtype=int))

xss,yss,rmss,foci,pixmax = [],[],[],[],[]

sizes,bcks = [],[]

doPlots = False

#for f,focusPosition in enumerate([5900,5950,6000]):
for f,focusPosition in enumerate(availableFocusPositions):

    runName = srowcol+f"{focusPosition}"
    
    print()
    print(f"{runName}")
    print()
    
    sh5 = h5py.File(dataDir+runName+".hdf5",'r')
    
    image = np.array(getSim(sh5,n=0))
    sizes.append(image.shape[0])

    xss.append([])
    yss.append([])
    rmss.append([])
    pixmax.append([])
    bcks.append([])
    
    #for n in [0]:
    for n in range(numExposures):

        # Do it here : if done right after savefig, it actually closes before savefig ends --> empty plot
        plt.close()

        print (f"          Image {n:3d}")

        image = np.array(getSim(sh5,n=n))
        
        """
        # assume image is square and extract image size
        if n==0: 
            sizes[f] = image.shape[0]
        """

        edge = arrayEdge1d(image,width=5)
        bckgnd = np.median(edge)
        noise  = np.std(edge)
        backgroundSubtrImage = image - bckgnd


        
        sep_background = sep.Background(image)                  # Spatially varying background        
        sep_backgroundImage = sep_background.back()                 # Evaluate background as 2D array (same dimensions as image)
        sep_backgroundSubtrImage = image - sep_backgroundImage      # Subtract the background
        

        """
        #imflat = image.copy()
        image[np.where(np.abs(image)<3.*np.std(edge))] = 0.
        immax = np.where(image==np.max(image))
        xs,ys = cog(image,cen=[immax[0][0],immax[1][0]],width=int(image.shape[0]/4))
        """
        

        
        pixmax[f].append(np.max(backgroundSubtrImage))
        
        # Source extraction (SExtractor)
        threshold = 5.
        sources = sep.extract(sep_backgroundSubtrImage, threshold, err = sep_background.globalrms)
        #rows = sources["y"]
        #columns = sources["x"]
        
        if doPlots:
            # plot background-subtracted image
            objects = sources
            data_sub = backgroundSubtrImage
            fig, ax = plt.subplots(figsize=(8,8))
            m, s, mx = np.mean(data_sub), np.std(data_sub),np.max(data_sub)
            im = ax.imshow(data_sub, interpolation='nearest', cmap='gray', vmin=m-s, vmax=mx, origin='lower')
            
            # plot an ellipse for each object
            for i in range(len(objects)):
                e = Ellipse(xy=(objects['x'][i], objects['y'][i]),
                            width=6*objects['a'][i],
                            height=6*objects['b'][i],
                            angle=objects['theta'][i] * 180. / np.pi)
                e.set_facecolor('none')
                e.set_edgecolor('red')
                ax.add_artist(e)
                plt.title(runName)
                plt.savefig(pngDir+runName+f'_{str(n).zfill(2)}_sources.png')        
                
        xs,ys = cog(backgroundSubtrImage,cen=[int(image.shape[0]//2),int(image.shape[1]//2)],width=int(image.shape[0]//2-1))
        rms = psff.psfRms(backgroundSubtrImage)
        
        foci.append(focusPosition)
        xss[f].append(xs)
        yss[f].append(ys)
        rmss[f].append(rms)
        bcks[f].append(bckgnd)
        
        if image.shape[0] != image.shape[1]:
                print ("WARNING : NOT SQUARE IMAGE")
        if n==0:
                sizes[f] = image.shape[0]
        elif (image.shape[0] != sizes[f]):
            print ("WARNING : NOT ALL IMAGES HAVE THE SAME SIZE")
        
foci = np.array(foci)
xss = np.array(xss)
yss = np.array(yss)
rmss = np.array(rmss)
sizes = np.array(sizes)
pixmax = np.array(pixmax)
#maxnsources = np.array(maxnsources)

###########

plt.figure()
for i in range(rmss.shape[1]):
    plt.plot(rmss[:,i],label=f"Focus {foci[i]}")

plt.legend()


plt.figure('imageSize')
plt.plot(availableFocusPositions, sizes, 'k.-',label="image size")
plt.grid(alpha=0.25)
plt.xlabel("Focus Position")
plt.ylabel("Size of simulated image")
plt.title(f"Size of simulated image\n [{str(row):4s},{str(column):4s}] CCD {ccdCode}")
plt.savefig(pngDir+f"{srowcol[:-1]}_imageSize.png")

plt.figure('pixmax')
plt.plot(availableFocusPositions, pixmax[:,0], 'k.-')
plt.grid(alpha=0.25)
plt.xlabel("Focus Position")
plt.ylabel("Pix Max [adu]")
plt.title(f"Flux in highest pixel (backgnd subtracted)\n [{str(row):4s},{str(column):4s}] CCD {ccdCode}")
plt.savefig(pngDir+f"{srowcol[:-1]}_pixmax_aboveBackgnd.png")

plt.figure('rmss0')
plt.plot(availableFocusPositions, rmss[:,0], 'k.-')
plt.grid(alpha=0.25)
plt.xlabel("Focus Position")
plt.ylabel("psf RMS [pix]")
plt.title(f"PSF RMS\n [{str(row):4s},{str(column):4s}] CCD {ccdCode}")
plt.ylim(0,120)
plt.savefig(pngDir+f"{srowcol[:-1]}_rmss_image0_edgeBckgnd.png")


plt.figure('rmss')
for i in range(numExposures):
  plt.plot(availableFocusPositions, rmss[:,i], marker='.',c=(0,i/float(2.*numExposures),1-i/float(numExposures)),lw=1,ls='-',label=f"n={i}")

plt.plot(availableFocusPositions, sizes, 'k.-',label="image size")
plt.legend()
plt.grid(alpha=0.25)
plt.xlabel("Focus Position")
plt.ylabel("psf RMS [pix]")
plt.title(f"PSF RMS\n [{str(row):4s},{str(column):4s}] CCD {ccdCode}")
plt.savefig(pngDir+f"{srowcol[:-1]}_rmss_image_all_edgeBckgnd.png")
plt.ylim(0,120)

"""
## USING FIND SOURCES DOESN'T WORK WITH EXTENDED PSFs AT AMBIENT, IT FINDS MULTIPLE SOURCES WITHIN THE PSF
allok = True
    
        xs,ys,rms = psff.sourcesRms(image)
    
        # We expect only one source ==> reduce the last dimension (sources) to one single element
        nbSources = len(xs)
        
        if nbSources == 0:
            print (f"                   Source Detection Failed")
            allok = False
            break
        else:
            if nbSources > 1:
                print (f"WARNING: More than one source detected : len(xs) = {len(xs)}")
            tmpx[n],tmpy[n],tmprms[n] = xs[0],ys[0],rms[0]
            tmpnb = max(tmpnb,nbSources)

    if allok:
            foci.append(focusPosition)
            xss[f].append(tmpx)
            yss[f].append(tmpy)
            rmss[f].append(tmprms)
            maxnsources.append(tmpnb)
"""
    

row,column,ccdCode = 1000,3000,2
srowcol = f"egse_{str(row).zfill(4):4s}_{str(column).zfill(4):4s}_{ccdCode}_"

focusPosition = 5500

runName = srowcol+f"{focusPosition}"
    
sh5 = h5py.File(dataDir+runName+".hdf5",'r')
    
image = np.array(getSim(sh5,n=0))

showImage(image)

edge = arrayEdge1d(image,width=5)
bckgnd = np.median(edge)
noise  = np.std(edge)
backgroundSubtrImage = image - bckgnd

showImage(backgroundSubtrImage,cmap='gray')
plt.colorbar()



img = backgroundSubtrImage

img = cv2.imread('/Users/pierre/fun/BrainHands.jpg',0)

ksize = 11
laplacian = cv2.Laplacian(img, cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=ksize)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=ksize)
sobelxy = np.sqrt(sobelx**2. + sobely**2.)

plt.figure(2)
plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
#plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
#plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
 
plt.subplot(2,2,2),plt.imshow(sobelxy,cmap = 'gray')
plt.title('SobelXY'), plt.xticks([]), plt.yticks([])

plt.show()


















