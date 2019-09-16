#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 09:50:45 2019

@author: pierre
"""
#import fnmatch

# Source Extractor in Python
# https://sep.readthedocs.io/en/v1.0.x/tutorial.html


###
###
### FOR THE NOISELESS PSFS : see cslPsfs.py  (read in via psfUtilsAmbient.getCslPsf)
###
###
###
###



import sep
import numpy as np
from matplotlib.patches import Ellipse
from skimage.feature import canny
from skimage import morphology
from imageUtils import backgroundSubtraction

import scipy.ndimage as ndi
from testSimulation.simulationUtils import coaddSimulatedImages as coadd

from analysis.ambient.analysisFocusSweep import focusInterpolation

dataDir = "/Users/pierre/plato/pr/simout/egse/smearingNo/"
#dataDir = "/Users/pierre/plato/pr/simout/egse/seeds/"

pngDir = "/Users/pierre/plato/pr/pngs/egse/smearingNo/"
#pngDir = "/Users/pierre/plato/pr/pngs/egse/seeds/"

row,column,ccdCode = 3000,1000,2

allfiles = fs.fileSelect([f"{str(row).zfill(4)}",f"{str(column).zfill(4)}","hdf5"], location=dataDir)

# First simulations ("smearingNo") without changing the random seeds : 20 exposures
# Second batch ("seeds"), changing the random seeds : 25 exposures
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


###############
# DISPLAY IMAGES
###############

dataDir = "/Users/pierre/plato/pr/simout/egse/smearingNo/"
#dataDir = "/Users/pierre/plato/pr/simout/egse/seeds/"

pngDir = "/Users/pierre/plato/pr/pngs/egse/smearingNo/"
#pngDir = "/Users/pierre/plato/pr/pngs/egse/seeds/"

row,column,ccdCode = 4000, 500,2  #  4 degrees
row,column,ccdCode = 3000,1000,2  #  8 degrees
row,column,ccdCode = 1000,3000,2  # 18 degrees

allfiles = fs.fileSelect([f"{str(row).zfill(4)}",f"{str(column).zfill(4)}","hdf5"], location=dataDir, listOrder=1)

numExposures = 20

srowcol = f"egse_{str(row).zfill(4):4s}_{str(column).zfill(4):4s}_{ccdCode}_"

availableFocusPositions = []
for f in allfiles:
    availableFocusPositions.append(int(f.split('_')[-1][:4]))
    
availableFocusPositions = np.sort(np.array(availableFocusPositions,dtype=int))


# Center of gravity of Canny Edges
cannyxs,cannyys = [],[]
sigmaxs,sigmays = [],[]
# Simple center of image
imCenxs,imCenys = [],[]

boxcannys = []
boxsigmas = []
pixmaxs = []

rmss = []

# Simulated image number (> 1 image is simulated at every x,y,ccd,focus)
n = 0

# Bounding box widening : widen = 0.5 means 50% width added left & right == width x 2
widen = 0.

doPlot,savePlot,verbose=0,0,0

sigma = 4

for focusPosition in availableFocusPositions:
    if doPlot: plt.close()
    runName = srowcol+f"{focusPosition}"
    
    print()
    print(f"{runName}")
    print()
    
    sh5 = h5py.File(dataDir+runName+".hdf5",'r')
    
    numExp = h5get(sh5,["ObservingParameters","numExposures"])
    
    #for n in range(numExp):
    for n in [0]:
    
      imorig = np.array(getSim(sh5,n=n))
    
      censigma,boxsigma = psff.psfBox(imorig,method='sigma',sigma=sigma,cosmicRemoval=True,kernel=None,verbose=1)
      sigmaxs.append(censigma[0])
      sigmays.append(censigma[1])
    
      cencanny,boxcanny = psff.psfBox(imorig,method='canny',sigma=sigma,cosmicRemoval=True,kernel=None,verbose=verbose)
      cannyxs.append(cencanny[0])
      cannyys.append(cencanny[1])

      boxsigmas.append(boxsigma)
      boxcannys.append(boxcanny)

      image,background = backgroundSubtraction(imorig,method='edge',width=5,verbose=verbose)
      #image = backgroundSubtraction(imorig,method='sep')#
    
      pixmaxs.append(np.max(image))
    
      noise = np.std(background)
      significant = np.zeros_like(image)
      significant[np.where(image>sigma*noise)] = 1
      
      ## Kernel is optional. The cosmic-ray clipping without affecting the shape is actually very good with the default
      ## Default kernel is the smallest possible cross (3x3 square with 0's in the corners)
      #significant = morphology.binary_opening(significant)#,selem=kernel)
        
      filtered = ndi.gaussian_filter(image, 5)

      lowThreshold = 0.
      highThreshold = .99
      edges = canny(image, sigma=sigma, low_threshold=lowThreshold, high_threshold=highThreshold, use_quantiles=True)

      ## NB: xmin,xmax,ymin,ymax == box
      xmin,xmax = np.min(np.where(edges)[0]),np.max(np.where(edges)[0])
      ymin,ymax = np.min(np.where(edges)[1]),np.max(np.where(edges)[1])
      #printm([xmin,xmax,ymin,ymax])
    
      ## NB: xc,yc == cen (when method='canny')
      xc,yc = com(edges)
      #printm([xc,yc])
      #cannyxs.append(xc)
      #cannyys.append(yc)
      
      xci,yci = int(round(xc)), int(round(yc))
      factor = np.max(image)
    
      imCenx,imCeny = image.shape[0]//2,image.shape[1]//2
      imCenxs.append(imCenx)
      imCenys.append(imCeny)


      """
    ## PSF PLOT
    
      fig, ax = plt.subplots(figsize=(8,8))
      mn, std, mx = np.mean(image), np.std(image),np.max(image)
      im = ax.imshow(image, interpolation='nearest', cmap='gray', vmin=mn-std, vmax=mx, origin='lower')
      plt.title(runName)
      plt.savefig(pngDir+runName+f'_{str(n).zfill(2)}.png')

    edge = arrayEdge1d(image,width=5)
  
    ### X and Y CUTS THROUGH THE PSF    
    plt.figure('cut2')
    plt.plot(image[xci,:],label="x-cut")
    plt.plot(image[:,yci],label="y-cut")
    plt.plot(np.array(edges[xci,:],dtype=int)*factor,label="x-edges")
    plt.plot(np.array(edges[:,yci],dtype=int)*factor,label="y-edges")
    plt.legend()
    plt.grid(alpha=0.25)

    plt.savefig(pngDir+runName+f'_{str(n).zfill(2)}_cuts.png')
      """

      ### CANNY EDGE DETECTION OF THE PSF
      if doPlot:
          fig, [[ax0, ax1], [ax2, ax3]] = plt.subplots(nrows=2, ncols=2, figsize=(14, 14), sharex=True, sharey=True)

          ax0.imshow(image, cmap=plt.cm.gray,interpolation='nearest',origin='lower')
          ax0.axis('off')
          ax0.set_title(f'Image\n{focusPosition}', fontsize=20)

          ax1.imshow(image, cmap=plt.cm.gray,interpolation='nearest',origin='lower')
          ax1.axis('off')
          ax1.set_title('+ : c.o.g.(edges)   x : image center\n* centroid significant', fontsize=20)
    
          ax2.imshow(filtered, cmap=plt.cm.gray,interpolation='nearest',origin='lower')
          ax2.axis('off')
          ax2.set_title('Filtered', fontsize=20)
    
          ax3.imshow(edges, cmap=plt.cm.gray,interpolation='nearest',origin='lower')
          ax3.axis('off')
          ax3.set_title('Edges', fontsize=20)
    
          # Canny
          ax1.scatter(yc,xc,marker="+",c='y',s=300)
          ax2.scatter(yc,xc,marker="+",c='y',s=300)
          ax3.scatter(yc,xc,marker="+",c='y',s=300)
    
          # Sigma
          ax1.scatter(sigmays[-1],sigmaxs[-1],marker="*",c='g',s=300)
          ax2.scatter(sigmays[-1],sigmaxs[-1],marker="*",c='g',s=300)
          ax3.scatter(sigmays[-1],sigmaxs[-1],marker="*",c='g',s=300)

          # Image Center
          ax1.scatter(imCeny,imCenx,marker="x",c='r',s=100)
          ax2.scatter(imCeny,imCenx,marker="x",c='r',s=100)
          ax3.scatter(imCeny,imCenx,marker="x",c='r',s=100)

      # Display significan Bounding Box
      xwidth = xmax-xmin
      ywidth = ymax-ymin
      cornersIn = [xmin,ymin,xwidth,ywidth]

      #cornersOut = [int(xmin-widen*xwidth),int(ymin-widen*ywidth),int((1.+2*widen)*xwidth),int((1.+2*widen)*ywidth)] ## asymetric : must start from center, not from xmin
      boxcen = [(xmin+xmax)/2.,(ymin+ymax)/2.]
      cornersOut = [int(boxcen[0]-(0.5+widen)*xwidth),int(boxcen[1]-(0.5+widen)*ywidth),int((1.+2*widen)*xwidth),int((1.+2*widen)*ywidth)]
      cornerscanny = [cornersIn,cornersOut]
    
      if doPlot:
          imshowrect(cornerscanny, figure=ax1,color='y')
          imshowrect(cornerscanny, figure=ax2,color='y')
          imshowrect(cornerscanny, figure=ax3,color='y')
    
      xmin,xmax,ymin,ymax = boxsigma
      boxcen = [(xmin+xmax)/2.,(ymin+ymax)/2.]

      xwidth = xmax-xmin
      ywidth = ymax-ymin
      #cornersIn = [xmin,ymin,xwidth,ywidth]
      #cornersOut = [int(xmin-widen*xwidth),int(ymin-widen*ywidth),int((1.+2*widen)*xwidth),int((1.+2*widen)*ywidth)]
    
      xywidth = max(xwidth,ywidth)
      cornersIn = [xmin,ymin,xwidth,ywidth]
      #cornersOut = [int(xmin-widen*xywidth),int(ymin-widen*xywidth),int((1.+2*widen)*xywidth),int((1.+2*widen)*xywidth)]
      cornersOut = [int(boxcen[0]-(0.5+widen)*xywidth),int(boxcen[1]-(0.5+widen)*xywidth),int((1.+2*widen)*xywidth),int((1.+2*widen)*xywidth)]

      cornerssigma = [cornersIn,cornersOut]
      
      if doPlot:
          imshowrect([cornerssigma[0]], figure=ax1,color='g',ls='--',lw=2)
          imshowrect([cornerssigma[0]], figure=ax3,color='g',ls='--',lw=2)

          if savePlot:
              plt.savefig(pngDir+runName+f'_{str(n).zfill(2)}_canny_{sigma}_marked.png')
    
              #distracting matplotlib so it actually saves the image
              time.sleep(0.5)
          plt.close()
    
      cropxmin = max(cornersOut[0],0)
      cropxmax = min(cornersOut[0]+cornersOut[2],image.shape[0])
      cropymin = max(cornersOut[1],0)
      cropymax = min(cornersOut[1]+cornersOut[3],image.shape[1])
      croppedImage = image[cropxmin:cropxmax,cropymin:cropymax]
      
      print(f"Cropped Image Size    ---    {croppedImage.shape}")
      
      fluxcen = com(croppedImage)
      censigmacropped = [censigma[0]-cropxmin, censigma[1]-cropymin]

      if doPlot:
          showImage(croppedImage,figsize=(8,8))
          plt.scatter(croppedImage.shape[1]/2.,croppedImage.shape[0]/2.,marker="+",c='b',s=300)    
          plt.scatter(fluxcen[1],fluxcen[0],marker="x",c='w',s=400)    
          plt.scatter(censigmacropped[1],censigmacropped[0],marker="*",c='g',s=300)
          plt.title(runName, fontsize=16)
          #if savePlot:
          plt.savefig(pngDir+runName+f'_{str(n).zfill(2)}_cropped_widen025.png')
    
      ## PSF RMS  
      ## [0] --> c.o.g. cropped image
      ## [1] --> c.o.g. of significant pixels
      smearedImage = croppedImage.copy()
      colintegral = np.sum(croppedImage,axis=0) * 90/4.e6
      colint2d = np.stack([colintegral for i in range(croppedImage.shape[0])])
      smearedImage += colint2d
      
      #rmss.append([psff.psfRms(croppedImage,center=fluxcen),psff.psfRms(croppedImage,center=censigmacropped),psff.psfRms(smearedImage,center=fluxcen)])
      rmss.append([psff.psfRms(croppedImage,center=fluxcen),psff.psfRms(croppedImage,center=censigmacropped),psff.psfRms(image,center=fluxcen),psff.psfRms(image,center=censigmacropped),psff.psfRms(smearedImage,center=fluxcen)])
    
      """
    # Smearing Profile
    plt.figure("smearing")
    plt.plot(colintegral)
    plt.grid(alpha=0.25)
    plt.title(runName, fontsize=16)
    plt.xlabel('column',fontsize=14)
    plt.ylabel('adu',fontsize=14)
    plt.savefig(pngDir+runName+f'_{str(n).zfill(2)}_smearingProfile.png')
    

    # Smearing Pattern
    showImage(smearedImage-croppedImage,figsize=(8,8))
    plt.scatter(smearedImage.shape[1]/2.,smearedImage.shape[0]/2.,marker="+",c='b',s=300)    
    plt.scatter(fluxcen[1],fluxcen[0],marker="x",c='w',s=400)    
    plt.scatter(censigmacropped[1],censigmacropped[0],marker="*",c='g',s=300)
    plt.title(runName, fontsize=16)
    plt.savefig(pngDir+runName+f'_{str(n).zfill(2)}_smearingPattern.png')

    # Smeared Image
    showImage(smearedImage,figsize=(8,8))
    plt.scatter(smearedImage.shape[1]/2.,smearedImage.shape[0]/2.,marker="+",c='b',s=300)    
    plt.scatter(fluxcen[1],fluxcen[0],marker="x",c='w',s=400)    
    plt.scatter(censigmacropped[1],censigmacropped[0],marker="*",c='g',s=300)
    plt.title(runName, fontsize=16)
    plt.savefig(pngDir+runName+f'_{str(n).zfill(2)}_smearedImage.png')
      """
      if doPlot:
          if savePlot:
              #distracting matplotlib so it actually saves the image
              time.sleep(0.5)
          plt.close()

          fig2, [ax4, ax5, ax6] = plt.subplots(nrows=1, ncols=3, figsize=(20, 6), sharex=True, sharey=True)
          
          ax4.imshow(image, cmap=plt.cm.gray,interpolation='nearest',origin='lower')
          ax4.axis('off')
          ax4.set_title(f'Image\n{focusPosition}', fontsize=20)
          
          ax5.imshow(significant, cmap=plt.cm.gray,interpolation='nearest',origin='lower')
          ax5.axis('off')
          ax5.set_title(f'> {sigma} sigmas', fontsize=20)
          
          ax6.imshow(image, cmap=plt.cm.gray,interpolation='nearest',origin='lower')
          ax6.axis('off')
          ax6.set_title('+ : c.o.g.(edges)   x : image center\n* centroid significant', fontsize=20)
          
          # Canny
          ax5.scatter(yc,xc,marker="+",c='y',s=300)
          ax6.scatter(yc,xc,marker="+",c='y',s=300)
          
          # Sigma
          ax5.scatter(sigmays[-1],sigmaxs[-1],marker="*",c='g',s=300)
          ax6.scatter(sigmays[-1],sigmaxs[-1],marker="*",c='g',s=300)
          
          # Image Center
          ax5.scatter(imCeny,imCenx,marker="x",c='r',s=100)
          ax6.scatter(imCeny,imCenx,marker="x",c='r',s=100)

          #imshowrect([cornerscanny[0]], figure=ax6,color='y',ls='--',lw=1)
          
          imshowrect(cornerssigma, figure=ax5,color='g',ls='--')
          imshowrect(cornerssigma, figure=ax6,color='g',ls='--')
          
          if savePlot:
              plt.savefig(pngDir+runName+f'_{str(n).zfill(2)}_significant_{sigma}_squared.png')

imCenxs = np.array(imCenxs)
imCenys = np.array(imCenys)
cannyxs = np.array(cannyxs)
cannyys = np.array(cannyys)
sigmaxs = np.array(sigmaxs)
sigmays = np.array(sigmays)
boxsigmas = np.array(boxsigmas)
boxcannys = np.array(boxcannys)
rmss    = np.array(rmss)
pixmaxs = np.array(pixmaxs)

"""
plt.figure("CannyCen",figsize=(10,8))
plt.plot(availableFocusPositions,cannyxs-imCenxs,"k.-",label="+ Canny x")
plt.plot(availableFocusPositions,imCenys-cannyys,"r.-",label="- Canny y")
plt.grid(alpha=0.25)
plt.legend()
plt.xlabel("Focus Posisition")
plt.ylabel("Center of mass (canny edges)")
plt.title(srowcol.replace("_"," ")+"\n Centroid Canny edges ($\Delta$ wrt center)", size=14)
#plt.savefig(pngDir+srowcol+f'{str(n).zfill(2)}_canny_sigma_{sigma}_centroid_vs_focus.png')
plt.plot(availableFocusPositions,sigmaxs-imCenxs,c=gray,ls="-",lw=2,label=f"+ {sigma} Sigma x")
plt.plot(availableFocusPositions,imCenys-sigmays,c=orange,ls="-",lw=2,label=f"- {sigma} Sigma y")
"""

plt.figure("SigniCen2",figsize=(10,8))
plt.plot(availableFocusPositions,cannyxs-imCenxs,c=gray,ls="-",lw=1,marker='.',label="+ Canny x")
plt.plot(availableFocusPositions,imCenys-cannyys,c=orange,ls="-",lw=1,marker='.',label="- Canny y")
plt.plot(availableFocusPositions,sigmaxs-imCenxs,"k.-",label=f"+ {sigma} Sigma x")
plt.plot(availableFocusPositions,imCenys-sigmays,"r.-",label=f"- {sigma} Sigma y")
plt.grid(alpha=0.25)
plt.legend()
plt.xlabel("Focus Posisition")
plt.ylabel("Center of mass (4 sigma)")
plt.title(srowcol.replace("_"," ")+"\n Centroid @ 4sigma ($\Delta$ wrt center)", size=14)
plt.savefig(pngDir+srowcol+f'{str(n).zfill(2)}_significant_vs_canny_sigma_{sigma}_centroid_vs_focus.png')


wx = boxsigmas[:,1]-boxsigmas[:,0]
wy = boxsigmas[:,3]-boxsigmas[:,2]
wmax = np.max(np.stack([wx,wy]), axis=0)
plt.figure("spotRMS",figsize=(10,8))
plt.plot(availableFocusPositions, wmax,c=gray,lw=2,marker='.',ls="-",label="Sel. image half-size",alpha=0.5)
plt.plot(availableFocusPositions, rmss[:,0],"k.-",label="c.o.g. flux (cropped)")
plt.plot(availableFocusPositions, rmss[:,1],"r.-",label="c.o.g. $\sigma$-clipped pixels (cropped)")
plt.plot(availableFocusPositions, rmss[:,4],c=lightblue,lw=2,marker='.',ls="-",label="c.o.g. flux (incl. smearing)")
#plt.plot(availableFocusPositions, rmss[:,2],c=gray,lw=2,marker='.',ls="-",label="c.o.g. flux (orig)")
#plt.plot(availableFocusPositions, rmss[:,3],c=orange,lw=2,marker='.',ls="-",label="c.o.g. pixels (orig)")
plt.legend(fontsize=14)
plt.grid(alpha=0.25)
plt.xlabel("Focus Posisition",fontsize=12)
plt.ylabel("Spot RMS Diameter",fontsize=12)
plt.title(srowcol.replace("_"," ")+f" - bbox x {1+widen*2}\n Spot RMS Diameter", size=14)
plt.savefig(pngDir+srowcol+f'{str(n).zfill(2)}_rmsSpotSize_vs_focus_w15_cen_flux_vs_pixel.png')
#plt.savefig(pngDir+srowcol+f'{str(n).zfill(2)}_rmsSpotSize_vs_focus_includingPartialReadoutSmearing.png')
plt.ylim(0,55)


plt.figure('max',figsize=(10,8))
plt.plot(availableFocusPositions, pixmaxs, 'k.-')
plt.grid(alpha=0.25)
plt.xlabel("Focus Posisition",fontsize=12)
plt.ylabel("Flux [adu]",fontsize=12)
plt.title(srowcol.replace("_"," ")+"\n Highest pixel flux n=0 & 1", size=14)
plt.savefig(pngDir+srowcol+f'{str(n).zfill(2)}_pixelMaxFlux.png')

plt.plot(availableFocusPositions, pixmaxs, c=gray,ls='-',marker='.')


focii15 = availableFocusPositions
rmss0 = rmss.copy()
rmss15 = rmss.copy()
rmss2 = rmss.copy()

plt.figure("Cropsize")
plt.plot(availableFocusPositions,wx,'k.-',label="x")
plt.plot(availableFocusPositions,wy,'r.-',label="y")
plt.legend()
plt.grid(alpha=0.25)


plt.figure("spotRMS",figsize=(10,8))
plt.plot(availableFocusPositions, rmss0[:,0],c=lightblue,lw=2,marker='.',ls="-",label="c.o.g. flux -- widen x 1. (w=0)")
plt.plot(availableFocusPositions, rmss15[:,0],c='k',lw=2,marker='.',ls="-",label="c.o.g. flux -- widen x 1.5 (w=0.25)")
plt.plot(availableFocusPositions, rmss2[:,0],"r.-",label="c.o.g. flux -- widen x 2 (w=0.5)")
#plt.plot(availableFocusPositions, rmss[:,1],"r.-",label="c.o.g. pixels widen x 1.5")
#plt.plot(availableFocusPositions, rmss5[:,1],c=orange,lw=2,marker='.',ls="-",label="c.o.g. pixels widen x 2)")
plt.legend(fontsize=14)
plt.grid(alpha=0.25)
plt.xlabel("Focus Posisition",fontsize=12)
plt.ylabel("Spot RMS Diameter",fontsize=12)
plt.title(srowcol.replace("_"," ")+"\n Spot RMS Diameter", size=14)
plt.savefig(pngDir+srowcol+f'{str(n).zfill(2)}_rmsSpotSize_vs_focus_widen0_vs025_vs_05.png')


################################################################################
##### FIT INTERSECTION PLAYGROUND ---> CHECK BEST VALUE OF W
################################################################################

xinit = availableFocusPositions
imax = np.where(xinit==4450)[0][0]
x = xinit[:imax+1]
iext = np.where(xinit==5200)[0][0]
xext = xinit[:iext+1]


y0 = rmss0[:,0][:imax+1]
y15 = rmss15[:,0][:imax+1]
y2 = rmss2[:,0][:imax+1]
y0ext = rmss0[:,0][:iext+1]
y15ext = rmss15[:,0][:iext+1]
y2ext = rmss2[:,0][:iext+1]

c0, yfit0, model0 = mypolyfit(x, y0, order=1)
c15, yfit15, model15 = mypolyfit(x, y15, order=1)
c2, yfit2, model2 = mypolyfit(x, y2, order=1)

yfit0ext  = model0(xext)
yfit15ext = model15(xext)
yfit2ext  = model2(xext)


plt.figure("Residuals")
plt.plot(xext, yfit0ext-y0ext,c=lightblue,lw=2,marker='.',ls="-",label="widen x 1. (w=0)")
plt.plot(xext, yfit15ext-y15ext,c='k',lw=2,marker='.',ls="-",label="widen x 1.5 (w=0.25)")
plt.plot(xext, yfit2ext-y2ext,"r.-",label="widen x 2 (w=0.5)")
#plt.plot(availableFocusPositions, rmss[:,1],"r.-",label="c.o.g. pixels widen x 1.5")
#plt.plot(availableFocusPositions, rmss5[:,1],c=orange,lw=2,marker='.',ls="-",label="c.o.g. pixels widen x 2)")
plt.legend(fontsize=14)
plt.grid(alpha=0.25)
plt.xlabel("Focus Posisition",fontsize=12)
plt.ylabel("Fit Residual",fontsize=12)
plt.title(srowcol.replace("_"," ")+"\n Fit Residual", size=14)
plt.savefig(pngDir+srowcol+f'{str(n).zfill(2)}_rmsSpotSize_vs_focus_widen0_vs025_vs_05_fitResidual_Left_zoom.png')

################################################################################
##### FIT INTERSECTION PLAYGROUND
################################################################################


# ORIGINAL PSF ANALYSIS
i = 1   #the 6 rows are for angle to opt. axis in ['00','04','08','10','14','18'] ==> i=1 ==> 4 deg  &  i=2 ==> 8 degrees
x = actualFocusPositions[i,:]   # (see cslPsfs.py -- header for where to look into the file) or load from /Users/pierre/plato/data/rtPsfs/csl/fociiArray.pickle
y = rms[i,:]    # (6,98) where the 6 rows are for angle to opt. axis in ['00','04','08','10','14','18'] (see cslPsfs.py  or load from /Users/pierre/plato/data/rtPsfs/csl/rmsArray.pickle)

# FOCUS SWEEP ANALYSIS
x = availableFocusPositions
y = rmss[:,0]


method = "include"  # Include N samples on both sides, counting from the edges of the scan
method = "exclude"  # Exclude N microns around a first estimate of the focus

# DATA SELECTION for the linear fit
# Keep equal number left and right
if method.lower().find('incl') >= 0:
    method= "include"
    kept = 35
    iminl,imaxl = 0,kept
    iminr,imaxr = x.size-kept,x.size

elif method.lower().find('excl') >= 0:
    method = "exclude"
    # DATA SELECTION INDICES for the linear fit
    # Exclude region around estimate
    estimate=5400
    estimateIndex = np.where(np.abs(x-estimate) == np.min(np.abs(x-estimate)))[0][0]

    avoidance = 500
    leftmax  = estimate-avoidance
    rightmin = estimate+avoidance

    imaxl = np.where(np.abs(x-leftmax) == np.min(np.abs(x-leftmax)))[0][0]
    iminr = np.where(np.abs(x-rightmin) == np.min(np.abs(x-rightmin)))[0][0]

    iminl = 0
    imaxr = x.size


## DATA SELECTION

# l & r symbolize the "left" and "right" sides of the focus curve
#xl = np.arange(imaxl-iminl) + iminl
#xr = np.arange(imaxr-iminr) + iminr

xl = x[iminl:imaxl]
xr = x[iminr:imaxr]


yl = y[iminl:imaxl]
yr = y[iminr:imaxr]

xc = x[imaxl:iminr]
yc = y[imaxl:iminr]


cl, yfitl, modell = mypolyfit(xl, yl, order=1)
cr, yfitr, modelr = mypolyfit(xr, yr, order=1)

xfocus = (cr[1]-cl[1]) / (cl[0] - cr[0])

print (f"Fit focus {xfocus:4.0f}")


## IMAGE ANALYSIS
saveplot=1

plt.figure(figsize=(12,8))
plt.plot(x,y,'k.-',label='measures')
plt.plot(xl,yl,'r.-',label='kept left & right')
plt.plot(xr,yr,'r.-')
plt.plot(xl,yfitl,c=orange,ls='-',marker='.',lw=2,label="fit left & right")
plt.plot(xr,yfitr,c=orange,ls='-',marker='.',lw=2)
plt.plot(xc,modell(xc),c=gray,ls='-',marker='.',lw=2, label="fit extrapolations")
plt.plot(xc,modelr(xc),c=gray,ls='-',marker='.',lw=2)
plt.plot([xfocus,xfocus],[-10,50],c=gray,ls="--",lw=1,label=f"Estimated Focus {xfocus:4.0f}")
plt.legend(fontsize=14)
plt.grid(alpha=0.25)
plt.xlabel("Focus Positions",fontsize=14)
plt.ylabel("Spot RMS Diameter",fontsize=14)
plt.ylim(-10,60)
if method == 'include':
    plt.title(srowcol.replace("_"," ")+f"\n Spot RMS Diameter -- Incl {kept} samples on both sides", size=14)
    #if saveplot: plt.savefig(pngDir+srowcol+f'{str(n).zfill(2)}_rmsSpotSize_Fit_n_intersection_including_{kept}_samples_widen{widen}_zoom.png')
elif method == 'exclude':
    plt.title(srowcol.replace("_"," ")+f"\n Spot RMS Diameter -- Excl 2 x {avoidance} microns around {estimate}", size=14)
    if saveplot: plt.savefig(pngDir+srowcol+f'{str(n).zfill(2)}_rmsSpotSize_Fit_n_intersection_excluding_{avoidance}_umAround{estimate}_widen{widen}.png')
plt.xlim(4800,6000)
plt.ylim(-5,15)

## ORIGINAL CSL PSFS

plt.figure(figsize=(12,8))
plt.plot(x,y,'k.-',label='measures')
plt.plot(xl,yl,'r.-',label='kept left & right')
plt.plot(xr,yr,'r.-')
plt.plot(xl,yfitl,c=orange,ls='-',marker='.',lw=2,label="fit left & right")
plt.plot(xr,yfitr,c=orange,ls='-',marker='.',lw=2)
plt.plot(xc,modell(xc),c=gray,ls='-',marker='.',lw=2, label="fit extrapolations")
plt.plot(xc,modelr(xc),c=gray,ls='-',marker='.',lw=2)
plt.plot([xfocus,xfocus],[-10,50],c=gray,ls="--",lw=1,label=f"Estimated Focus {xfocus:4.0f}")
plt.legend(fontsize=14)
plt.grid(alpha=0.25)
plt.xlabel("Focus Positions",fontsize=14)
plt.ylabel("Spot RMS Diameter",fontsize=14)
plt.ylim(-10,60)
if method=='include':
    plt.title(f"Zemax PSFs\n Spot RMS Diameter -- Incl {kept} samples on both sides", size=14)
    plt.savefig(pngDir+f"csl_psf_rmsSpotSize_Fit_n_intersection_04_including_{kept}_samples.png")
elif method=='exclude':
    plt.title(f"Zemax PSFs\n Spot RMS Diameter -- Excl 2 x {avoidance} microns around {estimate}", size=14)
    plt.savefig(pngDir+f"csl_psf_rmsSpotSize_Fit_n_intersection_04_excluding_{avoidance}_umAround{estimate}_zoom.png")
plt.ylim(-5,15)
plt.xlim(4800,6000)


# ORIGINAL PSF ANALYSIS
i = 2   #the 6 rows are for angle to opt. axis in ['00','04','08','10','14','18'] ==> i=1 ==> 4 deg  &  i=2 ==> 8 degrees
x = actualFocusPositions[i,:]   # (see cslPsfs.py -- header for where to look into the file) or load from /Users/pierre/plato/data/rtPsfs/csl/fociiArray.pickle
y = rms[i,:]    # (6,98) where the 6 rows are for angle to opt. axis in ['00','04','08','10','14','18'] (see cslPsfs.py  or load from /Users/pierre/plato/data/rtPsfs/csl/rmsArray.pickle)

# FOCUS SWEEP ANALYSIS
x = availableFocusPositions
y = rmss[:,0]

method = 'exclude'
focusInterpolation(x,y,method=method)
if method == 'include':
    plt.title(srowcol.replace("_"," ")+f"\n Spot RMS Diameter -- Incl {kept} samples on both sides", size=14)
    if saveplot: plt.savefig(pngDir+srowcol+f'{str(n).zfill(2)}_rmsSpotSize_Fit_n_intersection_including_{kept}_samples_widen{widen}.png')
elif method == 'exclude':
    plt.title(srowcol.replace("_"," ")+f"\n Spot RMS Diameter -- Excl 2 x {avoidance} microns around {estimate}", size=14)
    if saveplot: plt.savefig(pngDir+srowcol+f'{str(n).zfill(2)}_rmsSpotSize_Fit_n_intersection_excluding_{avoidance}_umAround{estimate}_widen{widen}_zoom.png')


def focusInterpolation(focii,figureOfMerit,method='include',parameters=None, doPlot=True, verbose=True):
    """
    focii         : array of focus values (delta Z vs L6S2)

    figureOfMerit : array of results from the estimation of the figure of merit at each focus value

    method        : 
            include : fit on a given number of samples, counting from both edges of the focus-sweep
                      parameters = [number_of_samples_kept_on_each_side]
            exclude : fit on all samples, excluding a given range of 'focii' around 
                      parameters = [focus_guesstimate, range_to_be_excluded_on_each_side_of_focus_guesstimate]  (same units as focii)

    if paramters is not None, it must be the list of all parameters required by the chosen "method" (see above)
    
    return : best focus estimated from the intersection of the 2 fits    
    """
    from convenienceFunctions import mypolyfit

    x = focii
    y = figureOfMerit
    
    if method.lower().find('incl')>=0:
        method = 'include'
        if parameters is None:
            kept = 35
            parameters = [kept]
        else:
            kept = parameters[0]
        iminl,imaxl = 0,kept
        iminr,imaxr = x.size-kept,x.size

    elif method.lower().find('excl') >=0:
        # DATA SELECTION INDICES for the linear fit
        # Exclude region around estimate
        method = 'exclude'
        if parameters is None:
            estimate=5300
            avoidance = 300       
            parameters = [estimate,avoidance]
        else:
            estimate, avoidance = parameters
        
        #estimateIndex = np.where(np.abs(x-estimate) == np.min(np.abs(x-estimate)))[0][0]
        
        leftmax  = estimate-avoidance
        rightmin = estimate+avoidance
        
        imaxl = np.where(np.abs(x-leftmax) == np.min(np.abs(x-leftmax)))[0][0]
        iminr = np.where(np.abs(x-rightmin) == np.min(np.abs(x-rightmin)))[0][0]
        
        iminl = 0
        imaxr = x.size

    ## DATA SELECTION
    
    # l & r symbolize the "left" and "right" sides of the focus curve
    
    xl = x[iminl:imaxl]
    xr = x[iminr:imaxr]
     
    yl = y[iminl:imaxl]
    yr = y[iminr:imaxr]
        
    xc = x[imaxl:iminr]
    yc = y[imaxl:iminr]
    
    cl, yfitl, modell = mypolyfit(xl, yl, order=1)
    cr, yfitr, modelr = mypolyfit(xr, yr, order=1)
    
    xfocus = (cr[1]-cl[1]) / (cl[0] - cr[0])
        
    if verbose: print (f"Method {method} -- Parameters {parameters} -- Fit focus {xfocus:4.0f}")
 
    if doPlot:
        plt.figure(figsize=(12,8))
        plt.plot(x,y,'k.-',label='measures')
        plt.plot(xl,yl,'r.-',label='kept left & right')
        plt.plot(xr,yr,'r.-')
        plt.plot(xl,yfitl,c=orange,ls='-',marker='.',lw=2,label="fit left & right")
        plt.plot(xr,yfitr,c=orange,ls='-',marker='.',lw=2)
        plt.plot(xc,modell(xc),c=gray,ls='-',marker='.',lw=2, label="fit extrapolations")
        plt.plot(xc,modelr(xc),c=gray,ls='-',marker='.',lw=2)
        plt.plot([xfocus,xfocus],[-10,50],c=gray,ls="--",lw=1,label=f"Estimated Focus {xfocus:4.0f}")
        plt.legend(fontsize=14)
        plt.grid(alpha=0.25)
        plt.xlabel("Focus Positions",fontsize=14)
        plt.ylabel("Spot RMS Diameter",fontsize=14)
        plt.ylim(-10,60)
        if method == 'include':
            plt.title(f"Spot RMS Diameter -- Incl {kept} samples on both sides", size=14)
        elif method == 'exclude':
            plt.title(f"\n Spot RMS Diameter -- Excl 2 x {avoidance} microns around {estimate}", size=14)
        #    #if saveplot: plt.savefig(pngDir+srowcol+f'{str(n).zfill(2)}_rmsSpotSize_Fit_n_intersection_excluding_{avoidance}_umAround{estimate}_widen{widen}_zoom.png')

    return xfocus


################################################################################
### RANDOM SEEDS --> MONTE CARLO ANALYSIS
################################################################################


dataDir = "/Users/pierre/plato/pr/simout/egse/seeds/"

pngDir = "/Users/pierre/plato/pr/pngs/egse/seeds/"

row,column,ccdCode = 3000,1000,2
row,column,ccdCode = 4000,500,2

allfiles = fs.fileSelect([f"{str(row).zfill(4)}",f"{str(column).zfill(4)}","hdf5"], location=dataDir, listOrder=1)

numExposures = 25

srowcol = f"egse_{str(row).zfill(4):4s}_{str(column).zfill(4):4s}_{ccdCode}_"

availableFocusPositions = []
for f in allfiles:
    availableFocusPositions.append(int(f.split('_')[-1][:4]))
    
availableFocusPositions = np.sort(np.array(availableFocusPositions,dtype=int))

# Simple center of image
imCenxs = np.zeros([len(availableFocusPositions),numExposures])
imCenys = np.zeros([len(availableFocusPositions),numExposures])

# Centroid based on sigma clipping + simple centroid of significant pixels (boolean map) -- arrays previously named sigmaxs and sigmays
sigCenxs = np.zeros([len(availableFocusPositions),numExposures])
sigCenys = np.zeros([len(availableFocusPositions),numExposures])


# Highest pixels in background subtracted image
pixmaxs = np.zeros([len(availableFocusPositions),numExposures])

boxsigmas = np.zeros([len(availableFocusPositions),numExposures,4])
croppedImageSizes = np.zeros([len(availableFocusPositions),numExposures,2])

# PSF RMS
rmssSigma   = np.zeros([len(availableFocusPositions),numExposures])  # Centroid based on boolean map of significant pixels (sigma clipped)
rmssFlux    = np.zeros([len(availableFocusPositions),numExposures])  # Centroid based on flux, within cropped image (cropping based sigma clipping in flux)

# Bounding box widening : widen = 0.5 means 50% width added left & right == width x 2
#widen = .5
widen = 0.25

verbose=0

for f,focusPosition in enumerate(availableFocusPositions):
    plt.close()
    runName = srowcol+f"{focusPosition}"
    
    print()
    print(f"{runName}")
    print()
    
    sh5 = h5py.File(dataDir+runName+".hdf5",'r')
    
    numExp = h5get(sh5,["ObservingParameters","numExposures"],verbose=verbose)
    if numExp != numExposures:
        print(f"focusPosition {focusPosition}")
        print("WARNING: NOT THE EXPECTED NUMBER OF SIMULATED IMAGES --> BREAKING HERE")
        break
    
    for n in range(numExp):
      
      imorig = np.array(getSim(sh5,n=n))

      # Centroid and bounding box of significant pixel map. Centroid is simply based on boolean map.
      censigma,boxsigma = psff.psfBox(imorig,method='sigma',sigma=4,cosmicRemoval=True,kernel=None,verbose=verbose)

      # print(f,n,censigma)
      
      # Centroid based on boolean barycenter from sigma-clipped pixels
      sigCenxs[f,n] = censigma[0]
      sigCenys[f,n] = censigma[1]
    
      boxsigmas[f,n,:] = boxsigma

      image,background = backgroundSubtraction(imorig,method='edge',width=5,verbose=verbose)
    
      pixmaxs[f,n] = np.max(image)
    
      sigma = 4

      noise = np.std(background)
      significant = np.zeros_like(image)
      significant[np.where(image>sigma*noise)] = 1
      
      imCenx,imCeny = image.shape[0]//2,image.shape[1]//2
      imCenxs[f,n] = imCenx
      imCenys[f,n] = imCeny

      xmin,xmax,ymin,ymax = boxsigma
      boxcen = [(xmin+xmax)/2.,(ymin+ymax)/2.]

      xwidth = xmax-xmin
      ywidth = ymax-ymin
    
      xywidth = max(xwidth,ywidth)
      cornersIn = [xmin,ymin,xwidth,ywidth]
      cornersOut = [int(boxcen[0]-(0.5+widen)*xywidth),int(boxcen[1]-(0.5+widen)*xywidth),int((1.+2*widen)*xywidth),int((1.+2*widen)*xywidth)]

      cornerssigma = [cornersIn,cornersOut]
      
      cropxmin = max(cornersOut[0],0)
      cropxmax = min(cornersOut[0]+cornersOut[2],image.shape[0])
      cropymin = max(cornersOut[1],0)
      cropymax = min(cornersOut[1]+cornersOut[3],image.shape[1])
      croppedImage = image[cropxmin:cropxmax,cropymin:cropymax]
      
      print(f"Cropped Image Size    ---    {croppedImage.shape}")
      croppedImageSizes[f,n] = croppedImage.shape
      
      # Flux centroid of the cropped image (cropping based on sigma-clipping)
      fluxcen = com(croppedImage)
      # Centroid based on boolean barycenter from sigma-clipped pixels (relocated wrt the image cropping)
      censigmacropped = [censigma[0]-cropxmin, censigma[1]-cropymin]
      

      # SIMULATION OF READOUT SMEARING : just add a fraction of the column integral to every pixel
      smearedImage = croppedImage.copy()
      colintegral = np.sum(croppedImage,axis=0) * 90/4.e6
      colint2d = np.stack([colintegral for i in range(croppedImage.shape[0])])
      smearedImage += colint2d

      ## PSF RMS  
      ## [0] --> c.o.g. cropped image
      ## [1] --> c.o.g. of significant pixels
      
      #rmss.append([psff.psfRms(croppedImage,center=fluxcen),psff.psfRms(croppedImage,center=censigmacropped),psff.psfRms(smearedImage,center=fluxcen)])
      #rmss.append([psff.psfRms(croppedImage,center=fluxcen),psff.psfRms(croppedImage,center=censigmacropped),psff.psfRms(image,center=fluxcen),psff.psfRms(image,center=censigmacropped)])

      rmssSigma[f,n] = psff.psfRms(croppedImage,center=censigmacropped,verbose=verbose)
      rmssFlux[f,n]  = psff.psfRms(croppedImage,center=fluxcen,verbose=verbose)
      

zfocus     = np.zeros([numExposures,2]) # Best focus from line-fit intersection. method -- 0:include; 1:exclude

# 1st iteration
kept = 35

# 1nd iteration
avoidance = 300

for n in range(numExposures):

    # FOCUS SWEEP ANALYSIS
    x = availableFocusPositions
    y = rmssFlux[:,n]

    method = 'include'
    zfocus[n,0] = focusInterpolation(x,y,method=method,parameters=[kept],doPlot=False)
    
    roundedTo50 = int(np.round(zfocus[n,0]/50.)*50)

    method = 'exclude'
    zfocus[n,1] = focusInterpolation(x,y,method=method,parameters=[roundedTo50,avoidance],doPlot=False)





# Are all cropped images square ?
np.allclose(croppedImageSizes[:,:,0],croppedImageSizes[:,:,1])
# Are some cropped image sizes empty
np.any(croppedImageSizes == 0)


# Image sizes
imsize    = np.nanmean(np.nanmean(croppedImageSizes,axis=1),axis=1)
imsizestd = np.nanstd(np.nanstd(croppedImageSizes,axis=1),axis=1)
imsize = np.nanmean(croppedImageSizes[:,0],axis=1)
imsizex = croppedImageSizes[:,0,0]
imsizey = croppedImageSizes[:,0,1]


""" Error bar demo
x = [1, 2, 3, 4]
y = [1, 4, 9, 16]
e = [0.5, 1., 1.5, 2.]
plt.errorbar(x, y, yerr=e, fmt='o')
"""

plt.figure("size")
plt.plot(availableFocusPositions,imsize,c='k',ls='-',marker='.')#,label=f"Exp {n}")
plt.xlabel('Focus [$\mu m$]',size=14)
plt.ylabel('Pixel',size=14)
plt.title(f"Cropped Image Size",size=14)
plt.grid(linewidth=0.5,alpha=0.3)
plt.savefig(pngDir+f"egse_randomSeeds_{srowcol}meanCroppedImageSize.png")

plt.figure("sizestd")
plt.plot(availableFocusPositions,imsizestd,c='k',ls='-',marker='.')#,label=f"Exp {n}")
plt.xlabel('Focus [$\mu m$]',size=14)
plt.ylabel('Pixel',size=14)
plt.title(f"Cropped Image Size",size=14)
plt.grid(linewidth=0.5,alpha=0.3)
plt.savefig(pngDir+f"egse_randomSeeds_{srowcol}stdCroppedImageSize.png")

croppedImageSizes[np.where(availableFocusPositions == 7000)[0],:,:]

# Spot size RMS
rmsmean   = np.nanmean(rmssFlux,axis=1)
rmsstd    = np.nanstd(rmssFlux,axis=1)
plt.figure("rms")
#plt.plot(availableFocusPositions,imsize,c=gray,ls='-',marker='.',alpha=0.25,label="Cropped image size")
#plt.plot(availableFocusPositions,rmsmean,c='k',ls='-',marker='.',label=f"RMS")
plt.errorbar(availableFocusPositions,rmsmean,yerr=rmsstd,c='k',ls='-',marker='.',label=f"RMS")
plt.xlabel('Focus [$\mu m$]',size=14)
plt.ylabel('Pixel',size=14)
plt.title(f"RMS",size=14)
plt.grid(linewidth=0.5,alpha=0.3)
plt.legend()
plt.title()
plt.savefig(pngDir+f"egse_randomSeeds_{srowcol}rms.png")
plt.ylim(0,120)
plt.savefig(pngDir+f"egse_randomSeeds_{srowcol}rms_n_CroppedImageSize.png")


focus1st = f"{int(np.round(np.mean(zfocus[:,0]))):4d}$\,\pm\,${np.std(zfocus[:,0]):4.1f}"
focus2nd = f"{int(np.round(np.mean(zfocus[:,1]))):4d}$\,\pm\,${np.std(zfocus[:,1]):4.1f}"

codeV = [5416,5416] # 4 degree, 2nd iteration
# Focus Solution
plt.figure("zfocus")
#plt.plot(availableFocusPositions,imsize,c=gray,ls='-',marker='.',alpha=0.25,label="Cropped image size")
plt.plot(np.arange(numExposures),zfocus[:,0],c='k',ls='--',marker='.',label=f"1st iteration {focus1st}",alpha=0.5)
plt.plot([0,numExposures],[np.mean(zfocus[:,0]),np.mean(zfocus[:,0])],c=gray,ls='-',alpha=0.25)

codeV = [5339,5339] # 8 degree, 1st iteration
plt.plot([0,numExposures],codeV,c=orange,ls='--',alpha=0.5,label=f"Code V {codeV[0]} (1st it)")

plt.plot(np.arange(numExposures),zfocus[:,1],c='k',ls='-',marker='.',label=f"2nd iteration {focus2nd}")
plt.plot([0,numExposures],[np.mean(zfocus[:,1]),np.mean(zfocus[:,1])],c=gray,ls='-',alpha=0.25)

codeV = [5343,5343] # 8 degree, 2nd iteration
plt.plot([0,numExposures],codeV,c=orange,ls='-',alpha=0.5,label=f"Code V {codeV[0]} (2nd it)")

plt.xlabel('Exposure #',size=14)
plt.ylabel('Focus [$\mu$m from L6]',size=14)
plt.title(f"Pixel {srowcol[5:-1]} (8 deg from OA)\nAvoidance $\pm\ ${avoidance}$\mu$m",size=14)
plt.grid(linewidth=0.5,alpha=0.3)
plt.legend()
plt.ylim(5385,5430)
plt.savefig(pngDir+f"egse_randomSeeds_{srowcol}zfocus_excluding{avoidance}um.png")



