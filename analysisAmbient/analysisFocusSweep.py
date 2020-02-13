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


################################################################################
### CSL ROOM TEMPERATURE REDUCTION PROCESS
"""
Summary of reduction parameters
-	Background removal: background estimated from the median of the edges of the image
-	Imagette selection (source identification):
    o	Bounding box estimation: selection of significant pixels via sigma clipping (4 sigma)
    o	Clean the isolated significant pixels via binary opening 
        	1st pass kernel = 3x3 cross
        	2nd pass kernel: remove the left and bottom branches from the cross
        	If no pixel is left after filtering  skip filtering
    o	widening to a square of width of (2w+1)*max(size of bounding box)
-	Centroid: flux centroid within the imagette
-	Straight Fit : 
    o	1st pass : include 35 data point, i.e. 1750 microns on both edges of the focus sweep
    o	2nd pass : include all data, except 500 microns on each side of the 1st pass solution
-	Repeat measure ~ 25 times to mitigate the stochastic variations
"""

################################################################################
## MODULE 1 : BASICS
################################################################################

import sep
import numpy as np
from matplotlib.patches import Ellipse
from skimage.feature import canny
from skimage import morphology
from imageUtils import backgroundSubtraction

import scipy.ndimage as ndi
from testSimulation.simulationUtils import coaddSimulatedImages as coadd

dataDir = "/Users/pierre/plato/pr/simout/egse/smearingNo/"
pngDir = "/Users/pierre/plato/pr/pngs/egse/smearingNo/"

#dataDir = "/Users/pierre/plato/pr/simout/egse/seeds/"
#pngDir = "/Users/pierre/plato/pr/pngs/egse/seeds/"

dataDir = "/Users/pierre/plato/pr/simout/egse/hartmann45/"
pngDir = "/Users/pierre/plato/pr/pngs/egse/hartmann45/"

row,column,ccdCode = 2500,2000,2

allfiles = fs.fileSelect([f"{str(row).zfill(4)}",f"{str(column).zfill(4)}","hdf5"], location=dataDir)

# First simulations ("smearingNo") without changing the random seeds : 20 exposures
# Second batch ("seeds"), changing the random seeds : 25 exposures
numExposures = 25

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


################################################################################
## MODULE 2 : DISPLAY SIMULATED IMAGES & FINE TUNE REDUCTION PARAMETERS
################################################################################


#dataDir = "/Users/pierre/plato/pr/simout/egse/smearingNo/"
#pngDir = "/Users/pierre/plato/pr/pngs/egse/smearingNo/"

#dataDir = "/Users/pierre/plato/pr/simout/egse/seeds/"
#pngDir = "/Users/pierre/plato/pr/pngs/egse/seeds/"

#dataDir = "/Users/pierre/plato/pr/simout/egse/tol/"
#pngDir = "/Users/pierre/plato/pr/pngs/egse/tol/"

dataDir = "/Users/pierre/plato/pr/simout/egse/hartmann45/"
pngDir = "/Users/pierre/plato/pr/pngs/egse/hartmann45/"

row,column,ccdCode = 4000, 500,2  #  4 degrees
row,column,ccdCode = 3000,1000,2  #  8 degrees
row,column,ccdCode = 2500,2000,2  # 12 degrees
row,column,ccdCode = 1000,3000,2  # 18 degrees

allfiles = fs.fileSelect([f"{str(row).zfill(4)}",f"{str(column).zfill(4)}","hdf5"], location=dataDir, listOrder=1)

numExposures = 25

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
widen = 0.25

doPlot,savePlot,verbose=1,1,0

sigma = 4

filtering   = False
hfiltering  = {True:"filtered", False: "unfiltered"}

#for focusPosition in availableFocusPositions:
for focusPosition in [5140]:
    if doPlot: plt.close()
    runName = srowcol+f"{focusPosition}"
    
    print()
    print(f"{runName}")
    print()
    
    sh5 = h5py.File(dataDir+runName+".hdf5",'r')
    
    numExp = h5get(sh5,["ObservingParameters","numExposures"])
    
    #for n in range(numExp):
    for n in [5]:
        
      print(f"STARTING n == {n}")
    
      imorig = np.array(getSim(sh5,n=n))
    
      censigma,boxsigma = psfBox(imorig,method='sigma',sigma=sigma,cosmicRemoval=filtering,kernel=None,verbose=1)
      sigmaxs.append(censigma[0])
      sigmays.append(censigma[1])
    
      cencanny,boxcanny = psff.psfBox(imorig,method='canny',sigma=sigma,cosmicRemoval=filtering,kernel=None,verbose=verbose)
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
      cornersOut = [int(xmin-widen*xywidth),int(ymin-widen*xywidth),int((1.+2*widen)*xywidth),int((1.+2*widen)*xywidth)]

      """      
      if int((1.+2*widen)*xywidth) >= 8:
          cornersOut = [int(boxcen[0]-(0.5+widen)*xywidth),int(boxcen[1]-(0.5+widen)*xywidth),int((1.+2*widen)*xywidth),int((1.+2*widen)*xywidth)]
      else:
          cornersOut = [int(boxcen[0])-4,int(boxcen[1])-4,8,8]
      """
      
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
          plt.savefig(pngDir+runName+f'_{str(n).zfill(2)}_{hfiltering[filtering]}_cropped_widen025.png')
    
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
          ax4.set_title(f'Image\n{focusPosition} - {n}', fontsize=20)
          
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
              plt.savefig(pngDir+runName+f'_{str(n).zfill(2)}_{hfiltering[filtering]}_significant_{sigma}_squared.png')

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
## MODULE 3 : straight fit pretests
################################################################################

"""
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
"""


################################################################################
## MODULE 4 : FIT INTERSECTION PLAYGROUND 
##                CHECK MINIMA (FOCUS) OF NOISELESS PSFs & simulations
##                CHECK BEST VALUE OF W
################################################################################

# CSL PSFs
oaangle = ['00','04','08','10','14','18']
oaangle = ['00','02','04','06','08','10','12','14','16','18']

# CSL PSFs TOL
oaangle = ['00','04','08','12','16']

# ORIGINAL PSF ANALYSIS
i = 6   #the 6 rows are for angle to opt. axis in ['00','04','08','10','14','18'] ==> i=1 ==> 4 deg  &  i=2 ==> 8 degrees
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
    estimate=5250
    estimateIndex = np.where(np.abs(x-estimate) == np.min(np.abs(x-estimate)))[0][0]

    avoidance = 300
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
    #if saveplot: plt.savefig(pngDir+srowcol+f'{str(n).zfill(2)}_rmsSpotSize_Fit_n_intersection_excluding_{avoidance}_umAround{estimate}_widen{widen}.png')
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
    plt.title(f"Zemax PSFs - {oaangle[i]} degrees\n Spot RMS Diameter -- Incl {kept} samples on both sides", size=14)
    plt.savefig(pngDir+f"csl_psf_rmsSpotSize_Fit_n_intersection_{oaangle[i]}_including_{kept}_samples.png")
elif method=='exclude':
    plt.title(f"Zemax PSFs - {oaangle[i]} degrees\n Spot RMS Diameter -- Excl 2 x {avoidance} microns around {estimate}", size=14)
    plt.savefig(pngDir+f"csl_psf_rmsSpotSize_Fit_n_intersection_{oaangle[i]}_excluding_{avoidance}_umAround{estimate}.png")
plt.ylim(-5,15)
plt.xlim(4400,5800)
if method=='include':
    plt.savefig(pngDir+f"csl_psf_rmsSpotSize_Fit_n_intersection_{oaangle[i]}_including_{kept}_samples_zoom.png")
elif method=='exclude':
    plt.savefig(pngDir+f"csl_psf_rmsSpotSize_Fit_n_intersection_{oaangle[i]}_excluding_{avoidance}_umAround{estimate}_zoom.png")

# NOMINAL PSFS
# 10
#   incl (35)        : 5291
#   excl (5250, 300) : 5193 
#   excl (5250, 500) : 5294
# 12
#   incl (35)        : 5237
#   excl (5250, 300) : 5240 
#   excl (5250, 500) : 5241
# 14
#   incl (35)        : 5176
#   excl (5150, 300) : 5178 
#   excl (5150, 500) : 5180

# TOLERANCE PSFS
# 04
#   incl (35)        : 5129
#   excl (5150, 300) : 5130 
#   excl (5150, 500) : 5130
# 08
#   incl (35)        : 5062
#   excl (5150, 300) : 5062 
#   excl (5150, 500) : 5062
# 12
#   incl (35)        : 4976
#   excl (5150, 300) : 4975
#   excl (5150, 500) : 4976

# ORIGINAL PSF ANALYSIS
i = 2   #the 6 rows are for angle to opt. axis in ['00','04','08','10','14','18'] ==> i=1 ==> 4 deg  &  i=2 ==> 8 degrees
x = actualFocusPositions[i,:]   # (see cslPsfs.py -- header for where to look into the file) or load from /Users/pierre/plato/data/rtPsfs/csl/fociiArray.pickle
y = rms[i,:]    # (6,98) where the 6 rows are for angle to opt. axis in ['00','04','08','10','14','18'] (see cslPsfs.py  or load from /Users/pierre/plato/data/rtPsfs/csl/rmsArray.pickle)

# FOCUS SWEEP ANALYSIS
x = availableFocusPositions
y = rmss[:,0]

method = 'exclude'
fitIntersection(x,y,method=method)
if method == 'include':
    plt.title(srowcol.replace("_"," ")+f"\n Spot RMS Diameter -- Incl {kept} samples on both sides", size=14)
    if saveplot: plt.savefig(pngDir+srowcol+f'{str(n).zfill(2)}_rmsSpotSize_Fit_n_intersection_including_{kept}_samples_widen{widen}.png')
elif method == 'exclude':
    plt.title(srowcol.replace("_"," ")+f"\n Spot RMS Diameter -- Excl 2 x {avoidance} microns around {estimate}", size=14)
    if saveplot: plt.savefig(pngDir+srowcol+f'{str(n).zfill(2)}_rmsSpotSize_Fit_n_intersection_excluding_{avoidance}_umAround{estimate}_widen{widen}_zoom.png')


################################################################################
## MODULE 5 : FOCUS FIT INTERSECTION
################################################################################


def fitIntersection(focii,figureOfMerit,method='include',parameters=None, doPlot=True, verbose=True):
    """
    focii         : array of focus values (delta Z vs L6S2)

    figureOfMerit : array of results from the estimation of the figure of merit at each focus value

    method        : 
            include : fit on a given number of samples, counting from both edges of the focus-sweep
                      parameters = [number_of_samples_kept_on_each_side]

            exclude : fit on all samples, excluding a given range of 'focii' around 
                      parameters = [focus_guesstimate, range_to_be_excluded_on_each_side_of_focus_guesstimate]  (same units as focii)

            init    : same as include, but additionally exclude 'init' samples on the left start
                      parameters = [number_of_samples_kept_on_each_side, numbers_of_samples_to_ignore_on_the_left_edge, starting_x_for_right_wing_fit]
                      starting_x_for_right_wing_fit : if None or negative, it is ignored ('kept' is then the only driver)

            2init    : same as exclude, but additionally exclude 'init' samples on the left start
                      parameters = [focus_guesstimate, range_to_be_excluded_on_each_side_of_focus_guesstimate, numbers_of_samples_to_ignore_on_the_left_edge, starting_x_for_right_wing_fit]
                      starting_x_for_right_wing_fit : if None or negative, it is ignored ('range_to_be_excluded' is then the only driver)

    Default parameters are provided for methods 'include' and 'exclude'
    For methods "init" and "2init", 'parameters' must be specified
    In any case, if parameters is not None, it must be the list of all parameters required by the chosen "method" (see above)
    
    return : best focus estimated from the intersection of the 2 fits    
    """
    from convenienceFunctions import mypolyfit

    x = focii
    y = figureOfMerit
    
    
    if method.lower().find('incl')>=0:
        # Include 'kept' samples on each edge of the focus sweep
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
        # Exclude region around estimate (2nd iteration after 'include')
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

    elif method.lower().find('2init')>=0:
        # EQUIVALENT TO 'exclude', but in addition we ignore init_excl samples on the left (2nd pass of 'init')
        
        method = '2init'
        estimate, avoidance,init_excl,xminr = parameters
        
        #estimateIndex = np.where(np.abs(x-estimate) == np.min(np.abs(x-estimate)))[0][0]
        
        leftmax  = estimate-avoidance
        rightmin = estimate+avoidance
        
        imaxl = np.where(np.abs(x-leftmax) == np.min(np.abs(x-leftmax)))[0][0]
        iminr = np.where(np.abs(x-rightmin) == np.min(np.abs(x-rightmin)))[0][0]
        if (xminr is not None) and (xminr > 0):
            iminr = max(iminr,np.where(x>=xminr)[0][0])
        
        iminl = init_excl
        imaxr = x.size
        

        print (f"{x.size}, Left {iminl},{imaxl}, Right {iminr},{imaxr}")

    elif method.lower().find('init')>=0:
        # EQUIVALENT TO 'include', but in addition we ignore init_excl samples on the left
        
        method = 'init'
        kept,init_excl,xminr = parameters
        
        iminl,imaxl = init_excl,kept
        imaxr = x.size
        iminr = x.size-kept

        if (xminr is not None) and (xminr > 0):
            iminr = max(iminr, np.where(x>=xminr)[0][0])
        
        print (f"{x.size}, Left {iminl},{imaxl}, Right {iminr},{imaxr}")
        

    ## DATA SELECTION
    
    # l & r symbolize the "left" and "right" sides of the focus curve
    
    xl = x[iminl:imaxl]
    xr = x[iminr:imaxr]
     
    yl = y[iminl:imaxl]
    yr = y[iminr:imaxr]
        
    xc = x[imaxl:iminr]
    yc = y[imaxl:iminr]
    
    # Eliminate potential NaNs
    sel = np.where(np.isfinite(yl))
    xl = xl[sel]
    yl = yl[sel]
    sel = np.where(np.isfinite(yr))
    xr = xr[sel]
    yr = yr[sel]
    
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
        plt.ylabel("Spot-size RMS",fontsize=14)
        plt.ylim(-10,60)
        if method == 'include':
            plt.title(f"Spot size RMS -- Incl {kept} samples on both sides", size=14)
        elif method == 'exclude':
            plt.title(f"\n Spot size RMS -- Excl 2 x {avoidance} microns around {estimate}", size=14)
        if method == 'init':
            plt.title(f"Spot size RMS -- Incl {kept} samples on both sides, but exclude the leftmost {init_excl}", size=14)
            if xminr is not None and xminr > 0:
                plt.title(f"Spot size RMS -- Incl {kept} samples on both sides, but exclude the leftmost {init_excl} & start right fit at {xminr}", size=14)                
        if method == '2init':
            plt.title(f"Spot size RMS -- Excl 2 x {avoidance} microns around {estimate} and the leftmost {init_excl}", size=14)
            if xminr is not None and xminr >= 0:
                plt.title(f"Spot size RMS -- Excl 2 x {avoidance} microns around {estimate} and the leftmost {init_excl} & start right fit at {xminr}", size=14)
        #    #if saveplot: plt.savefig(pngDir+srowcol+f'{str(n).zfill(2)}_rmsSpotSize_Fit_n_intersection_excluding_{avoidance}_umAround{estimate}_widen{widen}_zoom.png')

    return xfocus


## Analysis theoretical PSFs (read back from pickle files, see cslPsfs.py line ~ 1000)
## Toleranced PSFs --> fields = 0,4,8,12,16


################################################################################
## MODULE 5 : RANDOM SEEDS --> MONTE CARLO ANALYSIS
################################################################################

dataDir = "/Users/pierre/plato/pr/simout/egse/seeds/"
pngDir = "/Users/pierre/plato/pr/pngs/egse/seeds/"

dataDir = "/Users/pierre/plato/pr/simout/egse/tol/"
pngDir = "/Users/pierre/plato/pr/pngs/egse/tol/"

row,column,ccdCode,oangle = 4000, 500,2, 4    #  4 degrees
row,column,ccdCode,oangle = 3000,1000,2, 8   #  8 degrees
row,column,ccdCode,oangle = 2500,2000,2,12    # 12 degrees

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

sigma = 4

filtering  = 'sum'
hfiltering = {'open':"open", 'sum':'sum', False:"unfiltered"}

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
      censigma,boxsigma = psfBox(imorig,method='sigma',sigma=4,cosmicRemoval=filtering,kernel=None,verbose=verbose)

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
      #cornersOut = [int(boxcen[0]-(0.5+widen)*xywidth),int(boxcen[1]-(0.5+widen)*xywidth),int((1.+2*widen)*xywidth),int((1.+2*widen)*xywidth)]
      
      # IMPOSE A MINIMUM SIZE TO THE CROPPED IMAGE
      if int((1.+2*widen)*xywidth) >= 8:
          cornersOut = [int(boxcen[0]-(0.5+widen)*xywidth),int(boxcen[1]-(0.5+widen)*xywidth),int((1.+2*widen)*xywidth),int((1.+2*widen)*xywidth)]
      else:
          cornersOut = [int(boxcen[0])-4,int(boxcen[1])-4,8,8]

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
kept = 80

# 2nd iteration
avoidance = 300

for n in range(numExposures):

    # FOCUS SWEEP ANALYSIS
    x = availableFocusPositions
    y = rmssFlux[:,n]

    method = 'include'
    zfocus[n,0] = fitIntersection(x,y,method=method,parameters=[kept],doPlot=False)
    
    roundedTo50 = int(np.round(zfocus[n,0]/50.)*50)

    method = 'exclude'
    zfocus[n,1] = fitIntersection(x,y,method=method,parameters=[roundedTo50,avoidance],doPlot=False)





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
plt.title(f"Pixel {srowcol[5:-1]} ({oangle} deg from OA)\nToleranced PSFs",size=14)
plt.grid(linewidth=0.5,alpha=0.3)
plt.legend()
plt.savefig(pngDir+f"egse_randomSeeds_{srowcol}rms_.png")
plt.ylim(0,120)
plt.savefig(pngDir+f"egse_randomSeeds_{srowcol}rms_n_CroppedImageSize.png")


focus1st = f"{int(np.round(np.mean(zfocus[:,0]))):4d}$\,\pm\,${np.std(zfocus[:,0]):4.1f}"
focus2nd = f"{int(np.round(np.mean(zfocus[:,1]))):4d}$\,\pm\,${np.std(zfocus[:,1]):4.1f}"

#codeV = [5416,5416] # 4 degree, 2nd iteration

# Focus Solution Nominal
plt.figure("zfocus")
#plt.plot(availableFocusPositions,imsize,c=gray,ls='-',marker='.',alpha=0.25,label="Cropped image size")
plt.plot(np.arange(numExposures),zfocus[:,0],c='k',ls='--',marker='.',label=f"1st iteration {focus1st}",alpha=0.5)
plt.plot([0,numExposures],[np.mean(zfocus[:,0]),np.mean(zfocus[:,0])],c=gray,ls='-',alpha=0.25)

codeV = [5339,5339] #  8 degree, 1st iteration
codeV = [5237,5237] # 12 degree, 1st iteration
plt.plot([0,numExposures],codeV,c=orange,ls='--',alpha=0.5,label=f"Code V {codeV[0]} (1st it)")

plt.plot(np.arange(numExposures),zfocus[:,1],c='k',ls='-',marker='.',label=f"2nd iteration {focus2nd}")
plt.plot([0,numExposures],[np.mean(zfocus[:,1]),np.mean(zfocus[:,1])],c=gray,ls='-',alpha=0.25)

codeV = [5343,5343] #  8 degree, 2nd iteration
codeV = [5240,5240] # 12 degree, 2nd iteration
plt.plot([0,numExposures],codeV,c=orange,ls='-',alpha=0.5,label=f"Code V {codeV[0]} (2nd it)")

plt.xlabel('Exposure #',size=14)
plt.ylabel('Focus [$\mu$m from L6]',size=14)
plt.title(f"Pixel {srowcol[5:-1]} ({oangle} deg from OA)\nAvoidance $\pm\ ${avoidance}$\mu$m - Alt Filtering",size=14)
plt.grid(linewidth=0.5,alpha=0.3)
plt.legend()
plt.ylim(5385,5430)
plt.savefig(pngDir+f"egse_randomSeeds_{srowcol}zfocus_{oangle}_excluding{avoidance}um_sumFiltered.png")


focus1st = f"{int(np.round(np.mean(zfocus[:,0]))):4d}$\,\pm\,${np.std(zfocus[:,0]):4.1f}"
focus2nd = f"{int(np.round(np.mean(zfocus[:,1]))):4d}$\,\pm\,${np.std(zfocus[:,1]):4.1f}"

plt.figure("zfocustol",figsize=(10,8))
#plt.plot(availableFocusPositions,imsize,c=gray,ls='-',marker='.',alpha=0.25,label="Cropped image size")
plt.plot(np.arange(numExposures),zfocus[:,0],c='k',ls='--',marker='.',label=f"1st iteration {focus1st}",alpha=0.5)
plt.plot([0,numExposures],[np.mean(zfocus[:,0]),np.mean(zfocus[:,0])],c=gray,ls='--',alpha=0.25)

plt.plot(np.arange(numExposures),zfocus[:,1],c='k',ls='-',marker='.',label=f"2nd iteration {focus2nd}")
plt.plot([0,numExposures],[np.mean(zfocus[:,1]),np.mean(zfocus[:,1])],c=gray,ls='-',alpha=0.25)


codeV = [5129,5129] # TOL 4 degree, 1st iteration
codeV = [5062,5062] # TOL 8 degree, both iterations
#codeV = [4976,4976] # TOL 12 degree, both iterations
plt.plot([0,numExposures],codeV,c=orange,ls='--',alpha=0.5,label=f"Code V {codeV[0]} (1st it)")

#codeV = [5130,5130] # TOL 4 degree, 2nd iteration
plt.plot([0,numExposures],codeV,c=orange,ls='-',alpha=0.5,label=f"Code V {codeV[0]} (2nd it)")

plt.xlabel('Exposure #',size=14)
plt.ylabel('Focus [$\mu$m from L6]',size=14)
plt.title(f"Pixel {srowcol[5:-1]} ({oangle} deg from OA)\nAvoidance $\pm\ ${avoidance}$\mu$m",size=14)
plt.grid(linewidth=0.5,alpha=0.3)
plt.legend()
plt.savefig(pngDir+f"egse_randomSeeds_{srowcol}zfocus_{oangle}_excluding{avoidance}um_tolPSFs_minBoxSize_8.png")



################################################################################
## MODULE 7 : EFFECT OF FITTING DOMAIN
################################################################################

### A. Reduce the knowledge on the position of L6 : increase starting location of the left fit
### B. Reduce the knowledge on the position of the minimum (non-nominal PSFs minima between 4800 and 6200 microns) + exclusion zone

## Start by running module 6, gathering the RMS info, then analyse here

# highest nb of samples excluded on the left edge in method 'init' and '2init'
maxInit = 60
#zfocus     = np.zeros([numExposures,2]) # Best focus from line-fit intersection. method -- 0:include; 1:exclude
zfocus  = np.zeros([numExposures,2,maxInit]) # Best focus from line-fit intersection. method -- 0:include; 1:exclude

# 1st iteration
kept = 80

# 2nd iteration
avoidance = 500

xminr = 6500

doPlot,savePlot = 0,0

# FOCUS SWEEP ANALYSIS

#for n in [0]:
for n in range(numExposures):

  for init_excl in range(maxInit):

    x = availableFocusPositions
    y = rmssFlux[:,n]

    method = 'init'
    #xminr = 6500
    xminr = None
    zfocus[n,0,init_excl] = fitIntersection(x,y,method=method,parameters=[kept,init_excl,xminr],doPlot=doPlot)
    #method = 'include'
    #zfocus[n,0,init_excl] = fitIntersection(x,y,method=method,parameters=[kept],doPlot=doPlot)

    if doPlot:
          if savePlot:
              plt.savefig(pngDir+f"egse_randomSeeds_fitInter_{srowcol}zfocus_{oangle}_n{str(n).zfill(2)}_kept_{kept}_xminr_{xminr}_init_{str(init_excl).zfill(2)}.png")
              #distracting matplotlib so it actually saves the image
              time.sleep(0.5)
          plt.close()


    roundedTo50 = int(np.round(zfocus[n,0,init_excl]/50.)*50)

    xminr = None
    
    method = '2init'
    zfocus[n,1,init_excl] = fitIntersection(x,y,method=method,parameters=[roundedTo50,avoidance,init_excl,xminr],doPlot=doPlot)

    if doPlot:
          if savePlot:
              plt.savefig(pngDir+f"egse_randomSeeds_fitInter_{srowcol}zfocus_{oangle}_n{str(n).zfill(2)}_excl_{avoidance}_around_{roundedTo50}_xminr_{xminr}_init_{str(init_excl).zfill(2)}.png")
              #distracting matplotlib so it actually saves the image
              time.sleep(0.5)
          plt.close()


print(zfocus.shape)
print(np.any(np.isnan(zfocus)))

zmean = np.mean(zfocus,axis=0)
zstd  = np.std(zfocus,axis=0)

plt.figure("Init",figsize=(10,8))
plt.errorbar(availableFocusPositions[:maxInit],zmean[0,:],yerr=zstd[0,:],c=gray,ls='--',marker='.',ms=10,label=f"Focus 1st iteration",capsize=5)
plt.errorbar(availableFocusPositions[:maxInit],zmean[1,:],yerr=zstd[1,:],c='k',ls='-',marker='.',lw=2,ms=10,label=f"Focus 2nd iteration",capsize=5,elinewidth=2)

codeV = [5415,5415] # NOM 4 degree
codeV = [5339,5339] # NOM 8 degree
codeV = [5240,5240] # NOM 12 degree
codeV = [5130,5130] # TOL 4 degree
codeV = [5062,5062] # TOL 8 degree
codeV = [4976,4976] # TOL 12 degree
xx    = [availableFocusPositions[0]-50,availableFocusPositions[maxInit]]
plt.plot(xx,codeV,c='r',ls='--',alpha=1,lw=3,label=f"Code V {codeV[0]}")

plt.ylabel('Focus estimate [$\mu m$]',size=14)
plt.xlabel('Start of the fit [$\mu m$ from L6]',size=14)
plt.title(f"Pixel {srowcol[5:-1]} ({oangle} deg from OA)\n Focus estimate vs starting position of the fit",size=14)
plt.grid(linewidth=0.5,alpha=0.3)
plt.legend()

plt.savefig(pngDir+f"egse_randomSeeds_{srowcol}rms_{oangle}_focusVsStartOfFit_xminr_None_{hfiltering[filtering]}.png")
# NOM 4
plt.xlim(2500,4050)
plt.ylim(5390,5440)
# NOM 8
plt.xlim(2500,4050)
plt.ylim(5390,5440)
# TOL 4
plt.xlim(2500,4150)
plt.ylim(5080,5180)
# TOL 8
plt.xlim(2000,3900)
plt.ylim(5030,5080)
# TOL 12
plt.xlim(2000,3250)
plt.ylim(4930,5030)

plt.savefig(pngDir+f"egse_randomSeeds_{srowcol}rms_{oangle}_focusVsStartOfFit_tolPSFS.png")



################################################################################
## MODULE 8 : HARTMANN -- SINGLE IMAGE & parameter fine tuning
################################################################################

dataDir = "/Users/pierre/plato/pr/simout/egse/hartmann45/"
pngDir = "/Users/pierre/plato/pr/pngs/egse/hartmann45/"

row,column,ccdCode = 4000, 500,2  #  4 degrees
row,column,ccdCode = 3000,1000,2  #  8 degrees
row,column,ccdCode = 2500,2000,2  # 12 degrees
row,column,ccdCode = 1000,3000,2  # 18 degrees

allfiles = fs.fileSelect([f"{str(row).zfill(4)}",f"{str(column).zfill(4)}","hdf5"], location=dataDir, listOrder=1)

numExposures = 25

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
widen = 0.25

doPlot,savePlot,verbose=0,0,0

sigma = 5

filtering  = False
hfiltering = {'open':"open", 'sum':'sum', False:"unfiltered"}

nspots = 8

ellipses = np.zeros([len(availableFocusPositions),numExposures,5])
nspotskept = np.zeros([len(availableFocusPositions),numExposures])

#for f,focusPosition in enumerate([5140]):
for f,focusPosition in enumerate(availableFocusPositions):
    if doPlot: plt.close()
    runName = srowcol+f"{focusPosition}"
    
    print()
    print(f"{runName}")
    print()
    
    sh5 = h5py.File(dataDir+runName+".hdf5",'r')
    
    numExp = h5get(sh5,["ObservingParameters","numExposures"])
    
    #for n in [5]:
    for n in range(numExp):
        
      print(f"STARTING n == {n}")
    
      imorig = np.array(getSim(sh5,n=n))
    
      censigma,boxsigma = psfBox(imorig,method='sigma',sigma=sigma,cosmicRemoval=filtering,kernel=None,verbose=1)
      sigmaxs.append(censigma[0])
      sigmays.append(censigma[1])

      boxsigmas.append(boxsigma)

      image,background = backgroundSubtraction(imorig,method='edge',width=5,verbose=verbose)
      #image = backgroundSubtraction(imorig,method='sep')#
    
      pixmaxs.append(np.max(image))
    
      noise = np.std(background)
      significant = np.zeros_like(image)
      significant[np.where(image>sigma*noise)] = 1
          
      imCenx,imCeny = image.shape[0]//2,image.shape[1]//2
      imCenxs.append(imCenx)
      imCenys.append(imCeny)

      kernel = np.zeros([3,3])
      kernel[1,:] = 1
      kernel[:,1] = 1
      sigcleaned = morphology.binary_opening(significant,selem=kernel)
      
      """
      ## PSF PLOT    
      fig, ax = plt.subplots(figsize=(8,8))
      mn, std, mx = np.mean(image), np.std(image),np.max(image)
      im = ax.imshow(image, interpolation='nearest', cmap='gray', vmin=mn-std, vmax=mx, origin='lower')
      plt.title(runName)
      plt.savefig(pngDir+runName+f'_h45_{str(n).zfill(2)}.png')
      ## END PSF PLOT    
      """

      #labels = morphology.label(sigcleaned, background=0)
      #nlabels = np.max(labels)

      labels = morphology.label(significant, background=0)
      nlabelsorig = nlabels = np.max(labels)
      
      labelsclean = labels.copy()
      ilabels = [i for i in range(1,nlabels+1)]
      slabels = [len(np.where(labels==n)[0]) for n in ilabels]

      c = 0
      while(nlabels > nspots):
          # The spots to keep are most probably in the center --> in the event of equal sizes, kick out "from the outside"
          if c%2:
              # first occurence of the smallest 'blob' size
              kicklabeli = np.argmin(slabels)
          else:
              # last occurence of the smallest 'blob' size
              kicklabeli = len(slabels) - np.argmin(slabels[::-1]) - 1
          kicklabel = ilabels.pop(kicklabeli)
          kickedsize = slabels.pop(kicklabeli)
          print (f"c {c}:  Removing cluster with label {kicklabel} and size {kickedsize}")
          labelsclean[np.where(labelsclean==kicklabel)] = 0
          nlabels = len(ilabels)
          c+=1

      # CLASSICAL BOX - SIGMA
      #xmin,xmax,ymin,ymax = boxsigma
      #xmin,xmax = np.min(np.where(significant)[0]),np.max(np.where(significant)[0])
      #ymin,ymax = np.min(np.where(significant)[1]),np.max(np.where(significant)[1])
      # BASED ON CLEANED LABELS : I.E. THE RIGHT NB FOR THE HARTMANN MASK
      xmin,xmax = np.min(np.where(labelsclean)[0]),np.max(np.where(labelsclean)[0])
      ymin,ymax = np.min(np.where(labelsclean)[1]),np.max(np.where(labelsclean)[1])

      boxcen = [(xmin+xmax)/2.,(ymin+ymax)/2.]

      xwidth = xmax-xmin
      ywidth = ymax-ymin
    
      xywidth = max(xwidth,ywidth)
      cornersIn = [xmin,ymin,xwidth,ywidth]
      cornersOut = [int(xmin-widen*xywidth),int(ymin-widen*xywidth),int((1.+2*widen)*xywidth),int((1.+2*widen)*xywidth)]

      cornerssigma = [cornersIn,cornersOut]
      
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
          if savePlot:
              plt.savefig(pngDir+runName+f'_h45_{str(n).zfill(2)}_{hfiltering[filtering]}_cropped_widen025_labelsclean.png')
    
      ## PSF RMS  
      ## [0] --> c.o.g. cropped image
      ## [1] --> c.o.g. of significant pixels
      smearedImage = croppedImage.copy()
      colintegral = np.sum(croppedImage,axis=0) * 90/4.e6
      colint2d = np.stack([colintegral for i in range(croppedImage.shape[0])])
      smearedImage += colint2d
      
      #rmss.append([psff.psfRms(croppedImage,center=fluxcen),psff.psfRms(croppedImage,center=censigmacropped),psff.psfRms(smearedImage,center=fluxcen)])
      rmss.append([psff.psfRms(croppedImage,center=fluxcen),psff.psfRms(croppedImage,center=censigmacropped),psff.psfRms(image,center=fluxcen),psff.psfRms(image,center=censigmacropped),psff.psfRms(smearedImage,center=fluxcen)])
      
      # COLLECT ALL LABELLED POINTS INTO A DATASET & FIT AN ELLIPSE TO THAT SET OF COORDINATES
      sel = np.where(labelsclean != 0)
      ellipsin = np.vstack(sel).T
      # Swap x & y : NO
      #ellipsin = np.roll(ellipsin,1,1)
      
      try:
          ellipspars,ellipse=fitEllipse(ellipsin)
          ellipscoords = ellipse.get_verts()
      except:
          ellipspars   = [np.nan for i in range(5)]
          ellipscoords = [np.nan for i in range(5)]
      
      ellipses[f,n,:] = ellipspars
      nspotskept[f,n] = nlabels
      
      if doPlot:
          if savePlot:
              #distracting matplotlib so it actually saves the image
              time.sleep(0.5)
          plt.close()
          
          # For overplotting on the image, the x & y are inverted...
          patchellipse = matplotlib.patches.Ellipse((ellipspars[1],ellipspars[0]), width=2*ellipspars[2],height=2*ellipspars[3],angle=ellipspars[4], color="r", fill=False,lw=2,ls=':',alpha=0.5)

          fig2, [[ax00, ax01], [ax02,ax03]] = plt.subplots(nrows=2, ncols=2, figsize=(12, 12), sharex=True, sharey=True)
          
          ax00.imshow(image, cmap=plt.cm.gray,interpolation='nearest',origin='lower')
          #ax00.axis('off')
          ax00.set_title(f'Image\n{focusPosition} - {n}', fontsize=14)
          
          ax01.imshow(significant, cmap=plt.cm.gray,interpolation='nearest',origin='lower')
          #ax01.axis('off')
          ax01.set_title(f'> {sigma} sigmas\nx : image center   * : centroid significant', fontsize=14)
          
          ax02.imshow(sigcleaned, cmap=plt.cm.gray,interpolation='nearest',origin='lower')
          #ax02.axis('off')
          ax02.set_title(f'> {sigma} sigmas + Binary Opening', fontsize=14)
                    
          ax03.imshow(labelsclean, cmap=plt.cm.nipy_spectral,interpolation='nearest',origin='lower',clim=(0.,np.max(labelsclean)))
          #ax03.axis('off')
          ax03.set_title(f'Labels: {nlabelsorig} -> {nlabels}\nEllipse [a,b]   [{ellipspars[2]:5.1f},{ellipspars[3]:5.1f}]', fontsize=14)
          
          plt.gca().add_patch(patchellipse)
          #plt.plot(ellipscoords[:,1],ellipscoords[:,0],c='w',ls='--',alpha=0.5,label="coords")
          plt.plot([ellipspars[1]],[ellipspars[0]],ms=10,marker='+',c='w')

          # Sigma
          ax01.scatter(sigmays[-1],sigmaxs[-1],marker="*",c='g',s=300)
          ax02.scatter(sigmays[-1],sigmaxs[-1],marker="*",c='g',s=300)
          
          # Image Center
          ax01.scatter(imCeny,imCenx,marker="x",c='r',s=100)
          ax02.scatter(imCeny,imCenx,marker="x",c='r',s=100)

          imshowrect(cornerssigma, figure=ax01,color='g',ls='--')
          imshowrect(cornerssigma, figure=ax02,color='g',ls='--')
          #imshowrect([cornersOut], figure=ax01,color='g',ls='--')
          #imshowrect([cornersOut], figure=ax02,color='g',ls='--')
          
          if savePlot:
              plt.savefig(pngDir+runName+f'_h45_{str(n).zfill(2)}_{hfiltering[filtering]}_sig5_4panels_{nspots}largestSpotsEllipse.png')


fom = np.zeros([len(availableFocusPositions), numExposures, 3])
fom[:,:,0] = ella = ellipses[:,:,2] # a
fom[:,:,1] = ellb = ellipses[:,:,3] # b
fom[:,:,2] = np.sqrt(ella*ella + ellb*ellb)


fommean  = np.nanmean(fom,axis=1)
fomstd   = np.nanstd(fom,axis=1)

plt.figure("fom")
#plt.plot(availableFocusPositions,imsize,c=gray,ls='-',marker='.',alpha=0.25,label="Cropped image size")
#plt.plot(availableFocusPositions,rmsmean,c='k',ls='-',marker='.',label=f"RMS")
plt.errorbar(availableFocusPositions,fommean[:,0],yerr=fomstd[:,0],c='r',ls='-',marker='.',label='a')
plt.errorbar(availableFocusPositions,fommean[:,1],yerr=fomstd[:,1],c='g',ls='-',marker='.',label='b')
plt.errorbar(availableFocusPositions,fommean[:,2],yerr=fomstd[:,2],c='k',ls=':',marker='.',label='$\sqrt{a^2 + b^2}$')
plt.xlabel('Focus [$\mu m$]',size=14)
plt.ylabel('Ellipse dimension',size=14)
plt.title(f"Pixel {srowcol[5:-1]} ({oangle} deg from OA)\nHartmann 45  Ellipse dimension",size=14)
plt.grid(linewidth=0.5,alpha=0.3)
plt.legend()
plt.ylim(0,50)
plt.savefig(pngDir+f"egse_randomSeeds_{srowcol}ellipsePars.png")



################################################################################
## MODULE 9 : HARTMANN - FITTING ELLIPSE PARAMETERS
################################################################################

# 1st dim = exposures, second = method, 3rd = figure of merit (a, b, sqrt(a^2+b^2))
zfocus     = np.zeros([numExposures,2,3]) # Best focus from line-fit intersection. method -- 0:include; 1:exclude

# 1st iteration
kept = 35

# 2nd iteration
avoidance = 300

hfom={0:'ellipse_a',1:'ellipse_b',2:'ellipse_ab'}
ifom = 2

for n in range(numExposures):

    # FOCUS SWEEP ANALYSIS
    x = availableFocusPositions
    y = fom[:,n,ifom]

    method = 'include'
    zfocus[n,0,ifom] = fitIntersection(x,y,method=method,parameters=[kept],doPlot=False)
    
    roundedTo50 = int(np.round(zfocus[n,0,ifom]/50.)*50)

    method = 'exclude'
    zfocus[n,1,ifom] = fitIntersection(x,y,method=method,parameters=[roundedTo50,avoidance],doPlot=False)

robMean1 = int(np.round(robustf(zfocus[:,0,ifom],np.nanmean,sigma=2)))
robMean2 = int(np.round(robustf(zfocus[:,1,ifom],np.nanmean,sigma=2)))
focus1st = f"{robMean1:4d}$\,\pm\,${robustf(zfocus[:,0,ifom],np.nanstd,sigma=2):4.1f}" 
focus2nd = f"{robMean2:4d}$\,\pm\,${robustf(zfocus[:,1,ifom],np.nanstd,sigma=2):4.1f}" 

print (focus1st)
print (focus2nd)

#focus1st = f"{int(np.round(np.mean(zfocus[:,0,ifom]))):4d}$\,\pm\,${np.std(zfocus[:,0,ifom]):4.1f}"
#focus2nd = f"{int(np.round(np.mean(zfocus[:,1,ifom]))):4d}$\,\pm\,${np.std(zfocus[:,1,ifom]):4.1f}"

#codeV = [5416,5416] # 4 degree, 2nd iteration

# Focus Solution Nominal
plt.figure("zfocus"+str(ifom))
#plt.plot(availableFocusPositions,imsize,c=gray,ls='-',marker='.',alpha=0.25,label="Cropped image size")
plt.plot(np.arange(numExposures),zfocus[:,0,ifom],c='k',ls='--',marker='.',label=f"1st iteration {focus1st}",alpha=0.5)
plt.plot([0,numExposures],[robMean1,robMean1],c=gray,ls='-',alpha=0.25)

#codeV = [5339,5339] #  8 degree, 1st iteration
#codeV = [5237,5237] # 12 degree, 1st iteration
#plt.plot([0,numExposures],codeV,c=orange,ls='--',alpha=0.5,label=f"Code V {codeV[0]} (1st it)")

plt.plot(np.arange(numExposures),zfocus[:,1,ifom],c='k',ls='-',marker='.',label=f"2nd iteration {focus2nd}")
plt.plot([0,numExposures],[robMean2,robMean2],c=gray,ls='-',alpha=0.25)

#codeV = [5343,5343] #  8 degree, 2nd iteration
#codeV = [5240,5240] # 12 degree, 2nd iteration
#plt.plot([0,numExposures],codeV,c=orange,ls='-',alpha=0.5,label=f"Code V {codeV[0]} (2nd it)")

plt.xlabel('Exposure #',size=14)
plt.ylabel('Focus [$\mu$m from L6]',size=14)
plt.title(f"Pixel {srowcol[5:-1]} ({oangle} deg from OA)\nAvoidance $\pm\ ${avoidance}$\mu$m " + hfom[ifom],size=14)
plt.grid(linewidth=0.5,alpha=0.3)
plt.legend()
plt.savefig(pngDir+f"egse_hartmann45_{srowcol}zfocus_{oangle}_excluding{avoidance}um_"+hfom[ifom]+".png")
plt.ylim(4930,4940)






### A. Reduce the knowledge on the position of L6 : increase starting location of the left fit
### B. Reduce the knowledge on the position of the minimum (non-nominal PSFs minima between 4800 and 6200 microns) + exclusion zone

## Start by running module 6, gathering the RMS info, then analyse here

# highest nb of samples excluded on the left edge in method 'init' and '2init'
maxInit = 25
#zfocus     = np.zeros([numExposures,2]) # Best focus from line-fit intersection. method -- 0:include; 1:exclude
zfocus  = np.zeros([numExposures,2,maxInit,3]) # Best focus from line-fit intersection. method -- 0:include; 1:exclude

# 1st iteration
kept = 35

# 2nd iteration
avoidance = 300

xminr = None

doPlot,savePlot = 0,0

# FOCUS SWEEP ANALYSIS

hfom={0:'ellipse_a',1:'ellipse_b',2:'ellipse_ab'}

#for n in [0]:
for n in range(numExposures):

  for init_excl in range(maxInit):

    # y: 0=a 1=b 2=sqrt(a^2+b^2)
    x = availableFocusPositions
    #y = fom[:,n,1]

    
    for ifom in range(3):
      
      method = 'init'
      #xminr = 6500
      xminr = None

      y = fom[:,n,ifom]
      
      zfocus[n,0,init_excl,ifom] = fitIntersection(x,y,method=method,parameters=[kept,init_excl,xminr],doPlot=doPlot)
      if doPlot:
          if savePlot:
              plt.savefig(pngDir+f"egse_hartmann45_fitInter_{srowcol}zfocus_{oangle}_n{str(n).zfill(2)}_kept_{kept}_xminr_{xminr}_init_{str(init_excl).zfill(2)}.png")
              #distracting matplotlib so it actually saves the image
              time.sleep(0.5)
          plt.close()


      roundedTo50 = int(np.round(zfocus[n,0,init_excl,ifom]/50.)*50)

      xminr = None
    
      method = '2init'
      zfocus[n,1,init_excl,ifom] = fitIntersection(x,y,method=method,parameters=[roundedTo50,avoidance,init_excl,xminr],doPlot=doPlot)

      if doPlot:
          if savePlot:
              plt.savefig(pngDir+f"egse_randomSeeds_fitInter_{srowcol}zfocus_{oangle}_n{str(n).zfill(2)}_excl_{avoidance}_around_{roundedTo50}_xminr_{xminr}_init_{str(init_excl).zfill(2)}.png")
              #distracting matplotlib so it actually saves the image
              time.sleep(0.5)
          plt.close()


zmean = np.mean(zfocus,axis=0)
zstd  = np.std(zfocus,axis=0)

hfom={0:'ellipse_a',1:'ellipse_b',2:'ellipse_ab'}
ifom = 2

plt.figure("Init",figsize=(10,8))
plt.errorbar(availableFocusPositions[:maxInit],zmean[0,:,ifom],yerr=zstd[0,:,ifom],c=gray,ls='--',marker='.',ms=10,label=f"Focus 1st iteration",capsize=5)
plt.errorbar(availableFocusPositions[:maxInit],zmean[1,:,ifom],yerr=zstd[1,:,ifom],c='k',ls='-',marker='.',lw=2,ms=10,label=f"Focus 2nd iteration",capsize=5,elinewidth=2)

#codeV = [5240,5240] # HARTMANN 12 degree
#xx    = [availableFocusPositions[0]-50,availableFocusPositions[maxInit]]
#plt.plot(xx,codeV,c='r',ls='--',alpha=1,lw=3,label=f"Code V {codeV[0]}")

plt.ylabel('Hartmann solution [$\mu m$]',size=14)
plt.xlabel('Start of the fit [$\mu m$ from L6]',size=14)
plt.title(f"Pixel {srowcol[5:-1]} ({oangle} deg from OA)\n Focus estimate vs starting position of the fit",size=14)
plt.grid(linewidth=0.5,alpha=0.3)
plt.legend()

plt.savefig(pngDir+f"egse_randomSeeds_{srowcol}Hartmann45_{oangle}_focusVsStartOfFit_xminr_None.png")






"""
n = 0
ellcenx,ellceny,ella,ellb,ellangle = ellipses[:,n,0],ellipses[:,n,1],ellipses[:,n,2],ellipses[:,n,3],ellipses[:,n,4]
sel = np.where(nspotskept[:,n] >= 1)
plt.figure('Ellipse')
plt.plot(availableFocusPositions[sel], np.sqrt(ella[sel]**2. + ellb[sel]**2.),'ko', label='$\sqrt{a^2 + b^2}$')
plt.plot(availableFocusPositions[sel], ella[sel],'r.', label='a')
plt.plot(availableFocusPositions[sel], ellb[sel],'g.', label='b')
plt.ylim(0,60)
plt.grid(alpha=0.25)
plt.legend()


      edge = arrayEdge1d(image,width=5)
      sigth = np.std(edge)
      sig3 = np.zeros_like(image)
      sig3[np.where(image > 3 * sigth)] = 1
      sig4 = np.zeros_like(image)
      sig4[np.where(image > 4 * sigth)] = 1

      hist, bins_center = exposure.histogram(image)
        
      othreshold = threshold_otsu(image)
      otsignif = np.zeros_like(image)
      otsignif[image>=4*othreshold] = 1
    
      kernel = np.zeros([3,3])
      kernel[1,:] = 1
      kernel[:,1] = 1
      otcleaned = morphology.binary_opening(otsignif,selem=kernel)
      
      sig3opened = morphology.binary_opening(sig3.copy(),selem=kernel)
 
      kernel1Size = 10
      kernel1 = np.ones([kernel1Size,kernel1Size])
      c2d = convolve2d(sig3,kernel1,mode='same')
      sig3cleaned = sig3.copy()
      sig3cleaned[np.where(c2d==1)] = 0
      c2d = convolve2d(sig4,kernel1,mode='same')
      sig4cleaned = sig4.copy()
      sig4cleaned[np.where(c2d==1)] = 0

      plt.figure(figsize=(16,16))
      plt.subplot(331)
      plt.imshow(image, cmap='gray', interpolation='nearest',origin='lower')
      plt.title("Original")
      plt.axis('off')
      plt.subplot(332)
      plt.imshow(otsignif, cmap='gray', interpolation='nearest',origin='lower')
      plt.title("Otsu Threshold")
      plt.axis('off')
      plt.subplot(333)
      plt.imshow(otcleaned, cmap='gray', interpolation='nearest',origin='lower')
      plt.title("Otsu + Binary Opening")
      plt.axis('off')
      plt.subplot(334)
      plt.imshow(sig4, cmap='gray', interpolation='nearest',origin='lower')
      plt.title("Sigma clipping (4 stddev)")
      plt.axis('off')
      plt.subplot(335)
      plt.imshow(sig4cleaned, cmap='gray', interpolation='nearest',origin='lower')
      plt.title("Sigma clipping (4 stddev) + sum filtering")
      plt.axis('off')

      plt.subplot(337)
      plt.imshow(sig3, cmap='gray', interpolation='nearest',origin='lower')
      plt.title("Sigma clipping (3 stddev)")
      plt.axis('off')
      plt.subplot(338)
      plt.imshow(sig3cleaned, cmap='gray', interpolation='nearest',origin='lower')
      plt.title("Sigma clipping (3 stddev) + sum filtering")
      plt.axis('off')
      
      labels = morphology.label(sig3opened, background=0)
      nlabels = np.max(labels)
      labels += 3
      labels[np.where(labels==3)] = 0
      
      plt.subplot(339)
      plt.imshow(labels, cmap=plt.cm.nipy_spectral,clim=(0.,nlabels), interpolation='nearest',origin='lower')
      plt.title(f"Sigma clipping (3 stddev) + Binary Opening + {nlabels} labels")
      plt.axis('off')
      
      plt.subplot(336)
      plt.plot(bins_center, hist, lw=2)
      plt.axvline(othreshold, color='k', ls='--',label="Otsu")
      plt.axvline(3 * sigth, color=orange, ls='--',label="3 $\sigma$")
      plt.axvline(4 * sigth, color='r', ls='--',label="4 $\sigma$")
      plt.legend()
      plt.grid(alpha=0.25)
      plt.tight_layout()
      plt.title("Flux histogram & thresholds")
      plt.savefig(pngDir+runName+f'_h45_{str(n).zfill(2)}_Otsu_vs_SigmaClipping_9panels.png')
      """


      """
      # Same figure, but with the axes in synch --> prefer this one for manual zooming
      # The histogram can't be introduced here (synch zoom destroys the figure then) ==> prefer the regular plt.figure above for automatic plotting 
      fig9, [[ax00, ax01, ax02], [ax10,ax11,ax12], [ax20,ax21,ax22]] = plt.subplots(nrows=3, ncols=3, figsize=(16, 16), sharex=True, sharey=True)
          
      ax00.imshow(image, cmap=plt.cm.gray,interpolation='nearest',origin='lower')
      ax00.set_title(f'Image\n{focusPosition} - {n}', fontsize=14)
          
      ax01.imshow(otsignif, cmap=plt.cm.gray,interpolation='nearest',origin='lower')
      ax01.set_title(f"Otsu Threshold", fontsize=14)

      ax02.imshow(otcleaned, cmap=plt.cm.gray,interpolation='nearest',origin='lower')
      ax02.set_title(f"Otsu + Binary Opening", fontsize=14)

      ax10.imshow(sig4, cmap=plt.cm.gray,interpolation='nearest',origin='lower')
      ax10.set_title(f"Sigma clipping (4 stddev)", fontsize=14)

      ax11.imshow(sig4cleaned, cmap=plt.cm.gray,interpolation='nearest',origin='lower')
      ax11.set_title(f"Sigma clipping (4 stddev) + sum filtering", fontsize=14)

      #ax12.plot(bins_center, hist, lw=2)
      #ax12.axvline(othreshold, color='k', ls='--',label="Otsu")
      #ax12.axvline(3 * sigth, color=orange, ls='--',label="3 $\sigma$")
      #ax12.axvline(4 * sigth, color='r', ls='--',label="4 $\sigma$")
      #ax12.legend()
      #ax12.grid(alpha=0.25)
      #ax12.tight_layout()
      #ax12.title("Flux histogram & thresholds")
      
      ax20.imshow(sig3, cmap=plt.cm.gray,interpolation='nearest',origin='lower')
      ax20.set_title(f"Sigma clipping (3 stddev)", fontsize=14)

      ax21.imshow(sig3cleaned, cmap=plt.cm.gray,interpolation='nearest',origin='lower')
      ax21.set_title(f"Sigma clipping (3 stddev) + sum filtering", fontsize=14)

      ax22.imshow(labels, cmap=plt.cm.nipy_spectral,interpolation='nearest',origin='lower')
      ax22.set_title(f"Sigma clipping (3 stddev) + Binary opening + {nlabels} labels", fontsize=14)
      """
    
"""
labels += 5
labels[np.where(labels==5)] = 0
#labels[np.where(labels==2)]=np.max(labels)+2

plt.figure(figsize=(12,16))
plt.subplot(221)
plt.imshow(image, cmap='gray', interpolation='nearest',origin='lower')
plt.title("Original")
plt.axis('off')
plt.subplot(222)
plt.imshow(significant, cmap='gray', interpolation='nearest',origin='lower')
plt.title("Significant")
plt.axis('off')
plt.subplot(223)
plt.imshow(sigcleaned, cmap='gray', interpolation='nearest',origin='lower')
plt.title("Binary Opening")
plt.axis('off')
plt.subplot(224)
plt.imshow(labels, cmap='nipy_spectral', interpolation='nearest',origin='lower',clim=(0.,np.max(labels)))
plt.title("Labels")
plt.axis('off')

labelsfinal = labelsorig.copy()
nlabels = nlabelsorig
ilabels = range(1,nlabels+1)
slabels = [len(np.where(labelsorig==n)[0]) for n in ilabels]

c = 0
while(nlabels > 7):
    if c%2:
        # first occurence of the smallest 'blob' size
        kicklabel = np.argmin(slabels) + 1
    else:
        # last occurence of the smallest 'blob' size
        kicklabel = len(slabels) - np.argmin(slabels[::-1])
    print(f"Nb or labels: {nlabels}; rejecting label {kicklabel}")
    labelsfinal[np.where(labelsfinal==kicklabel)] = 0
    slabels.pop(np.argmin(slabels))
    nlabels = len(slabels)
    c += 1

plt.figure()
plt.imshow(labelsfinal, interpolation='nearest',origin='lower')
"""



