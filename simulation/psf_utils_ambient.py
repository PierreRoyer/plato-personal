#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:46:23 2019

@author: pierre
"""
import referenceFrames as simref
import h5py
import numpy as np
from camtest.numeric import next_power_of_2, get_nearest_value
from scipy.interpolate.interpolate import interp2d
from h5 import h5get
from scipy.ndimage.interpolation import  rotate

def extractCslPsf(angleFromAxis, focusPosition, psfInputFile = "/STER/platoman/data/psfs/csl/cslPsfsOrig.hdf5",verbose=0):

    """
    Extract the CSL PSF for the given field angle and focus position, together with the attributes.

    :param angleFromAxis: Requested angular distance to the optical axis, expressed in degrees.
                            Should be an integer or a string (with two characters) with values in [0:2:20].

    :param focusPosition: Requested focus position [µm].  This is the distance from L6S2, ranging
                          (unequidistantly between 2520 and 7480 micron).

    :param angleFromX: Requested position angle of the pixel in the FOV, expressed in degrees.

    :param psfInputFile: Name of the HDF5 with all defocused PSF from CSL, as simulated with Code V.

    :return psf: Original CSL PSF for the given field angle and focus position.

    :param dz: Focus position [µm].  This is the distance from L6S2, ranging (unequidistantly between
               2520 and 7480 micron).

    :return pixelSizeRatio: Ratio of the pixel size in the CSL PSF and the pixel size of the Plato detectors.

    :return nd: Number of samples in the CSL PSF, in the x- and in the y-direction.
    """

    # Try to open the HDF5 with all defocused PSFs from CSL, as simulated with Code V


    try:

        csl = h5py.File(psfInputFile, 'r')

    except:

        print("Could not access the file with the orginal PSFs")
        print("Filename: {0}".format(psfInputFile))



    # Verify the requested angular distance
    # (name of the groups in the HDF5 file)

    possibleAnglesFromAxis = [str(i).zfill(2) for i in np.arange(0, 20, 2)]

    if verbose: print (f"Chosen angleFromAxis : {angleFromAxis} (deg)")

    if angleFromAxis not in possibleAnglesFromAxis:

        if angleFromAxis in np.arange(0, 20, 2):

            angleFromAxis = str(angleFromAxis).zfill(2)

        else:

            if verbose: print("Requested field angleFromAxis should be in {0} (deg; integer or string)".format(possibleAnglesFromAxis))

            angleFromAxis = str(get_nearest_value(np.float(angleFromAxis), np.arange(0, 20, 2))).zfill(2)

            if verbose: print (f"Nearest available angleFromAxis : {angleFromAxis} (deg)")

            return



    # Verify the requested focus position
    # (name of the sub-groups in the HDF5 file)

    possibleFocusPositions = np.sort(h5get(csl, [str(angleFromAxis).zfill(2), "dzs"], verbose = 0))

    if verbose: print (f"Chosen focusPosition : {focusPosition}")

    if focusPosition not in possibleFocusPositions:

        if verbose: print("Requested pair of (angleFromAxis, focusPosition) = ({0}, {1}) does not exist in HDF5 file with CSL PSFs".format(angleFromAxis, focusPosition))
        focusPosition = get_nearest_value(focusPosition, possibleFocusPositions)
        if verbose: print ("Nearest focus position found for this field is taken instead: {0}".format(focusPosition))

    # Extract CSL PSF (for the requested field angle and focus position)
    #   - dz: focus position
    #   - dxy: dimension of the PSF image (side-to-side) [µm]
    #   - span: dimension of the PSF image (side-to-side) [mm]
    #   - sampling: 'pixel' size for the PSF image [mm/sample]   # Sampling = Span / nb of samples recorded in the data [mm/sample]
    #   - nd: number of pixels in the PSF image in the x- and in the y-direction

    if verbose:
        print (f"Nb or ITEMS in PSF : {len(h5get(csl, [angleFromAxis + '/' + str(focusPosition)], verbose = 0))}")
    dz, dxy, nd, sampling, span, psf = h5get(csl, [angleFromAxis + '/' + str(focusPosition)], verbose = 0)
    csl.close()

    pixelSizeDetectors = 0.018                             # Pixel size for the Plato detectors (not for the CSL PSF!) [mm]
    pixelSizeRatio = sampling / pixelSizeDetectors         # Number of Plato detector pixels that fit in a 'pixel' (sample) in the CSL PSF

    # Return the following information:
    #   - psf: CSL PSF for the given field angle and focus position
    #   - dz: focus position
    #   - pixelSizeRatio: Ratio of the pixel size in the CSL PSF and the pixel size for the Plato detectors
    #   - nd: Number of pixels in the CSL PSF image in the x- and in the y-direction

    return psf, dz, pixelSizeRatio, nd





def getCslPsf(angleFromAxis, focusPosition, angleFromX, ccdCode, numSubPixels = None, numPixels = None, psfInputFile = "/STER/platoman/data/psfs/csl/cslPsfsOrig.hdf5", outputFile = None,verbose=False):

    """
    Extracts the CSL PSF for the given field angle and focus position from the given HDF5 file and
    turns it into an HDF5 file that can be handles by PlatoSim.  This means that we have to (linearly)
    interpolate the original CSL PSF to a rectangular grid in detector sub-pixels and normalise the PSF.

    :param angleFromAxis: Requested angular distance from the optical axis, expressed in degrees.
                            Should be an integer or a string with values in [0:2:20].

    :param focusPosition: Requested focus position [µm].  This is the distance from L6S2, ranging
                          (unequidistantly between 2520 and 7480 micron).

    :param angleFromX: Requested position angle of the pixel in the FOV, expressed in degrees.

    :param numSubPixels: Number of sub-pixels per pixel in both directions for the PSF that will be fed
                         into PlatoSim (i.e. after interpolation and cropping).  Default: next power of
                         two w.r.t. the original sampling, with a minimum of 8 sub-pixels per pixel

    :param numPixels: Number of pixels in both directions for the (square) PSF that will be fed into
                      PlatoSim (i.e. after interpolation).  Default: Next power of two w.r.t.
                      the original span of the PSF, with a minimum of 8 pixels.

    :param psfInputFile: Name of the HDF5 with all defocused PSF from CSL, as simulated with Code V.

    :param outputFile: Name of the HDF5 file in which to store the output PSF (i.e. after interpolation
                       and cropping).

    :return result: PSF for the given field angle and focus position, interpolated to a rectangular grid
                    in detector sub-pixels and normalised.
    """

    angleFromAxis = get_nearest_value(np.float(angleFromAxis), np.arange(0, 20, 2))

    ccdAngle = np.rad2deg(simref.CCD[str(ccdCode)]["angle"])     # Orientation angle of the CCD [degrees]

    # Extract the requested PSF from the HDF5 file that contains all PSFs from CSL
    # This must be turned into a format that PlatoSim can handle:
    #   - the required PSF must be interpolated to a grid specified in detector sub-pixels
    #     (rather than an arbitrary equidistant spatial grid)
    #   - the interpolated PSF must be stored in a designated HDF5 file

    cslPsf, dz, pixelSizeRatio, numDatapoints = extractCslPsf(angleFromAxis, focusPosition, psfInputFile, verbose=verbose)

    # Rotation over the position angle

    #cslPsfRot = rotate(cslPsf,  - ccdAngle
    # -ccdAngle is kept out
    # cos' ccdAngle should not be considered for anything else than the deviations
    # from a multiple of 90 degrees (misalignment at FPA levels) (+ global CAM rotational misalignment ?)
    cslPsfRot = rotate(cslPsf,  (180 - angleFromX))# - ccdAngle)
    numDatapoints = cslPsfRot.shape[0]

    # Number of sub-pixels per pixel in the PSF that will be handed to PlatoSim
    #    - either specified by the user or derived from the sampling of the CSL PSF
    #    - must be a power of two (use next power of two)
    #    - enforce at least 8 sub-pixels

    # sampling[mm/datapoint] = psf_span[mm] / numberOfPoints --> saved in the original hdf5 file
    # pixelSizeRatio = sampling [mm/datapoint] / pixelsize [mm/pixel] = [pixel/datapoint] --> derived in extractCslPsf

    # pixelSizeRatio = sampling / pixelSizeDetectors
    # sampling = span / nd
    # span = extension of the PSF as delivered by CSL [mm]
    # numSubPixels = 1/pixelSizeRatio = [nb of datapoints in original psf] * pixelSize [0.018 mm] / [original psf size in mm]
    #                                 = [nb of datapoints in original psf] / [nb of pixels spanned by the PSF]  = [nb of datapoints/ pixel]

    numSubPixels = (1.0 / pixelSizeRatio if (numSubPixels is None) else numSubPixels)
    numSubPixels = max(next_power_of_2(numSubPixels), 8)



    # Spatial grid for the original CSL PSF, expressed in detector sub-pixels
    #   - range(0, numDatapoints): pixel coordinates in the CSL PSF
    #   - multiply with pixelSizeRatio: now expressed in detector pixels
    #   - multiply with numSubPixels: now expressed in detector sub-pixels
    # Most likely non-integer values -> we have to make sure we have values for the integer sub-pixel positions
    # (that is why we will perform a linear interpolation to an intermediate grid first)

    initialSpatialGrid = np.arange(0, numDatapoints) * pixelSizeRatio * numSubPixels       # [detector sub-pixels] (non-integer)

    interpolatedGridSize = int(np.ceil(initialSpatialGrid[-1]))
    interpolatedSpatialGrid = np.arange(interpolatedGridSize)                              # [detector sub-pixels] (integer)

    interpolator = interp2d(x = initialSpatialGrid, y = initialSpatialGrid, z = cslPsfRot, kind = "linear")    # Source spatial grid (non-integer)
    interpolatedPsf = interpolator(interpolatedSpatialGrid, interpolatedSpatialGrid)                           # Target spatial grid (integer)

    # We only need to copy the part of the interpolated PSF that is non-zero

    nonzero = np.where(interpolatedPsf != 0)                            # Non-zero region in the interpolated PSF
    xMin, xMax = np.min(nonzero[0]), np.max(nonzero[0])                 # Min & max x-coordinate for the non-zero region [detector sub-pixels]
    yMin, yMax = np.min(nonzero[1]), np.max(nonzero[1])                 # Min & max y-coordinate for the non-zero region [detector sub-pixels]
    realSpanSubPixelsX = xMax - xMin + 1                                # Size of the non-zero region in the x-direction [detector sub-pixels]
    realSpanSubPixelsY = yMax - yMin + 1                                # Size of the non-zero region in the y-direction [detector sub-pixels]
    realSpanPixelsX = int(np.ceil(realSpanSubPixelsX / numSubPixels))   # Size of the non-zero region in the x-direction [detector pixels]
    realSpanPixelsY = int(np.ceil(realSpanSubPixelsY / numSubPixels))   # Size of the non-zero region in the y-direction [detector pixels]



    # Number of pixels in the PSF that will be handed to PlatoSim
    #   - either specified by the user or derived from the real (i.e. non-zero) span of the interpolated PSF
    #   - must be a power of two (use next power of two)
    #   - enforce at least 8 pixels

    numPixels = (max(realSpanPixelsX, realSpanPixelsY) if (numPixels is None) else max(numPixels, realSpanPixelsX, realSpanPixelsY))
    numPixels = max(next_power_of_2(numPixels), 8)


    # Size of the PSF that will be handed to PlatoSim [detector sub-pixels]

    size = numPixels * numSubPixels
    psf = np.zeros([size, size])#,dtype=np.float32)

    # Make sure the centre is still the centre

    endCenter = size // 2
    initShiftX = endCenter - (realSpanSubPixelsX // 2)
    initShiftY = endCenter - (realSpanSubPixelsY // 2)

    psf[initShiftX : initShiftX + realSpanSubPixelsX, initShiftY : initShiftY + realSpanSubPixelsY] = interpolatedPsf[xMin : xMax + 1, yMin : yMax + 1]

    # Normalisation

    psf /= np.sum(psf)

    if outputFile is not None:

        outfile = h5py.File(outputFile, 'w')
        group = outfile.create_group("T6000/ar00000")
        dataset = group.create_dataset("az0", data = psf)
        #group.attrs["angleFromAxis"] = np.int(angleFromAxis)
        dataset.attrs.create(name = "orientation", data = -ccdAngle)
        outfile.attrs.create(name = 'dz', data = dz)
        outfile.attrs.create(name = 'numSubPixels', data = numSubPixels)
        outfile.attrs.create(name = 'numPixels', data = numPixels)
        outfile.attrs.create(name = 'size', data = size)

        outfile.close()


    # Ratio of peak and total flux

    rebinnedShape = (numPixels, numPixels)
    shape = (rebinnedShape[0], psf.shape[0] // rebinnedShape[0], rebinnedShape[1], psf.shape[1] // rebinnedShape[1])
    rebinnedPsf = psf.reshape(shape).mean(-1).mean(1)
    rho = np.max(rebinnedPsf) / np.sum(rebinnedPsf)

    result = {}
    result["psf"] = psf
    result["angleFromAxis"] = int(angleFromAxis)
    result["dz"] = dz
    result["numSubPixels"] = numSubPixels
    result["numPixels"] = numPixels
    result["size"] = size
    result["rho"] = rho

    return result

def getAnglesFromOpticalAxis():

    """
    Return list with all possible angular distances [degrees].

    :return: List with all possible angular distances [degrees].
    """

    return [str(i).zfill(2) for i in np.arange(2, 20, 2)]


def getFocusPositions(angleFromAxis, psfInputFile = "/STER/platoman/data/psfs/csl/cslPsfs_v02.hdf5"):

    """
    Return list with all possible focus positions [µm] for the given angular distance to the optical axis.

    :param angleFromAxis: Requested angular distance from the to the optical axis, expressed in degrees.

    :param psfInputFile: Name of the HDF5 with all defocused PSF from CSL, as simulated with Code V.

    :return focusPositions: List with all focus positions [µm] for the given angular distance to the optical axis.
    """

    angleFromAxis = get_nearest_value(np.float(angleFromAxis), np.arange(0, 20, 2))

    # Try to open the HDF5 with all defocused PSFs from CSL, as simulated with Code V

    try:

        csl = h5py.File(psfInputFile, 'r')

    except:

        print("Could not access the file with the orginal PSFs")
        print("Filename: {0}".format(psfInputFile))

    # Verify the requested field angle
    # (name of the groups in the HDF5 file)

    # Verify the requested angular distance
    # (name of the groups in the HDF5 file)

    possibleAnglesFromAxis = [str(i).zfill(2) for i in np.arange(0, 20, 2)]

    if angleFromAxis not in possibleAnglesFromAxis:

        if angleFromAxis in np.arange(0, 20, 2):

            angleFromAxis = str(angleFromAxis).zfill(2)

        else:

            print("Requested field angle must be in {0} (integer or string)".format(possibleAnglesFromAxis))
            return

    focusPositions = h5get(csl, [angleFromAxis, "dzs"], verbose = 0)

    print(f"Min: {np.min(focusPositions)} -- Max: {np.max(focusPositions)}")

    return np.sort(focusPositions)

