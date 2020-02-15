#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:43:44 2019

@author: Pierre Royer

Functions linked to PlatoSim (usually accepting a simulation object as input)
"""

import referenceFrames as simref
from scipy import constants
import numpy as np
import math

def getDarkTime(sim, numRowsWindow):

    """
    Calculate the time over which dark current is being accumulated, expressed in seconds. This is
    the sum of the exposure time and the readout time.  The calculation of the latter is explained
    in PLATO-KUL-PL-TN-0003 (PLATO FEE readout modes for AIV), in Sect. 5 (N-FEE, partial readout mode).  

    :param sim: Simulation object from which to extract the exposure time [s] and the charge transfer times [s].

    :param numRowsWindow: Number of rows in the window that will be read out in partial-readout mode.

    :return: Time over which dark current is being accumulated [s].
    """

    # Transfer times

    parallelTransferTimeFast = sim["CCD/ParallelTransferTimeFast"] * constants.micro   # Parallel transfer without reading out the readout register [µs] -> [s]
    parallelTransferTime     = sim["CCD/ParallelTransferTime"] * constants.micro       # Parallel transfer with reading out the readout register [µs] -> [s]
    serialTransferTime       = sim["CCD/SerialTransferTime"] * constants.nano          # Serial transfer [ns] -> [s]

    # Additional serial readouts (pre-scan & over-scan)

    numColumnsSerialPreScan  = sim["SubField/NumBiasPrescanColumns"]
    numColumnsSerialOverScan = sim["SubField/NumSmearingOverscanRows"]

    # Calculate the readout time

    numRowsDump = 4510 - numRowsWindow                                          # Number of rows that are not read out
    numColumns  = 2255 + numColumnsSerialPreScan + numColumnsSerialOverScan     # Number of columns that are read out from the readout register

    windowReadoutTime = (numRowsDump * parallelTransferTimeFast) + (numRowsWindow * (numColumns * serialTransferTime + parallelTransferTime))   # Readout time [s]

    # Exposure Time
    exposureTime           = sim["ObservingParameters/ExposureTime"]

    return (exposureTime + windowReadoutTime)


def getMagnitude(sim, darkCurrent, darkTime, rho):

    """
    Calculate the magnitude for the OGSE source for the given dark current, dark time, and
    ratio of the flux in the brightest pixels and the total flux.  Other required parameters
    (exposure time, photon flux for a V = 0 G2V-star, transmission efficiency, quantum 
    efficiency, light collecting, and throughput bandwidth area are extracted from the given 
    simulation object). The calculation is done based on Eq. (4) in PLATO-KUL-PL-TN-005 
    (Configuration Parameters for PlatoSim).

    :param sim: Simulation object from which to extract the exposure time [s], the photon flux
                [photons / s / m^2 / nm], transmission efficiency, quantum efficiency,
                light collecting area [cm^2], and throughput bandwidth [nm].

    :param darkCurrent: Dark current at room temperature [e- / s].

    :param darkTime: Time over which dark current is being accumulated [s].

    :param rho: Ratio of the flux in the brightest pixel of the point source and the total flux.
                This is a property of the PSF
    
    :return: Magnitude for the OGSE source, based on Eq. (1) in PLATO-KUL-PL-TN-003 
             (Configuration Parameters for PlatoSim).
    """
    import math
    # Make sure the total flux in the brightest pixel in the image (brightest contribution of the
    # OGSE source + accumulated dark signal) is well under the full-well saturation limit of
    # 900 000 e- / pixel (arbitrary choice: 800 000 e- / pixel).

    peakFlux = 8.0e5 - (darkCurrent * darkTime)     # Peak flux of OGSE source [e-]
    totalFlux = peakFlux / rho                      # Total flux of OGSE source [e-]

    exposureTime           = sim["ObservingParameters/ExposureTime"]                                                                 # Exposure time [s]
    photonFluxm0           = sim["ObservingParameters/Fluxm0"]                                                                       # Photon flux of a V = 0 G2V-star [photons / s / m^2 / nm]
    transmissionEfficiency = sim["Telescope/TransmissionEfficiency/BOL"]                                                             # Transmission efficiency
    quantumEfficiency      = sim["CCD/QuantumEfficiency/MeanQuantumEfficiency"] * sim["CCD/QuantumEfficiency/MeanAngleDependency"]   # Quantum efficiency
    lightCollectingArea    = sim["Telescope/LightCollectingArea"] * pow(constants.centi, 2)                                          # Light collecting area [cm^2] -> [m^2]
    bandWidth              = sim["Camera/ThroughputBandwidth"]                                                                       # Throughput bandwidth [nm]

    # Eq. (4) in PLATO-KUL-PL-TN-005
    magnitude = -2.5 * math.log10(totalFlux / exposureTime / photonFluxm0 / transmissionEfficiency / quantumEfficiency / lightCollectingArea / bandWidth)
   
    return magnitude


def createTemperatureFile(sim,temperature=[293.15,293.15],timerange=[0, 60 * 60 * 24 * 365.25 * 4]):

    """
    Create a temperature time series for FEE and CCD for the given room temperature.


    :param sim: PlatoSim simulation object
    :param temperature: temperature [start,end] [K].
    :param timerange  : [0,timeend] [s]

    :return: Name of the temperature file for FEE and CCD.
    """

    temperatureFilename = sim.configurationFilename[:sim.configurationFilename.rfind(".yaml")] + "_temperature.txt"
    
    np.savetxt(temperatureFilename, np.transpose([timerange, temperature]), fmt=['%11.0f', '%11.2f'])

    return temperatureFilename

#    """
#    Calculates the undistorted pixel coordinates (row, column) on the detector
#    with the given code, for the given angle from the optical axis and angle
#    from the x-axis of the camera. Other required parameters (pixel size, CCD
#    zeropoints, and CCD orientation are extracted from the given PlatoSim
#    simulation object).  The calculations are done based on Eqs. (16) and (14)
#    in PLATO-KUL-PL-TN-0001 (PlatoSim Reference Frames).
#
#    :param sim: PlatoSim simulation object from which to extract the focal
#                length [m] and the pixel size [µm].
#
#    :param fieldAngle: Angular distance from the optical axis [degrees].
#
#    :param inPlaneAngle: Angle from the x-axis of the focal plane [degrees].
#
#    """
#    ##########################################################################
#    # Field angle and in-plane angle [radians] -> focal-plane coordinates [mm]
#    ##########################################################################
#
#    deltaX = heightCollimatedBeam * math.tan(np.deg2rad(fieldAngle))  # [cm]
#    radialDistance = deltaX / heightCollimatedBeam * \
#        (distanceL6S2 * constants.centi + focusPosition * constants.micro) / \
#        constants.milli   # [mm]
#
#    xFP = radialDistance * math.cos(np.deg2rad(inPlaneAngle))  # [mm]
#    yFP = radialDistance * math.sin(np.deg2rad(inPlaneAngle))  # [mm]
#
#    # Correct for field distortion
#
#    if(sim["Camera/IncludeFieldDistortion"]):
#
#        inverseDistortionCoefficients = sim["Camera/FieldDistortion/ConstantInverseCoefficients"]
#        focalLength = sim["Camera/FocalLength/ConstantValue"] / constants.milli    # [m] -> [mm]
#
#        xFP, yFP = simref.distortedToUndistortedFocalPlaneCoordinates(xFP, yFP, inverseDistortionCoefficients, focalLength)
#
#    ###################################################
#    # Focal-plane coordinates [mm] -> pixel coordinates
#    ###################################################
#
#    # Required configuration parameters:
#    #  - Pixel size [µm]
#    #  - Detector zeropoint x-coordinate in the focal-plane reference frame
#    #    [mm]
#    #  - Detector zeropoint y-coordinate in the focal-plane reference frame
#    #    [mm]
#    #  - Detector orientation angle in the focal-plane reference frame
#    #    [radians]
#
#    pixelSize = sim["CCD/PixelSize"]
#    ccdZeroPointX = simref.CCD[str(ccdCode)]["zeroPointXmm"]
#    ccdZeroPointY = simref.CCD[str(ccdCode)]["zeroPointYmm"]
#    ccdAngle = simref.CCD[str(ccdCode)]["angle"]






# def anglesFromPixelCoordinates(sim, row, column, ccdCode, focusPosition,
#                                distanceL6S2=176):

#     """
#     Calculates the following angles from the given pixel coordinates
#     (row, column) coordinates for the detector with the given code:
#         - angular distance from the optical axis [radians];
#         - angle from the x-axis of the camera [radians].
#     Other required parameters (pixel size, CCD zeropoints, and CCD orientation
#     are extracted from the given PlatoSim simulation object). The calculations
#     are done based on Eqs. (13) and (16) in PLATO-KUL-PL-TN-0001 (PlatoSim
#     Reference Frames).

#     :param sim: PlatoSim simulation object from which to extract the focal
#                 length [m] and the pixel size [µm].

#     :param row: Pixel row coordinate on the CCD (corresponds to the
#                 y-coordinate).

#     :param column: Pixel column coordinate on the CCD (corresponds to the
#                    x-coordinate).

#     :param ccdCode: Identifier for the CCD (in [1, 2, 3, 4]).

#     :param focusPosition: Focus position [µm].

#     :param distanceL6S2: Distance between L6S2 and the entrance pupil [cm].

#     :return fieldAngle: Angular distance from the optical axis [degrees].

#     :return inPlaneAngle: Angle from the x-axis of the focal plane [degrees].
#     """

#     ###################################################
#     # Pixel coordinates -> focal-plane coordinates [mm]
#     ###################################################

#     # Required configuration parameters:
#     #  - Pixel size [µm]
#     #  - Detector zeropoint x-coordinate in the focal-plane reference frame
#     #    [mm]
#     #  - Detector zeropoint y-coordinate in the focal-plane reference frame
#     #    [mm]
#     #  - Detector orientation angle in the focal-plane reference frame
#     #    [radians]

#     pixelSize = sim["CCD/PixelSize"]
#     ccdZeroPointX = simref.CCD[str(ccdCode)]["zeroPointXmm"]
#     ccdZeroPointY = simref.CCD[str(ccdCode)]["zeroPointYmm"]
#     ccdAngle = simref.CCD[str(ccdCode)]["angle"]

#     xFP, yFP = simref.pixelToFocalPlaneCoordinates(column, row, pixelSize,
#                                                    ccdZeroPointX,
#                                                    ccdZeroPointY, ccdAngle)

#     ##########################################################################
#     # Focal-plane coordinates [mm] -> field angle and in-plane angle [degrees]
#     ##########################################################################

#     inPlaneAngle = np.arctan2(yFP, xFP)

#     radialDistance = math.sqrt(pow(xFP, 2) + pow(yFP, 2)) * constants.milli
#     height = distanceL6S2 * constants.centi + focusPosition * constants.micro

#     fieldAngle = np.arctan2(radialDistance, height)

#     return np.rad2deg(inPlaneAngle), np.rad2deg(fieldAngle)


def xyPixFromXYFP(sim,xFP,yFP,ccdCode):
    """
    :param sim: PlatoSim simulation object
    :param xFP: focal plane X coordinate    
    :param yFP: focal plane Y coordinate    
    :param ccdCode: CCD ID, in [1,2,3,4]
    """
    #focalLength   = sim["Camera/FocalLength/ConstantValue"] / constants.milli      # Focal length [m] -> [mm]
    pixelSize     = sim["CCD/PixelSize"]                                           # Pixel size [µm]
    ccdZeroPointX = simref.CCD[str(ccdCode)]["zeroPointXmm"]                                # Detector zeropoint x-coordinate in the focal-plane reference frame [mm]
    ccdZeroPointY = simref.CCD[str(ccdCode)]["zeroPointYmm"]                                # Detector zeropoint y-coordinate in the focal-plane reference frame [mm]
    ccdAngle      = simref.CCD[str(ccdCode)]["angle"]                                       # Detector orientation angle in the focal-plane reference frame [radians]

    column, row = simref.focalPlaneToPixelCoordinates(xFP, yFP, pixelSize, ccdZeroPointX, ccdZeroPointY, ccdAngle)
    
    return row,column
    

def xyFromAngles(sim,angleFromOpticalAxis,angleFromX, ccdCode):
    """
    :param sim: PlatoSim simulation object
    :param angleFromOpticalAxis: angular position of the source wrt the optical axis 
    :param angleFromX: angular positon of the source wrt the x-axis of the camera
    :param ccdCode: CCD ID, in [1,2,3,4]
    """

    focalLength   = sim["Camera/FocalLength/ConstantValue"] / constants.milli      # Focal length [m] -> [mm]
    pixelSize     = sim["CCD/PixelSize"]                                           # Pixel size [µm]
    ccdZeroPointX = simref.CCD[str(ccdCode)]["zeroPointXmm"]                                # Detector zeropoint x-coordinate in the focal-plane reference frame [mm]
    ccdZeroPointY = simref.CCD[str(ccdCode)]["zeroPointYmm"]                                # Detector zeropoint y-coordinate in the focal-plane reference frame [mm]
    ccdAngle      = simref.CCD[str(ccdCode)]["angle"]                                       # Detector orientation angle in the focal-plane reference frame [radians]


    # Gnomonic radial distance & in-field rotation -> focal plane coordinates -> pixel coordinates

    xFP, yFP = simref.focalPlaneCoordinatesFromGnomonicRadialDistance(math.radians(angleFromOpticalAxis), focalLength, inPlaneRotation = math.radians(angleFromX))
    column, row = simref.focalPlaneToPixelCoordinates(xFP, yFP, pixelSize, ccdZeroPointX, ccdZeroPointY, ccdAngle)

    row = int(row)
    column = int(column)

    return row, column

def anglesFromXY(sim,row,column,ccdCode,verbose=1):
    """
    :param sim: PlatoSim simulation object
    :param row: pixel coordinate on the CCD
    :param column: pixel coordinate on the CCDangular positon of the source wrt the x-axis of the camera (X_CAM = -Y_CCD)
    :param ccdCode: CCD ID, in [1,2,3,4]
    """
    
    focalLength   = sim["Camera/FocalLength/ConstantValue"] / constants.milli      # Focal length [m] -> [mm]
    pixelSize     = sim["CCD/PixelSize"]                                           # Pixel size [µm]
    ccdZeroPointX = simref.CCD[str(ccdCode)]["zeroPointXmm"]                                # Detector zeropoint x-coordinate in the focal-plane reference frame [mm]
    ccdZeroPointY = simref.CCD[str(ccdCode)]["zeroPointYmm"]                                # Detector zeropoint y-coordinate in the focal-plane reference frame [mm]
    ccdAngle      = simref.CCD[str(ccdCode)]["angle"]                                       # Detector orientation angle in the focal-plane reference frame [radians]

    # TODO VERIFY row vs column
    xFP,yFP = simref.pixelToFocalPlaneCoordinates(column, row, pixelSize, ccdZeroPointX, ccdZeroPointY, ccdAngle)
    angleFromOpticalAxis = simref.gnomonicRadialDistanceFromOpticalAxis(xFP, yFP, focalLength)

    angleFromX = np.arctan2(yFP,xFP)

    if verbose: print(f"Angle from opt. axis {angleFromOpticalAxis:7.3f} (rad) , rotation angle from X {angleFromX:7.3f} (rad)")
    if verbose: print(f"Angle from opt. axis {np.rad2deg(angleFromOpticalAxis):7.3f} (deg) , rotation angle from X {np.rad2deg(angleFromX):7.3f} (deg)")

    return angleFromOpticalAxis, angleFromX


def coaddSimulatedImages(simh5,nstart=None,nend=None,type="Image",average=True):
    """
    coaddSimImages(simh5,nstart=None,nend=None,type="Image",average=True)

    INPUTS
        simfile  : platosim file opened with h5py.File
        n = 0    : image number to display
        type     : first 2 characters are used to branch to 
                   im - pixelImage, 
                   su - subPixelImage
                   sm - smearingMap
                   bi - biasMap
                   fp - prnu -- FF pixel
                   fs - irnu -- FF subpixel
    """
    from h5 import h5get
    from showSim import getSim

    """
    # If the user didn't specify the figure name, create our own figure name that is based
    # on the requested data product. 
    
    dataProduct = {"im":'image', "su":'subPixelImage', "bi":"biasMap",\
       "sm":"smearingMap", "fp":"PRNU", "fs":"IRNU", "pp":"rebinnedPSFpixel",\
       "ps":"rebinnedPSFsubPixel", "pr":"rotatedPSF"}

    firstTwoLettersOfDataProduct = str(type).lower()[:2]
    productName = dataProduct[firstTwoLettersOfDataProduct]+str(n).zfill(6)
    """
    
    nSimulatedImages = h5get(simh5,["ObservingParameters","NumExposures"],verbose=0)

    # Extract the data from the HDF5 file
    
    if nstart is None: nstart = 0
    if nend is None  : nend = nSimulatedImages
    if (nstart < 0) or (nend < 0) or (nstart > nSimulatedImages) or (nend > nSimulatedImages):
        print(f"WARNING: invalid selection of products ")
        print(f"[nstart,nend] = [{nstart},{nend}]")
        print(f"[nstart,nend] forced to [0,{nSimulatedImages}]")
    
    # Initialise the co-added image with the first image (no need for its size)
    result = np.array(getSim(simh5,type=type,n=nstart))
    
    for i in range(nstart+1,nend):
        result += np.array(getSim(simh5,type=type,n=i))
    
    if average:
        result /= (nend-nstart)
    
    return result


