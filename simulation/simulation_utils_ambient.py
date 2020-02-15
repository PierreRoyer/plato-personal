#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:47:35 2019

@author: pierre
"""
import numpy as np

def getRoomTemperatureDarkCurrent(roomTemperature):

    """
    Calculate the dark current, expressed in e- / s, for the given room temperature,
    based on Sect. 2 in PLATO-KUL-PL-TN-0003 (PLATO FEE readout modes for AIV).

    :param roomTemperature: Room temperature [K].

    :return: Dark current [e- / s] at room temperature.
    """
    import math
    from scipy import constants
    A     = 3.816788e10
    Eg0   = 1.166         # [eV]
    alpha = 4.74e-4       # [eV / K]
    beta  = 636           # [K]

    EgT = Eg0 - (alpha * math.pow(roomTemperature, 2)) / (roomTemperature + beta)

    k = constants.k / constants.eV    # [J / K] * [eV / J] = [eV / K]

    dark = A * (roomTemperature**1.5) * math.exp(-EgT / (2 * k * roomTemperature))

    return dark


def configureForRoomTemperature(sim, roomTemperature, texp=None):

    """
    PURPOSE: Sets the parameters for the given simulation object that are specific for tests
             at room temperature.

    INPUT:
        - sim: Simulation object for which to set the parameters that are specific for
               tests at room temperature.
        - roomTemperature: Room temperature [K].
    """
    from testSimulation.simulationUtils import createTemperatureFile

    # Dark current
    sim["CCD/DarkSignal/DarkCurrent"] = getRoomTemperatureDarkCurrent(roomTemperature)  # [e-/s] Dark @ ambient, PLATO-KUL-PL-TN-0003
    sim["CCD/DarkSignal/Stability"]   = 0.0                                             # Variations with temperature off
    sim["CCD/Gain/Stability"]         = 0.0


    # Exposure Time isn't proper to RT, but is given for dark --> set it
    if texp is not None:
        sim["ObservingParameters/ExposureTime"] = texp

    ################
    # Sky parameters
    ################

    sim["Sky/SkyBackground"]               = 0
    sim["Sky/IncludeVariableSources"]      = 0          # def no
    sim["Sky/IncludeCosmicsInSubField"]    = 0          # def yes
    sim["Sky/IncludeCosmicsInSmearingMap"] = 0          # def yes
    sim["Sky/IncludeCosmicsInBiasMap"]     = 0          # def yes
    sim["Sky/Cosmics/CosmicHitRate"]       = 0          # def 10

    #####################
    # Platform parameters
    #####################

    sim["Platform/UseJitter"]              = 0

    ######################
    # Telescope parameters
    ######################

    sim["Telescope/GroupID"]      = "Custom"
    sim["Telescope/AzimuthAngle"] = 0.0
    sim["Telescope/TiltAngle"]    = 0.0
    sim["Telescope/UseDrift"]     = 0


    ######################
    # Temperatures (FEE & CCD)
    ######################

    temperatureFilename = createTemperatureFile(sim)

    sim["FEE/Temperature"] = "FromFile"
    sim["FEE/TemperatureFileName"] = temperatureFilename

    sim["CCD/Temperature"] = "FromFile"
    sim["CCD/TemperatureFileName"] = temperatureFilename

    ###################
    # FEE parameters
    ###################

    sim["FEE/Gain/Stability"]      = 0.0

    ###################
    # Camera parameters
    ###################

    sim["Camera/FocalPlaneOrientation/Source"] = "ConstantValue"
    sim["Camera/FocalLength/Source"]           = "ConstantValue"
    sim["Camera/IncludeAberrationCorrection"]  = 0      # def yes
    sim["Camera/IncludeFieldDistortion"]       = 0      # def yes


    ###################
    # CCD parameters
    ###################

    # TODO : verify this is the right parameters also in partial readout
    sim["CCD/IncludeOpenShutterSmearing"]      = "no"  # def yes

    sim["CCD/IncludeConvolution"]              = "yes" # def yes  -  Convolution with PSF

    sim["CCD/IncludeFlatfield"]                = "yes" # def yes  - Flatfielding
    sim["CCD/IncludeDarkSignal"]               = "yes" # def yes  - Dark signal
    sim["CCD/IncludeBFE"]                      = "yes" # def yes
    sim["CCD/IncludePhotonNoise"]              = "yes" # def yes  - Photon noise
    sim["CCD/IncludeReadoutNoise"]             = "yes" # def yes  - Readout noise
    sim["CCD/IncludeCTIeffects"]               = "yes" # def yes
    sim["CCD/IncludeQuantumEfficiency"]        = "yes" # def yes  - QE
    sim["CCD/IncludeVignetting"]               = "no"  # def yes
    sim["CCD/IncludePolarization"]             = "no"  # def yes
    sim["CCD/IncludeParticulateContamination"] = "no"  # def yes
    sim["CCD/IncludeMolecularContamination"]   = "no"  # def yes
    sim["CCD/IncludeDigitalSaturation"]        = "yes" # def yes  -  Digital saturation
    sim["CCD/IncludeFullWellSaturation"]       = "yes" # def yes  - Full-well saturation (blooming)
    sim["CCD/IncludeQuantisation"]             = "yes" # def yes  - Quantisation

    # Not present in the current yaml config file
    #sim["CCD/IncludeNaturalVignetting"]        = 1
    #sim["CCD/IncludeMechanicalVignetting"]     = 1

    return

def configureObsParameters(sim, exposureTime, numExposures, centralRow, numRows=500):#, numColumns=500):
    """
    """

    ######################
    # Observing parameters
    ######################

    sim["ObservingParameters/ExposureTime"] = exposureTime
    sim["ObservingParameters/NumExposures"] = int(numExposures)


    # Readout mode: partial
    sim["CCD/ReadoutMode/ReadoutMode"]             = "Partial"
    sim["CCD/ReadoutMode/Partial/FirstRowReadout"] = int(max(0,centralRow - numRows // 2))
    sim["CCD/ReadoutMode/Partial/NumRowsReadout"]  = int(numRows)

    return

#def configureForXY(sim, angleFromAxis, angleFromX, focusPosition, exposureTime, ccdCode, roomTemperature, numRows, numColumns):
def configureForXYZ(sim, row, column, ccdCode, focusPosition, exposureTime, numRows, numColumns, temperature, psfInputFile=None, starCatalogName=None,verbose=True):
    """
    Configurare parameters specific for a given field angle, angular distance, and focus position
    for the given CCD and the given room temperature.  Parameter specific for the dither positions are
    not configured here.

    :param angleFromAxis: Requested angular diastance from the field to the optical axis, expressed in degrees.
                            Should be an integer or a string with values in [0:2:20].

    :param angleFromX: Requested position angle of the field, expressed in degrees.

    :param focusPosition: Requested focus position [µm].  This is the distance from L6S2, ranging
                          (unequidistantly between 2520 and 7480 micron).

    :param ccdCode: Identifier for the CCD.  Should be "1", "2", "3", or "4".

    :param roomTemperature: Room temperature [K].

    :param numRows: Minimum Row dimension of the sub-field. If the extracted PSF is larger, next_power_of_2(size of the PSF) will be used instead
                    Also used as the number of rows that should be read out in partial-readout mode [pixels].


    :param numColumns: Minimum Column dimension of the sub-field. If the extracted PSF is larger, 2 x next_power_of_2(size of the PSF) will be used instead

    :param workDir: Directory in which the results will be stored.

    :return sim: Simulation object inhibiting all configuration parameters for the given field, focus
                 position, CCD, and room temperature.

    :return magnitude: Magnitude for the OGSE source.

    :return row: Row coordinate (integer) of the pixel with the given field angle and angular distance.

    :return column: Column coordinate (integer) of the pixel with the given field angle and angular distance.
    """
    from testSimulation.simulationUtils import anglesFromXY,getDarkTime,getMagnitude
    from testSimulation.psfUtilsAmbient import getCslPsf
    from camtest.numeric import next_power_of_2


    ccdCode = str(ccdCode)

    ################
    # PSF parameters
    ################

    # Master hdf5 file containing all PSFs
    if psfInputFile is None:
        psfInputFile="/STER/platoman/data/psfs/csl/cslPsfsOrig.hdf5"

    # File where the single PSF extracted here is written, to be used as input for PlatoSim
    inputfilesDir = sim.configurationFilename[:sim.configurationFilename.rfind('/')]
    psfFilename = inputfilesDir + "/psf.hdf5"

    angleFromAxis, angleFromX = anglesFromXY(sim,row,column,ccdCode)

    psf = getCslPsf(np.rad2deg(angleFromAxis), focusPosition, np.rad2deg(angleFromX), ccdCode, outputFile = psfFilename, psfInputFile=psfInputFile,verbose=verbose)
    numPixelsPsf = psf["numPixels"]
    rho = psf["rho"]

    sim["PSF/Model"]                                 = "MappedFromFile"
    sim["PSF/MappedFromFile/Filename"]               = psfFilename
    sim["PSF/MappedFromFile/NumberOfPixels"]         = numPixelsPsf
    sim["PSF/MappedFromFile/DistanceToOA"]           = 0.0
    sim["PSF/MappedFromFile/RotationAngle"]          = 0
    sim["PSF/MappedFromFile/IncludeChargeDiffusion"] = 0
    sim["PSF/MappedFromFile/IncludeJitterSmoothing"] = 0


    ######################
    # Sub-field parameters
    ######################

    numRows    = int(max(next_power_of_2(numPixelsPsf), numRows))
    numColumns = int(max(next_power_of_2(numPixelsPsf), numColumns))

    sim.setSubfieldAroundPixelCoordinates(ccdCode, column, row, numColumns, numRows)    # This resets the exposure time

    sim["ObservingParameters/ExposureTime"] = exposureTime

    numSubPixels = psf["numSubPixels"]
    sim["SubField/SubPixels"] = numSubPixels


    # TODO : verify if this is relevant with non-full frame subfield
    sim["SubField/NumBiasPrescanColumns"]   = 25        # Number of columns in the serial pre-scan (see PLATO-KUL-PL-TN-0003)
    sim["SubField/NumSmearingOverscanRows"] = 15        # Number of columns in the serial over-scan (see PLATO-KUL-PL-TN-0003)

    #
    darkCurrent = getRoomTemperatureDarkCurrent(temperature)
    darkTime    = getDarkTime(sim, numRows)                       # Time over which dark current is being accumulated [s]

    # Create the star catalog file: an ascii file will be written with the columns
    # ra, dec, and magnitude.
    # Then point the simulation object to the catalog

    magnitude = getMagnitude(sim, darkCurrent, darkTime, rho)

    if starCatalogName is None:
        starCatalogName = sim.runName + "_catalog.txt"
    starCatalogFileName = sim.outputDir + starCatalogName

    rows = np.array([row])
    cols = np.array([column])
    mags = np.array([magnitude])
    starIds = np.array([0])

    sim.createStarCatalogFileFromPixelCoordinates(rows, cols, mags, starIds, starCatalogFileName)

    sim["ObservingParameters/StarCatalogFile"] = starCatalogFileName

    return

