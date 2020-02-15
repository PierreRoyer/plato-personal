#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:13:40 2019

@author: pierre
"""
#import yaml
import os
import yaml
import numpy as np
from simulation import Simulation
from scipy import constants
from h5 import h5ls
#from egse.image_processing.findSources import findSources

#import genericFunctions as genf
import testSimulation.simulationUtils as simut
import testSimulation.simulationUtilsAmbient as simuta
#import simulation.psfFunctionsAmbient as psfa

from showSim import getSim, showSim


# Directory with input and output files for the simulation
inputDir    = os.getenv("PLATO_WORKDIR") + "inputfiles/"
outputDir   = os.getenv("PLATO_WORKDIR") + "simout/"

# Name of the simulation (--> names of output files)
simname = "egse"
temperature = 20. + 273.15


# Configuration filename
configFile = inputDir + simname + ".yaml"


##########   CREATE SIMULATION OBJECT ############
sim = Simulation(simname, configurationFile=configFile, outputDir=outputDir)

conf = sim.getYamlConfiguration()

"""
# Load config file (if you don't specify it in the Simulation input)
conf = sim.readConfigurationFile(configFile)
"""

"""
##########   DICTONARY OF CONFIG PARAMETERS ############
# The syntax is not the same for the yaml file and the sim object:
pixelSize = sim["CCD/PixelSize"]
pixelSize = conf["CCD"]["PixelSize"]
"""

###############################################################################
# UPDATE INPUT PARAMETERS
###############################################################################
dataConfigFile = open(os.path.dirname(simut.__file__)+'/configuration.yaml','r')
psfInputFile = yaml.load(dataConfigFile)["simulationDataDirectory"]+"cslPsfs_v01.hdf5"
dataConfigFile.close()


temperature = 20.0                                                            #C
temperature = constants.convert_temperature(temperature, 'Celsius', 'Kelvin') #K

exposureTime = 4.


# Focal Plate XY coordinates [mm]
xfp,yfp,ccdCode = 10,10,2
row,column = simut.xyPixFromXYFP(sim,xfp,yfp,ccdCode)

###  OR  ###

### CCD coordinates [pix]
row,column,ccdCode = 1000,3000,2


focusPosition      = 3000
centralRow = row
numRows,numColumns = 50,50

numExposures = 10


simuta.configureForRoomTemperature(sim, roomTemperature=temperature, texp=exposureTime)

simuta.configureObsParameters(sim, exposureTime=exposureTime, numExposures=numExposures, centralRow=centralRow, numRows=numRows)

simuta.configureForXYZ(sim, row, column, ccdCode, focusPosition, exposureTime, numRows, numColumns, temperature, psfInputFile=psfInputFile, starCatalogName=None)

##########   INSPECT INPUT PARAMETERS ############
# Inspect input parameters
print (f'nExp: {sim["ObservingParameters/NumExposures"]}')
print (f'tExp: {sim["ObservingParameters/ExposureTime"]}')
print (f'subPix: {sim["SubField/NumRows"]}')
print (f'subPix: {sim["SubField/NumColumns"]}')
print (f'subPix: {sim["SubField/SubPixels"]}')
print (f'Dark {sim["CCD/DarkSignal/DarkCurrent"]:8.1f} [e-/s]')
print (f'Incl. Aberration Corr.: {sim["Camera/IncludeAberrationCorrection"]}')
print (f'Incl. Field Distortion: {sim["Camera/IncludeFieldDistortion"]}')
print (f'Incl. Flat Field : {sim["CCD/IncludeFlatfield"]}')

########## RUN SIMULATION ############

sh = sim.run(removeOutputFile = True)

sh5 = sh.hdf5file

h5ls(sh5)

n = 0
showSim(sh5,n=n,figname=f"Im_{row}_{column}_{focusPosition}_{n}",cmap='gray')#,clim=[8000,10000])

image = np.array(getSim(sh5,n=n))

