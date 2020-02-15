#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 13:50:40 2019

@author: pierre
"""

import os
import yaml
import numpy as np
from simulation import Simulation
from scipy import constants
from h5 import h5ls

import testSimulation.simulationUtils as simut
import testSimulation.simulationUtilsAmbient as simuta
from testSimulation.psfUtilsAmbient import getFocusPositions

simname = "egse"
temperature = 20. + 273.15

#outputDir = "/Users/pierre/plato/pr/simout/egse/smearingNo/"
# Directory with input and output files for the simulation
inputDir    = os.getenv("PLATO_WORKDIR") + "inputfiles/"
outputDir   = os.getenv("PLATO_WORKDIR") + "simout/"
#outputDir += "egse/smearingNo/"    # NOMINAL PSFs, 1 unique random seed
#outputDir += "egse/seeds/"         # NOMINAL PSFs, changing seeds
#outputDir += "egse/tol/"            # TOLERANCED PSFs, changing seeds
outputDir += "egse/hartmann45/"        # HARTMANN MASK, changing seeds, Large spot
outputDir += "egse/hartmann35/"        # HARTMANN MASK, changing seeds, Small spot

configFile = inputDir + simname + ".yaml"

##########   CREATE SIMULATION OBJECT ############
sim = Simulation(simname, configurationFile=configFile, outputDir=outputDir)

conf = sim.getYamlConfiguration()


###############################################################################
# INPUT PARAMETERS
###############################################################################

# default: None --> /STER/plato/data/psfs/csl/cslPsfsOrig.hdf5
dataConfigFile = open(os.path.dirname(simut.__file__)+'/configuration.yaml','r')
#psfInputFile = yaml.load(dataConfigFile)["simulationDataDirectory"]+"cslPsfs_v02.hdf5"  # NOISELESS PSFs
#psfInputFile = yaml.load(dataConfigFile)["simulationDataDirectory"]+"cslPsfs_tol.hdf5"  # TOLERANCED PSFs
#psfInputFile = yaml.load(dataConfigFile)["simulationDataDirectory"]+"hartmann_35_v01.hdf5"  # HARTMANN 35
psfInputFile = yaml.load(dataConfigFile)["simulationDataDirectory"]+"hartmann_45_v01.hdf5"  # HARTMANN 45
dataConfigFile.close()

temperature = 20.0                                                            #C
temperature = constants.convert_temperature(temperature, 'Celsius', 'Kelvin') #K

exposureTime = 4.

# Focal Plate XY coordinates [mm]
xfp,yfp,ccdCode = 10,10,2
row,column = simut.xyPixFromXYFP(sim,xfp,yfp,ccdCode)

###  OR  ###

### CCD coordinates [pix]

row,column,ccdCode = 4000,500,2  #  4 degrees
row,column,ccdCode = 3000,1000,2 #  8 degrees
row,column,ccdCode = 2500,2000,2 # 12 degrees
#row,column,ccdCode = 2000,2000,2 # 14 degrees


focusPosition      = 5140
focusPosition      = 5520
centralRow = row
numRows,numColumns = 50,50

numExposures = 25


########## UPDATE INPUT PARAMETERS ############


simuta.configureForRoomTemperature(sim, roomTemperature=temperature, texp=exposureTime)

simuta.configureObsParameters(sim, exposureTime=exposureTime, numExposures=numExposures, centralRow=centralRow, numRows=numRows)

# TODO : extract the dz range available for the input [x,y]
# Next line is only to extract the angleFromOpticalAxis corresponding to the chosen row,column (watch the output on the console)
simuta.configureForXYZ(sim, row, column, ccdCode, focusPosition, exposureTime, numRows, numColumns, temperature, psfInputFile=psfInputFile, starCatalogName=None)
# Determine angleFromOpticalAxis, then
#anglefromopticalax = 18.  # row,column = 1000,3000
anglefromopticalax = 12.  # row,column = 2500,2000
#anglefromopticalax = 8.  # row,column = 3000,1000
#anglefromopticalax = 4.  # row,column = 4000,500

dzs = getFocusPositions(anglefromopticalax, psfInputFile=psfInputFile)
dzmin,dzmax,dzstep = 2550,7450,50

seedsseed, cseed = 1424477999, 0

#for f,focusPosition in enumerate(range(dzmin,dzmax,dzstep)):
for f,focusPosition in enumerate(dzs):

    runName = f"egse_{str(row).zfill(4):4s}_{str(column).zfill(4):4s}_{ccdCode}_{focusPosition}"
    sim.runName = runName

    print()
    print(runName)
    print()

    simuta.configureForXYZ(sim, row, column, ccdCode, focusPosition, exposureTime, numRows, numColumns, temperature, psfInputFile=psfInputFile, starCatalogName=None)
    
    
    ########## UPDATE RANDOM SEEDS ############
    sim["RandomSeeds/ReadOutNoiseSeed"] = seedsseed + cseed
    sim["RandomSeeds/PhotonNoiseSeed"]  = seedsseed + cseed + 1
    sim["RandomSeeds/JitterSeed"]       = seedsseed + cseed + 2
    sim["RandomSeeds/FlatFieldSeed"]    = seedsseed + cseed + 3
    sim["RandomSeeds/DriftSeed"]        = seedsseed + cseed + 4
    sim["RandomSeeds/CosmicSeed"]       = seedsseed + cseed + 5
    sim["RandomSeeds/DarkSignalSeed"]   = seedsseed + cseed + 6
    cseed += 7
    
    ########## RUN SIMULATION ############

    sh = sim.run(removeOutputFile = True)



