###########################################################
# Serial program that performs a post-processing analysis #
# on a large set of simulations results. In particular it #
# computes the mean contact map for each given value of Q #
###########################################################

###########################################################
# ----------------------- ISSUES ------------------------ #
###########################################################

###########################################################
# ----------------------- TO DO ------------------------- #
###########################################################

###########################################################
# ---------------------- IMPORTS ------------------------ #
# Dependencies:                                           #
# mdtraj   - Package for the analysis of MD simulations   #
#                                                         #
# warnings - Avoids calling Python warnings. Sometimes    #
#            warnings occur when the pdb file loaded via  #
#            mdtraj shows unexpected features             #
#														  #
# argparse - needed for parser                            #
###########################################################

from __future__ import division
import numpy as np
import warnings
import os
import sys
import argparse
import contacts.PopulationHistogramClass as PoHC
warnings.filterwarnings("ignore")

###########################################################
# ---------------------- FUNCTIONS ---------------------- #
# Notes:                                                  #
###########################################################

def rm_r(path):
#intelligent rm -r: check if the directory exists and delete it!

    if( os.path.isdir(path) ):
        bash = 'rm -r ' + path
        os.system(bash)
    elif( os.path.isfile(path) ):
        bash = 'rm ' + path
        os.system(bash)
#-----------------------------------------------------------------------------------------------------------------------

#function which provides the way to obtain the input using a parser
#all that is done inside is practically self-explaining
def get_args():
    parser = argparse.ArgumentParser(
                                     description = 'Script for the computation of the population histogram for each given value of the fraction \
                                     of native contacts')
    parser.add_argument(
                    '-q','--q_dir', type = str, help = 'Name of the directory in which Q to time files are located (mandatory)', required = True)
    parser.add_argument(
                    '-b', '--bins', type = int, help = 'Number of bins for the mean calculation (optional, default = 20)',
                    required = False, default = 20)
    parser.add_argument(
                    '-o', '--out_dir', type = str, help = 'Name of the directory in which to store the final results (optional, default = results/',
                    required = False, default = 'results')
    parser.add_argument(
    				'-gq', '--global_q', type = str,
    				help = 'Global name of the Q files (optional, default no global name)', required = False, default = '')

    args = parser.parse_args()

    return args.q_dir, args.bins, args.out_dir, args.global_q
#-----------------------------------------------------------------------------------------------------------------------

###########################################################
# ------------------------ MAIN ------------------------- #
# Notes:                                                  #
# this part of code will not be executed if this file is  #
# imported as a module instead of being directly ran      #
###########################################################

if __name__ == '__main__':

	#get parser arguments
	Q_dir, B, out_dir, global_q = get_args()

	print()
	print("####################################################################")
	print("#                               Inputs                             #")
	print("####################################################################")
	print()
	print("# Q files:", Q_dir)
	print("# Number of bins:", B)
	print("# Output:", out_dir)

	if( os.path.isdir(out_dir) ):
		pass
	else:
		bash = 'mkdir ' + out_dir
		os.system(bash)

	#convert adresses to absolute paths
	Q_dir = os.path.abspath(Q_dir)

	print()
	print("####################################################################")
	print("#                               Analysis                           #")
	print("####################################################################")
	print()

	ph = PoHC.PopulationHistogram(Q_dir, out_dir, B, global_q)
	counters = ph.histogram()

	print()
	print("# Normalizing histogram")
	ph.normalize_histogram(counters)

	print("# Done")
