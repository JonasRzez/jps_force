#! /usr/bin/env python
import numpy as np
import os, argparse, logging, time, sys
from os import path, system
from sys import argv ,exit
import subprocess, glob
import multiprocessing
import matplotlib.pyplot as plt
import re

#=========================
testnr = int(argv[0].split("_")[-1].split(".")[0])
#========================

must_time = 10  # 10 m corridor with 1m/s 
SUCCESS = 0
FAILURE = 1

#--------------------------------------------------------
logfile="log_test_%d.txt"%testnr
f=open(logfile, "w")
f.close()
logging.basicConfig(filename=logfile, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

#-------------------- DIRS ------------------------------
HOME = path.expanduser("~")
DIR= os.path.dirname(os.path.realpath(argv[0]))
CWD = os.getcwd()
#--------------------------------------------------------
    

if __name__ == "__main__":
    if CWD != DIR:
        logging.info("working dir is %s. Change to %s"%(os.getcwd(), DIR))
        os.chdir(DIR)


    logging.info("change directory to ..")
    os.chdir("..")
    logging.info("call makeini.py with -f %s/master_ini.xml"%DIR)
    subprocess.call(["python", "makeini.py", "-f", "%s/master_ini.xml"%DIR])
    os.chdir(DIR)
    #-------- get directory of the code TRUNK
    os.chdir("../..")
    TRUNK = os.getcwd()
    os.chdir(DIR)
    lib_path = os.path.abspath("%s/Utest"%TRUNK)
    sys.path.append(lib_path)
    from utils import *
    #----------------------------------------
    logging.info("change directory back to %s"%DIR)

    geofile = "%s/geometry.xml"%DIR
    inifiles = glob.glob("inifiles/*.xml")
    if not path.exists(geofile):
        logging.critical("geofile <%s> does not exist"%geofile)
        exit(FAILURE)
        
    executable = "%s/bin/jpscore"%TRUNK
    if not path.exists(executable):
        logging.critical("executable <%s> does not exist yet."%executable)
        exit(FAILURE)
        
    for inifile in inifiles:
        if not path.exists(inifile):
            logging.critical("inifile <%s> does not exist"%inifile)
            exit(FAILURE)
        #--------------------- SIMULATION ------------------------  
        #os.chdir(TRUNK) #cd to the simulation directory      
        cmd = "%s --inifile=%s"%(executable, inifile)
        logging.info('start simulating with exe=<%s>'%(cmd))
        #------------------------------------------------------
        subprocess.call([executable, "--inifile=%s"%inifile])
        #------------------------------------------------------
        logging.info('end simulation ...\n--------------\n')
        trajfile = "trajectories/traj" + inifile.split("ini")[2]
        logging.info('trajfile = <%s>'%trajfile)
        #--------------------- PARSING & FLOW-MEASUREMENT --------
        if not path.exists(trajfile):
            logging.critical("trajfile <%s> does not exist"%trajfile)
            exit(FAILURE)

        logsim = "inifiles/log.P0.dat"
        if not path.exists(logsim):
            logging.critical("logsim <%s> does not exist"%logsim)
            exit(FAILURE)
    
        logging.info("open  <%s> "%logsim)
        f = open(logsim, "r")
        for line in f:
            if line.startswith("Exec"):
                exec_time = float( line.split()[-1])

            if line.startswith("Evac"):
                evac_time = float(line.split()[-1])

        f.close()
        
        
        if evac_time < exec_time:
            logging.info("%s exits with FAILURE evac_time = %f, exec_time = %f)"%(argv[0], evac_time, exec_time))
            exit(FAILURE)
        else:
            logging.info("evac_time = %f  exec_time = %f)"%(evac_time, exec_time))
        
    logging.info("%s exits with SUCCESS"%(argv[0]))
    exit(SUCCESS)