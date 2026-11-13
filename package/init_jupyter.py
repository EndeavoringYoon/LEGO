import sys
import time
import os
import mujoco
import scipy
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from IPython import get_ipython
from PyQt5.QtWidgets import QApplication

# Matrix operations
from scipy.linalg import logm, expm

# Get directory where this script lives
THIS_DIR = Path(__file__).resolve().parent

# Set paths and import modules
sys.path.append(str(THIS_DIR / './mujoco_helper')) # for 'mujoco_parser'
sys.path.append(str(THIS_DIR / './kinematics_helper')) # for 'transforms','ik','torch_chain','kinematic_chain
sys.path.append(str(THIS_DIR / './utility')) # for 'utils'
sys.path.append(str(THIS_DIR / './bvh_helper')) # for 'bvh_parser'
sys.path.append(str(THIS_DIR / './smpl')) # for 'smpl_model'
sys.path.append(str(THIS_DIR / './motion_retarget')) # for 'preretarget'



# Import modules for lego
sys.path.append(str(THIS_DIR / '../asset'))
sys.path.append(str(THIS_DIR / './yoon'))
sys.path.append(str(THIS_DIR / './kinematics'))
sys.path.append(str(THIS_DIR / './irvae_tools'))
from ae import *
from data_utils import *
from util_yoon import *
from clustering_robots import *
from curation import *
from dataset import *
from geom import *
from models import *
from representations import *
from geometry import relaxed_distortion_measure


# Import modules
from mujoco_parser import *
from transforms import *
from ik import *
from torch_chain import *
from utils import *
from bvh_parser import *

from smpl_model import *
from dir_retarget import *
from kinematic_chain import *

# Inline commands for Jupyter Notebook
ipython = get_ipython()
if ipython:
    ipython.run_line_magic('config', 'InlineBackend.figure_format = "retina"')
    ipython.run_line_magic('matplotlib', 'inline')
    ipython.run_line_magic('gui', 'qt')
    print_green ("Jupyter Notebook GUI initialized: Retina, inline, qt")

# Print which packages are imported
print_green ("Following packages are imported:")
print_green (" [mujoco_parser, transforms, ik, torch_chain, utils, bvh_parser, smpl_model]")

np.set_printoptions(precision=2,suppress=True,linewidth=100)
plt.rc('xtick',labelsize=6); plt.rc('ytick',labelsize=6)
if QApplication.instance() is None:
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
else:
    app = QApplication.instance()
app.setStyle('Fusion')

# Print the MuJoCo version
mv = tuple(map(int,mujoco.__version__.split('.')))
print ("MuJoCo version:",mv)


""" It is somewhat equivalent to the following code:

%config InlineBackend.figure_format = 'retina'
%matplotlib inline
%gui qt
print("Jupyter Notebook GUI initialized: Retina, inline, qt")

import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication
from pathlib import Path

# Set paths
sys.path.append('../../package/mujoco_helper/')       # for 'mujoco_parser'
sys.path.append('../../package/kinematics_helper/')   # for 'transforms'
sys.path.append('../../package/utility/')             # for 'utils'
sys.path.append('../../package/bvh_helper/')          # for 'bvh_parser'
sys.path.append('../../package/ml/')                  # for 'gp'

# Import modules
from mujoco_parser import *
from transforms import *
from ik import *
from torch_chain import *
from utils import *
from bvh_parser import *
from gp import *

# Set print options for numpy and matplotlib
np.set_printoptions(precision=2, suppress=True, linewidth=100)
plt.rc('xtick', labelsize=6)
plt.rc('ytick', labelsize=6)

# Set up QApplication for PyQt5
if QApplication.instance() is None:
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
else:
    app = QApplication.instance()
app.setStyle('Fusion')

print("MuJoCo version:", MUJOCO_VERSION)

"""
