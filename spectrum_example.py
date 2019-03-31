# Compute energy spectrum tendency from a given stream function
# on a periodic unit square domain

from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
from spectrum_example_tools import FFT


##############################################################################
# Mesh and spaces initialization #############################################
##############################################################################

# Mesh parameters
Ne = 128 #Number of elements in the mesh
r = 2    #Polynomial order (stream function and vorticity)

mesh = PeriodicRectangleMesh(Ne, Ne,1.0 ,1.0, direction="both", quadrilateral=False, reorder=None)

# Stream function space (also used for vorticity)
W = FunctionSpace(mesh, "CG",r)

##############################################################################
# Spectrum Computation #######################################################
##############################################################################

# Stream Function
psi0 = Function(W,name="SF")
psi0.interpolate(Expression("x[0]*(1.0-x[0])*x[1]*(1.0-x[1])"))

# Set up FFT tools
Ng = Ne*(r+1) #Number of grid points for function evaluation
threshold = 86 #Threshold for filter in frequency space
Fourier = FFT(Ng,threshold)

# Spectrum computation
Es = Fourier.E_spectrum(psi0, filter_flag = True)

# Plot
plt.loglog(Es)
plt.xlabel("k")
plt.ylabel("E(k)")
plt.show()

