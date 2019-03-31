# Compute energy/enstrophy spectrum tendencies as in Natale and Cotter (2017) for
# SUPG discretisation of the incompressible Euler equations in a periodic unit square.

# Legend:

# dt - time step
# spectrum_skip - number of time steps between spectrum computations

# Et - Energy tendency at given time t, i.e. time derivative of energy density in frequency domain
# Zt - Enstrophy tendency at given time t, i.e. time derivative of energy density in frequency domain

# E_mat - Array storing in row n the energy tendency at time t = n*spectrum_skip*dt
# Z_mat - Array storing in row n the enstrophy tendency at time t = n*spectrum_skip*dt


import time
from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
from spectrum_example_tools import FFT


##############################################################################
# Simulation parameters ######################################################
##############################################################################


# Mesh parameters
Ne = 128 #Number of elements in the mesh
r = 2    #Polynomial order (stream function and vorticity)

#Final time 
t_end = 300

#Computing spectrum at spectrum_skip (number time steps) intervals
Spectrum_flag = True #If True computes spectrum every spectrum_skip time steps
spectrum_skip = 10
threshold = 85 # Wavenumbers above threshold are filtered (max resolved wavenumber k = Ne*r/3)

# Stabilisation coefficient for SUPG discretization
beta = 1.0

##############################################################################
# Mesh and spaces initialization #############################################
##############################################################################

mesh = PeriodicRectangleMesh(Ne, Ne,1.0 ,1.0, direction="both", quadrilateral=False, reorder=None)

# Stream function space (also used for vorticity)
W = FunctionSpace(mesh, "CG",r)

# Mixed space
M = MixedFunctionSpace([W,W])

nullspaceM = MixedVectorSpaceBasis(M, [VectorSpaceBasis(constant=True),M[1]])
nullspace = VectorSpaceBasis(constant = True)

uu = Function(M)
psi, omega = split(uu)
vv = TestFunction(M)
phi, theta = split(vv)

##############################################################################
# Initial conditions #########################################################
##############################################################################
nu = FacetNormal(mesh)

# Stream function and vorticity at time t 
uun = Function(M)
psin,omegan = split(uun)


# Initial Conditions
psi0 = Function(W,name="SF") #Stream function at time t=0
omega0 = Function(W,name="V") #Vorticity at time t=0
omega0.interpolate(Expression("sin(8*pi*x[0])*sin(8*pi*x[1])+0.4*cos(6*pi*x[0])*cos(6*pi*x[1])+ 0.3*cos(10*pi*x[0])*cos(4*pi*x[1]) + 0.02*sin(2*pi*x[0])+0.02*sin(2*pi*x[1])"))

# Compute stream function psi0 from omega0
gamma_ic = TestFunction(W)
xi_ic = TrialFunction(W)
b_ic = dot(grad(xi_ic),grad(gamma_ic))*dx
M_ic = - omega0*gamma_ic*dx
solve(b_ic == M_ic,psi0, nullspace = nullspace)


# Initialization 
uun.sub(0).assign(psi0)
uun.sub(1).assign(omega0)


##############################################################################
# Forms for weak Jacobian Computation ########################################
##############################################################################

# Set up system to compute the Jacobian as appears in vorticity equation
Wpsi = TrialFunction(W)
Wphi = TestFunction(W)

# Velocities from stream functions
gpsin = grad(psin)
un = perp(gpsin)

Wgphi = grad(Wphi)
Wv = perp(Wgphi)

# Stabilisation coefficient
C_s = 1.0/(2*r*sqrt(2)*Ne*beta)

F = project(Expression("0.1*sin(32*pi*x[0])"),W)
epsilon =1e-5 

a = Wpsi*(Wphi +C_s*dot(un,grad(Wphi)/(dot(un,un)**0.5+epsilon)))*dx
L = dot(un,grad(omegan))*(Wphi+C_s*dot(un,grad(Wphi))/(dot(un,un)**0.5+epsilon))*dx
omega_dt = Function(W)
solve(a==L,omega_dt)

a_psi = dot(grad(Wphi),grad(Wpsi))*dx
L_psi = -omega_dt*Wphi*dx

psi_dt = Function(W)
solve(a_psi==L_psi,psi_dt,nullspace=nullspace)

# L2 projection to treat mixed variables
a_L2 = Wpsi*Wphi*dx
L_L2 = psin*Wphi*dx
vpsi = Function(W)
solve(a_L2==L_L2,vpsi)

# Set up FFT tools
Ng = Ne*(r+1) #Number of grid points for function evaluation
Fourier = FFT(Ng,threshold)

##############################################################################
# Form assembly ##############################################################
##############################################################################

# Time stepping
t = 0
dt = 1.0/50 
tend = t_end

counter = 0 # Iteration counter for integration loop
counter1 =0 # Iteration counter for spectrum evaluation loop 

# Velocities from stream functions
gpsi = grad(psi)
u = perp(gpsi)
gpsin = grad(psin)
un = perp(gpsin)
gphi = grad(phi)
v = perp(gphi)

# Governing equations
Eq1 = (omega- omegan+\
          dt*dot(0.5*(u+un),0.5*grad(omega+omegan))+\
          dt*(omega*0.01- F))*(theta+0.5*C_s*dot(u+un,grad(theta))/(dot(un,un)**0.5+ epsilon))*dx

Eq2 = dot(u,v)*dx + omega*phi*dx

F_psi = Eq1 +Eq2   


##############################################################################
# Time integrator ############################################################
##############################################################################

# Energy and enstrophy at time zero
E0 = 0.5*assemble(dot(un,un)*dx)
Z0 = 0.5*assemble(omegan*omegan*dx)

# Vectors initialisation 
Eval= [] # Kinetic energy
Zval =[] # Enstrophy
Time = [] # Time

Zval.append(Z0)
Eval.append(E0)


solver_parameters={'ksp_type':'preonly',
                   'mat_type': 'aij',
                   'pc_type':'lu',
                   "snes_lag_preconditioner": 1,
                   'pc_factor_mat_solver_package':'mumps',
                   'snes_monitor': True,
                   }


# Save stream function for Paraview visualization
#out_stream = File("stream.pvd")
#out_stream.write(project(psin,W,name="SF"))

psi_prob = NonlinearVariationalProblem(F_psi,uu)
psi_solver = NonlinearVariationalSolver(psi_prob, 
                                        solver_parameters= solver_parameters,
                                        nullspace = nullspaceM)



# Setting up matrices to store spectra
solve(a_L2==L_L2,vpsi)
Et,Zt = Fourier.EZ_spectra(vpsi,psi_dt)
E_mat= np.ones([int(round(tend/dt/spectrum_skip)+1),np.size(Et)])
Z_mat= np.ones([int(round(tend/dt/spectrum_skip)+1),np.size(Zt)])
E_mat[0,:]=Et
Z_mat[0,:]=Zt


while (t<=tend):
        
        t1 = time.time()         
        psi_solver.solve()
        uun.assign(uu) 
        counter +=1
        t +=dt
        
        psin,omegan = split(uun)
        solve(a_L2==L_L2,vpsi)
       
        # Energy and enstrophy computation
        Znt = 0.5*assemble(omegan*omegan*dx)
        Ent = 0.5*assemble(dot(grad(vpsi),grad(vpsi))*dx)
        print ("Total energy: %lf" % Ent)
        print ("Iteration number: %d" %counter)
        Eval.append(Ent)
        Zval.append(Znt)
        #out_stream.write(project(psin,W,name ="SF"))
      

        if Spectrum_flag & ((counter1*spectrum_skip)==(counter-spectrum_skip)):
            # Spectra computation
            solve(a==L,omega_dt) # Time derivative of vorticity 
            solve(a_psi==L_psi,psi_dt,nullspace=nullspace) # Time derivative of stream function 
            Et,Zt = Fourier.EZ_spectra(vpsi,psi_dt) # Spectra
            counter1 +=1
            E_mat[counter1,:]=Et
            Z_mat[counter1,:]=Zt

        deltat = time.time()-t1
        print ("Elapsed time: %f" % deltat)

# Plot of the Energy tendency at final time
plt.plot(Et)
plt.xlabel("$k$")
plt.ylabel("$\dot{E}(k)$") 
plt.show()


