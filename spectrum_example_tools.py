import numpy as np
from firedrake import *
from itertools import product 
import scipy.fftpack as fft 
import scipy.signal as sg 

class FFT:
    """
    Class containg evaluation and FFT routines to
    calculate spectra:

    E_spectrum - Computes energy spectrum
    EZ_spectra - Computes energy and enstrophy tendency in frequency space

    Domain is unit square in physical space. See Natale and Cotter (2017) for more details.
    """

    def __init__(self, Ng, threshold):
        
        """
        Initialisation

        Ng - number of grid points for evaluation
        threshold - frequency threshold for FFT top-hat filter
        """
       
        self.Ng = Ng

        #Get array Mg for point evaluation in physical space
        #Excludes last point because of periodicity and avoids cell boundaries
        Xg = np.linspace(0,1,Ng+1)[:-1]+0.00001*pi
        self.Mg = list(product(Xg, Xg))
                
        #Frequency domain (same size as Xg)
        Ff = np.arange(-(Ng-1)/2,(Ng-1)/2+1) #wave number 
        self.Xf, self.Yf = np.meshgrid(Ff*2*pi, Ff*2*pi)

        # Square of wave number
        self.Ksq = self.Xf**2+self.Yf**2

        # Filter
        self.threshold = threshold
        t2 = threshold**2*(2*pi)**2
        self.Filter = (self.Ksq<=t2)

        # Calculate the indices from the image
        y, x = np.indices((Ng,Ng))

        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])
        r = np.hypot(x - center[0], y - center[1])

        # Get sorted radii
        self.ind = np.argsort(r.flat)
        r_sorted = r.flat[self.ind]
    
        # Get the integer part of the radii (bin size = 1)
        r_int = r_sorted.astype(int)

        # Find all pixels that fall within each radial bin.
        deltar = r_int[1:] - r_int[:-1]    # Assumes all radii represented
        self.rind = np.where(deltar)[0]    # location of changed radius
        

    def Eval(self,fun):
        """
        Evaluates fft of finite element function fun
        """
        
        print "Point evaluation"
        
        #Evaluate & convert to list of arrays to array
        fun_grid = np.vstack(fun.at(self.Mg))
        # Reshape to matrix
        fun_grid_mat = np.reshape(fun_grid,(self.Ng,self.Ng))
      
        # Stream function in frequency domain
        fun_fft =fft.fftshift(fft.fft2(fun_grid_mat))
       
        print "Evaluation completed"

        return fun_fft

    def E_spectrum(self,psi, filter_flag = False):
        
        """
        Compute 1D Energy spectrum 
        
        psi - stream function
        filter_flag - apply top hat filter up to self.threshold

        Note: sum of elements in Esp = total kinetic energy  0.5 * int u^2 dx
        """

        #Evaluate & fft of stream function
        Pf = self.Eval(psi)

        #Vorticity computation in frequency domain
        Of = Pf*self.Ksq

        if filter_flag:
            #Filter high freqency
            Pf_f = Pf*self.Filter
            Of_f = Of*self.Filter
            Esp_f = 0.5*self.azimuthalInt(np.real(np.conj(Pf_f)*Of_f)/(self.Ng**4))
            Esp =  Esp_f[range(self.threshold)]
        else: 
            Esp = 0.5*self.azimuthalInt(np.real(np.conj(Pf)*Of)/(self.Ng**4))
 
        return Esp

    def EZ_spectra(self,psi,psi_dt):
        """
        Calculates the 1D spectrum of energy and enstrophy tendency

        psi - stream function
        psi_dt - time derivative of stream function
        
        Compuation made as in Natale and Cotter (2017) via sub-grid Jacobian J_SG.
        Domain is unit square in physical space.
        """
       
        # Stream function in frequency domain
        Pf = self.Eval(psi)

        # Time derivative of stream function in frequency domain 
        Pdtf = self.Eval(psi_dt)
       
        # Jacobian from time derivative of stream function
        Jf = -Pdtf*self.Ksq
        
        # Vorticity computation in frequency domain
        Of = Pf*self.Ksq         

        # Filtered quantities 
        Pf_f = Pf*self.Filter
        Of_f = Of*self.Filter

        # Construction of Jacobian perp(grad(psi) dot grad(omega) 
        Jx = -fft.ifft2(fft.ifftshift(1j*Pf_f*self.Yf))*fft.ifft2(fft.ifftshift(1j*Of_f*self.Xf))+fft.ifft2(fft.ifftshift(1j*Pf_f*self.Xf))*fft.ifft2(fft.ifftshift(1j*Of_f*self.Yf))

        Jf_spect = self.Filter*fft.fftshift(fft.fft2(Jx))
        
        # Subgrid Jacobian (difference of Jacobian from discretisation and "true" Jacobian)
        J_SG = Jf*self.Filter-Jf_spect
           
        # Energy/Enstrophy tendency
        Edt = self.azimuthalInt(np.real(np.conj(Pf_f)*J_SG)/(self.Ng**4)/(2*np.pi))
        Zdt = self.azimuthalInt(np.real(np.conj(Of_f)*J_SG)/(self.Ng**4)/(2*np.pi))
        
        return Edt, Zdt

    def azimuthalInt(self, image):
        """
        Calculates the azimuthally averaged radial profile.

        image - the 2D image
       
        """
        i_sorted = image.flat[self.ind]

        # Cumulative sum to figure out sums for each radius bin
        csim = np.cumsum(i_sorted, dtype=float)
        tbin = csim[self.rind[1:]] - csim[self.rind[:-1]]
       
        return tbin


