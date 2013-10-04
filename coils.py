# -*- coding: iso-8859-1 -*-
from scipy.special import ellipe,ellipk
import numpy as np
from numpy.linalg import norm,inv
import pylab as pb

""" Calculates magnetic field due to a coil or coil array (multi-winding coil)

This module contains two classes: Coil and CoilArray.
Mostly one will use CoilArray to define a multi-winding coil at some location and orientation.
These can then be summed up to calculate the field for any coil configuration made of circular coils.

Running the module will run an example of a Helmholtz coil configuration

"""
mu_0 = 1.25663706144e-06 # N/A^2


class Coil(object):
    """
    returns the magnetic field from one arbitrary current loop calculated from
    eqns (1) and (2) in Phys Rev A Vol. 35, N 4, pp. 1535-1546; 1987.

    This is used in CoilArray which creates an array of coil objects given a
    positions and spacings and normal vector
    
    arguments (all units in MKS):
    the class is defined for a coil where:

    * n is normal vector to the plane of the loop at the center (nx,ny,nz), 
    * current is oriented by the right-hand-rule.
    * r0 is the location of the center of the loop (x,y,z)
    * R is the radius of the loop
    * we assume the finite width of the wire is insignificant

    the Bvector method can then be used to calculate the B field (Tesla/amp)
    at a point r 
    * r is a position vector where the Bfield is evaluated: [x y z]

    """
    def __init__(self,r0,R,n):
        self.length = 1                     #length unit (meters)
        self.unit = mu_0/(2*np.pi)     #T/A assuming length in m
        self.R = R                          #Coil radius
        self.wireLength = 2*np.pi*self.R
        self.ax = np.asarray([1.0, 0, 0])  #define lab coordinates
        self.ay = np.asarray([0, 1.0, 0])
        self.az = np.asarray([0, 0, 1.0])
        self.r0 = np.asarray(r0)         #Coil position
        n = np.asarray(n)                #Coil axis vector
        #make coil coordinates = unit vectors (a,b,c)
        self.c = n/norm(n)
        if np.dot(self.c,self.az)==1:  
            #normal is on z axis
            self.a = self.ax
            self.b = self.ay
        elif np.dot(self.c,self.az)==-1:  
            #normal is on -z axis
            self.a = self.ax
            self.b = -self.ay
        else:   
            #normal is off z axis, make a perp to c and z
            aa = np.cross(self.c,self.az)
            self.a = aa/norm(aa)
            bb = np.cross(self.c,self.a)
            self.b = bb/norm(bb)
        # make transform matrices
        self.T = np.column_stack((self.a,self.b,self.c))
        self.invT = inv(self.T)

    def toLab(self,rc):
        # rotate vector in coil frame to lab frame
        rc = np.asarray(rc)
        return np.inner(self.T,rc)

    def toCoil(self,r):
        # rotate vector in lab frame to coil frame
        r = np.asarray(r)
        return np.inner(self.invT,r)

    def shift_toLab(self,rc):
        # move postion vector from coil origin to lab origin
        rc = np.asarray(rc)
        rcoil = self.r0 + rc
        return rcoil
        
    def shift_toCoil(self,r):
        # move postion vector from lab origin to coil origin
        r = np.asarray(r)
        rclab = r - self.r0
        return rclab

    def posToLab(self,rc):
        # combine shift and rotation to take position vector in lab to postion vector rel to coil
        rcoil = self.shift_toLab(rc)
        return self.toLab(rcoil)

    def posToCoil(self,r):
        # combine shift and rotation to take position vector in coil to postion vector rel to lab
        rclab = self.shift_toCoil(r)
        return self.toCoil(rclab)
        
    def Bvector(self,r):
        '''
        calculate B vector (T/amp) at a point r.
        convert r to coil frame. then get B then convert b
        back to lab frame.
        '''
        x,y,z = self.posToCoil(r)
        R = self.R
        rho = norm([x,y])
        d = np.sqrt( (R+rho)**2 + z**2 )
        if d == 0:  # No Coil
            return np.asarray([0,0,0])
        d2 = ( (R-rho)**2 + z**2 )
        if d2 == 0: # on coil
            return np.asarray([0,0,0])
        k2 = (4*R*rho)/d**2
        K = ellipk(k2)
        E = ellipe(k2)
        Bc = (1/d)*(K + E*(R**2 - rho**2 - z**2)/d2)
        if rho == 0:
            Br=Ba=Bb = 0
        else:
            Br = (1/rho)*(z/d)*(-K + E*(R**2 + rho**2 + z**2)/d2)
            Ba = Br*x/rho
            Bb = Br*y/rho
        B = np.asarray([Ba,Bb,Bc])*self.unit
        return self.toLab(B)


class CoilArray(object):
    """
    Initialized with arguments r0,R0,dr,Nr,dR,NR,n (described below)
    this will initialize a 'coils' array where each element is a single coil (winding) with its own size and position
    From this we can calculate the field at any point due to the array as the sum of the contributions from 
    each coil.
    arguments:
    'dr', 'dR' axial  and radial spacing of windings (coils) 
                which may depend on wire size and number of layers or total thickness
    'r0', 'R0' position and radius of the first coil 
    'Nr','NR', axial and radial number of windings
    'n' is the axial direction of array as defined in Coil.
    'r' is thickness in axial direction (calculated)
    'R' is radial thickness (calculated)

    to get proper Resistance set self.resistance (ohms/m)
    """
    def __init__(self,r0,R0,dr,Nr,dR,NR,n):
        self.ohmspermeter = .0023378
        self.coils = []
        self.wireLength = 0
        self.turns = Nr*NR
        self.R0 = R0
        self.Nr = Nr
        self.NR = NR
        self.thickness = dr*Nr
        self.width = dR*NR
        self.inner_radius = R0 - dR/2.0
        self.outer_radius = self.inner_radius + self.width
        self.inner_thickness = r0[2] - dr/2.0
        self.outer_thickness = self.inner_thickness + self.thickness
        for i in range(Nr):
            for j in range(NR):
                r = [r0[0],r0[1],r0[2] + i*dr]
                R = R0 + j*dR
                C = Coil(r,R,n)
                self.wireLength += C.wireLength
                self.coils.append(C)

    def Bvector(self,r):
        B = np.zeros([3])
        for C in self.coils:
            B = B + C.Bvector(r)
        return B

    @property
    def resistance(self):
        return self.ohmspermeter*self.wireLength

    def inductance(self):
        N = self.turns
        Rmean = self.R0 + (self.width - dR)/2.0
        a = self.width/2.0
        L = (N**2)*Rmean*mu_0*( np.log(8*Rmean/a)-2 )
        return float(L)

    def tau(self):
        return self.inductance()/self.resistance

    @property
    def dimensions(self):
        ''' return string describing dimesions and array characteristics
        '''
        NR,Nr,n = self.NR,self.Nr,self.NR*self.Nr
        L = self.inductance()
        tau = L/self.resistance
        critical_capacitance = 4*tau/self.resistance
        s = "\
        N = %d x %d = %d\n\
        inner radius = %.6f\n\
        outer radius = %.6f\n\
        radial width =  %.6f\n\
        inner thickness = %.6f\n\
        outer thickness = %.6f\n\
        thickness = %.6f\n\
        inductance = %.6f\n\
        resistance = %.6f\n\
        tau=L/R = %.6f\n\
        critical_damping_capacitance = %.0f nF\n\
        lenth of wire = %.6f\n\
        "%(NR,Nr,n,self.inner_radius,self.outer_radius,self.width,\
        self.inner_thickness,self.outer_thickness,self.thickness,\
        L,self.resistance,tau,critical_capacitance*1e9,self.wireLength)
        return s

def gradient(CoilArray1,CoilArray2,r,dr=.001):
    """
    given two CoilArrays this estimates the gradient vector direction and magnitude at r.
    Simple gradient estimate based on spacial step on a step size dr.
    """
    rx1 = r
    rx2 = r+np.asarray([dr,0,0])
    rz1 = r
    rz2 = r+np.asarray([0,0,dr])
    bxp = CoilArray1.Bvector(rx1)+CoilArray2.Bvector(rx1)
    bxm = CoilArray1.Bvector(rx2)+CoilArray2.Bvector(rx2)
    bzp = CoilArray1.Bvector(rz1)+CoilArray2.Bvector(rz1)
    bzm = CoilArray1.Bvector(rz2)+CoilArray2.Bvector(rz2)
    dBx = bxp-bxm
    dBz = bzp-bzm
    dbx = norm(dBx)/dr
    dbz = norm(dBz)/dr
    return dbx,dbz,dBx,dBz
    
def time_to_I(I,tau,Imax):
    """
    using a time constant for the LR circuit this figures time to 
    reach a current I given an Imax of the coil driver based on 
    V and R (not on an artificial current limit)
     assume I(t) = Imax*(1-exp(-t/tau)) so t = -tau*ln(1-(I/Imax) )
    """
    if I>=Imax:
        print 'I greater than Imax:',I,Imax
        return 0
    t = -tau*np.log( 1-(I/Imax) )
    return t

if __name__ == "__main__":
    '''
    Example with basic Helmholtz coils
    Here there are two coils separated by 6.4 cm of radius 5.2cm, each with 228 windings
    wound 12 high with wire separation of 1 mm and 19 radially with wire separation of 
    2 mm (spacers in winding)
    '''

    #our coil small wire (this give 31.9 G/A which matches our calibration pretty well)
    dr = .001 #axial wire separation
    dR = .002 #radial wire separation
    r0 = np.asarray((0,0,-0.032)) #position of center of coil1 bottom
    R0 = .052      #inner radius
    Nr = 12
    NR = 19
    n = np.asarray([0,0,1])
    Ca1 = CoilArray(r0,R0,dr,Nr,dR,NR,n)
    Ca2 = CoilArray(-r0,R0,-dr,Nr,dR,NR,n) 
    Ca1.ohmspermeter = .003    
    Ca2.ohmspermeter = .003    
    Itest = 5
    Vmax = 15

    #make grid
    plotMargin = 1.2
    sizex = Ca1.outer_radius*plotMargin
    sizez = Ca1.outer_thickness*plotMargin
    print sizex,sizez
    size = np.max([sizex,sizez])
    N=20.0    # number of grid points per axis
    i=j=pb.arange(N)
    n=float(N)
    x = ((i/(n-1))-.5)*size*2
    z = ((j/(n-1))-.5)*size*2
    X,Z = pb.meshgrid(x,z)
    Bx = pb.zeros_like(X)
    Bz = pb.zeros_like(X)
    Bnorm = pb.zeros_like(X)
    #get B field
    gaussPerTesla = 1e4    
    for ii in i:
        for jj in j:
            x = X[ii,jj]
            z = Z[ii,jj]
            if jj==0:
                print ii,jj,x,z
            B=(Ca1.Bvector([x,0,z])+Ca2.Bvector([x,0,z]))*gaussPerTesla
            Bnorm[ii,jj] = norm(B)
            Bx[ii,jj] = B[0]*np.log1p(np.fabs(1000*norm(B)))/norm(B)
            Bz[ii,jj] = B[2]*np.log1p(np.fabs(1000*norm(B)))/norm(B)
            
            
    #make figure
    pb.figure()
    
    pb.axis([-size, size, -size, size])
    CS = pb.contour(X, Z, Bnorm,30,linewidths=0.5,colors='k')
    pb.clabel(CS, inline=1, fontsize=10)
    CS = pb.contourf(X, Z, Bnorm,30,cmap=pb.cm.jet)#,extent= (-.1,.1,-.1,.1)
    pb.xlim(-size,size)
    pb.ylim(-size,size)
    pb.colorbar() # draw colorbar
    Q = pb.quiver( X, Z, Bx, Bz, units='width')
    B0 = ( Ca1.Bvector([0,0,0])+Ca2.Bvector([0,0,0]) )*gaussPerTesla
    s = '%.0f G/A'%norm(B0)
    pb.text(0,0,s,size=8)
    pb.title('gauss per amp')
    print Ca1.dimensions
    pb.show()