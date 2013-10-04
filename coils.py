# -*- coding: iso-8859-1 -*-
from scipy.special import ellipe,ellipk
import numpy as np
from numpy.linalg import norm,inv
from physcon import mu_0,u,mu_B
import pylab

"""
this version is just modified to look at the freq and B of our main coils at different distances
"""
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
                self.coils.append(C3

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

    def dimensions(self):
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
    given two CoilArrays this estimates the gradient vector and magnitude
    at a point are based on a step size dr.
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
    ## You can define various coil configs here and then just comment or 
    ## uncomment as needed.
    ## note in the example a coil pair can go from helmholtz to anti by changing
    ## the n vector of one of the pair.


    ##Large comp coil
    dr = .0254/15  
    dR = .0254/30
    r0 = np.asarray((0,0,0.3)) #inside separation
    r0[2] = r0[2] + dr/2.0
    R0 = .3      #inside rad
    R0 = R0 + dR/2.0
    Nr = 15
    NR = 10
    n = np.asarray([0,0,1])
    Ca1 = CoilArray(r0,R0,dr,Nr,dR,NR,n)
    Ca2 = CoilArray(-r0,R0,-dr,Nr,dR,NR,n)   
    Ca1.ohmspermeter = 16./300.
    Ca2.ohmspermeter = 16./300.
    Itest = 5
    Vmax = 15  
    
    ##weilei  coil
    #dr = .0254/8  
    #dR = .0254/8
    r0 = np.asarray((0,0,0.3)) #inside separation
    r0[2] = r0[2] + dr/2.0
    R0 = .3      #inside rad
    R0 = R0 + dR/2.0
    Nr = 15
    NR = 10
    n = np.asarray([0,0,1])
    Ca1 = CoilArray(r0,R0,dr,Nr,dR,NR,n)
    Ca2 = CoilArray(-r0,R0,-dr,Nr,dR,NR,n)   
    #Ca1.ohmspermeter = 16./300.
    #Ca2.ohmspermeter = 16./300.
    Itest = 5
    Vmax = 15  
    
    ##confinement coil for optical trap
    #dr = .0015
    #dR = .0015
    r0 = np.asarray((0,0,0.025)) #inside separation
    r0[2] = r0[2] + dr/2.0
    R0 = .025  #inside rad
    R0 = R0 + dR/2.0
    Nr = 25
    NR = 20
    n = np.asarray([0,0,1])
    Ca1 = CoilArray(r0,R0,dr,Nr,dR,NR,n)
    Ca2 = CoilArray(-r0,R0,-dr,Nr,dR,NR,n)   
    #Ca1.ohmspermeter = 16./300.
    #Ca2.ohmspermeter = 16./300.
    Itest = 5
    Vmax = 15  
    
    ##Nolan/Deep helm coil
    #dr = .0254/8  
    #dR = .0254/8
    #r0 = np.asarray((0,0,0.02210)) #inside separation
    #r0[2] = r0[2] + dr/2.0
    #R0 = .05015      #inside rad
    #R0 = R0 + dR/2.0
    #Nr = 5
    #NR = 6
    #n = np.asarray([0,0,1])
    #Ca1 = CoilArray(r0,R0,dr,Nr,dR,NR,n)
    #Ca2 = CoilArray(-r0,R0,-dr,Nr,dR,NR,n)     
   
    #Nolan/Deep antihelm coil
    #dr = .0254/8  
    #dR = .0254/8
    #r0 = np.asarray((0,0,0.02210)) #inside separation
    #r0[2] = r0[2] + dr/2.0
    #R0 = .03747      #inside diam
    #R0 = R0 + dR/2.0
    #Nr = 10
    #NR = 2
    #n = np.asarray([0,0,1])
    #Ca1 = CoilArray(r0,R0,dr,Nr,dR,NR,n)
    #Ca2 = CoilArray(-r0,R0,-dr,Nr,dR,NR,-n) 
    
   #Mariusz antihelm coil
    #dr = .0254/8  
    #dR = .0254/8
    #r0 = np.asarray((0,0,0.0127)) #inside separation
    #r0[2] = r0[2] + dr/2.0
    #R0 = .025      #inside diam
    #R0 = R0 + dR/2.0
    #Nr = 8
    #NR = 10
    #n = np.asarray([0,0,1])
    #Ca1 = CoilArray(r0,R0,dr,Nr,dR,NR,n)
    #Ca2 = CoilArray(-r0,R0,-dr,Nr,dR,NR,-n) 
    #Ca1.ohmspermeter = .003    
    #Ca2.ohmspermeter = .003    
    #Itest = 5
    #Vmax = 15

   #Li oven heating coil coil
    #dr = .002
    #dR = .002
    #r0 = np.asarray((0,0,0.0)) #inside separation
    #r0[2] = r0[2] + dr/2.0
    #R0 = .01      #inside diam
    #R0 = R0 + dR/2.0
    #Nr = 8
    #NR = 1
    #n = np.asarray([0,0,1])
    #Ca1 = CoilArray(r0,R0,dr,Nr,dR,NR,n)
    #Ca2 = CoilArray(r0,R0,-dr,Nr,dR,NR,n) 
    #Ca1.ohmspermeter = .003   
    #Ca2.ohmspermeter = .003   
    #Itest = 5
    #Vmax = 15
    
    #our coil small wire (this give 31.9 G/A which matches our calibration pretty well)
    dr = .002
    dR = .002
    r0 = np.asarray((0,0,0.025)) #position of center of coil1 bottom
    R0 = .05      #inner radius
    dr = .001
    dR = .00139
    r0[2] = r0[2] + dr/2.0 #effective separation
    R0 = R0 + dR/2 #effective radius
    Nr = 15
    NR = 15
    n = np.asarray([0,0,1])
    Ca1 = CoilArray(r0,R0,dr,Nr,dR,NR,n)
    Ca2 = CoilArray(-r0,R0,-dr,Nr,dR,NR,n) 
    Ca1.ohmspermeter = .003    
    Ca2.ohmspermeter = .003    
    Itest = 5
    Vmax = 15

   ##our coil
    #r0 = np.asarray((0,0,0.0250)) #position of center of coil1 bottom
    #R0 = .05        #inner radius
    #dr = .0254/8
    #dR = .0254/8
    #r0[2] = r0[2] + dr/2.0 #effective separation
    #R0 = R0 + dR/2 #effective radius
    #Nr = 5
    #NR = 12
    #n = np.asarray([0,0,1])
    #Ca1 = CoilArray(r0,R0,dr,Nr,dR,NR,n)
    #Ca2 = CoilArray(-r0,R0,-dr,Nr,dR,NR,-n) 
    #Itest = 220
    #Vmax = 46

    ###our fast coil
    #r0 = np.asarray((0,0,0.0250)) #position of center of coil1 bottom
    #R0 = .05        #inner radius
    #dr = .0254/8
    #dR = .0254/8
    #r0[2] = r0[2] + dr/2.0 #effective separation
    #R0 = R0 + dR/2 #effective radius
    #Nr = 5
    #NR = 2
    #n = np.asarray([0,0,1])
    #Ca1 = CoilArray(r0,R0,dr,Nr,dR,NR,n)
    #Ca2 = CoilArray(-r0,R0,-dr,Nr,dR,NR,-n) 
    #Itest = 300
    #Vmax = 46

    ###our fast coil2  on top of other one = 25mm + 16mm
    #r0 = np.asarray((0,0,0.0250+.016)) #position of center of coil1 bottom
    #R0 = .05        #inner radius
    #dr = .0254/8
    #dR = .0254/8
    #r0[2] = r0[2] + dr/2.0 #effective separation
    #R0 = R0 + dR/2 #effective radius
    #Nr = 2
    #NR = 5
    #n = np.asarray([0,0,1])
    #Ca1 = CoilArray(r0,R0,dr,Nr,dR,NR,n)
    #Ca2 = CoilArray(-r0,R0,-dr,Nr,dR,NR,-n) 
    #Itest = 300
    #Vmax = 46

    ##single coil radius R0
    #r0 = np.asarray((0,0,0.0250))
    #R0 = 1#.05
    #dr = .0254/8
    #dR = .0254/8
    #Nr = 1#5
    #NR = 1#6
    #n = np.asarray([0,0,1])
    #Ca1 = CoilArray(r0,R0,dr,Nr,dR,NR,n)
    #Ca2 = CoilArray(-r0,R0,-dr,Nr,dR,NR,n) 
    #Ca2 = CoilArray(-50*r0,50*R0,-dr,Nr,dR,1,n) 
    Bcenters=[]
    fs_trap=[]
    seps=[]
    #pylab.plot([1,2],[3,3.4])
    #pylab.plot(2,2)
    #pylab.show()
    for dsep in range(0,50,5):
	#our coil small wire (this give 31.9 G/A which matches our calibration pretty well)
	r0 = np.asarray((0,0,0.025+dsep/1000.)) #position of center of coil1 bottom
	R0 = .05      #inner radius
	dr = .001
	dR = .00139
	r0[2] = r0[2] + dr/2.0 #effective separation
	R0 = R0 + dR/2 #effective radius
	Nr = 15
	NR = 15
	n = np.asarray([0,0,1])
	Ca1 = CoilArray(r0,R0,dr,Nr,dR,NR,n)
	Ca2 = CoilArray(-r0,R0,-dr,Nr,dR,NR,n) 
	Ca1.ohmspermeter = .003    
	Ca2.ohmspermeter = .003    
	Itest = 5
	Vmax = 15
	
	#make grid
	N=3.0
	i=j=pylab.arange(N)
	n=float(N)
	x = ((i/(n-1))-.5)*.2
	z = ((j/(n-1))-.5)*.2
	X,Z = pylab.meshgrid(x,z)
	Bx = pylab.zeros_like(X)
	Bz = pylab.zeros_like(X)
	Bnorm = pylab.zeros_like(X)
	#get B field
	for ii in i:
	    for jj in j:
		x = X[ii,jj]
		z = Z[ii,jj]
		B = Ca1.Bvector([x,0,z])+Ca2.Bvector([x,0,z])
		Bnorm[ii,jj] = norm(B)
		#print B[0],B[0]*np.log1p(np.fabs(1000*norm(B)))
		Bx[ii,jj] = B[0]*np.log1p(np.fabs(1000*norm(B)))/norm(B)
		Bz[ii,jj] = B[2]*np.log1p(np.fabs(1000*norm(B)))/norm(B)
		#print '%.4f\t%.4f\t%s'%(x,z ,str(B))
	B = Ca1.Bvector([0,0,0])+Ca2.Bvector([0,0,0])
	print 'field at 0,0 = ',norm(B)
	Bcenters.append(norm(B))
	print Ca1.dimensions()
	print Ca2.dimensions()
	Imax = Vmax/(Ca1.resistance+Ca2.resistance)
	tau = Ca1.tau()
	t_test = time_to_I(Itest,tau,Imax)
	print 'time to current %.2f is %.6f\n'%(Itest,t_test)
	#get gradient
	gradientpt = [-.000,0,0]
	dbx,dbz,dBx,dBz = gradient(Ca1,Ca2,gradientpt,dr=.0001)
	print 'the gradient vectors at point:',gradientpt,'(m) are:'
	print 'dBx = ',dBx, 'in G/m'
	print 'dBz = ',dBz, 'in G/m'
	print 'the magnetude gradients in x and z are:'
	print 'dbx = %.2g,  dbz = %.2g (G/cm)'%(dbx/100.,dbz/100.)
	for x in np.arange(0,.014,.001):
	    gradientpt = [x,0,0]
	    dbx,dbz,dBx,dBz = gradient(Ca1,Ca2,gradientpt,dr=.0001)
	    print 'x = %.4f mm:  dbx = %.2g,  dbz = %.2g (G/cm)'%(x*1000,dbx/100.,dbz/100.)
	    
	print 'mu_B/(u*6)',(.0255/1e-4)*1e-4*mu_B/((u*6)*6.28**2),np.sqrt(.0255*1e-4*mu_B/(u*6))
	gradientpt = [0,0,0]
	dbx1,dbz,dBx,dBz = gradient(Ca1,Ca2,gradientpt,dr=.0001)
	gradientpt = [.001,0,0]
	dbx2,dbz,dBx,dBz = gradient(Ca1,Ca2,gradientpt,dr=.0001)
	Bcurv=(dbx2-dbx1)*10/100 # x10 for to convert to G/cm**2 because we used 1mm and /100 to convert grad to G/cm
	BcurvSalim=.0255*1200
	#Bcurv=BcurvSalim
	print 'B" in x (G/cm*2) =',dbx1/100,dbx2/100,Bcurv
	(Bcurv/1e-4)*1e-4*mu_B/((u*6)*6.28**2)
	f_trap=np.sqrt( (Bcurv/1e-4)*1e-4*mu_B/((u*6)*6.28**2) )
	print 'f_trap=',f_trap
	fs_trap.append(f_trap*10)
	print r0[2],norm(B)
	print r0[2],f_trap
	seps.append(r0[2]*2000)
    print Bcenters
    print fs_trap
    pylab.plot(seps,Bcenters,'gd',label='B at center G/A')
    pylab.plot(seps,fs_trap,'bo',label='trap frequency Hz@1A (x10) (goes as sqrt(A)')
    pylab.xlabel('vertical inner separation of coils mm')
    pylab.legend()
    pylab.show()
    
	
    #make figure
    pylab.figure()
    #quiverclip=.02
    #np.clip(Bx,-quiverclip,quiverclip,out=Bx)
    #np.clip(Bz,-quiverclip,quiverclip,out=Bz)
    #Bx=Bx*np.log(np.abs(Bx))
    #Bz=Bz*np.log(np.abs(Bz))
    Q = pylab.quiver( X, Z, Bx, Bz, units='width')
    qk = pylab.quiverkey(Q, 0.9, 0.95, 2, 'array'+r'$2 \frac{m}{s}$',
                labelpos='E',
                coordinates='figure',
                fontproperties={'weight': 'bold'})
    pylab.axis([-.1, .1, -.1, .1])
    #np.clip(Bnorm,0.,.1,out=Bnorm)
    CS = pylab.contour(X, Z, Bnorm,30,linewidths=0.5,colors='k')
    CS = pylab.contourf(X, Z, Bnorm,30,cmap=pylab.cm.jet)#,extent= (-.1,.1,-.1,.1)
    #pylab.xlim(-.1,.1)
    #pylab.ylim(-.1,.1)
    pylab.colorbar() # draw colorbar
    Q = pylab.quiver( X, Z, Bx, Bz, units='width')
    pylab.title('gauss per amp')
    pylab.show()
    
        
        
        