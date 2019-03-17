import numpy as np
from vertexF import vertexR
from propG import scaleProp
from math import factorial
import sys

def printBar(steP,current):
    """A status bar for duration of an fRG implementation."""
    width=20
    per=int(current*100)
    curWidth=int(current*width)
    remWidth=width-curWidth
    sys.stdout.write('Step: '+str(steP)+\
                         ' [%s]'%('#'*curWidth+ ' '*remWidth)+\
                         str(per)+'%\r')
    sys.stdout.flush()
    
def calcScale(xV,yV,cScaleS):
    """A simple estimate (via Newton's Method) of scale for an algebraic map."""
    diff=cScaleS
    tol=0.001

    deltaW=min(xV[1::])/2.0
    wPrev=xV[1:]-deltaW
    wPrev=np.append(xV[0],wPrev)
        
    uPrev=np.interp(wPrev,xV,yV)

    wFor=xV[:-1]+deltaW
    wFor=np.append(wFor,xV[-1])
    uFor=np.interp(wFor,xV,yV)
    
    uDer=(uFor-uPrev)/(wFor-wPrev)
    uDer=np.append(uDer,np.zeros(1))
    
    while diff>tol:

        zFull=np.linspace(0,1,100)
        (zFullX,gFullX)=gaussianInt(zFull,2)
    
        zPos=cScaleS*(zFullX+1)/(1-zFullX)
        zNeg=cScaleS*(-zFullX+1)/(1+zFullX)
        
        wMax=zPos.max()
        xVa=np.append(xV,wMax)
        
        uPos=np.interp(zPos,xVa,uDer)
        uNeg=np.interp(zNeg,xVa,uDer)

        uValu=np.append(yV,np.zeros(1))
        uSPos=np.interp(zPos,xVa,uValu)
        uSNeg=np.interp(zNeg,xVa,uValu)
        
        jaCBN=2.0/(1.0+zFullX)**2
        jaCBP=2.0/(1.0-zFullX)**2
        
        signU=np.sign(np.sum(gFullX*(uSPos*jaCBP*cScaleS-uSNeg*jaCBN*cScaleS)))
        
        fXv=np.abs(np.sum(gFullX*(jaCBP*uSPos*cScaleS-jaCBN*uSNeg*cScaleS)))
        fDerv=signU*np.sum(gFullX*(cScaleS*jaCBP*uPos*(1+zFullX)/(1-zFullX)-\
                                 cScaleS*jaCBN*uNeg*(-zFullX+1)/(1+zFullX)+\
                                       jaCBP*uSPos-jaCBN*uSNeg))
        
        cScale=cScaleS-fXv/(fDerv+tol/2)
        diff=np.abs(cScale-cScaleS)

        cScaleS=cScale
    return cScale
def linInterp(xI,yI,xQ):
    """A linear interpolator along 0 dimension of a 3-d array"""
    y1=yI.shape[1]
    y2=yI.shape[2]

    yQ = np.zeros((len(xQ),y1,y2),dtype=np.complex_)
    for i in range(y1):
        for j in range(y2):
            yQ[:,i,j]=np.interp(xQ,xI,yI[:,i,j].real)
            yQ[:,i,j]+=1j*np.interp(xQ,xI,yI[:,i,j].imag)
    return yQ
def linInterp2(xI,yI,xQ):
    """A vectorized linear interpolator along 0 dimension of a 3-d array"""
    xLQ=len(xQ)
    xL=len(xI)
    yL=yI.size/(len(xI))
    yL1=len(yI[0,:,0])
    yL2=len(yI[0,0,:])

    yI2=np.reshape(yI,(xL,yL))    
    yI2=np.concatenate((yI2,yI2[-1::,...]),axis=0)
    xQI=np.searchsorted(xI,xQ)

    y1=yI2[xQI-1,:]
    y2=yI2[xQI,:]
    
    xI=np.append(xI,max(xQ))
    x1=xI[xQI-1]
    x1=np.tile(x1[:,np.newaxis],(1,yL))
    x2=xI[xQI]
    x2=np.tile(x2[:,np.newaxis],(1,yL))
    
    xQE=np.tile(xQ[:,np.newaxis],(1,yL))
    yQ=np.divide(np.multiply(xQE-x1,y1-y2),(x1-x2))+y1
    
    return np.reshape(yQ,(xLQ,yL1,yL2))
    
def padeWeights(wMax,beTa):
    """Okazaki matsubara frequencies and weights for fermionic matsubara sums"""
    tol=0.01
    xV=beTa*wMax*2
    eRR=np.tanh(xV/2.0)
    iMax=10
    while eRR>tol: 
        bB=np.zeros((iMax,iMax))
        for i in range(iMax-1):
            bB[i,i+1]=1.0/(2.0*np.sqrt((2.0*(i+1)-1.0)*(2.0*(i+1)+1.0)))
            bB[i+1,i]=bB[i,i+1]

        bEig,aV=np.linalg.eig(bB)
        freQ=1.0/bEig
        wghT=np.zeros(iMax)
        for i in range(iMax):
            wghT[i]=(aV[0,i]**2)/(4*(bEig[i]**2))

            fW=zip(np.abs(freQ),freQ,wghT)
            fW=sorted(fW)
        for i in range(iMax):
            (a,freQ[i],wghT[i])=fW[i]
        est=0.0
        for i in range(iMax):
            est+=(2*wghT[i]*xV)/(xV**2+freQ[i]**2)
        eRR=np.abs(est-np.tanh(xV/2.0))
        
        iMax+=2
    return np.abs(freQ[::2])/beTa,wghT[::2]


def hermiteExpansion(wV,j):
    """Orthogonal Hermite functions."""
    lInJ=np.zeros(j+1)
    lInJ[j]=1
    
    return (1/np.sqrt(np.sqrt(np.pi)*(2**j)*np.math.factorial(j)))*np.exp(-0.5*wV**2)*np.polynomial.hermite.hermval(wV,lInJ)
def freqExpansion(zV,j):
    """Orthonormal Legendre functions."""
    lInJ=np.zeros(j+1)
    lInJ[j]=1
    
    return np.sqrt((2*j+1)/2.0)*np.polynomial.legendre.legval(zV,lInJ)

def gaussianInt(xGrid,nG):
    """Gauss-Legendre points and weights for smooth integrals."""
    lIn=np.zeros(nG+1)
    lIn[nG]=1

    xI=np.polynomial.legendre.legroots(lIn)
    dlIn=np.polynomial.legendre.legder(lIn)
    dLi=np.polynomial.legendre.legval(xI,dlIn)
    wI=2.0/((1-xI**2)*(dLi**2))

    xG=np.zeros(nG*(len(xGrid)-1))
    wG=np.zeros(nG*(len(xGrid)-1))

    for i in range(len(xGrid)-1):
        xG[(i*nG):((i+1)*nG)]=(0.5*(xGrid[i+1]-xGrid[i])*xI+\
                                   0.5*(xGrid[i+1]+xGrid[i]))
        wG[(i*nG):((i+1)*nG)]=0.5*(xGrid[i+1]-xGrid[i])*wI

    return xG,wG

def freqPoints(beTa,wMax,nPoints):
    """Logarthmic set of fermionic and bosonic matsubara frequencies."""
    nMax=np.floor(0.5*((wMax*beTa/np.pi)-1))

    wTF=(np.pi/beTa)*(2*np.arange(0,nMax,1)+1)
    wTB=(np.pi/beTa)*2*np.arange(0,nMax,1)

    wIndex=np.logspace(0,np.log(len(wTF))/np.log(10),nPoints)-1
    wIndex=np.unique(wIndex.astype(int))
    
    wF=wTF[wIndex]
    wB=wTB[wIndex]

    i=1
    while len(wF)<nPoints:
        if len(wF)<len(wTF):
            wA=wIndex+i
            wA=wA[wA<len(wTF)]
            
            i+=1
            lEnd=min([len(wA),nPoints-len(wF)])
            wF=np.unique(np.append(wF,wTF[wA[:lEnd]]))
        else:
            wA=(np.pi/beTa)*(2*np.arange(nMax,nMax+nPoints-len(wF),1)+1)
            wF=np.append(wF,wA)

    i=1
    while len(wB)<nPoints:
        if len(wB)<len(wTB):
            wA=wIndex+i
            wA=wA[wA<len(wTB)]
            
            i+=1
            lEnd=min([len(wA),nPoints-len(wB)])
            wB=np.unique(np.append(wB,wTB[wA[:lEnd]]))
        else:
            wA=(np.pi/beTa)*2*np.arange(nMax,nMax+nPoints-len(wB),1)
            wB=np.append(wB,wA)


    return wF,wB
def forMap(wV,cScale):
    """Algebraic map to finite domain."""
    return wV/np.sqrt(wV**2+cScale)
def backMap(zV,cScale):
    """Inverse algebraic map to unbounded domain."""
    return zV/np.sqrt(cScale-zV**2)

def aScale(lC,A0):
    return A0*np.exp(-lC)

def litim0(wQ,AC):
    """Litim Regulator for frequency domain."""
    stepR=0.5*(np.sign(AC-np.abs(wQ))+1)
    return 1j*(np.sign(wQ)*AC-wQ)*stepR

def dLitim0(wQ,AC):
    """Scale derivative of Litim regulator."""
    stepR=0.5*(np.sign(AC-np.abs(wQ))+1)
    return 1j*np.sign(wQ)*stepR

def additiveR0(wQ,AC):
    """Smooth additive regualtor for frequency domain."""
    return 1j*(np.sign(wQ)*np.sqrt(wQ**2+AC**2)-wQ)

def dAdditiveR0(wQ,AC):
    """Scale derivative of additive regulator."""
    return (1j*np.sign(wQ)*AC)/np.sqrt(wQ**2+AC**2)

def sharpR(wQ,AC):
    """A smoothed hard regulator for frequency domain."""
    kSm=1
    return (1-np.exp(-(np.abs(wQ)/AC)**kSm))

def dSharpR(wQ,AC):
    """Scale derivative of sharp regulator."""
    kSm=1
    return ((kSm*np.abs(wQ)**kSm)/AC**(kSm+1))*np.exp(-(np.abs(wQ)/AC)**kSm)

def softR(wQ,AC):
    """A soft regulator for frequency domain."""
    return (wQ**2+AC**2)/(wQ**2)

def dSoftR(wQ,AC):
    """Scale derivative of smooth regulator."""
    return 2*AC/(wQ**2)

def projectedVertex(UnX,UnF,AC,flag=None):
    """Calculates projected contributions to each channel."""

    #Expand vertices in a basis set
    #for projection across channels
    UnPPI=UnF.legndExpand(UnX[0],AC)
    UnPHI=UnF.legndExpand(UnX[1],AC)
    UnPHEI=UnF.legndExpand(UnX[2],AC)

    if flag is 'b':
        UnF.UnPPI=UnPPI
        UnF.UnPHI=UnPHI
        UnF.UnPHEI=UnPHEI

    #Projection from each channel
    #to the other channels
    uPPtoPH,uPPtoPHE=UnF.projectChannel(UnPPI,AC,'PP')
    uPHtoPP,uPHtoPHE=UnF.projectChannel(UnPHI,AC,'PH')
    uPHEtoPP,uPHEtoPH=UnF.projectChannel(UnPHEI,AC,'PHE')

    uPP=uPHtoPP+uPHEtoPP
    uPH=uPPtoPH+uPHEtoPH
    uPHE=uPPtoPHE+uPHtoPHE
    
    return uPP,uPH,uPHE
def basisDerv(UnF,AC):
    """Additional vertex for a scale dependent basis set"""
    scaleD=UnF.scaleDerv

    UnPPs=(AC/(AC**2+1))*np.sum(UnF.UnPP[:,:,:,None,None]*scaleD[None,...],axis=(1,2))
    UnPHs=(AC/(AC**2+1))*np.sum(UnF.UnPH[:,:,:,None,None]*scaleD[None,...],axis=(1,2))
    UnPHEs=(AC/(AC**2+1))*np.sum(UnF.UnPHE[:,:,:,None,None]*scaleD[None,...],axis=(1,2))
    
    return UnPPs,UnPHs,UnPHEs
def betaF(UnF,propT,AC,nL):
    """One-loop beta function for decoupled fRG."""
    
    #Project every vertex across channels
    #to construct full vertex
    UnX=(UnF.UnPP,UnF.UnPH,UnF.UnPHE)
    uPP,uPH,uPHE=projectedVertex(UnX,UnF,AC,'b')

    UnPPX=UnF.UnPPO+UnF.UnPP+uPP
    UnPHX=UnF.UnPHO+UnF.UnPH+uPH
    UnPHEX=UnF.UnPHEO+UnF.UnPHE+uPHE

    dSE=calcSE(UnF,propT,AC)
    mixPP, mixPH=propT.xBubbles(UnF.wB,dSE,AC,UnF.NW)    
    
    #Contributions from scale dependent basis functions
    UnPPs,UnPHs,UnPHEs=basisDerv(UnF,AC)
    
    dUnPP=UnPPs-np.matmul(UnPPX,np.matmul(mixPP,UnPPX))
    dUnPH=UnPHs+np.matmul(UnPHX-UnPHEX,np.matmul(mixPH,UnPHX))+\
        np.matmul(UnPHX,np.matmul(mixPH,UnPHX-UnPHEX))
    
    dUnPHE=UnPHEs-np.matmul(UnPHEX,np.matmul(mixPH,UnPHEX))

    return -AC*dSE, -AC*dUnPP, -AC*dUnPH, -AC*dUnPHE
def betaF2L(UnF,propT,AC,nL):
    """Two-loop beta function for decoupled fRG."""
    
    #Project every vertex across channels
    #to construct full vertex
    UnX=(UnF.UnPP,UnF.UnPH,UnF.UnPHE)
    uPP,uPH,uPHE=projectedVertex(UnX,UnF,AC,'b')

    UnPPX=UnF.UnPPO+UnF.UnPP+uPP
    UnPHX=UnF.UnPHO+UnF.UnPH+uPH
    UnPHEX=UnF.UnPHEO+UnF.UnPHE+uPHE


    gPP, gPH=propT.gBubbles(UnF.wB,AC,UnF.NW)
    dSE=calcSE(UnF,propT,AC)
    mixPP, mixPH=propT.xBubbles(UnF.wB,dSE,AC,UnF.NW) 
    
    dUnPP=-np.matmul(UnPPX,np.matmul(mixPP,UnPPX))
    dUnPH=np.matmul(UnPHX-UnPHEX,np.matmul(mixPH,UnPHX))+\
        np.matmul(UnPHX,np.matmul(mixPH,UnPHX-UnPHEX))
    dUnPHE=-np.matmul(UnPHEX,np.matmul(mixPH,UnPHEX))
    
    #Two-Loop contributions from 
    #scale derivative of vertex
    UnX=(dUnPP,dUnPH,dUnPHE)
    dUnPPX,dUnPHX,dUnPHEX=projectedVertex(UnX,UnF,AC)
    
    UnPPs,UnPHs,UnPHEs=basisDerv(UnF,AC)

    dUnPP+=UnPPs-np.matmul(np.matmul(dUnPPX,gPP),UnPPX)-\
        np.matmul(UnPPX,np.matmul(gPP,dUnPPX))
    dUnPH+=UnPHs+np.matmul(np.matmul(dUnPHX-dUnPHEX,gPH),UnPHX)+\
        np.matmul(UnPHX-UnPHEX,np.matmul(gPH,dUnPHX))+\
        np.matmul(UnPHX,np.matmul(gPH,dUnPHX-dUnPHEX))+\
        np.matmul(dUnPHX,np.matmul(gPH,UnPHX-UnPHEX))

    dUnPHE+=UnPHEs-np.matmul(UnPHEX,np.matmul(gPH,dUnPHEX))-\
        np.matmul(dUnPHEX,np.matmul(gPH,UnPHEX))   
    
    return -AC*dSE, -AC*dUnPP, -AC*dUnPH, -AC*dUnPHE

def betaFNL(UnF,propT,AC,nL):
    """Multi-Loop beta function for decoupled fRG."""

    UnX=(UnF.UnPP,UnF.UnPH,UnF.UnPHE)
    uPP,uPH,uPHE=projectedVertex(UnX,UnF,AC,'b')

    UnPPX=UnF.UnPPO+UnF.UnPP+uPP
    UnPHX=UnF.UnPHO+UnF.UnPH+uPH
    UnPHEX=UnF.UnPHEO+UnF.UnPHE+uPHE

    gPP, gPH=propT.gBubbles(UnF.wB,AC,UnF.NW)
    dSE=calcSE(UnF,propT,AC)
    mixPP, mixPH=propT.xBubbles(UnF.wB,dSE,AC,UnF.NW)

    dUnPP=-np.matmul(UnPPX,np.matmul(mixPP,UnPPX))
    dUnPH=np.matmul(UnPHX-UnPHEX,np.matmul(mixPH,UnPHX))+\
        np.matmul(UnPHX,np.matmul(mixPH,UnPHX-UnPHEX))
    dUnPHE=-np.matmul(UnPHEX,np.matmul(mixPH,UnPHEX))
     
    UnX=(dUnPP,dUnPH,dUnPHE)
    dUnPP2X,dUnPH2X,dUnPHE2X=projectedVertex(UnX,UnF,AC)
    
    dUnPPL=-np.matmul(np.matmul(dUnPP2X,gPP),UnPPX)
    dUnPPR=-np.matmul(UnPPX,np.matmul(gPP,dUnPP2X))
    dUnPPC=dUnPPL+dUnPPR

    dUnPHL=np.matmul(np.matmul(dUnPH2X-dUnPHE2X,gPH),UnPHX)+\
        np.matmul(dUnPH2X,np.matmul(gPH,UnPHX-UnPHEX))
    dUnPHL2=np.matmul(np.matmul(dUnPH2X-dUnPHE2X,gPH),UnPHX-UnPHEX)
    dUnPHR=np.matmul(np.matmul(UnPHX-UnPHEX,gPH),dUnPH2X)+\
        np.matmul(np.matmul(UnPHX,gPH),dUnPH2X-dUnPHE2X)
    dUnPHC=dUnPHL+dUnPHR

    dUnPHEL=-np.matmul(np.matmul(dUnPHE2X,gPH),UnPHEX)
    dUnPHER=-np.matmul(np.matmul(UnPHEX,gPH),dUnPHE2X)
    dUnPHEC=dUnPHEL+dUnPHER

    dUnPP+=dUnPPC
    dUnPH+=dUnPHC
    dUnPHE+=dUnPHEC

    dUnPPSEc=np.zeros(dUnPP.shape,dtype=np.complex_)
    dUnPHESEc=np.zeros(dUnPP.shape,dtype=np.complex_)
    for i in range(nL-2):
        UnX=(dUnPPX,dUnPHX,dUnPHEC)
        dUnPPNX,dUnPHNX,dUnPHENX==projectedVertex(UnX,UnF,AC)
 
        dUnPPc=-np.matmul(np.matmul(UnPPX,gPP),dUnPPL)
        dUnPPL=-np.matmul(np.matmul(dUnPPNX,gPP),UnPPX)
        dUnPPR=-np.matmul(np.matmul(UnPPX,gPP),dUnPPNX)
        dUnPPC=dUnPPc+dUnPPL+dUnPPR
        dUnPPSEc+=dUnPPc

        dUnPHc=np.matmul(np.matmul(UnPHX,gPH),dUnPHL2)+\
            np.matmul(np.matmul(UnPHX-UnPHEX,gPH),dUnPHL)
        dUnPHL=np.matmul(np.matmul(dUnPHNX-dUnPHENX,gPH),UnPHX)+\
            np.matmul(np.matmul(dUnPHNX,gPH),UnPHX-UnPHEX)
        dUnPHL2=np.matmul(np.matmul(dUnPHNX-dUnPHENX,gPH),UnPHX-UnPHEX)
        dUnPHR=np.matmul(np.matmul(UnPHX-UnPHEX,gPH),dUnPHNX)+\
            np.matmul(np.matmul(UnPHX,gPH),dUnPHNX-dUnPHENX)
        dUnPHC=dUnPHc+dUnPHL+dUnPHR

        dUnPHEc=-np.matmul(np.matmul(UnPHEX,gPH),dUnPHEL)
        dUnPHEL=-np.matmul(np.matmul(dUnPHENX,gPH),UnPHEX)
        dUnPHER=-np.matmul(np.matmul(UnPHEX,gPH),dUnPHENX)
        dUnPHEC=dUnPHEc+dUnPHEL+dUnPHER
        dUnPHESEc+=dUnPHEc

        dUnPP+=dUnPPC
        dUnPH+=dUnPHC
        dUnPHE+=dUnPHEC

    UnPPs,UnPHs,UnPHEs=basisDerv(UnF,AC)

    dUnPP+=UnPPs
    dUnPH+=UnPHs
    dUnPHE+=UnPHEs
    
    return -AC*dSE, -AC*dUnPP, -AC*dUnPH, -AC*dUnPHE

def calcSE(UnF,propT,AC):
    """Evaluates the scale derivative of the self energy."""
    wF=propT.wF
    wFX=propT.wFX
    wFG=propT.wFG

    (wFE,wFXE)=np.meshgrid(wF,wFX,indexing='ij')

    wPP=wFE+wFXE
    wPHZ=wFXE-wFXE
    wPH=wFXE-wFE
    
    uV1=UnF.uEvaluate(wPP,wPHZ,wPH,AC)
    uV2=UnF.uEvaluate(wPP,wPH,wPHZ,AC)

    sProp=wFG*propT.sF(wFX,AC)
    
    sPropE=np.tile(sProp[np.newaxis,:],(len(wF),1))
    dSE=(1.0/propT.beTa)*np.sum((2.0*uV1-uV2)*sPropE,axis=1)
    
    return dSE

