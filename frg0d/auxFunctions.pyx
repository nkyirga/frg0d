import numpy as np
cimport numpy as np
from vertexF import vertexR
from propG import scaleProp
from math import factorial
import time

def calcScale(xV,yV,cScaleS):
    
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
cpdef linInterp(np.ndarray[np.float64_t,ndim=1] xI,
                np.ndarray[np.complex128_t,ndim=3] yI,
                np.ndarray[np.float64_t,ndim=1] xQ):

    cdef int y1=yI.shape[1]
    cdef int y2=yI.shape[2]

    cdef np.ndarray[np.complex128_t,ndim=3] yQ = np.zeros((len(xQ),y1,y2),dtype=np.complex_)
    cdef int i,j
    for i in range(y1):
        for j in range(y2):
            yQ[:,i,j]=np.interp(xQ,xI,yI[:,i,j].real)
            yQ[:,i,j]+=1j*np.interp(xQ,xI,yI[:,i,j].imag)
    return yQ
def linInterp2(np.ndarray[double,ndim=1] xI,np.ndarray[np.complex_t,ndim=3] yI,np.ndarray[double,ndim=1] xQ):
    cdef int xLQ=len(xQ)
    cdef int xL=len(xI)
    cdef int yL=yI.size/(len(xI))
    cdef int yL1=len(yI[0,:,0])
    cdef int yL2=len(yI[0,0,:])

    cdef np.ndarray[np.complex_t,ndim=2] yI2=np.reshape(yI,(xL,yL))    
    yI2=np.concatenate((yI2,yI2[-1::,...]),axis=0)
    cdef np.ndarray[long,ndim=1] xQI=np.searchsorted(xI,xQ)

    cdef np.ndarray[np.complex_t,ndim=2] y1=yI2[xQI-1,:]
    cdef np.ndarray[np.complex_t,ndim=2] y2=yI2[xQI,:]
    
    xI=np.append(xI,max(xQ))
    x1=xI[xQI-1]
    x1=np.tile(x1[:,np.newaxis],(1,yL))
    x2=xI[xQI]
    x2=np.tile(x2[:,np.newaxis],(1,yL))
    
    cdef np.ndarray[double,ndim=2] xQE=np.tile(xQ[:,np.newaxis],(1,yL))
    cdef np.ndarray[np.complex_t,ndim=2] yQ=np.divide(np.multiply(xQE-x1,y1-y2),(x1-x2))+y1
    
    return np.reshape(yQ,(xLQ,yL1,yL2))
    
def padeWeights(wMax,beTa):
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
    lInJ=np.zeros(j+1)
    lInJ[j]=1
    
    return (1/np.sqrt(np.sqrt(np.pi)*(2**j)*np.math.factorial(j)))*np.exp(-0.5*wV**2)*np.polynomial.hermite.hermval(wV,lInJ)
def freqExpansion(zV,j):
    lInJ=np.zeros(j+1)
    lInJ[j]=1
    
    return np.sqrt((2*j+1)/2.0)*np.polynomial.legendre.legval(zV,lInJ)

def gaussianInt(xGrid,nG):
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
    return wV/np.sqrt(wV**2+cScale)
def backMap(zV,cScale):
    return zV/np.sqrt(cScale-zV**2)

def aScale(lC,A0):
    return A0*np.exp(-lC)

def litim0(wQ,AC):
    stepR=0.5*(np.sign(AC-np.abs(wQ))+1)
    return 1j*(np.sign(wQ)*AC-wQ)*stepR

def dLitim0(wQ,AC):
    stepR=0.5*(np.sign(AC-np.abs(wQ))+1)
    return 1j*np.sign(wQ)*stepR

def additiveR0(wQ,AC):
    return 1j*(np.sign(wQ)*np.sqrt(wQ**2+AC**2)-wQ)

def dAdditiveR0(wQ,AC):
    return (1j*np.sign(wQ)*AC)/np.sqrt(wQ**2+AC**2)

def sharpR(wQ,AC):
    kSm=1
    return (1-np.exp(-(np.abs(wQ)/AC)**kSm))

def dSharpR(wQ,AC):
    kSm=1
    return ((kSm*np.abs(wQ)**kSm)/AC**(kSm+1))*np.exp(-(np.abs(wQ)/AC)**kSm)

def softR(wQ,AC):
    return (wQ**2+AC**2)/(wQ**2)

def dSoftR(wQ,AC):
    return 2*AC/(wQ**2)

def betaF(UnF,propT,AC,nL):
    UnF.UnPPI=UnF.legndExpand(UnF.UnPP,AC)
    UnF.UnPHI=UnF.legndExpand(UnF.UnPH,AC)
    UnF.UnPHEI=UnF.legndExpand(UnF.UnPHE,AC)

    uPPtoPH,uPPtoPHE=UnF.projectChannel(UnF.UnPPI,AC,'PP')
    uPHtoPP,uPHtoPHE=UnF.projectChannel(UnF.UnPHI,AC,'PH')
    uPHEtoPP,uPHEtoPH=UnF.projectChannel(UnF.UnPHEI,AC,'PHE')

    dSE=calcSE(UnF,propT,AC)
    mixPP, mixPH=propT.xBubbles(UnF.wB,dSE,AC,UnF.NW)
    
    UnPPX=UnF.UnPPO+UnF.UnPP+uPHtoPP+uPHEtoPP
    UnPHX=UnF.UnPHO+UnF.UnPH+uPPtoPH+uPHEtoPH
    UnPHEX=UnF.UnPHEO+UnF.UnPHE+uPPtoPHE+uPHtoPHE

    scaleD=UnF.scaleDerv
    scaleF=np.tile(scaleD[np.newaxis,:,:,:,:],(len(UnF.wB),1,1,1,1))
    UnPPs=np.tile(UnF.UnPP[:,:,:,np.newaxis,np.newaxis],(1,1,1,UnF.NW,UnF.NW))
    UnPHs=np.tile(UnF.UnPH[:,:,:,np.newaxis,np.newaxis],(1,1,1,UnF.NW,UnF.NW))
    UnPHEs=np.tile(UnF.UnPHE[:,:,:,np.newaxis,np.newaxis],(1,1,1,UnF.NW,UnF.NW))

    UnPPs=(AC/(AC**2+1))*np.sum(UnPPs*scaleF,axis=(1,2))
    UnPHs=(AC/(AC**2+1))*np.sum(UnPHs*scaleF,axis=(1,2))
    UnPHEs=(AC/(AC**2+1))*np.sum(UnPHEs*scaleF,axis=(1,2))

    dUnPP=UnPPs-np.matmul(UnPPX,np.matmul(mixPP,UnPPX))
    dUnPH=UnPHs+np.matmul(UnPHX-UnPHEX,np.matmul(mixPH,UnPHX))+\
        np.matmul(UnPHX,np.matmul(mixPH,UnPHX-UnPHEX))
    
    dUnPHE=UnPHEs-np.matmul(UnPHEX,np.matmul(mixPH,UnPHEX))

    return -AC*dSE, -AC*dUnPP, -AC*dUnPH, -AC*dUnPHE
def betaF2L(UnF,propT,AC,nL):
    UnF.UnPPI=UnF.legndExpand(UnF.UnPP,AC)
    UnF.UnPHI=UnF.legndExpand(UnF.UnPH,AC)
    UnF.UnPHEI=UnF.legndExpand(UnF.UnPHE,AC)

    uPPtoPH,uPPtoPHE=UnF.projectChannel(UnF.UnPPI,AC,'PP')
    uPHtoPP,uPHtoPHE=UnF.projectChannel(UnF.UnPHI,AC,'PH')
    uPHEtoPP,uPHEtoPH=UnF.projectChannel(UnF.UnPHEI,AC,'PHE')

    dSE=calcSE(UnF,propT,AC)
    mixPP, mixPH=propT.xBubbles(UnF.wB,dSE,AC,UnF.NW)

    UnPPX=UnF.UnPPO+UnF.UnPP+uPHtoPP+uPHEtoPP
    UnPHX=UnF.UnPHO+UnF.UnPH+uPPtoPH+uPHEtoPH
    UnPHEX=UnF.UnPHEO+UnF.UnPHE+uPPtoPHE+uPHtoPHE

    dUnPP=-np.matmul(UnPPX,np.matmul(mixPP,UnPPX))
    dUnPH=np.matmul(UnPHX-UnPHEX,np.matmul(mixPH,UnPHX))+\
        np.matmul(UnPHX,np.matmul(mixPH,UnPHX-UnPHEX))
    
    dUnPHE=-np.matmul(UnPHEX,np.matmul(mixPH,UnPHEX))
    
    dUnPPI=UnF.legndExpand(dUnPP,AC)
    dUnPHI=UnF.legndExpand(dUnPH,AC)
    dUnPHEI=UnF.legndExpand(dUnPHE,AC)

    uPPtoPH2,uPPtoPHE2=UnF.projectChannel(dUnPPI,AC,'PP')
    uPHtoPP2,uPHtoPHE2=UnF.projectChannel(dUnPHI,AC,'PH')
    uPHEtoPP2,uPHEtoPH2=UnF.projectChannel(dUnPHEI,AC,'PHE')
   
    dUnPPX=uPHtoPP2+uPHEtoPP2
    dUnPHX=uPPtoPH2+uPHEtoPH2
    dUnPHEX=uPPtoPHE2+uPHtoPHE2
    
    gPP, gPH=propT.gBubbles(UnF.wB,AC,UnF.NW)
  
    scaleD=UnF.scaleDerv
    scaleF=np.tile(scaleD[np.newaxis,:,:,:,:],(len(UnF.wB),1,1,1,1,))
    UnPPs=np.tile(UnF.UnPP[:,:,:,np.newaxis,np.newaxis],(1,1,1,UnF.NW,UnF.NW))
    UnPHs=np.tile(UnF.UnPH[:,:,:,np.newaxis,np.newaxis],(1,1,1,UnF.NW,UnF.NW))
    UnPHEs=np.tile(UnF.UnPHE[:,:,:,np.newaxis,np.newaxis],(1,1,1,UnF.NW,UnF.NW))

    UnPPs=(AC/(AC**2+1))*np.sum(UnPPs*scaleF,axis=(1,2))
    UnPHs=(AC/(AC**2+1))*np.sum(UnPHs*scaleF,axis=(1,2))
    UnPHEs=(AC/(AC**2+1))*np.sum(UnPHEs*scaleF,axis=(1,2))

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
    UnF.UnPPI=UnF.legndExpand(UnF.UnPP,AC)
    UnF.UnPHI=UnF.legndExpand(UnF.UnPH,AC)
    UnF.UnPHEI=UnF.legndExpand(UnF.UnPHE,AC)

    uPPtoPH,uPPtoPHE=UnF.projectChannel(UnF.UnPPI,AC,'PP')
    uPHtoPP,uPHtoPHE=UnF.projectChannel(UnF.UnPHI,AC,'PH')
    uPHEtoPP,uPHEtoPH=UnF.projectChannel(UnF.UnPHEI,AC,'PHE')

    dSE=calcSE(UnF,propT,AC)
    mixPP, mixPH=propT.xBubbles(UnF.wB,dSE,AC,UnF.NW)
    gPP, gPH=propT.gBubbles(UnF.wB,AC,UnF.NW)
  
    UnPPX=UnF.UnPPO+UnF.UnPP+uPHtoPP+uPHEtoPP
    UnPHX=UnF.UnPHO+UnF.UnPH+uPPtoPH+uPHEtoPH
    UnPHEX=UnF.UnPHEO+UnF.UnPHE+uPPtoPHE+uPHtoPHE

    dUnPP=-np.matmul(UnPPX,np.matmul(mixPP,UnPPX))
    dUnPH=np.matmul(UnPHX-UnPHEX,np.matmul(mixPH,UnPHX))+\
        np.matmul(UnPHX,np.matmul(mixPH,UnPHX-UnPHEX))
    
    dUnPHE=-np.matmul(UnPHEX,np.matmul(mixPH,UnPHEX))
    
    dUnPPI=UnF.legndExpand(dUnPP,AC)
    dUnPHI=UnF.legndExpand(dUnPH,AC)
    dUnPHEI=UnF.legndExpand(dUnPHE,AC)

    uPPtoPH2,uPPtoPHE2=UnF.projectChannel(dUnPPI,AC,'PP')
    uPHtoPP2,uPHtoPHE2=UnF.projectChannel(dUnPHI,AC,'PH')
    uPHEtoPP2,uPHEtoPH2=UnF.projectChannel(dUnPHEI,AC,'PHE')
   
    dUnPP2X=uPHtoPP2+uPHEtoPP2
    dUnPH2X=uPPtoPH2+uPHEtoPH2
    dUnPHE2X=uPPtoPHE2+uPHtoPHE2
    
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

    for i in range(nL-2):
        dUnPPI=UnF.legndExpand(dUnPPC,AC)
        dUnPHI=UnF.legndExpand(dUnPHC,AC)
        dUnPHEI=UnF.legndExpand(dUnPHEC,AC)

        uPPtoPHN,uPPtoPHEN=UnF.projectChannel(dUnPPI,AC,'PP')
        uPHtoPPN,uPHtoPHEN=UnF.projectChannel(dUnPHI,AC,'PH')
        uPHEtoPPN,uPHEtoPHN=UnF.projectChannel(dUnPHEI,AC,'PHE')
      
        dUnPPNX=uPHtoPPN+uPHEtoPPN
        dUnPHNX=uPPtoPHN+uPHEtoPHN
        dUnPHENX=uPPtoPHEN+uPHtoPHEN
    
        dUnPPc=-np.matmul(np.matmul(UnPPX,gPP),dUnPPL)
        dUnPPL=-np.matmul(np.matmul(dUnPPNX,gPP),UnPPX)
        dUnPPR=-np.matmul(np.matmul(UnPPX,gPP),dUnPPNX)
        dUnPPC=0*dUnPPc+dUnPPL+dUnPPR

        dUnPHc=np.matmul(np.matmul(UnPHX,gPH),dUnPHL2)+\
            np.matmul(np.matmul(UnPHX-UnPHEX,gPH),dUnPHL)
        dUnPHL=np.matmul(np.matmul(dUnPHNX-dUnPHENX,gPH),UnPHX)+\
            np.matmul(np.matmul(dUnPHNX,gPH),UnPHX-UnPHEX)
        dUnPHL2=np.matmul(np.matmul(dUnPHNX-dUnPHENX,gPH),UnPHX-UnPHEX)
        dUnPHR=np.matmul(np.matmul(UnPHX-UnPHEX,gPH),dUnPHNX)+\
            np.matmul(np.matmul(UnPHX,gPH),dUnPHNX-dUnPHENX)
        dUnPHC=0*dUnPHc+dUnPHL+dUnPHR

        dUnPHEc=-np.matmul(np.matmul(UnPHEX,gPH),dUnPHEL)
        dUnPHEL=-np.matmul(np.matmul(dUnPHENX,gPH),UnPHEX)
        dUnPHER=-np.matmul(np.matmul(UnPHEX,gPH),dUnPHENX)
        dUnPHEC=0*dUnPHEc+dUnPHEL+dUnPHER

        dUnPP+=dUnPPC
        dUnPH+=dUnPHC
        dUnPHE+=dUnPHEC

    scaleD=UnF.scaleDerv
    scaleF=np.tile(scaleD[np.newaxis,:,:,:,:],(len(UnF.wB),1,1,1,1,))
    UnPPs=np.tile(UnF.UnPP[:,:,:,np.newaxis,np.newaxis],(1,1,1,UnF.NW,UnF.NW))
    UnPHs=np.tile(UnF.UnPH[:,:,:,np.newaxis,np.newaxis],(1,1,1,UnF.NW,UnF.NW))
    UnPHEs=np.tile(UnF.UnPHE[:,:,:,np.newaxis,np.newaxis],(1,1,1,UnF.NW,UnF.NW))

    UnPPs=(AC/(AC**2+1))*np.sum(UnPPs*scaleF,axis=(1,2))
    UnPHs=(AC/(AC**2+1))*np.sum(UnPHs*scaleF,axis=(1,2))
    UnPHEs=(AC/(AC**2+1))*np.sum(UnPHEs*scaleF,axis=(1,2))

    dUnPP+=UnPPs
    dUnPH+=UnPHs
    dUnPHE+=UnPHEs

    return -AC*dSE, -AC*dUnPP, -AC*dUnPH, -AC*dUnPHE

def calcSE(UnF,propT,AC):
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

