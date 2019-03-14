import numpy as np

def linInterp(xI,yI,xQ):
    xShape=xQ.shape
    yShape=(yI.shape)[1:]
    xL=len(xI)
    yL=yI.size/(len(xI))
    yI=np.reshape(yI,(xL,yL))
        
    yI=np.concatenate((yI,yI[-1:,...]),axis=0)
    xQI=np.searchsorted(xI,xQ)

    y1=yI[xQI-1,:]
    y2=yI[xQI,:]
    
    xI=np.append(xI,max(xQ))
    x1=xI[xQI-1]
    x1=np.tile(x1[:,np.newaxis],(1,yL))
    x2=xI[xQI]
    x2=np.tile(x2[:,np.newaxis],(1,yL))
    
    xQE=np.tile(xQ[:,np.newaxis],(1,yL))
    yQ=(xQE-x1)*(y1-y2)/(x1-x2)+y1
    
    return np.reshape(yQ,xShape+yShape)

def cubInterp(xI,yI,xQ):
    nGrid=len(xI)
    xShape=xQ.shape
    yShape=(yI.shape)[1:]
    yL=yI.size/nGrid
    yI=np.reshape(yI,(nGrid,yL))
    
    aDiag=(1/3.0)*(xI[2:]-xI[:-2])
    aDiag=np.tile(aDiag[:,np.newaxis],(1,yL))
    aDn=np.append(np.zeros(1),(1/6.0)*(xI[2:-1]-xI[1:-2]))
    aDn=np.tile(aDn[:,np.newaxis],(1,yL))
    aUp=np.append((1/6.0)*(xI[2:-1]-xI[1:-2]),np.zeros(1))
    aUp=np.tile(aUp[:,np.newaxis],(1,yL))

    xIE=np.tile(xI[:,np.newaxis],(1,yL))
    bRight=((yI[2:,:]-yI[1:-1,:])/(xIE[2:,:]-xIE[1:-1,:])-
            (yI[1:-1,:]-yI[:-2,:])/(xIE[1:-1,:]-xIE[:-2,:]))

    yI2=np.zeros((nGrid-2,yL))
    tempY=np.zeros((nGrid-2,yL))
    denoM=float(aDiag[0,:])
    yI2[0,:]=bRight[0,:]/denoM
    for i in range(1,len(yI2)):
        tempY[i,:]=aUp[i-1,:]/denoM
        denoM=aDiag[i,:]-aDn[i,:]*tempY[i,:]

        yI2[i,:]=(bRight[i,:]-aDn[i,:]*yI2[i-1,:])/denoM

    for i in range(len(yI2)-2,-1,-1):
        yI2[i,:]-=tempY[i+1,:]*yI2[i+1,:]

    endYI=np.zeros((1,yL))
    yI2E=np.concatenate((endYI,yI2,endYI,endYI),axis=0)
    
    xQI=np.searchsorted(xI,xQ)
    yIE=np.concatenate((yI,yI[-1:,:]),axis=0)

    yV1=yIE[xQI-1,:]
    yD1=yI2E[xQI-1,:]

    yV2=yIE[xQI,:]
    yD2=yI2E[xQI,:]
    
    xIE=np.append(xI,max(xQ))
    x1=xIE[xQI-1]
    x2=xIE[xQI]
    
    A=(x2-xQ)/(x2-x1)
    AE=np.tile(A[:,np.newaxis],(1,yL))
    B=(xQ-x1)/(x2-x1)
    BE=np.tile(B[:,np.newaxis],(1,yL))

    C=(1/6.0)*(A**3-A)*((x2-x1)**2)
    CE=np.tile(C[:,np.newaxis],(1,yL))
    D=(1/6.0)*(B**3-B)*((x2-x1)**2)
    DE=np.tile(D[:,np.newaxis],(1,yL))

    yQ=AE*yV1 + BE*yV2 + CE*yD1 + DE*yD2
    return np.reshape(yQ,xShape+yShape)
