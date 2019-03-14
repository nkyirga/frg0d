import numpy as np
cimport numpy as np
def linInterp(np.ndarray[double,ndim=1] xI,np.ndarray[np.complex_t,ndim=3] yI,np.ndarray[double,ndim=1] xQ):
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

def cubInterp(np.ndarray[double,ndim=1] xI,np.ndarray[np.complex_t,ndim=3] yI,np.ndarray[double,ndim=1] xQ):
    cdef int nGrid=len(xI)
    cdef int xLQ=len(xQ)
    cdef int yL=yI.size/nGrid
    cdef int yL1=len(yI[0,:,0])
    cdef int yL2=len(yI[0,0,:])

    cdef np.ndarray[np.complex_t,ndim=2] yIR=np.reshape(yI,(nGrid,yL))
    
    cdef np.ndarray[double,ndim=1] aDiag=(1/3.0)*(xI[2:]-xI[:-2])
    cdef np.ndarray[double,ndim=2] aDiagE=np.tile(aDiag[:,np.newaxis],(1,yL))
    cdef np.ndarray[double,ndim=1] aDn=np.append(np.zeros(1),(1/6.0)*(xI[2:-1]-xI[1:-2]))
    cdef np.ndarray[double,ndim=2] aDnE=np.tile(aDn[:,np.newaxis],(1,yL))
    cdef np.ndarray[double,ndim=1] aUp=np.append((1/6.0)*(xI[2:-1]-xI[1:-2]),np.zeros(1))
    cdef np.ndarray[double,ndim=2] aUpE=np.tile(aUp[:,np.newaxis],(1,yL))

    cdef np.ndarray[double,ndim=2] xIE=np.tile(xI[:,np.newaxis],(1,yL))
    cdef np.ndarray[np.complex_t,ndim=2] bRight=(np.divide(yIR[2:,:]-yIR[1:-1,:],xIE[2:,:]-xIE[1:-1,:])-
            np.divide(yIR[1:-1,:]-yIR[:-2,:],xIE[1:-1,:]-xIE[:-2,:]))

    cdef np.ndarray[np.complex_t,ndim=2] yI2=np.zeros((nGrid-2,yL),dtype=np.complex_)
    cdef np.ndarray[double,ndim=2] tempY=np.zeros((nGrid-2,yL))
    cdef np.ndarray[double,ndim=1] denoM=aDiagE[0,:]
    yI2[0,:]=np.divide(bRight[0,:],denoM)

    cdef int i
    for i in range(1,len(yI2)):
        tempY[i,:]=np.divide(aUpE[i-1,:],denoM)
        denoM=aDiagE[i,:]-np.multiply(aDnE[i,:],tempY[i,:])

        yI2[i,:]=np.divide(bRight[i,:]-np.multiply(aDnE[i,:],yI2[i-1,:]),denoM)

    for i in range(len(yI2)-2,-1,-1):
        yI2[i,:]-=np.multiply(tempY[i+1,:],yI2[i+1,:])

    cdef np.ndarray[np.complex_t,ndim=2] endYI=np.zeros((1,yL),dtype=np.complex_)
    cdef np.ndarray[np.complex_t,ndim=2] yI2E=np.concatenate((endYI,yI2,endYI,endYI),axis=0)
    
    cdef np.ndarray[long,ndim=1] xQI=np.searchsorted(xI,xQ)
    cdef np.ndarray[np.complex_t,ndim=2] yIE=np.concatenate((yIR,yIR[-1:,:]),axis=0)

    cdef np.ndarray[np.complex_t,ndim=2] yV1=yIE[xQI-1,:]
    cdef np.ndarray[np.complex_t,ndim=2] yD1=yI2E[xQI-1,:]

    cdef np.ndarray[np.complex_t,ndim=2] yV2=yIE[xQI,:]
    cdef np.ndarray[np.complex_t,ndim=2] yD2=yI2E[xQI,:]
    
    cdef np.ndarray[double,ndim=1] xIE2=np.append(xI,max(xQ))
    cdef np.ndarray[double,ndim=1] x1=xIE2[xQI-1]
    cdef np.ndarray[double,ndim=1] x2=xIE2[xQI]
    
    cdef np.ndarray[double,ndim=1] A=np.divide((x2-xQ),(x2-x1))
    cdef np.ndarray[double,ndim=2] AE=np.tile(A[:,np.newaxis],(1,yL))
    cdef np.ndarray[double,ndim=1] B=np.divide((xQ-x1),(x2-x1))
    cdef np.ndarray[double,ndim=2] BE=np.tile(B[:,np.newaxis],(1,yL))

    cdef np.ndarray[double,ndim=1] C=(1/6.0)*np.multiply((np.power(A,3)-A),np.multiply(x2-x1,x2-x1))
    cdef np.ndarray[double,ndim=2] CE=np.tile(C[:,np.newaxis],(1,yL))
    cdef np.ndarray[double,ndim=1] D=(1/6.0)*np.multiply((np.power(B,3)-B),np.multiply(x2-x1,x2-x1))
    cdef np.ndarray[double,ndim=2] DE=np.tile(D[:,np.newaxis],(1,yL))

    cdef np.ndarray[np.complex_t,ndim=2] yQ=(np.multiply(AE,yV1) + np.multiply(BE,yV2) 
                                             + np.multiply(CE,yD1) + np.multiply(DE,yD2))
    
    return np.reshape(yQ,(xLQ,yL1,yL2))

