import numpy as np
import auxFunctions as auxF

class vertexR:
    """
    A class to contain the two particle vertex

    ...
    Attributes
    ----------
    wB : array_like(float, ndim=1)
        An array of bosonic frequencies at which the 
        value of the vertex is known

    beTa : float
        Inverse temperature of the system

    NLmax : int
        Max number of basis functions used to expand
        the vertices in each channel
        
    NW : int
        Current number of basis functions used to 
        approximate the vertex

    uF : function(wPP,wPH,wPHE)
        Initial two particle vertex

    UnPP, UnPPO : array_like(complex, ndim=3)
        Intial and generated particle-particle components 
        of the vertex
        
    UnPH, UnPHO : array_like(complex, ndim=3)
        Particle-particle component of the vertex

    UnPHE : array_like(complex, ndim=3)
        Particle-particle component of the vertex

    zFX,zFG : array_like(float, ndim=1)
        Gauss-Legendre points and weights for integration 
        over Legendre polynomials

    scaleDerv : array_like(float, ndim=4)
        Projection of the derivative of scale dependent
        basis functions
   
    wTransXtoY : arraylike(float, nidm=5)
        Six internal variables for projecting between the 
        the three channels of the vertex

    Methods
    -------
    initializeVertex()
        Projects initial two particle vertex over the three
        channels

    projectionW()
        One shot calculation of projection arrays between the
        various channels. Calculates wTransXtoY for run. 
 
    legndExpand(UnX,AC)
        Expands the bosonic frequency dependence of the vertex 
        at scale AC via NLmax basis functions

    uEvaluate(wPP,wPH,wPHE,AC)
        Evaluates the full vertex at scale AC at the given 
        frequecies

    _expandChannel(wS,wSX,wSY,AC,chnL)
        Fully expands a channel of the vertex at scale AC 

    projectChannel(UnL,AC,chnL):
        Projects a channel at scale AC into the other two 
        channels
    """
    def __init__(self,wB,NW,beTa,UnF):
        """
        Parameters
        ----------
        wB : array_like(float, ndim=1)
            An array of bosonic frequencies at which the 
            value of the vertex is known

        NW : int
            Current number of basis functions
        beTa : float
            Inverse temperature of the system

        UnF : function(wPP,wPH,wPHE)
            Initial two particle vertex
        
        """

        self.wB=wB
        self.beTa=beTa
        self.NLmax=20
        self.NW=NW

        UnPP = np.zeros((len(self.wB),NW,NW),dtype=np.complex_)
        UnPH = np.zeros((len(self.wB),NW,NW),dtype=np.complex_)
        UnPHE = np.zeros((len(self.wB),NW,NW),dtype=np.complex_)
        
        self.UnPP = UnPP
        self.UnPH = UnPH
        self.UnPHE = UnPHE

        self.uF=UnF
        wMidI=np.append(-wB[:0:-1],wB)
        self.wMidI=wMidI
        zFX,b=auxF.gaussianInt([-1,1],20)
        zFX=np.unique(np.append(auxF.forMap(wMidI,1.0),zFX))
        self.zFX,self.zFG=auxF.gaussianInt(zFX,8)
        
        self.UnPPO,self.UnPHO,self.UnPHEO=self.initializeVertex()
        self.projectionW()        
        
    def initializeVertex(self):
        """Calculates the intial vertex in the three channels"""

        wB,NW,beTa=self.wB,self.NW,self.beTa
        NLmax=self.NLmax
        zFX,zFG=auxF.gaussianInt([-1,1],30)

        UnPPO=np.zeros((len(wB),NW,NW),dtype=np.complex_)
        UnPHO=np.zeros((len(wB),NW,NW),dtype=np.complex_)
        UnPHEO=np.zeros((len(wB),NW,NW),dtype=np.complex_)

        wFX=auxF.backMap(zFX,1.0)
        wP1=np.tile(wFX[:,np.newaxis],(1,len(wFX)))
        wP2=np.tile(wFX[np.newaxis,:],(len(wFX),1))
        zP1=np.tile(zFX[:,np.newaxis],(1,len(zFX)))
        zP2=np.tile(zFX[np.newaxis,:],(len(zFX),1))
        zPG=np.tile(zFG[:,np.newaxis],(1,len(zFX)))
        zPG=zPG*np.tile(zFG[np.newaxis,:],(len(zFX),1))

        for i in range(len(wB)):
            wS=np.zeros((len(wFX),len(wFX)))+wB[i]
            
            wPH=wP1+wP2
            wPHE=wP2-wP1
            uPP=self.uF(wS,wPH,wPHE)

            wPP=wP1+wP2
            wPHE=wP2-wP1
            uPH=self.uF(wPP,wS,wPHE)

            wPP=wP1+wP2
            wPH=wP1-wP2
            uPHE=self.uF(wPP,wPH,wS)

            for j in range(NW):
                lTemp1=auxF.freqExpansion(zP1,2*j)
                for k in range(NW):
                    lTemp2=auxF.freqExpansion(zP2,2*k)
                    UnPPO[i,j,k]=np.sum(zPG*uPP*lTemp1*lTemp2)
                    UnPHO[i,j,k]=np.sum(zPG*uPH*lTemp1*lTemp2)
                    UnPHEO[i,j,k]=np.sum(zPG*uPHE*lTemp1*lTemp2)
        
        return UnPPO,UnPHO,UnPHEO       

    def projectionW(self):
        """Calculates the arrays for projection between the channels at the 
        start of the flow"""

        NW=self.NW
        NLmax=self.NLmax
        wB=self.wB
        zFX,zFG=auxF.gaussianInt([-1,1],30)
        cScale=1.0

        scaleDerv=np.zeros((NW,NW,NW,NW))
        for i in range(NW):
            for j in range(NW):
                scaleTemp1=np.zeros((NW,NW))
                scaleTemp2=np.zeros((NW,NW))
                for k in range(1,NW):
                    intG=(zFX**2)*auxF.freqExpansion(zFX,2*k)-zFX*auxF.freqExpansion(zFX,2*k-1)
                    scaleTemp1[k,j]=2*k*np.sum(zFG*intG*auxF.freqExpansion(zFX,2*i))

                    intG=(zFX**2)*auxF.freqExpansion(zFX,2*k)-zFX*auxF.freqExpansion(zFX,2*k-1)
                    scaleTemp2[i,k]=2*k*np.sum(zFG*intG*auxF.freqExpansion(zFX,2*j))
                scaleDerv[:,:,i,j]=-(scaleTemp1+scaleTemp2)
        self.scaleDerv=scaleDerv

        wTransPHtoPP=np.zeros((NLmax,NW,NW,len(wB),NW,NW))
        wTransPHEtoPP=np.zeros((NLmax,NW,NW,len(wB),NW,NW))
        wTransPPtoPH=np.zeros((NLmax,NW,NW,len(wB),NW,NW))
        wTransPHEtoPH=np.zeros((NLmax,NW,NW,len(wB),NW,NW))
        wTransPPtoPHE=np.zeros((NLmax,NW,NW,len(wB),NW,NW))
        wTransPHtoPHE=np.zeros((NLmax,NW,NW,len(wB),NW,NW))

        wFX=auxF.backMap(zFX,cScale)
        wP1=np.tile(wFX[:,np.newaxis,np.newaxis],(1,len(wFX),len(wB)))
        wP2=np.tile(wFX[np.newaxis,:,np.newaxis],(len(zFX),1,len(wB)))
        wBE=np.tile(wB[np.newaxis,np.newaxis,:],(len(zFX),len(zFX),1))

        wFXE=np.tile(wFX[:,np.newaxis],(1,len(wB)))
        wBEe=np.tile(wB[np.newaxis,:],(len(zFX),1))

        zP1=np.tile(zFX[:,np.newaxis,np.newaxis],(1,len(zFX),len(wB)))
        zP2=np.tile(zFX[np.newaxis,:,np.newaxis],(len(zFX),1,len(wB)))
        zPE=np.tile(zFG[:,np.newaxis,np.newaxis],(1,len(zFX),len(wB)))
        zPE=zPE*np.tile(zFG[np.newaxis,:,np.newaxis],(len(zFX),1,len(wB)))

        lTempXtoY1=np.zeros((len(zFX),len(zFX),len(wB),NLmax))
        lTempXtoY2=np.zeros((len(zFX),len(zFX),len(wB),NLmax))

        for i in range(NLmax):
            zT=auxF.forMap((wP1+wP2),cScale)
            lTempXtoY1[...,i]=zPE*auxF.freqExpansion(zT,2*i)

            zT=auxF.forMap((wP2-wP1),cScale)
            lTempXtoY2[...,i]=zPE*auxF.freqExpansion(zT,2*i)

        lTempP1=np.zeros((len(zFX),len(zFX),len(wB),NW))
        lTempP2=np.zeros((len(zFX),len(zFX),len(wB),NW))
        lTempP3=np.zeros((len(zFX),len(zFX),len(wB),NW))
        lTempP4=np.zeros((len(zFX),len(zFX),len(wB),NW))

        lTempW1=np.zeros((len(zFX),len(zFX),len(wB),NW))
        lTempWn=np.zeros((len(zFX),len(wB),NW))

        for i in range(NW):
            zT=auxF.forMap(0.5*(wBE-(wP2-wP1)),cScale)
            lTempP1[...,i]=auxF.freqExpansion(zT,2*i)

            zT=auxF.forMap(0.5*(wBE+(wP2-wP1)),cScale)
            lTempP2[...,i]=auxF.freqExpansion(zT,2*i)

            zT=auxF.forMap(0.5*(wBE-(wP1+wP2)),cScale)
            lTempP3[...,i]=auxF.freqExpansion(zT,2*i)

            zT=auxF.forMap(0.5*(wBE+(wP1+wP2)),cScale)
            lTempP4[...,i]=auxF.freqExpansion(zT,2*i)

            lTempW1[...,i]=auxF.freqExpansion(zP1,2*i)
            lTempWn[...,i]=auxF.freqExpansion(auxF.forMap(wFXE,cScale),2*i)
        
        wTemp1=np.zeros((len(zFX),NW,NW,len(wB),NW))
        wTemp2=np.zeros((len(zFX),NW,NW,len(wB),NW))
        for i in range(NLmax):
            for j in range(NW):
                for k in range(NW):
                    for l in range(NW):
                        wTemp1[:,j,k,:,l]=np.sum(lTempXtoY1[...,i]*lTempP1[...,j]*lTempP2[...,k]*lTempW1[...,l],axis=0)
                        wTemp2[:,j,k,:,l]=np.sum(lTempXtoY2[...,i]*lTempP3[...,j]*lTempP4[...,k]*lTempW1[...,l],axis=0)
                    
            for j in range(NW):
                intG=lTempWn[...,j]
                intG=np.tile(intG[:,np.newaxis,np.newaxis,:,np.newaxis],(1,NW,NW,1,NW))
                wTransPHtoPP[i,:,:,:,:,j]=np.sum(intG*wTemp1,axis=0)
                wTransPHEtoPP[i,:,:,:,:,j]=np.sum(intG*wTemp2,axis=0)
        wTransPPtoPH=wTransPHtoPP[:]
        for j in range(NW):
            wTransPHEtoPH[:,j,:,:,:,:]=wTransPHEtoPP[:,j,:,:,:,:]
            wTransPPtoPHE[:,j,:,:,:,:]=wTransPHtoPP[:,j,:,:,:,:]
            wTransPHtoPHE[:,j,:,:,:,:]=wTransPHEtoPP[:,j,:,:,:,:]

        self.wTransPHtoPP=wTransPHtoPP
        self.wTransPHEtoPP=wTransPHEtoPP
        self.wTransPPtoPH=wTransPPtoPH
        self.wTransPHEtoPH=wTransPHEtoPH
        self.wTransPPtoPHE=wTransPPtoPHE
        self.wTransPHtoPHE=wTransPHtoPHE

    def legndExpand(self,UnX,AC):
        """Expands the frequency dependence of UnX in terms of basis set""" 
        wB=self.wB
        beTa=self.beTa
        NW=self.NW
        NLmax=self.NLmax
        zFX,zFG=self.zFX,self.zFG
        wMidI=self.wMidI
        cScale=1.0
        
        UnXi=np.swapaxes(UnX,1,2)
        UnXi=np.concatenate((np.conj(UnXi[:0:-1,:,:]),UnX),axis=0)

        UnXiF=auxF.linInterp(wMidI,UnXi,auxF.backMap(zFX,cScale)*np.sqrt(AC**2+1))
        UnL=np.zeros((NLmax,NW,NW),dtype=np.complex_)
        
        zFXE=np.tile(zFX[:,np.newaxis,np.newaxis],(1,NW,NW))
        zFGE=np.tile(zFG[:,np.newaxis,np.newaxis],(1,NW,NW))
        for i in range(NLmax):
            UnL[i,:,:]=np.sum(zFGE*auxF.freqExpansion(zFXE,2*i)*UnXiF,axis=0)

        return UnL
    def uEvaluate(self,wPP,wPH,wPHE,AC):
        """Evaluates the vertex at the given frequency and scale"""
        uShape=wPP.shape

        wPP=np.reshape(wPP,wPP.size)
        wPH=np.reshape(wPH,wPH.size)
        wPHE=np.reshape(wPHE,wPHE.size)
        

        uPP=self._expandChannel(wPP,0.5*(wPH-wPHE),0.5*(wPH+wPHE),AC,'PP')
        uPH=self._expandChannel(wPH,0.5*(wPP-wPHE),0.5*(wPP+wPHE),AC,'PH')
        uPHE=self._expandChannel(wPHE,0.5*(wPP-wPH),0.5*(wPP+wPH),AC,'PHE')
        
        u0=self.uF(wPP,wPH,wPHE)
        UnV=u0+uPP+uPH+uPHE
        
        return np.reshape(UnV,uShape)

    def _expandChannel(self,wS,wSX,wSY,AC,chnL):
        if chnL is 'PP':
            UnX = self.UnPP
        elif chnL is 'PH':
            UnX = self.UnPH  
        elif chnL is 'PHE':
            UnX = self.UnPHE
        
        beTa=self.beTa
        NW=self.NW
        wB=self.wB
        NLmax=self.NLmax
        nPoints=len(wS)
        
        wMidI=self.wMidI

        UnXi=np.swapaxes(UnX,1,2)
        UnXi=np.concatenate((np.conj(UnX[:0:-1,:,:]),UnX),axis=0)
        
        UnXiF=auxF.linInterp(wMidI,UnXi,wS)
        
        zSX=auxF.forMap(wSX/np.sqrt(AC**2+1),1.0)
        zSY=auxF.forMap(wSY/np.sqrt(AC**2+1),1.0)
        
        lTempXN=np.zeros((nPoints,1,NW))
        lTempYN=np.zeros((nPoints,NW,1))
        for i in range(NW):
            lTempXN[:,0,i]=auxF.freqExpansion(zSX,2*i)
            lTempYN[:,i,0]=auxF.freqExpansion(zSY,2*i)

        UnE=np.zeros(len(wS),dtype=np.complex_)
        UnE=np.squeeze(np.matmul(lTempXN,np.matmul(UnXiF,lTempYN)))

        return UnE

    def vertExpand(self,UnX,wS,wSX,wSY,AC):
        beTa=self.beTa
        NW=self.NW
        wB=self.wB
        NLmax=self.NLmax
        
        wMidI=self.wMidI

        uShape=wS.shape
        wS=np.reshape(wS,wS.size)
        wSX=np.reshape(wSX,wSX.size)
        wSY=np.reshape(wSY,wSY.size)
        nPoints=len(wS)
        
        UnXi=np.swapaxes(UnX,1,2)
        UnXi=np.concatenate((np.conj(UnX[:0:-1,:,:]),UnX),axis=0)

        UnXiF=auxF.linInterp(wMidI,UnXi,wS)
        
        zSX=auxF.forMap(wSX/np.sqrt(AC**2+1),1.0)
        zSY=auxF.forMap(wSY/np.sqrt(AC**2+1),1.0)
        
        lTempXN=np.zeros((nPoints,1,NW))
        lTempYN=np.zeros((nPoints,NW,1))
        for i in range(NW):
            lTempXN[:,0,i]=auxF.freqExpansion(zSX,2*i)
            lTempYN[:,i,0]=auxF.freqExpansion(zSY,2*i)

        UnE=np.zeros(len(wS),dtype=np.complex_)
        UnE=np.squeeze(np.matmul(lTempXN,np.matmul(UnXiF,lTempYN)))

        return np.reshape(UnE,uShape)

    def projectChannel(self,UnL,AC,chnL):
        """Projects the vertex UnL in the channel chnL into 
        the other channels"""
        if chnL is 'PP':
           wTrans1=self.wTransPPtoPH
           wTrans2=self.wTransPPtoPHE
           
        elif chnL is 'PH':
           wTrans1=self.wTransPHtoPP
           wTrans2=self.wTransPHtoPHE

        elif chnL is 'PHE':
           wTrans1=self.wTransPHEtoPP
           wTrans2=self.wTransPHEtoPH

        NW=self.NW
        wB=self.wB
        NLmax=self.NLmax

        UnR1=np.sum(UnL[:,:,:,None,None,None]*wTrans1,axis=(0,1,2))
        UnR1=auxF.linInterp(np.sqrt(AC**2+1)*wB,UnR1,wB)
        
        UnR2=np.sum(UnL[:,:,:,None,None,None]*wTrans2,axis=(0,1,2))
        UnR2=auxF.linInterp(np.sqrt(AC**2+1)*wB,UnR2,wB)
        
        return UnR1,UnR2

                
                
                
        

        
        
