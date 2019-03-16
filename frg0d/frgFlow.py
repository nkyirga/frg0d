import copy
import time
import numpy as np
import auxFunctions as auxF
from vertexF import vertexR
from propG import scaleProp

class fRG0D:
    def __init__(self, nPatches, deltal, beTa, maxW, NT,cutoffR):
        self.step=deltal
        self.beTa=beTa
        self.NW=NT
        self.maxW=maxW
        self.cutoffT=cutoffR
        
        a,self.wB=auxF.freqPoints(beTa,maxW,nPatches)
        
    def initializeFlow(self,hybriD,uVertex,nLoop,Mu=0,lStart=0):
        self.l=lStart
        self.Mu=Mu
        NW=self.NW

        self.UnF=vertexR(self.wB,self.NW,self.beTa,uVertex)
        self.propG=scaleProp(self.maxW,hybriD,self.beTa,self.Mu,cutoff=self.cutoffT)
        self.nLoop=nLoop

        if nLoop is 1:
            self.betaFC=auxF.betaF
        elif nLoop is 2:
            self.betaFC=auxF.betaF2L
        elif (nLoop>=3):
            self.betaFC=auxF.betaFNL
    
    def advanceRK4(self):
        nL=self.nLoop
        UnFTemp=copy.copy(self.UnF)
        propT=copy.copy(self.propG)
        lC=copy.copy(self.l)
        
        AC=auxF.aScale(lC,self.maxW)    
        
        shapeSE=propT.wF.shape
        sERK=np.zeros(shapeSE+(4,),dtype=np.complex_)
        
        UnPPRK=[]
        UnPHRK=[]
        UnPHERK=[]

        UnPPRK.append(np.zeros(UnFTemp.UnPP[0].shape+(4,),dtype=np.complex_))
        UnPPRK.append(np.zeros(UnFTemp.UnPP[1].shape+(4,),dtype=np.complex_))
        UnPPRK.append(np.zeros(UnFTemp.UnPP[2].shape+(4,),dtype=np.complex_))
        UnPPRK.append(np.zeros(UnFTemp.UnPP[3].shape+(4,),dtype=np.complex_))

        UnPHRK.append(np.zeros(UnFTemp.UnPH[0].shape+(4,),dtype=np.complex_))
        UnPHRK.append(np.zeros(UnFTemp.UnPH[1].shape+(4,),dtype=np.complex_))
        UnPHRK.append(np.zeros(UnFTemp.UnPH[2].shape+(4,),dtype=np.complex_))
        UnPHRK.append(np.zeros(UnFTemp.UnPH[3].shape+(4,),dtype=np.complex_))
        
        UnPHERK.append(np.zeros(UnFTemp.UnPHE[0].shape+(4,),dtype=np.complex_))
        UnPHERK.append(np.zeros(UnFTemp.UnPHE[1].shape+(4,),dtype=np.complex_))
        UnPHERK.append(np.zeros(UnFTemp.UnPHE[2].shape+(4,),dtype=np.complex_))
        UnPHERK.append(np.zeros(UnFTemp.UnPHE[3].shape+(4,),dtype=np.complex_))
        
        sERK[...,0],UnPPRKT,UnPHRKT,UnPHERKT=\
            betaFC(UnFTemp,propT,AC,nL)

        for i in range(4):
            UnPPRK[i][...,0]=UnPPRKT[i]
            UnPHRK[i][...,0]=UnPHRKT[i]
            UnPHERK[i][...,0]=UnPHERKT[i]

        stepB=np.zeros(3)
        stepB[:]=[1.0/2.0,1.0/2,1.0]
    
        stepB=stepB*self.step

        for i in range(3):
            lC+=stepB[i]
            sETemp=copy.copy(self.propG.sE)
            UnPPTemp=copy.copy(self.UnF.UnPP)
            UnPHTemp=copy.copy(self.UnF.UnPH)
            UnPHETemp=copy.copy(self.UnF.UnPHE)
                
            sETemp+=stepB[i]*sERK[...,i]
            for j in range(4):
                UnPPTemp[j]+=stepB[i]*UnPPRK[j][...,i]
                UnPHTemp[j]+=stepB[i]*UnPHRK[j][...,i]
                UnPHETemp[j]+=stepB[i]*UnPHERK[j][...,i]
                
            propT.setSE(sETemp)
            UnFTemp.UnPP=UnPPTemp
            UnFTemp.UnPPI=UnFTemp.legndExpand(UnPPTemp)
            UnFTemp.UnPH=UnPHTemp
            UnFTemp.UnPHI=UnFTemp.legndExpand(UnPHTemp)
            UnFTemp.UnPHE=UnPHETemp
            UnFTemp.UnPHEI=UnFTemp.legndExpand(UnPHETemp)
            
            AC=auxF.aScale(lC,self.maxW)
            sERK[...,i+1],UnPPRKT,UnPHRKT,UnPHERKT=\
                betaFC(UnFTemp,propT,AC,nL)
 
            for j in range(4):
                UnPPRK[j][...,i+1]=UnPPRKT[j]
                UnPHRK[j][...,i+1]=UnPHRKT[j]
                UnPHERK[j][...,i+1]=UnPHERKT[j]

        sE=copy.copy(self.propG.sE)
        UnPP=copy.copy(self.UnF.UnPP)
        UnPH=copy.copy(self.UnF.UnPH)
        UnPHE=copy.copy(self.UnF.UnPHE)
            
        o4=np.zeros(4)
        o4[:]=[1.0/6.0,2.0/6.0,2.0/6.0,1.0/6.0]
        o4=o4*self.step

        for i in range(4):
            sE+=o4[i]*sERK[...,i]
            for j in range(4):
                UnPP[j]+=o4[i]*UnPPRK[j][...,i]
                UnPH[j]+=o4[i]*UnPHRK[j][...,i]
                UnPHE[j]+=o4[i]*UnPHERK[j][...,i]
        
        return sE,UnPP,UnPH,UnPHE
    
    def susFunctions(self):
        gPHL,gPHR,gPHL0,gPHR0=self.propG.susBubbles(self.UnF.wB,self.UnF.NW)
        
        AC=0.001
        UnPPI=self.UnF.legndExpand(self.UnF.UnPP,AC)
        UnPHI=self.UnF.legndExpand(self.UnF.UnPH,AC)
        UnPHEI=self.UnF.legndExpand(self.UnF.UnPHE,AC)

        uPPtoPH,uPPtoPHE=self.UnF.projectChannel(UnPPI,AC,'PP')
        uPHtoPP,uPHtoPHE=self.UnF.projectChannel(UnPHI,AC,'PH')
        uPHEtoPP,uPHEtoPH=self.UnF.projectChannel(UnPHEI,AC,'PHE')

        zR=np.zeros(1)
        wB=self.UnF.wB
        cSus=np.zeros(len(wB))
        sSus=np.zeros(len(wB))

        UnPHX=self.UnF.UnPHO+self.UnF.UnPH+uPPtoPH+uPHEtoPH
        UnPHEX=self.UnF.UnPHEO+self.UnF.UnPHE+uPPtoPHE+uPHtoPHE
        cSus=2*np.squeeze(np.matmul(gPHL,np.matmul(2*UnPHX-UnPHEX,gPHR)))
        sSus=-2*np.squeeze(np.matmul(gPHL,np.matmul(UnPHEX,gPHR)))

        cSus=np.squeeze(np.matmul(gPHL,np.matmul(UnPHX,gPHR)))
        nMax=np.floor(0.5*((self.maxW*self.beTa/np.pi)-1))
        wQ=(np.pi/self.beTa)*2*np.arange(0,nMax,1)

        dF=np.interp(wQ,self.wB,cSus.real)
        dF=np.append(dF[:0:-1],dF)

        print 'dFac',(1.0/self.beTa)*np.sum(dF)
        dF=np.interp(wQ,self.wB,gPHL0.real)
        dF=np.append(dF[:0:-1],dF)

        print 'dFacNI',(1.0/self.beTa)*np.sum(dF)

        cSus=-cSus-gPHL0
        
        return cSus,sSus

    def adaptiveRGFlow(self,lMax):
        while self.l<lMax or self.step<(self.step/10): 
            sE,UnPP,UnPH,UnPHE,eRR=self.advanceRKF()
            auxF.printBar(np.round(self.step,2),self.l/lMax)

            absERR=0.01
            stepN=self.step*(absERR/eRR)**0.2
            if stepN>=self.step:
                self.l+=self.step
                self.propG.sE=sE
 
                self.UnF.UnPP=UnPP
                self.UnF.UnPH=UnPH
                self.UnF.UnPHE=UnPHE
                
            else:
                self.step=0.9*stepN
        from matplotlib import pyplot as plt
        plt.plot(self.propG.wF,sE.imag,'o-')
        plt.show()
            
        return self.UnF.UnPP,self.UnF.UnPH,self.UnF.UnPHE,self.propG.sE
    
    def advanceRKF(self):
        UnFTemp=copy.deepcopy(self.UnF)
        propT=copy.deepcopy(self.propG)
        lC=copy.copy(self.l)
        nL=self.nLoop
        
        AC=auxF.aScale(lC,self.maxW)    
        
        shapeSE=propT.wF.shape
        sERK=np.zeros(shapeSE+(6,),dtype=np.complex_)

        UnPPRK=np.zeros(UnFTemp.UnPP.shape+(6,),dtype=np.complex_)
        UnPHRK=np.zeros(UnFTemp.UnPH.shape+(6,),dtype=np.complex_)
        UnPHERK=np.zeros(UnFTemp.UnPHE.shape+(6,),dtype=np.complex_)
         
        sERK[...,0],UnPPRK[...,0],UnPHRK[...,0],UnPHERK[...,0]=\
            self.betaFC(UnFTemp,propT,AC,nL)

        stepB=np.zeros(5)
        stepB[:]=[1.0/4,3.0/8,12.0/13,1.0,1.0/2]
        
        stepB=stepB*self.step
        bTable=np.zeros(15)
        bTable[:]=[1.0/4,3.0/32,9.0/32,1932.0/2197,-7200.0/2197,
                   7296.0/2197,439.0/216,-8.0,3680.0/513,-845.9/4104,
                   -8.0/27,2.0,-3544.0/2565,1859.0/4104,-11.0/40]

        for i in range(5):
            lC=self.l+stepB[i]
            sETemp=copy.copy(self.propG.sE)
            UnPPTemp=copy.copy(self.UnF.UnPP)
            UnPHTemp=copy.copy(self.UnF.UnPH)
            UnPHETemp=copy.copy(self.UnF.UnPHE)
            for j in range(i+1):
                bWeight=bTable[sum(range(i+1))+j]
                
                sETemp+=stepB[i]*bWeight*sERK[...,j]
                UnPPTemp+=stepB[i]*bWeight*UnPPRK[...,j]
                UnPHTemp+=stepB[i]*bWeight*UnPHRK[...,j]
                UnPHETemp+=stepB[i]*bWeight*UnPHERK[...,j]
            
            AC=auxF.aScale(lC,self.maxW)
            
            propT.sE=sETemp
            UnFTemp.UnPP=UnPPTemp
            UnFTemp.UnPH=UnPHTemp
            UnFTemp.UnPHE=UnPHETemp
            
            sERK[...,i+1],UnPPRK[...,i+1],UnPHRK[...,i+1],UnPHERK[...,i+1]=\
                self.betaFC(UnFTemp,propT,AC,nL)
            
        o5=np.zeros(6)
        o4=np.zeros(6)
        o5[:]=[16.0/135.0,0.0,6656.0/12825.0,28561.0/56430.0,-9.0/50.0,2.0/55.0]
        o4[:]=[25.0/216,0.0,1408.0/2565,2197.0/4104,-1.0/5,0.0]
    
        o5=self.step*o5
        o4=self.step*o4

        sE=copy.copy(self.propG.sE)
        UnPP=copy.copy(self.UnF.UnPP)
        UnPH=copy.copy(self.UnF.UnPH)
        UnPHE=copy.copy(self.UnF.UnPHE)
            
        sEerr=np.zeros(sE.shape,dtype=np.complex_)
        UnPPerr=np.zeros(UnPP.shape,dtype=np.complex_)
        UnPHerr=np.zeros(UnPH.shape,dtype=np.complex_)
        UnPHEerr=np.zeros(UnPHE.shape,dtype=np.complex_)

        for i in range(6):
            sE+=o5[i]*sERK[...,i]
            sEerr+=(o5[i]-o4[i])*sERK[...,i]

            UnPP+=o5[i]*UnPPRK[...,i]
            UnPH+=o5[i]*UnPHRK[...,i]                
            UnPHE+=o5[i]*UnPHERK[...,i]

            UnPPerr+=(o5[i]-o4[i])*UnPPRK[...,i]
            UnPHerr+=(o5[i]-o4[i])*UnPHRK[...,i]
            UnPHEerr+=(o5[i]-o4[i])*UnPHERK[...,i]
            
        sEerrMax=np.abs(sEerr).max()
        UnPPerrMax=np.abs(UnPPerr).max()
        UnPHerrMax=np.abs(UnPHerr).max()
        UnPHEerrMax=np.abs(UnPHEerr).max()
        
        eRR=max([sEerrMax,UnPPerrMax,UnPHerrMax,UnPHEerrMax])
        return sE,UnPP,UnPH,UnPHE,eRR

            
