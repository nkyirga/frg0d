import numpy as np
import auxFunctions as auxF
import interpP1F as intPP

class scaleProp:
    def __init__(self,maxW,freeG,beTa,Mu,cutoff='litim'):
        wF,wFG=auxF.padeWeights(maxW,beTa)
        
        self.g0=freeG
        self.beTa=beTa
        self.Mu=Mu
        
        self.wF=wF[wF<maxW]
        self.wFI=np.append(-self.wF[::-1],self.wF)
        
        self.wFX=np.append(-wF[::-1],wF)
        self.wFG=np.append(wFG[::-1],wFG)
        self.sE=np.zeros(len(self.wF),dtype=np.complex_)
        
        self.sEF=self.sEInterp 


        if cutoff=='litim':
            self.gF=self._gLitim
            self.sF=self._sScale_litim
            
        elif cutoff=='additive':
            self.gF=self._gAdditive
            self.sF=self._sScale_additive
        
        elif cutoff=='sharp':
            self.gF=self._gSharp
            self.sF=self._sScale_sharp
        
        elif cutoff=='soft':
            self.gF=self._gSoft
            self.sF=self._sScale_soft
    
    def sEInterp(self,wQ):
        sE=self.sE
        sEI=np.append(np.conj(sE[::-1]),sE)
        
        sEQ=np.zeros(len(wQ),dtype=np.complex_)
        sEQ+=np.interp(wQ,self.wFI,sEI.real)
        sEQ+=1j*np.interp(wQ,self.wFI,sEI.imag)

        return sEQ
    
    def xBubbles(self,wQ,dSEwMid,AC,NW):
        beTa=self.beTa
        wFX=self.wFX
        wFG=self.wFG
        wFI=self.wFI
        
        dSE=np.zeros(len(wFX),dtype=np.complex_)
        dSEI=np.append(np.conj(dSEwMid[::-1]),dSEwMid)
        dSE+=np.interp(wFX,wFI,dSEI.real)
        dSE+=1j*np.interp(wFX,wFI,dSEI.imag)

        gProp=self.gF
        sProp=self.sF(wFX,AC)+dSE*(gProp(wFX,AC)**2)
        
        sPropPP=np.zeros((len(wQ),len(wFX)),dtype=np.complex_)
        sPropPH1=np.zeros(sPropPP.shape,dtype=np.complex_)
        sPropPH2=np.zeros(sPropPP.shape,dtype=np.complex_)
        for i in range(len(wQ)):
            sPropPP[i,:]=wFG*sProp*gProp(wQ[i]-wFX,AC)
            sPropPH1[i,:]=wFG*sProp*gProp(wQ[i]+wFX,AC)
            sPropPH2[i,:]=wFG*sProp*gProp(-wQ[i]+wFX,AC)    
    
        (wS,wX)=np.meshgrid(wQ,wFX,indexing='ij')
        wPP1=0.5*(wS-2*wX)/np.sqrt(AC**2+1)
        wPP2=0.5*(-wS+2*wX)/np.sqrt(AC**2+1)

        wPH1=0.5*(wS+2*wX)/np.sqrt(AC**2+1)
        wPH2=0.5*(-wS+2*wX)/np.sqrt(AC**2+1)

        lTempPP1=np.zeros((len(wQ),len(wFX),NW))
        lTempPP2=np.zeros((len(wQ),len(wFX),NW))
        
        lTempPH1=np.zeros((len(wQ),len(wFX),NW))
        lTempPH2=np.zeros((len(wQ),len(wFX),NW))
        for i in range(NW):
            lTempPP1[...,i]=auxF.freqExpansion(auxF.forMap(wPP1,1.0),2*i)
            lTempPP2[...,i]=auxF.freqExpansion(auxF.forMap(wPP2,1.0),2*i)
            
            lTempPH1[...,i]=auxF.freqExpansion(auxF.forMap(wPH1,1.0),2*i)
            lTempPH2[...,i]=auxF.freqExpansion(auxF.forMap(wPH2,1.0),2*i)
            
        mixPP=np.zeros((len(wQ),NW,NW),dtype=np.complex_)
        mixPH=np.zeros((len(wQ),NW,NW),dtype=np.complex_)
        for i in range(NW):
            for j in range(NW):
                intGPP=(lTempPP1[...,i]*lTempPP2[...,j]+\
                            lTempPP2[...,i]*lTempPP1[...,j])*sPropPP

                intGPH=lTempPH1[...,i]*lTempPH1[...,j]*sPropPH1+\
                    lTempPH2[...,i]*lTempPH2[...,j]*sPropPH2
                
                mixPP[:,i,j]=(1/beTa)*np.sum(intGPP,axis=1)
                mixPH[:,i,j]=(1/beTa)*np.sum(intGPH,axis=1)
        
        return mixPP,mixPH

    
    def gBubbles(self,wQ,AC,NW):
        wFX=self.wFX
        wFG=self.wFG
        beTa=self.beTa

        gProp=self.gF
        gPropX=gProp(wFX,AC)

        gPropPP=np.zeros((len(wQ),len(wFX)),dtype=np.complex_)
        gPropPH1=np.zeros(gPropPP.shape,dtype=np.complex_)
        gPropPH2=np.zeros(gPropPP.shape,dtype=np.complex_)
        for i in range(len(wQ)):
            gPropPP[i,:]=wFG*gPropX*gProp(wQ[i]-wFX,AC)
            gPropPH1[i,:]=wFG*gPropX*gProp(wQ[i]+wFX,AC)
            gPropPH2[i,:]=wFG*gPropX*gProp(-wQ[i]+wFX,AC)
        
        
        (wS,wX)=np.meshgrid(wQ,wFX,indexing='ij')
        wPP1=0.5*(wS-2*wX)/np.sqrt(AC**2+1)
        wPP2=0.5*(-wS+2*wX)/np.sqrt(AC**2+1)

        wPH1=0.5*(wS+2*wX)/np.sqrt(AC**2+1)
        wPH2=0.5*(-wS+2*wX)/np.sqrt(AC**2+1)

        lTempPP1=np.zeros((len(wQ),len(wFX),NW))
        lTempPP2=np.zeros((len(wQ),len(wFX),NW))
        
        lTempPH1=np.zeros((len(wQ),len(wFX),NW))
        lTempPH2=np.zeros((len(wQ),len(wFX),NW))
        for i in range(NW):
            lTempPP1[...,i]=auxF.freqExpansion(auxF.forMap(wPP1,1.0),2*i)
            lTempPP2[...,i]=auxF.freqExpansion(auxF.forMap(wPP2,1.0),2*i)
            
            lTempPH1[...,i]=auxF.freqExpansion(auxF.forMap(wPH1,1.0),2*i)
            lTempPH2[...,i]=auxF.freqExpansion(auxF.forMap(wPH2,1.0),2*i)
            
        gPP=np.zeros((len(wQ),NW,NW),dtype=np.complex_)
        gPH=np.zeros((len(wQ),NW,NW),dtype=np.complex_)
        for i in range(NW):
            for j in range(NW):
                intGPP=(lTempPP1[...,i]*lTempPP2[...,j]+\
                            lTempPP2[...,i]*lTempPP1[...,j])*gPropPP

                intGPH=lTempPH1[...,i]*lTempPH1[...,j]*gPropPH1+\
                    lTempPH2[...,i]*lTempPH2[...,j]*gPropPH2
        
                gPP[:,i,j]=(0.5/beTa)*np.sum(intGPP,axis=1)
                gPH[:,i,j]=(0.5/beTa)*np.sum(intGPH,axis=1)
                
        return gPP,gPH
    
    def susBubbles(self,wQ,NW):
        wFX=self.wFX
        wFG=self.wFG
        beTa=self.beTa

        gProp=self.gF
        gPropX=gProp(wFX,0.0)
        
        gPropPHL=np.zeros((len(wQ),len(wFX)),dtype=np.complex_)
        gPropPHR=np.zeros((len(wQ),len(wFX)),dtype=np.complex_)
        for i in range(len(wQ)):
            gPropPHL[i,:]=wFG*gPropX*gProp(-wQ[i]+wFX,0.0)
            gPropPHR[i,:]=wFG*gPropX*gProp(-wQ[i]+wFX,0.0)

        gPHl=(1/beTa)*np.sum(gPropPHL,axis=1)
        gPHr=(1/beTa)*np.sum(gPropPHR,axis=1)
        (wS,wX)=np.meshgrid(wQ,wFX,indexing='ij')
        
        sPHL=np.zeros((len(wQ),1,NW),dtype=np.complex_)
        sPHR=np.zeros((len(wQ),NW,1),dtype=np.complex_)
        for i in range(NW):
            lTempWPHL=auxF.freqExpansion(auxF.forMap(0.5*(2*wX-wS),1.0),2*i)
            lTempWPHR=auxF.freqExpansion(auxF.forMap(0.5*(2*wX-wS),1.0),2*i)

            sPHL[:,0,i]=(1/beTa)*np.sum(lTempWPHL*gPropPHL,axis=1)
            sPHR[:,i,0]=(1/beTa)*np.sum(lTempWPHR*gPropPHR,axis=1)
            
        return sPHL,sPHR,gPHl,gPHr
    def _gSoft(self,wQ,AC):
        sEQ=self.sEF(wQ)
        freeG=self.g0(wQ)
        Mu=self.Mu

        regL=auxF.softR(wQ,AC)
        return 1/((1j*wQ+Mu+freeG)*regL-sEQ)
    def _sScale_soft(self,wQ,AC):
        sEQ=self.sEF(wQ)
        freeG=self.g0(wQ)
        Mu=self.Mu

        regL=auxF.softR(wQ,AC)
        dSoft=auxF.dSoftR(wQ,AC)
        
        return -(dSoft*(1j*wQ+Mu+freeG))/(((1j*wQ+Mu+freeG)*regL-sEQ)**2)
    def _gSharp(self,wQ,AC):
        sEQ=self.sEF(wQ)
        freeG=self.g0(wQ)
        Mu=self.Mu

        regL=auxF.sharpR(wQ,AC)
        return regL/((1j*wQ+Mu+freeG)-regL*sEQ)
    def _sScale_sharp(self,wQ,AC):
        sEQ=self.sEF(wQ)
        freeG=self.g0(wQ)
        Mu=self.Mu

        regL=auxF.sharpR(wQ,AC)
        dSharp=auxF.dSharpR(wQ,AC)
        
        dDsharp=-((1j*wQ+Mu+freeG)*dSharp)/((1j*wQ+Mu+freeG-regL*sEQ)**2)
        return dDsharp
    def _gAdditive(self,wQ,AC):
        sEQ=self.sEF(wQ)
        freeG=self.g0(wQ)
        Mu=self.Mu
        
        regL=auxF.additiveR0(wQ,AC)

        return 1/(1j*wQ+Mu+freeG+regL-sEQ)
    def _sScale_additive(self,wQ,AC):
        sEQ=self.sEF(wQ)
        freeG=self.g0(wQ)
        Mu=self.Mu
        regL=auxF.additiveR0(wQ,AC)
        
        dAdditive=auxF.dAdditiveR0(wQ,AC)

        return -(dAdditive)/((1j*wQ+Mu+freeG+regL-sEQ)**2)
    
    def _gLitim(self,wQ,AC):
        sEQ=self.sEF(wQ)
        freeG=self.g0(wQ)
        Mu=self.Mu
        
        regL=auxF.litim0(wQ,AC)

        return 1/(1j*wQ+Mu+freeG+regL-sEQ)
    def _sScale_litim(self,wQ,AC):
        sEQ=self.sEF(wQ)
        freeG=self.g0(wQ)
        Mu=self.Mu
        
        regL=auxF.litim0(wQ,AC)
        dLitim=auxF.dLitim0(wQ,AC)
        
        return -dLitim/((1j*wQ+Mu+freeG+regL-sEQ)**2)
    
        
        
        
