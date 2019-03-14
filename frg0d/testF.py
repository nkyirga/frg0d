from frgFlow import fRG0D
from propG import scaleProp
from vertexF import vertexR
import auxFunctions as auxF
from auxFunctions import freqPoints
from auxFunctions import gaussianInt
from auxFunctions import padeWeights
from auxFunctions import calcScale
import numpy as np
from matplotlib import pyplot as plt

def hF(wM):
    return 1j*(np.sign(wM))

def uF(wPP,wPH,wPHE):
    return 6.0

def algForMap(wX,fA):
    return wX/np.sqrt(wX**2+fA**2)
def algBackMap(zX,fA):
    return (fA*zX)/np.sqrt(1-zX**2)

def logForMap(wX,fA):
    return np.tanh(wX/fA)
def logBackMap(zX,fA):
    return fA*np.arctan(zX)

beTa=50.0
maxW=100
nPatches=20
wF,wFG=auxF.padeWeights(maxW,beTa)

wFX=np.append(-wF[::-1],wF)
wFG=np.append(wFG[::-1],wFG)
        
a,wB=auxF.freqPoints(beTa,maxW,nPatches)

wBE=np.append(-wB[:0:-1],wB)
wBe=wBE[:]
def slitim(wQ,AC):
    freeG=1j*np.sign(wQ)
        
    regL=auxF.litim0(wQ,AC)
    dLitim=auxF.dLitim0(wQ,AC)
        
    return -dLitim/((1j*wQ+freeG+regL)**2)
    
def gLitim(wQ,AC):
    freeG=1j*np.sign(wQ)
        
    regL=auxF.litim0(wQ,AC)

    return 1/(1j*wQ+freeG+regL)
    
deltaL=0.1
NW=20

wBX=2*np.arange(-600,601)*(np.pi/beTa)
wBXX,wBGG=auxF.gaussianInt(wBE,4)

gPPo=np.zeros(len(wBE),dtype=np.complex_)
for j in range(len(wBE)):
    gPPo[j]=(1/beTa)*np.sum(wFG*gLitim(wFX,0)*gLitim(wBE[j]-wFX,0))

zFX,zFG=auxF.gaussianInt([-1,1],30)
cScale=1.0

uInt=3
uPPl=np.zeros((len(wBE),100),dtype=np.complex_)
logPPl=np.zeros((len(wBE),100),dtype=np.complex_)
algPPl=np.zeros((len(wBE),100),dtype=np.complex_)

algN=np.zeros(NW,dtype=np.complex_)
logN=np.zeros(NW,dtype=np.complex_)
ACl=np.zeros(100)
for i in range(1,100):
    AC=maxW*np.exp(-deltaL*i)
    ACl[i]=AC
    gPP=np.zeros(len(wBE))
    for j in range(len(wBE)):
        gPP[j]=(1/beTa)*np.sum(wFG*slitim(wFX,AC)*gLitim(wBE[j]+wFX,AC))
        gPP[j]+=(1/beTa)*np.sum(wFG*slitim(wFX,AC)*gLitim(-wBE[j]+wFX,AC))

    uPPl[:,i]=uPPl[:,i-1]+deltaL*AC*(uInt+uPPl[:,i-1])*gPP*(uInt+uPPl[:,i-1])

    fA=np.sqrt(AC**2+0.1)
    zFX,a=auxF.gaussianInt([-1,1],30)
    zFX=np.append(zFX,algForMap(wBE,fA))
    zFX=np.unique(zFX)
    zFAlgX,zFAlgG=auxF.gaussianInt(zFX,8)

    zFX,a=auxF.gaussianInt([-1,1],30)
    zFX=np.append(zFX,logForMap(wBE,fA))
    zFX=np.unique(zFX)
    zFLogX,zFLogG=auxF.gaussianInt(zFX,8)

    uPPElog=np.zeros(len(zFLogX),dtype=np.complex_)
    uPPEalg=np.zeros(len(zFAlgX),dtype=np.complex_)

    uPPElog=np.interp(logBackMap(zFLogX,fA),wBE,uPPl[:,i].real)+0*1j
    uPPElog+=1j*np.interp(logBackMap(zFLogX,fA),wBE,uPPl[:,i].imag)
    
    uPPEalg=np.interp(algBackMap(zFAlgX,fA),wBE,uPPl[:,i].real)+0*1j
    uPPEalg+=1j*np.interp(algBackMap(zFAlgX,fA),wBE,uPPl[:,i].imag)
    
    bTempLog=np.zeros(len(wBE),dtype=np.complex_)
    bTempAlg=np.zeros(len(wBE),dtype=np.complex_)
    
    for j in range(NW):
        algN[j]=np.sum(zFAlgG*auxF.freqExpansion(zFAlgX,j)*uPPEalg)
        logN[j]=np.sum(zFLogG*auxF.freqExpansion(zFLogX,j)*uPPElog)
        
        bTempLog+=logN[j]*auxF.freqExpansion(logForMap(wBE,fA),j)
        bTempAlg+=algN[j]*auxF.freqExpansion(algForMap(wBE,fA),j)
    
    logPPl[:,i]=bTempLog.real
    algPPl[:,i]=bTempAlg.real

    #logPPl[:,i]=np.interp(wBE,fA*wBE,bTempLog.real)
    #algPPl[:,i]=np.interp(wBE,fA*wBE,bTempAlg.real)

print ACl[10],ACl[40],ACl[80]
plt.style.use('seaborn-paper')

plt.plot(np.arange(0,NW,2),logN[::2],'-o',label='$X_n^{Log}$',color='darkorchid')
plt.plot(np.arange(1,NW,2),logN[1::2],'o',color='red')
plt.plot(np.arange(0,NW,2),algN[::2],'-s',label='$X_n^{Alg}$',color='deepskyblue')
plt.plot(np.arange(1,NW,2),algN[1::2],'s',color='red')
plt.legend(loc='upper right',shadow=False)
plt.xlabel('$N_\omega$')
plt.ylabel('$X_n$')
plt.savefig('uPPN.eps',format='eps',dpi=1200)
plt.close()
plt.style.use('seaborn-paper')
fg=plt.figure(figsize=(6,9))

ax0=plt.subplot(311)
ax1=plt.subplot(312)
ax2=plt.subplot(313)
plt.subplots_adjust(left=0.08,right=0.96,top=0.96,bottom=0.08,hspace=0.05)
ax0.plot(wBE,uPPl[:,10],label='$X^{\Lambda=36\Delta}(\omega)$',linewidth=1.0,color='royalblue')
ax0.plot(wBE,logPPl[:,10],'o-',markersize=5,label='$X_{Log}(\omega)$',linewidth=1.0,color='darkorchid')
ax0.plot(wBE,algPPl[:,10],'s-',markersize=5,label='$X_{Alg}(\omega)$',linewidth=1.0,color='deepskyblue')

ax1.plot(wBE,uPPl[:,40],label='$X^{\Lambda=1.8\Delta}(\omega)$',linewidth=1.0,color='royalblue')
ax1.plot(wBE,logPPl[:,40],'o-',markersize=5,label='$X_{Log}(\omega)$',linewidth=1.0,color='darkorchid')
ax1.plot(wBE,algPPl[:,40],'s-',markersize=5,label='$X_{Alg}(\omega)$',linewidth=1.0,color='deepskyblue')

ax2.plot(wBE,uPPl[:,80],label='$X^{\Lambda=0.03\Delta}(\omega)$',linewidth=1.0,color='royalblue')
ax2.plot(wBE,logPPl[:,80],'o-',markersize=5,label='$X_{Log}(\omega)$',linewidth=1.0,color='darkorchid')
ax2.plot(wBE,algPPl[:,80],'s-',markersize=5,label='$X_{Alg}(\omega)$',linewidth=1.0,color='deepskyblue')

ax0.tick_params(labelbottom=False)
ax1.tick_params(labelbottom=False)
ax2.set_xlabel('$\omega/\Delta$')
ax0.set_ylabel('$X^\Lambda(\omega)$',labelpad=-6)
ax1.set_ylabel('$X^\Lambda(\omega)$',labelpad=-4)
ax2.set_ylabel('$X^\Lambda(\omega)$',labelpad=-2)
ax0.legend(loc='upper right',shadow=False,frameon=True)
ax1.legend(loc='upper right',shadow=False,frameon=True)
ax2.legend(loc='upper right',shadow=False,frameon=True)
plt.savefig('uPPLambda.eps',format='eps',dpi=1200)

besselTest=np.zeros(30,dtype=np.complex_)
bTest=np.zeros(len(wBE),dtype=np.complex_)

gPP=np.zeros(len(wBE),dtype=np.complex_)
for i in range(len(wBE)):
    gPP[i]=np.sum(wFG*(1/((1j*wFX+hF(wFX))*(1j*(wBE[i]-wFX)+hF(wBE[i]-wFX)))))

(zFull,b)=auxF.gaussianInt([-1,1],10)
tFull=(beTa/2)*(zFull+1)
print tFull

xMin=-np.log(0.01/(beTa-0.01))
xL=np.linspace(-xMin,xMin,20)
tFull=np.unique(np.append(tFull,beTa*np.exp(xL)/(1+np.exp(xL))))
(tFX,tFG)=auxF.gaussianInt(tFull,2)

wFE=2*np.arange(-200,201)*(np.pi/beTa)
gTT=np.zeros(len(tFX),dtype=np.complex_)
gPPE=np.interp(wFE,wBE,gPP.real)

l=np.linspace(0,10,20)
Al=(np.max(wFE)/5)*np.exp(-l)
fTT=np.zeros((len(tFX),20),dtype=np.complex_)
for i in range(len(tFX)):
    for j in range(20):
        fTT[i,j]=(1/beTa)*np.sum(np.exp(1j*wFE*tFX[i])*((wFE**2)/(wFE**2+Al[j]**2)))
    gTT[i]=(1/beTa)*np.sum(np.exp(1j*wFE*tFX[i])*gPPE)

for j in range(20):
    plt.plot(tFX,fTT[:,j])
    plt.show()

for i in range(30):
    besselTest[i]=(1/beTa)*np.sum(gPPE*np.sqrt(beTa)*auxF.besselExpansion(wFE*beTa,i))
    bTest+=besselTest[i]*np.sqrt(beTa)*auxF.besselExpansion(-wBE*beTa,i)
plt.plot(besselTest,'o')
plt.show()
plt.plot(wBE,bTest.real,'-o')
plt.plot(wBE,bTest.imag,'-.')
plt.plot(wBE,gPP)
plt.show()

gExpand=np.interp(zFX/np.sqrt(1-zFX**2),wBE,gPP.real)
plt.plot(zFX/np.sqrt(1-zFX**2),gExpand)
plt.show()
lCoeff=np.zeros(20)
gTest=np.zeros(len(zFX))
for i in range(20):
    print i
    lCoeff[i]=np.sum(zFG*gExpand*auxF.freqExpansion(zFX,i))
    gTest+=auxF.freqExpansion(zFX,i)*lCoeff[i]
    plt.plot(zFX/np.sqrt(1-zFX**2),gTest)
    plt.plot(wBE,gPP,'o-')
    plt.show()
"""
tMin=0.01
NW=10
xM=-np.log(tMin/(beTa-tMin))
zI=np.linspace(-xM,xM,NW)
tVals=beTa*(np.exp(zI)/(1+np.exp(zI)))
zVals=np.log(tVals/(beTa-tVals))
plt.plot(tVals,np.exp(-0.5*(zVals**2)),'.-')
plt.show()
zT=zVals[1]-zVals[0]
jacB=beTa*(np.exp(zVals)/((1+np.exp(zVals))**2))



wBF=((2*np.pi)/beTa)*np.arange(-200,200)
gTarget=np.zeros(len(wBF),dtype=np.complex_)
gTarget=np.interp(wBF,wBE,a[:,0,0].real)

plt.plot(wBF,gTarget)
plt.show()
ftG=np.zeros(len(tVals),dtype=np.complex_)
for i in range(len(tVals)):
    ftG[i]=(1/beTa)*np.sum(np.exp(1j*tVals[i]*wBF)*gTarget)
plt.plot(tVals,ftG.real,'o-')
plt.plot(tVals,ftG.imag)
plt.show()
fInv=np.zeros(len(wBF),dtype=np.complex_)
for i in range(len(wBF)):
    fInv[i]=np.sum(zT*jacB*np.exp(-1j*tVals*wBF[i])*ftG)

plt.plot(wBF,fInv,'o-')
plt.plot(wBF,gTarget)
plt.show()

plt.plot(wBF,fInv.real/gTarget.real,'.-')
plt.show()
(wFX,wFG)=padeWeights(100,50)
print len(wFX)
wB=4*testRun.UnF.wB
wFO=wFX[wFX<=100]
wFX=np.append(-wFX[::-1],wFX)
wFG=np.append(wFG[::-1],wFG)


(wFi,wFXi)=np.meshgrid(wFO,wFX,indexing='ij')
wShape=wFi.shape
wFi=np.reshape(wFi,wFi.size)
wFXi=np.reshape(wFXi,wFXi.size)
wFGi=np.reshape(np.tile(wFG[np.newaxis,:],(len(wFO),1)),wFXi.size)

mixPP2=np.zeros(len(wB))
for i in range(len(wB)):
    intG=(wFG*np.exp(1j*wB[i]*0.01))/((1j*wFX+hF(wFX))*(1j*(wB[i]-wFX)+hF(wB[i]-wFX)))
    mixPP2[i]=(1/50.0)*np.sum(intG)
wMidI=np.append(-wB[:0:-1],wB)
mixPPI=np.append(mixPP2[:0:-1],mixPP2)
mixPPX=np.interp(wFi+wFXi,wMidI,mixPPI)

plt.plot(mixPPX,'.-')
plt.show()

mixPPF=-(1/50.0)*np.sum(np.reshape(mixPPX*np.exp(1j*wFXi*0.01)*(wFGi/(1j*wFXi+hF(wFXi))),wShape),axis=1).imag
plt.plot(wFO,mixPPF,'o-')
plt.plot(wFO,-0.25/wFO,'-x')
plt.show()

testRun=fRG0D(20,0.1,40.0,100.0,10,'litim')
testRun.initializeFlow(hF,uF,1)
(c,d)=testRun.propG.gBubbles(testRun.UnF.wB,0.0,10,1.0)

testRun=fRG0D(20,0.1,80.0,100.0,10,'litim')
testRun.initializeFlow(hF,uF,1)
(e,f)=testRun.propG.gBubbles(testRun.UnF.wB,0.0,10,1.0)

testRun=fRG0D(20,0.1,100.0,100.0,10,'litim')
testRun.initializeFlow(hF,uF,1)
(g,h)=testRun.propG.gBubbles(testRun.UnF.wB,0.0,10,1.0)

plt.plot(testRun.UnF.wB,b[:,0,0])
plt.plot(testRun.UnF.wB,d[:,0,0])
plt.plot(testRun.UnF.wB,f[:,0,0])
plt.plot(testRun.UnF.wB,h[:,0,0])
plt.show()

(c,s)=testRun.susFunctions()

#plt.plot(testRun.UnF.wB,testRun.UnF.UnPHE[:,0,0].real)
#plt.show()
#plt.plot(cayleyMap(testRun.UnF.wB,1.0),testRun.UnF.UnPHE[:,0,0].real)
#plt.show()
plt.plot(testRun.propG.wF,testRun.propG.sE.imag,'o-')
plt.show()
plt.plot(testRun.UnF.wB,c,'o-')
plt.show()

print c[0],(2/(2*np.pi))*np.trapz(np.transpose(c.real),testRun.UnF.wB)
print s[0],(2/(2*np.pi))*np.trapz(np.transpose(s.real),testRun.UnF.wB)
plt.plot(testRun.UnF.wB,s,'o-')
plt.show()
"""
