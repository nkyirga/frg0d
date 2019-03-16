from frgFlow import fRG0D
from propG import scaleProp
from vertexF import vertexR
from tempfile import TemporaryFile
import numpy as np

def frgRun(args):
    wF=(np.pi/args.beTa[0])*(2*np.arange(10000)+1)
    wF=np.append(-wF[:0:-1],wF)
    
    kX=np.linspace(-np.pi,np.pi,1000)
    gLoc=np.zeros(len(wF),dtype=np.complex_)
    for i in range(len(wF)):
        gKint=1.0/(1j*wF[i]+2*np.cos(kX))
        gLoc[i]=(1.0/(2*np.pi))*np.trapz(gKint,kX)
       
    hFI=1j*wF-1/gLoc
    def hF(wM):
        hFr=np.interp(wM,wF,hFI.real)
        hFi=np.interp(wM,wF,hFI.imag)
        return hFr+1j*hFi

    def hF2(wM):
        return 1j*(np.sign(wM))

    def uF(wPP,wPH,wPHE):
        return args.couplingU[0]+np.zeros(wPP.shape)

    wMin=(np.pi)/args.beTa[0]
    lMax=-np.log(wMin/args.wMax)

    fRGrun=fRG0D(args.nPatches,args.step,args.beTa[0],args.wMax,args.NW,args.cutoffR)
    fRGrun.initializeFlow(hF,uF,args.nLoop,args.Mu[0])

    fRGrun.adaptiveRGFlow(lMax)
    (c,s)=fRGrun.susFunctions()
    gamma0=fRGrun.UnF.uEvaluate(0*fRGrun.propG.wF[0],np.zeros(1),np.zeros(1),0.00)
    wFI=fRGrun.propG.wF[0:2]
    wFI=np.append(-wFI[::-1],wFI)
    sEI=fRGrun.propG.sE[0:2]
    sEI=np.append(np.conj(sEI[::-1]),sEI)
    sEd=np.polyfit(wFI,sEI.imag,2)[1]

    cX=(1/np.pi)*(1-sEd-gamma0/np.pi)
    xx=np.linspace(0,100,100000)
    uU=args.couplingU[0]
    xxI=(np.exp(-xx**2/(2.0*uU/np.pi))*np.cos(np.pi*xx/2.0))/(1.0-xx**2)
    cSexact=np.trapz(xxI,xx)*np.exp(np.pi*uU/8.0)*(1/np.pi)*np.sqrt(2.0/uU)
    cS=(1/np.pi)*(1-sEd+gamma0/np.pi)
    xxI=(np.exp(-xx**2/(2.0*uU/np.pi))*np.cosh(np.pi*xx/2.0))/(1.0+xx**2)
    cXexact=np.trapz(xxI,xx)*np.exp(-np.pi*uU/8.0)*(1/np.pi)*np.sqrt(2.0/uU)

    nMax=np.floor(0.5*((args.wMax*args.beTa[0]/np.pi)-1))
    wQ=(np.pi/args.beTa[0])*2*np.arange(0,nMax,1)

    dF=np.interp(wQ,fRGrun.UnF.wB,c.real)
    dF=np.append(dF[:0:-1],dF)

    print 'dFac',(1.0/args.beTa[0])*np.sum(dF)

    print cX,cXexact,cS,cSexact
    print 'QuasiP: ',1-sEd,(np.pi/2.0)*(cXexact+cSexact)
    print 'VertexG: ',gamma0,(np.pi**2/2.0)*(cSexact-cXexact)
    print c[0],s[0],cXexact,cSexact
    filename=''.join(['couplingU',str(int(args.couplingU[1])),'Mu',str(int(args.Mu[1])),'R',args.cutoffR,
                  'SIAMnL',str(args.nLoop),'nf',str(args.nPatches),'NW',str(args.NW),'bT',str(int(args.beTa[1]))])
    np.savez(filename,xC=c,xS=s,wB=fRGrun.UnF.wB,wF=fRGrun.propG.wF,sE=fRGrun.propG.sE,lM=fRGrun.l,\
             bT=args.beTa[0],uU=args.couplingU[0],g00=gamma0,gPP=fRGrun.UnF.UnPP[0,:,:],gPH=fRGrun.UnF.UnPH[0,:,:],
             gPHE=fRGrun.UnF.UnPHE[0,:,:])

