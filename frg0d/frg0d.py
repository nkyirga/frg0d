import argparse
from genFRG0 import frgRun

def main():
    parser = argparse.ArgumentParser(description='fRG flow for SIAM')
    parser.add_argument('nPatches',metavar='Nf',type=int,
                   help='resolution of frequency axis')
    parser.add_argument('step',metavar='dl',type=float,
                    help='initial step')
    parser.add_argument('beTa',metavar='T',type=float,nargs=2,
                    help='Inverse temperature')
    parser.add_argument('wMax',metavar='A',type=float,
                    help='Initial scale of flow')
    parser.add_argument('NW',metavar='NW',type=int,
                    help='Order of expansion for non singular contributions')
    parser.add_argument('couplingU',metavar='U',type=float,nargs=2,
                    help='strength of couloumb interaction at impurity site')
    parser.add_argument('cutoffR',metavar='R(A)',type=str,
                    help='type of regulator for flow: Litim,Additive,...')
    parser.add_argument('nLoop',metavar='nL',type=int,
                    help='number of loops')
    parser.add_argument('Mu',metavar='Mu',type=float,nargs=2,
                    help='chemical potential')

    args = parser.parse_args()
    frgRun(args)
