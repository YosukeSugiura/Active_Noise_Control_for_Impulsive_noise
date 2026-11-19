import argparse
from pathlib import Path

import numpy as np
import scipy.signal as ss
M=512; fs=16000
def wien(p,s,M,col=None):
    # white (or coloured by filter col) input: min || c*(p - s*w) ||
    if col is not None: p=np.convolve(p,col); s=np.convolve(s,col)
    L=len(p); A=np.zeros((L,M))
    for k in range(M):
        n=min(len(s),L-k); A[k:k+n,k]=s[:n]
    w=np.linalg.lstsq(A,p,rcond=None)[0]
    return 10*np.log10(np.sum((p-A@w)**2)/np.sum(p**2))
lp=ss.firwin(129,1000,fs=fs); lp2=ss.firwin(129,2000,fs=fs)
def ph(S,S2,fmax):
    N=8192; A=np.fft.rfft(S,N); B=np.fft.rfft(S2,N); f=np.fft.rfftfreq(N,1/fs); m=f<fmax
    d=np.abs(np.angle(B*np.conj(A)))[m]; wgt=(np.abs(A)**2)[m]
    return np.sum(wgt*(d>np.pi/2))/np.sum(wgt)
def main():
    parser = argparse.ArgumentParser(description="White-input causal least-squares ANC bound")
    parser.add_argument("--paths16", action="store_true")
    parser.add_argument("--off", type=int)
    args = parser.parse_args()
    offset = args.off if args.off is not None else (0 if args.paths16 else 1880)
    if offset < 0:
        parser.error("--off must be nonnegative")
    directory = Path(__file__).resolve().parent / ("paths16" if args.paths16 else "paths")
    if args.paths16:
        for fn in sorted(directory.glob("pair_*.npz")):
            with np.load(fn, allow_pickle=False) as z:
                secondary = z["S"][offset:offset + 2048]
                result = [f"{key} {wien(z[key][offset:offset + 4096], secondary, M):.4f} dB"
                          for key in ("P", "P_near", "P_far")]
                print(" | ".join([fn.stem, *result]))
        return
    for fn in sorted(directory.glob("pair_*.npz")):
        z=np.load(fn); P=z["P"][offset:offset+4096]; S=z["S"][offset:offset+2048]
        out=[fn.stem, "ANRopt full %.1f"%wien(P,S,M), "<2k %.1f"%wien(P,S,M,lp2), "<1k %.1f"%wien(P,S,M,lp)]
        for d in ["5cm","10cm","20cm","40cm"]:
            S2=z["S_"+d][offset:offset+2048]
            out.append("%s bad-phase frac full/2k/1k %.2f/%.2f/%.2f"%(d,ph(S,S2,8000),ph(S,S2,2000),ph(S,S2,1000)))
        print(" | ".join(out))


if __name__ == "__main__":
    main()
