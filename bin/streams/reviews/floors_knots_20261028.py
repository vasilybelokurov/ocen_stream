"""Fable review 2026-10-28 (docs/reviews/error_floors_fable_codex_20261028.md): rescore logged models with per-observable floors. Usage: OCEN_RUNS_DIR=... PYTHONPATH=src python <this>"""
import numpy as np, glob, os
exec(open(os.path.join(os.path.dirname(__file__),"floors_rescore_20261028.py")).read().split("res={}")[0])
z=np.load(os.path.join(os.environ.get("TMPDIR", "/tmp"), "floors_rescore.npz")); Fprof=z["Fprof"]; chi2_0=z["chi2"]
def resid(i):
    out={}
    for n,sl,Cn in blocks:
        r=D[n]-M[i,sl]; e=E[i,sl]
        de=np.sqrt(np.diag(D[Cn])) if Cn else D[n+"e"]
        out[n]=np.round(r/np.sqrt(de**2+e**2),1)
    return out
for lab,i in (("chi2 best",np.argmin(chi2_0)),("floor best",np.argmin(Fprof)),("scan-like (angle 5-10 best chi2)",np.nonzero((angle>5)&(angle<10))[0][np.argmin(chi2_0[(angle>5)&(angle<10)])])):
    print(lab, f"angle {angle[i]:.1f} chi2 {chi2_0[i]:.1f}"); 
    for n,v in resid(i).items(): print("   ",n,"resid/sigma per knot",v)
print("pmdec data",np.round(D["pmdec"],3),"sigma",np.round(np.sqrt(np.diag(D["pmdec_C"])),3))
print("pmra  data",np.round(D["pmra"],3),"sigma",np.round(np.sqrt(np.diag(D["pmra_C"])),3))
print("pmdec corr matrix\n",np.round(D["pmdec_C"]/np.sqrt(np.outer(np.diag(D["pmdec_C"]),np.diag(D["pmdec_C"]))),2))
# variant A: drop the last pmdec knot (phi1 28) and last pmra knot (28); plain chi2 profile along angle
def chi2_drop(dropA,dropD):
    tot=np.zeros(len(M))
    for n,sl,Cn in blocks:
        r=D[n]-M[:,sl]; e=E[:,sl]
        if Cn is None: de=D[n+"e"]; tot+=np.sum(r**2/(de**2+e**2),1); continue
        keep=np.ones(r.shape[1],bool)
        if n=="pmra": keep[dropA]=False
        if n=="pmdec": keep[dropD]=False
        Cd=D[Cn][np.ix_(keep,keep)]; rr=r[:,keep]; ee=e[:,keep]
        Cf=Cd[None]+np.einsum('ij,jk->ijk',ee**2,np.eye(keep.sum())); sol=np.linalg.solve(Cf,rr[...,None])[...,0]; tot+=np.sum(rr*sol,1)
    return tot
cA=chi2_drop([5],[4])
# variant B: half-normal prior on floors with scale 1 sigma_data (tight), profiled (MAP in s)
res2={}
for n,sl,Cn in blocks:
    c2,ld=block_chi2(n,sl,Cn,grid[n]**2); F=c2+ld+(grid[n]/sig[n])[None,:]**2
    res2[n]=np.min(F,1)
FhN=sum(res2.values())
print("\nangle bin | min chi2 | min chi2 (drop end PM knots) | min F half-normal(1 sigma) floors | (all minus own min)")
for lo in np.arange(0,50,2.5):
    s=(angle>=lo)&(angle<lo+2.5)
    if s.sum(): print(f"{lo:4.1f} | {chi2_0[s].min()-chi2_0.min():6.1f} | {cA[s].min()-cA.min():6.1f} | {FhN[s].min()-FhN.min():6.1f}")
for lab,v in (("drop-end",cA),("half-normal",FhN)):
    i=np.argmin(v); print(lab,"best at angle",round(angle[i],1),"value",round(v[i],1),"plain chi2",round(chi2_0[i],1))
