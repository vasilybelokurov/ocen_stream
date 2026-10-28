"""Fable review 2026-10-28 (docs/reviews/error_floors_fable_codex_20261028.md): rescore logged models with per-observable floors. Usage: OCEN_RUNS_DIR=... PYTHONPATH=src python <this>"""
"""Rescore logged real-data models (m, e) under per-observable floors; profile along bar angle."""
import numpy as np, glob, os
R = os.environ["OCEN_RUNS_DIR"]
U=[];C=[];M=[];E=[];B=[];T=[]
for d in ("pilots_20261027","pilots_20261027_scan"):
    for f in sorted(glob.glob(f"{R}/{d}/log/0*.npz")):
        z=np.load(f); U.append(z["u"]);C.append(z["chi2"]);M.append(z["m"]);E.append(z["e"]);B.append(z["bank"]);T.append(z["tag"])
U=np.concatenate(U);C=np.concatenate(C);M=np.concatenate(M);E=np.concatenate(E);B=np.concatenate(B);T=np.concatenate(T)
S=np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../results/gpu/score_inputs.npz"))
D={k[2:]:np.asarray(S[k],float) for k in S.files if k.startswith("D_")}
k=7; na=6
blocks=[("sky_c",slice(0,7),None),("sky_s",slice(7,14),None),("pmra",slice(14,20),"pmra_C"),("pmdec",slice(20,25),"pmdec_C")]
ok=np.isfinite(C)&(B==7)&np.all(np.isfinite(M),1)&np.all(np.isfinite(E),1)
print("models logged",len(C),"usable (bank 7, finite)",ok.sum())
U,C,M,E=U[ok],C[ok],M[ok],E[ok]
angle=50*U[:,0]
# verify chi2 reconstruction
def block_chi2(name,sl,Cn,s2):
    """returns (chi2_block, logdet_rel) for an array of floor variances s2 (shape (ns,)) -> shapes (nm, ns)"""
    m=M[:,sl]; e=E[:,sl]; r=D[name]-m
    if Cn is None:
        de=D[name+"e"]; var=de[None,None,:]**2+e[:,None,:]**2+s2[None,:,None]
        return np.sum(r[:,None,:]**2/var,2), np.sum(np.log(var/de**2),2)
    Cd=D[Cn]; out=np.empty((len(m),len(s2))); ld=np.empty_like(out)
    for j,s in enumerate(s2):
        Cf=Cd[None]+np.einsum('ij,jk->ijk',e**2+s,np.eye(r.shape[1]))
        sol=np.linalg.solve(Cf,r[...,None])[...,0]; out[:,j]=np.sum(r*sol,1); ld[:,j]=np.linalg.slogdet(Cf)[1]-np.linalg.slogdet(Cd)[1]
    return out,ld
tot=sum(block_chi2(n,sl,Cn,np.array([0.]))[0][:,0] for n,sl,Cn in blocks)
print("chi2 reconstruction max |diff|",np.max(np.abs(tot-C)))
# floor grids: log-uniform in s from 0.01 to 30 x median data sigma of the block
sig={n:np.sqrt(np.median(np.diag(D[Cn])) if Cn else np.median(D[n+"e"]**2)) for n,sl,Cn in blocks}
print("median data sigma per block",{n:round(v,4) for n,v in sig.items()})
ns=61; grid={n:sig[n]*np.logspace(-2,1.5,ns) for n in sig}
res={}
for n,sl,Cn in blocks:
    c2,ld=block_chi2(n,sl,Cn,grid[n]**2); F=c2+ld          # -2 ln L_k(s) relative to no-floor normalisation
    # add s=0 column
    c20,ld0=block_chi2(n,sl,Cn,np.array([0.]))
    prof=np.min(F,1); jbest=np.argmin(F,1)
    # marginal with log-uniform prior on s over the grid (trapezoid in log s)
    w=np.exp(-0.5*(F-prof[:,None])); marg=prof-2*np.log(np.trapezoid(w,np.log(grid[n]),axis=1)/np.log(grid[n][-1]/grid[n][0]))
    res[n]=dict(c0=c20[:,0],prof=prof,marg=marg,sbest=grid[n][jbest],chi2_at_prof=c2[np.arange(len(c2)),jbest])
chi2_0=sum(res[n]["c0"] for n in res); Fprof=sum(res[n]["prof"] for n in res); Fmarg=sum(res[n]["marg"] for n in res)
ib=np.argmin(chi2_0); ip=np.argmin(Fprof); im=np.argmin(Fmarg)
def desc(i): return f"angle {angle[i]:5.1f} Omega {20+30*U[i,1]:4.1f} q {0.65+0.6*U[i,8]:.2f} vc {200+50*U[i,7]:.0f} d {5.28+0.5*U[i,4]:.2f}"
print("best chi2        ",round(chi2_0[ib],1),desc(ib))
print("best profiled F  ",round(Fprof[ip],1),desc(ip),"floors(sigma units)",{n:round(res[n]["sbest"][ip]/sig[n],2) for n in res}, "chi2 at floors",round(sum(res[n]["chi2_at_prof"][ip] for n in res),1),"plain chi2",round(chi2_0[ip],1))
print("best marginal F  ",round(Fmarg[im],1),desc(im),"plain chi2",round(chi2_0[im],1))
# profile along angle: min per 2.5-deg bin for chi2, Fprof, Fmarg; also per-block floors at the bin's best
edges=np.arange(0,52.5,2.5); print("\nangle bin | n | min chi2 | min Fprof (-min) | min Fmarg (-min) | floors sky_c sky_s pmra pmdec (sigma units) at Fprof best")
for lo in edges[:-1]:
    s=(angle>=lo)&(angle<lo+2.5)
    if s.sum()==0: continue
    j=np.nonzero(s)[0]; jp=j[np.argmin(Fprof[s])]
    print(f"{lo:4.1f}-{lo+2.5:4.1f} | {s.sum():5d} | {chi2_0[s].min()-chi2_0.min():6.1f} | {Fprof[s].min()-Fprof.min():6.1f} | {Fmarg[s].min()-Fmarg.min():6.1f} | "+" ".join(f"{res[n]['sbest'][jp]/sig[n]:.2f}" for n in res))
# single global scale: chi2/f^2 + 25 log f^2 profiled -> f^2 = chi2/25, F = 25 log(chi2/25)+25: monotone in chi2 -> same ranking. State it.
# per-observable multiplicative: F_k = n_k log(chi2_k/n_k)+n_k (profiled), floor scale >=1
nk={"sky_c":7,"sky_s":7,"pmra":6,"pmdec":5}
Fmul=sum(np.where(res[n]["c0"]>nk[n], nk[n]*np.log(res[n]["c0"]/nk[n])+nk[n], res[n]["c0"]) for n in res)
imu=np.argmin(Fmul); print("\nbest per-observable multiplicative (profiled)",round(Fmul[imu],1),desc(imu),"plain chi2",round(chi2_0[imu],1))
print("angle bin | min Fmul - min")
for lo in edges[:-1]:
    s=(angle>=lo)&(angle<lo+2.5)
    if s.sum(): print(f"{lo:4.1f} | {Fmul[s].min()-Fmul.min():6.1f}")
# correlation of e with chi2 (bias risk of log|C|): mean log(1+e^2/de^2) sum vs chi2
ld0=sum(block_chi2(n,sl,Cn,np.array([0.]))[1][:,0] for n,sl,Cn in blocks)
print("\nlog|C| term (no floor): median",np.median(ld0),"range",ld0.min(),ld0.max(), "corr with chi2 (chi2<200)",np.corrcoef(ld0[chi2_0<200],chi2_0[chi2_0<200])[0,1])
# e / sigma_data ratios
for n,sl,Cn in blocks:
    de=np.sqrt(np.diag(D[Cn])) if Cn else D[n+"e"]; good=chi2_0<160
    print(n,"median e/sigma_data over good models per knot",np.round(np.median(E[good][:,sl]/de,0),2))
np.savez(os.path.join(os.environ.get("TMPDIR", "/tmp"), "floors_rescore.npz"),angle=angle,chi2=chi2_0,Fprof=Fprof,Fmarg=Fmarg,Fmul=Fmul,U=U)
