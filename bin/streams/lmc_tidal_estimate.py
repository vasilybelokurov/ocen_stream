"""Order-of-magnitude: LMC tidal (differential) acceleration on omega Cen's orbit over the last 1 Gyr.
Usage: PYTHONPATH=src python bin/streams/lmc_tidal_estimate.py   (1 core, ~10 s)
LMC orbit: AGAMA example_lmc_mw_interaction.py recipe (rigid MW + rigid LMC, Chandrasekhar DF on LMC), MW = Hunter24 axi.
omega Cen orbit: Hunter24 axi only (no bar, no LMC) -- estimate only."""
import os, numpy as np, scipy.integrate, scipy.special
from ocen_dm.streams.restricted import agama_kpc, HUNTER24_DIR, AGAMA_T_MYR
from ocen_dm.streams.frames import to_model, ocen_today
ag = agama_kpc()
mw = ag.Potential(file=os.path.join(HUNTER24_DIR, "MWPotentialHunter24_axi.ini"))
for r in (8, 50, 100, 200):
    print(f"Hunter24 axi: r {r:4d} kpc  M(<r) {-r*r*mw.force([r,0,0])[0]/ag.G:.3e}  vc {np.sqrt(-r*mw.force([r,0,0])[0]):.1f}")
# LMC today (same observables as the AGAMA example), project solar frame
lmc0 = to_model(81.28, -69.78, 49.6, 1.858, 0.385, 262.2)[0]
mwsph = ag.Potential(type="Multipole", potential=mw, lmax=0, rmin=0.01, rmax=1000)
rg = np.logspace(0, 3, 30)
# isotropic Jeans sigma of the total spherical MW (approximation for the DF formula)
rho = mwsph.density(np.column_stack((rg, 0*rg, 0*rg))); Fr = -mwsph.force(np.column_stack((rg, 0*rg, 0*rg)))[:, 0]
rr = np.logspace(0, 3.5, 2000); rho_f = np.interp(np.log(rr), np.log(rg), np.log(rho)); 
rhoF = np.exp(rho_f)*(-mwsph.force(np.column_stack((rr, 0*rr, 0*rr)))[:, 0])
P = np.array([np.trapezoid(rhoF[i:], rr[i:]) for i in range(len(rr))]); sig = np.sqrt(P/np.exp(rho_f))
sigf = lambda r: np.interp(r, rr, sig)
T1 = -1.0/(AGAMA_T_MYR/1e3)*1.0   # 1 Gyr back [AGAMA units] (we integrate 3 Gyr to be safe)
for M in (1.0e11, 1.5e11, 2.0e11):
    a = (M/1e11)**0.6*8.5
    lmc = ag.Potential(type="spheroid", mass=M, scaleradius=a, outercutoffradius=10*a, gamma=1, beta=3)
    bmin = 2*a
    def f(v, t):
        dx = v[6:9]-v[0:3]; dv = v[9:12]-v[3:6]; d = np.linalg.norm(dx); vm = np.linalg.norm(dv)
        X = vm/(np.sqrt(2)*sigf(d)); cl = max(0, np.log(d/bmin))**0.5
        drag = -4*np.pi*mw.density(dx)*dv/vm*(scipy.special.erf(X)-2/np.sqrt(np.pi)*X*np.exp(-X*X))*M*ag.G**2/vm**2*cl
        return np.hstack((v[3:6], lmc.force(-dx), v[9:12], mw.force(dx)+drag))
    tg = np.linspace(0, -3.0/(AGAMA_T_MYR/1e3), 3001)
    s = scipy.integrate.odeint(f, np.hstack((np.zeros(6), lmc0)), tg)
    rel = s[:, 6:9]-s[:, 0:3]; dl = np.linalg.norm(rel, axis=1)
    ip = np.argmin(dl[:600])
    # omega Cen orbit back 1 Gyr in the axisymmetric MW
    t1 = 1000/AGAMA_T_MYR; tt, oc = ag.orbit(potential=mw, ic=ocen_today(), timestart=0, time=-t1, trajsize=2001, accuracy=1e-10)
    relI = np.array([np.interp(tt, tg[::-1], rel[::-1, i]) for i in range(3)]).T
    aL = lmc.force(oc[:, :3]-relI) - lmc.force(-relI)   # tidal (differential) LMC acceleration on omega Cen
    aM = np.linalg.norm(mw.force(oc[:, :3]), axis=1)
    dv = np.abs(np.trapezoid(aL, tt, axis=0)); dvmag = np.linalg.norm(np.trapezoid(aL, tt, axis=0))
    # cumulative |int a dt| over last 250 Myr
    m250 = tt > -250/AGAMA_T_MYR
    dv250 = np.linalg.norm(np.trapezoid(aL[m250], tt[m250], axis=0))
    print(f"M_LMC {M:.1e}: peri {dl[ip]:.1f} kpc {-tg[ip]*AGAMA_T_MYR:.0f} Myr ago; |a_tide|/|a_MW| at omega Cen: median "
          f"{np.median(np.linalg.norm(aL,axis=1)/aM):.2e}, today {np.linalg.norm(aL[0])/aM[0]:.2e}; |int a_tide dt| 1 Gyr {dvmag:.2f}, "
          f"250 Myr {dv250:.2f} km/s; max |a_tide| {np.linalg.norm(aL,axis=1).max():.1f} (km/s)^2/kpc; MW reflex |a_MW0| today "
          f"{np.linalg.norm(lmc.force(-rel[0])):.0f} (km/s)^2/kpc")
