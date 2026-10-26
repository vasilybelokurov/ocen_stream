/* Per-particle DOP853 spray integration in C (GPU pipeline step 1, 2026-10-26; from Fable's prototype dev/kernel/kern.c).
   Same code shape as the intended CUDA kernel (DEV = __device__ there). Mirrors jaxspray.phi/accel/centre and jaxmodel._init_step /
   _attempts with analytic B-spline derivatives (vs jax.grad: max 5e-12, median 3e-14 relative) and AGAMA's DOP853 controller
   (math_ode.cpp: initTimeStep, doStep incl. nbad/preverr, sk == 0 skip, three square roots; orbit.cpp accuracy factor).
   Entry points: force_many, integrate_many (from release), integrate_resume (from a mid-integration state: the GPU hand-off of the
   last particles), orbit_back (host-only centre orbit from T back to 0, every accepted step recorded).
   Built by ocen_stream.cspray with -O2 -ffp-contract=off (fixed flags: results depend on contraction at the 1e-16 level).        */
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#ifdef __CUDACC__
#define DEV __device__ __forceinline__
#else
#define DEV static inline
#endif

typedef struct {
    const double *axi;      /* (nu+4)*(nw+4), padded, row-major [u][w]            (per model)   */
    const double *bar;      /* nm*(nu+4)*(nw+4), padded, [mode][u][w]              (shared)      */
    const double *m;        /* nm Fourier orders                                                 */
    int nm, nu, nw;
    double du, dw, rs, zs;
    double A, S, theta_T, omega, T;                                   /* A,S,theta_T,omega per model */
    const double *sat; int nr; double lr0, dlr, mtot, G;             /* progenitor profile */
    const double *orb; int norb; double tc0, dtc;                     /* centre orbit (norb, 6) uniform in t (per model) */
} Tables;

/* cubic B-spline weights and their t-derivatives for nodes i-1..i+2, t in [0,1) */
DEV void bw(double t, double w[4], double d[4])
{
    double t2 = t*t, t3 = t2*t, u = 1.0 - t;
    w[0] = u*u*u/6.0;            d[0] = -0.5*u*u;
    w[1] = (3*t3 - 6*t2 + 4)/6.0; d[1] = 0.5*(3*t2 - 4*t);
    w[2] = (-3*t3 + 3*t2 + 3*t + 1)/6.0; d[2] = 0.5*(-3*t2 + 2*t + 1);
    w[3] = t3/6.0;               d[3] = 0.5*t2;
}

/* cell index and fractional position as jaxspray._spline2: floor clipped to [0, n-2], weights not clipped (extrapolation) */
DEV void cell(double x, int n, int *i, double *t)
{
    double f = floor(x); if (f < 0) f = 0; if (f > n - 2) f = n - 2;
    *i = (int)f + 1; *t = x - f;
}

/* tensor cubic B-spline value and partial derivatives (in node units) of one padded table c[(nu+4)*(nw+4)] */
DEV void spl2(const double *c, int ldw, int i, int j, const double wx[4], const double dx[4], const double wy[4], const double dy[4],
              double *f, double *fx, double *fy)
{
    double s = 0, sx = 0, sy = 0;
    for (int k = 0; k < 4; k++) {
        const double *row = c + (size_t)(i + k)*ldw + j;
        double r = 0, ry = 0;
        for (int l = 0; l < 4; l++) { double g = row[l]; r += g*wy[l]; ry += g*dy[l]; }   /* 16 gathers */
        s += r*wx[k]; sx += r*dx[k]; sy += ry*wx[k];
    }
    *f = s; *fx = sx; *fy = sy;
}

/* progenitor centre at time t: cubic Hermite on the uniform orbit grid (jaxspray.centre) */
DEV void centre(const Tables *tb, double t, double c[3])
{
    double s = (t - tb->tc0)/tb->dtc, fk = floor(s);
    if (fk < 0) fk = 0; if (fk > tb->norb - 2) fk = tb->norb - 2;
    int k = (int)fk; double h = s - fk, h2 = h*h, h3 = h2*h;
    const double *o0 = tb->orb + 6*(size_t)k, *o1 = o0 + 6;
    double a0 = 2*h3 - 3*h2 + 1, a1 = h3 - 2*h2 + h, b0 = -2*h3 + 3*h2, b1 = h3 - h2;
    for (int d = 0; d < 3; d++) c[d] = a0*o0[d] + a1*o0[3+d]*tb->dtc + b0*o1[d] + b1*o1[3+d]*tb->dtc;
}

/* acceleration a = -grad Phi, potential Phi (host + progenitor) and the chi integrand |r| - |r_prog| at (x, t) */
DEV void force(const Tables *tb, const double x[3], double t, double a[3], double *pot, double *dchi, int with_sat)
{
    const int ldw = tb->nw + 4;
    double R = hypot(x[0], x[1]), az = fabs(x[2]), sz = x[2] >= 0 ? 1.0 : -1.0;
    /* axisymmetric part at (u, w) */
    double u = asinh(R/tb->rs)/tb->du, w = asinh(az/tb->zs)/tb->dw;
    int i, j; double tu, tw, wu[4], du_[4], ww[4], dw_[4];
    cell(u, tb->nu, &i, &tu); cell(w, tb->nw, &j, &tw); bw(tu, wu, du_); bw(tw, ww, dw_);
    double p, pu, pw; spl2(tb->axi, ldw, i, j, wu, du_, ww, dw_, &p, &pu, &pw);
    double dudR = 1.0/(tb->du*tb->rs*sqrt(1.0 + (R/tb->rs)*(R/tb->rs)));
    double dwdz = sz/(tb->dw*tb->zs*sqrt(1.0 + (az/tb->zs)*(az/tb->zs)));
    double dR = pu*dudR, dz = pw*dwdz, dang = 0.0;
    /* bar modes at (R/S, z/S) */
    double S = tb->S, uS = asinh(R/(S*tb->rs))/tb->du, wS = asinh(az/(S*tb->zs))/tb->dw;
    cell(uS, tb->nu, &i, &tu); cell(wS, tb->nw, &j, &tw); bw(tu, wu, du_); bw(tw, ww, dw_);
    double duSdR = 1.0/(S*tb->du*tb->rs*sqrt(1.0 + (R/(S*tb->rs))*(R/(S*tb->rs))));
    double dwSdz = sz/(S*tb->dw*tb->zs*sqrt(1.0 + (az/(S*tb->zs))*(az/(S*tb->zs))));
    double ang = atan2(x[1], x[0]) - (tb->theta_T + tb->omega*(t - tb->T));
    size_t npl = (size_t)(tb->nu + 4)*ldw;
    for (int q = 0; q < tb->nm; q++) {
        double B, Bu, Bw, mq = tb->m[q], cq = cos(mq*ang), sq = sin(mq*ang);
        spl2(tb->bar + q*npl, ldw, i, j, wu, du_, ww, dw_, &B, &Bu, &Bw);
        p += tb->A*B*cq; dR += tb->A*Bu*duSdR*cq; dz += tb->A*Bw*dwSdz*cq; dang += -tb->A*B*mq*sq;
    }
    double gx = dR*x[0]/R - dang*x[1]/(R*R), gy = dR*x[1]/R + dang*x[0]/(R*R), gz = dz;
    if (!with_sat) { a[0] = -gx; a[1] = -gy; a[2] = -gz; if (pot) *pot = p; return; }
    /* progenitor: frozen profile Phi_sat(ln r) around the centre; -G M / r beyond the table */
    double c[3]; centre(tb, t, c);
    double d0 = x[0] - c[0], d1 = x[1] - c[1], d2 = x[2] - c[2], r = sqrt(d0*d0 + d1*d1 + d2*d2);
    double s = (log(r) - tb->lr0)/tb->dlr, ps, dpdr;
    if (s < tb->nr - 1) {
        double sm = s > 0 ? s : 0.0; int k; double ts, ws[4], ds[4]; cell(sm, tb->nr, &k, &ts); bw(ts, ws, ds);
        double f = 0, fs = 0; for (int l = 0; l < 4; l++) { double g = tb->sat[k + l]; f += g*ws[l]; fs += g*ds[l]; }
        ps = f; dpdr = s > 0 ? fs/(tb->dlr*r) : 0.0;                 /* jnp.maximum(s, 0): zero gradient for s < 0 */
    } else { ps = -tb->G*tb->mtot/r; dpdr = tb->G*tb->mtot/(r*r); }
    gx += dpdr*d0/r; gy += dpdr*d1/r; gz += dpdr*d2/r;
    a[0] = -gx; a[1] = -gy; a[2] = -gz;
    if (pot) *pot = p + ps;
    *dchi = sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]) - sqrt(c[0]*c[0] + c[1]*c[1] + c[2]*c[2]);
}

/* chi integrand |r| - |r_prog(t)| alone */
DEV double dchi_at(const Tables *tb, const double x[3], double t)
{
    double c[3]; centre(tb, t, c);
    return sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]) - sqrt(c[0]*c[0] + c[1]*c[1] + c[2]*c[2]);
}

/* dy/dt of (x, v); fac = AGAMA's accuracy factor min(1, |E| / max(|Epot|, Ekin)) when requested */
DEV void deriv(const Tables *tb, const double y[6], double t, double k[6], double *dchi, double *fac)
{
    double a[3], pot;
    force(tb, y, t, a, fac ? &pot : 0, dchi, 1);
    k[0] = y[3]; k[1] = y[4]; k[2] = y[5]; k[3] = a[0]; k[4] = a[1]; k[5] = a[2];
    if (fac) { double ek = 0.5*(y[3]*y[3] + y[4]*y[4] + y[5]*y[5]); double e = fabs(pot + ek), m = fabs(pot) > ek ? fabs(pot) : ek;
               *fac = e/m < 1.0 ? e/m : 1.0; }
}

#define NDIM 6
static const double C_[11] = {0.05260015195876773187856, 0.07890022793815159781784, 0.11835034190722739672676, 0.28164965809277260327324,
    0.33333333333333333333333, 0.25, 0.30769230769230769230769, 0.65128205128205128205128, 0.6, 0.85714285714285714285714, 1.0};
static const double A_[11][11] = {
    {0.05260015195876773187856},
    {0.01972505698453789945446, 0.05917517095361369836338},
    {0.02958758547680684918169, 0., 0.08876275643042054754507},
    {0.24136513415926668550237, 0., -0.88454947932828608534486, 0.92483400326179200311574},
    {0.03703703703703703703704, 0., 0., 0.17082860872947387127960, 0.12546768756682242501669},
    {0.037109375, 0., 0., 0.17025221101954403931498, 0.06021653898045596068502, -0.017578125},
    {0.03709200011850479271088, 0., 0., 0.17038392571223999381021, 0.10726203044637328465181, -0.01531943774862440175279, 0.00827378916381402288758},
    {0.62411095871607571711443, 0., 0., -3.36089262944694129406857, -0.86821934684172600681819, 27.5920996994467083049416, 20.1540675504778934086187, -43.4898841810699588477366},
    {0.47766253643826436589043, 0., 0., -2.48811461997166764192642, -0.59029082683684299637145, 21.2300514481811942347289, 15.2792336328824235832597, -33.2882109689848629194453, -0.02033120170850862613582},
    {-0.93714243008598732571704, 0., 0., 5.18637242884406370830024, 1.09143734899672957818500, -8.14978701074692612513997, -18.5200656599969598641566, 22.7394870993505042818970, 2.49360555267965238987089, -3.04676447189821950038237},
    {2.27331014751653820792360, 0., 0., -10.5344954667372501984067, -2.00087205822486249909676, -17.9589318631187989172766, 27.9488845294199600508500, -2.85899827713502369474066, -8.87285693353062954433549, 12.3605671757943030647266, 0.64339274601576353035597}};
static const double B_[12] = {0.05429373411656876223805, 0, 0, 0, 0, 4.45031289275240888144114, 1.89151789931450038304282, -5.80120396001058478146721,
    0.31116436695781989440892, -0.15216094966251607855618, 0.20136540080403034837478, 0.04471061572777259051769};
static const double BHH_[12] = {0.24409448818897637795276, 0, 0, 0, 0, 0, 0, 0, 0.73384668828161185734136, 0, 0, 0.02205882352941176470588};
static const double ER_[12] = {0.01312004499419488073250, 0, 0, 0, 0, -1.22515644637620444072057, -0.49575894965725019152141, 1.66437718245498653696153,
    -0.35032884874997368168865, 0.33417911871301747902973, 0.08192320648511571246571, -0.02235530786388629525884};

DEV double clip(double x, double lo, double hi) { return x < lo ? lo : (x > hi ? hi : x); }

/* AGAMA initTimeStep (accAbs = 0) x accuracy factor; returns h and fills k1 */
DEV double init_step(const Tables *tb, const double y[6], double t, double acc, double k1[6])
{
    double fac, dchi, k2[6], k3[6], xt[6], ny = 0, nk = 0;
    deriv(tb, y, t, k1, &dchi, &fac);
    for (int i = 0; i < NDIM; i++) { ny += y[i]*y[i]; nk += k1[i]*k1[i]; }
    double h1 = clip(sqrt(ny/nk)*0.01, 1e-12, 1e12);
    for (int i = 0; i < NDIM; i++) xt[i] = y[i] + h1*k1[i];
    deriv(tb, xt, t + h1, k2, &dchi, 0);
    for (int i = 0; i < NDIM; i++) xt[i] = y[i] + 0.5*h1*(k1[i] + k2[i]);
    deriv(tb, xt, t + h1, k3, &dchi, 0);
    double h2 = INFINITY;
    for (int i = 0; i < NDIM; i++) {
        double d0 = fmax(fabs(y[i]), fabs(xt[i])), d1 = fmax(fabs(k1[i]), fabs(k3[i]));
        double d2 = fabs((3*k2[i] - k1[i] - 2*k3[i])/h1), d3 = fabs(6*(k3[i] - k2[i])/(h1*h1));
        h2 = fmin(h2, (d0*d2 + d1*d1)/(d1*d3 + d2*d2));
    }
    return clip(pow(acc, 1.0/8)*sqrt(h2), 1e-12, 1e12)*fac;
}

/* DOP853 attempts for one particle from the state (t, y, k1, h, chi, nbad, prev) to T (AGAMA doStep + OrbitIntegrator::run). The
   state is updated in place; returns the number of attempts made (accepted + rejected) or -1 on a non-finite error / attempt cap. */
DEV int run_state(const Tables *tb, double *t_, double y[6], double k1[6], double *h_, double *chi_, int *nbad_, double *prev_,
                  double acc, int max_att, long *nforce)
{
    double K[12][6], D[12], xt[6], kb[6], k13[6], chi = *chi_, t = *t_, T = tb->T, prev = *prev_, fac, d13, h = *h_;
    int nbad = *nbad_, natt = 0; long nf = 0;
    while (t < T) {
        if (natt >= max_att || !(h > 0) || !isfinite(h)) { natt = -1; break; }
        double rem = T - t, dt = h < rem ? h : rem;
        for (int i = 0; i < NDIM; i++) K[0][i] = k1[i];
        D[0] = dchi_at(tb, y, t);                                                  /* cheap: centre + two norms, no force */
        for (int s = 0; s < 11; s++) {                                             /* stages 2..12 */
            for (int i = 0; i < NDIM; i++) { double acc_ = 0; for (int j = 0; j <= s; j++) if (A_[s][j] != 0) acc_ += A_[s][j]*K[j][i]; xt[i] = y[i] + dt*acc_; }
            deriv(tb, xt, t + C_[s]*dt, K[s+1], &D[s+1], 0);
        }
        nf += 12;
        for (int i = 0; i < NDIM; i++) { double a = 0; for (int j = 0; j < 12; j++) if (B_[j] != 0) a += B_[j]*K[j][i]; kb[i] = a; xt[i] = y[i] + dt*kb[i]; }
        deriv(tb, xt, t + dt, k13, &d13, &fac); nf += 1;
        double e3 = 0, e5 = 0;
        for (int i = 0; i < NDIM; i++) {
            double sk = acc*fmax(fabs(y[i]), fabs(xt[i]))*fac; if (sk == 0) continue;
            double b3 = 0, b5 = 0; for (int j = 0; j < 12; j++) { if (BHH_[j] != 0) b3 += BHH_[j]*K[j][i]; if (ER_[j] != 0) b5 += ER_[j]*K[j][i]; }
            double q3 = (kb[i] - b3)/sk, q5 = b5/sk; e3 += q3*q3; e5 += q5*q5;
        }
        double den = sqrt(NDIM*(e5 + 0.01*e3)), err = den == 0 ? 0 : e5*dt/den;
        if (!isfinite(err)) { natt = -1; break; }
        double f8 = sqrt(sqrt(sqrt(err))); natt++;
        int worse = err > 0.5*prev, nb = worse ? nbad + 1 : 0;
        int ok = (err <= 1) || ((nbad >= 3) && worse);
        if (ok) {
            double dchi_sum = 0; for (int j = 0; j < 12; j++) if (B_[j] != 0) dchi_sum += B_[j]*D[j];
            chi += dt*dchi_sum;
            t = (dt == rem) ? T : t + dt;
            for (int i = 0; i < NDIM; i++) { y[i] = xt[i]; k1[i] = k13[i]; }
            h = (err <= 1) ? dt*fmin(6.0, 0.9/f8) : dt; nbad = 0; prev = INFINITY;
        } else {
            h = dt*fmax(0.333, 0.9/(nb >= 2 ? 1/0.333 : f8)); nbad = nb; prev = err;
        }
    }
    *t_ = t; *h_ = h; *chi_ = chi; *nbad_ = nbad; *prev_ = prev; if (nforce) *nforce += nf;
    return natt;
}

/* one particle from t0 to T; returns attempts (or -1) */
DEV int integrate(const Tables *tb, const double y0[6], double t0, double acc, int max_att, double yout[6], double *chi_out, long *nforce)
{
    double y[6], k1[6], t = t0, chi = 0, prev = INFINITY; int nbad = 0;
    for (int i = 0; i < NDIM; i++) y[i] = y0[i];
    double h = init_step(tb, y, t, acc, k1); if (nforce) *nforce = 3;
    int natt = run_state(tb, &t, y, k1, &h, &chi, &nbad, &prev, acc, max_att, nforce);
    for (int i = 0; i < NDIM; i++) yout[i] = y[i];
    *chi_out = chi;
    return natt;
}

/* ---- host entry points for the ctypes harness ---- */
void force_many(const Tables *tb, int n, const double *x, const double *t, double *a, double *pot, double *dchi)
{
    for (int p = 0; p < n; p++) force(tb, x + 3*p, t[p], a + 3*p, pot + p, dchi + p, 1);
}
void integrate_many(const Tables *tb, int n, const double *y0, const double *t0, double acc, int max_att, double *yout, double *chi, int *natt, long *nforce)
{
    for (int p = 0; p < n; p++) natt[p] = integrate(tb, y0 + 6*p, t0[p], acc, max_att, yout + 6*p, chi + p, nforce + p);
}

/* continue n particles from mid-integration states (the GPU hand-off); arrays updated in place, natt = attempts made here (-1 = fail) */
void integrate_resume(const Tables *tb, int n, double *t, double *y, double *k1, double *h, double *chi, int *nbad, double *prev,
                      double acc, int max_att, int *natt)
{
    for (int p = 0; p < n; p++) natt[p] = run_state(tb, t + p, y + 6*p, k1 + 6*p, h + p, chi + p, nbad + p, prev + p, acc, max_att, 0);
}

/* host-only centre orbit from T back to 0 with DOP853 (accuracy acc) in tau = T - t: dx/dtau = -v, dv/dtau = -a(x, T - tau).
   Records every accepted step (t, x, v, a) into buffers of length nmax (entry 0 = today). Returns the number of steps, or -1 if the
   buffer is exhausted or the integration fails.                                                                              */
DEV void deriv_back(const Tables *tb, const double y[6], double tau, double k[6], double *fac)
{
    double a[3], pot, dummy; force(tb, y, tb->T - tau, a, fac ? &pot : 0, &dummy, 0);
    k[0] = -y[3]; k[1] = -y[4]; k[2] = -y[5]; k[3] = -a[0]; k[4] = -a[1]; k[5] = -a[2];
    if (fac) { double ek = 0.5*(y[3]*y[3] + y[4]*y[4] + y[5]*y[5]); double e = fabs(pot + ek), m = fabs(pot) > ek ? fabs(pot) : ek;
               *fac = e/m < 1.0 ? e/m : 1.0; }
}
int orbit_back(const Tables *tb, const double xv0[6], double acc, int nmax, double *tout, double *yout, double *aout)
{
    double y[6], k1[6], K[12][6], xt[6], kb[6], k13[6], k2[6], k3[6], T = tb->T, tau = 0, prev = INFINITY, fac;
    int nbad = 0, n = 0;
    for (int i = 0; i < NDIM; i++) y[i] = xv0[i];
    /* AGAMA initTimeStep */
    deriv_back(tb, y, 0, k1, &fac);
    double ny = 0, nk = 0; for (int i = 0; i < NDIM; i++) { ny += y[i]*y[i]; nk += k1[i]*k1[i]; }
    double h1 = clip(sqrt(ny/nk)*0.01, 1e-12, 1e12);
    for (int i = 0; i < NDIM; i++) xt[i] = y[i] + h1*k1[i];
    deriv_back(tb, xt, h1, k2, 0);
    for (int i = 0; i < NDIM; i++) xt[i] = y[i] + 0.5*h1*(k1[i] + k2[i]);
    deriv_back(tb, xt, h1, k3, 0);
    double h2 = INFINITY;
    for (int i = 0; i < NDIM; i++) {
        double d0 = fmax(fabs(y[i]), fabs(xt[i])), d1 = fmax(fabs(k1[i]), fabs(k3[i]));
        double d2 = fabs((3*k2[i] - k1[i] - 2*k3[i])/h1), d3 = fabs(6*(k3[i] - k2[i])/(h1*h1));
        h2 = fmin(h2, (d0*d2 + d1*d1)/(d1*d3 + d2*d2));
    }
    double h = clip(pow(acc, 1.0/8)*sqrt(h2), 1e-12, 1e12)*fac;
    tout[0] = T; for (int i = 0; i < 6; i++) yout[i] = y[i]; for (int i = 0; i < 3; i++) aout[i] = -k1[3+i];
    while (tau < T) {
        if (!(h > 0) || !isfinite(h) || n >= nmax - 1) return -1;
        double rem = T - tau, dt = h < rem ? h : rem;
        for (int i = 0; i < NDIM; i++) K[0][i] = k1[i];
        for (int s = 0; s < 11; s++) {
            for (int i = 0; i < NDIM; i++) { double acc_ = 0; for (int j = 0; j <= s; j++) if (A_[s][j] != 0) acc_ += A_[s][j]*K[j][i]; xt[i] = y[i] + dt*acc_; }
            deriv_back(tb, xt, tau + C_[s]*dt, K[s+1], 0);
        }
        for (int i = 0; i < NDIM; i++) { double a = 0; for (int j = 0; j < 12; j++) if (B_[j] != 0) a += B_[j]*K[j][i]; kb[i] = a; xt[i] = y[i] + dt*kb[i]; }
        deriv_back(tb, xt, tau + dt, k13, &fac);
        double e3 = 0, e5 = 0;
        for (int i = 0; i < NDIM; i++) {
            double sk = acc*fmax(fabs(y[i]), fabs(xt[i]))*fac; if (sk == 0) continue;
            double b3 = 0, b5 = 0; for (int j = 0; j < 12; j++) { if (BHH_[j] != 0) b3 += BHH_[j]*K[j][i]; if (ER_[j] != 0) b5 += ER_[j]*K[j][i]; }
            double q3 = (kb[i] - b3)/sk, q5 = b5/sk; e3 += q3*q3; e5 += q5*q5;
        }
        double den = sqrt(NDIM*(e5 + 0.01*e3)), err = den == 0 ? 0 : e5*dt/den;
        if (!isfinite(err)) return -1;
        double f8 = sqrt(sqrt(sqrt(err)));
        int worse = err > 0.5*prev, nb = worse ? nbad + 1 : 0, ok = (err <= 1) || ((nbad >= 3) && worse);
        if (ok) {
            tau = (dt == rem) ? T : tau + dt;
            for (int i = 0; i < NDIM; i++) { y[i] = xt[i]; k1[i] = k13[i]; }
            n++; tout[n] = T - tau; for (int i = 0; i < 6; i++) yout[6*n + i] = y[i]; for (int i = 0; i < 3; i++) aout[3*n + i] = -k13[3+i];
            h = (err <= 1) ? dt*fmin(6.0, 0.9/f8) : dt; nbad = 0; prev = INFINITY;
        } else {
            h = dt*fmax(0.333, 0.9/(nb >= 2 ? 1/0.333 : f8)); nbad = nb; prev = err;
        }
    }
    return n + 1;
}
