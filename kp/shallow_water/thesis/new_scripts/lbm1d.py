# NumPy replica of TCLB sw (D2Q9 MRT shallow water), uniform in y (ny=1, periodic), OLD grid: nx=8416, outlet wall x>=8400.
# Mirrors Dynamics.c.Rt (S4 overridable; S5=S6=S7=1; S8=S9=omega) - NO side walls, so it shows the damping of the collision model alone.
# Usage: python3 lbm1d.py [S4=1.3333] [out_dir=frames] [inlet_csv]   (writes 1-D Rho profiles every 200 it, 355 frames)
import sys, time, os
import numpy as np, pandas as pd
S4 = float(sys.argv[1]) if len(sys.argv) > 1 else 1.3333
outdir = sys.argv[2] if len(sys.argv) > 2 else "frames"
os.makedirs(outdir, exist_ok=True)
g, H, nu = 0.00209424, 9.55, 0.000228
omega = 1.0 / (3 * nu + 0.5)
NX, NSTEP, EVERY, XWALL = 8416, 71000, 200, 8400
CSV = sys.argv[3] if len(sys.argv) > 3 else "kp/shallow_water/thesis/inlet_packet.csv"
hin = pd.read_csv(CSV)["h_inlet"].to_numpy()

M = np.array([[1]*9,
 [0,1,0,-1,0,1,-1,-1,1],[0,0,1,0,-1,1,1,-1,-1],
 [-4,-1,-1,-1,-1,2,2,2,2],[4,-2,-2,-2,-2,1,1,1,1],
 [0,-2,0,2,0,1,-1,-1,1],[0,0,-2,0,2,1,1,-1,-1],
 [0,1,-1,1,-1,0,0,0,0],[0,0,0,0,0,1,-1,1,-1]], float)
Mw = (M**2).sum(1)
S = np.array([0, 0, 0, S4, 1.0, 1.0, 1.0, omega, omega])
w = np.ones(NX); w[XWALL:] = 0.0

d0 = H
Req0 = np.array([d0, 0, 0, -4*d0 + 3*d0*d0*g, 4*d0 - 4.5*d0*d0*g, 0, 0, 0, 0])
f = np.tile((M.T @ (Req0 / Mw))[:, None], (1, NX))

def eq(R, d, ux, uy, usq):
    Q = np.empty_like(R)
    Q[3] = -4*d + 3*usq + 3*d*d*g
    Q[4] = 4*d - 3*usq - 4.5*d*d*g
    Q[5] = -ux; Q[6] = -uy
    Q[7] = (ux*ux - uy*uy)/d; Q[8] = ux*uy/d
    return Q

t0 = time.time(); frames = []
for it in range(NSTEP):
    fn = f.copy()                                    # streaming (pull)
    for i in (1, 5, 8): fn[i, 1:] = f[i, :-1]
    for i in (3, 6, 7): fn[i, :-1] = f[i, 1:]
    f = fn
    rho = (hin[it] if it < len(hin) else 0.0) + H    # WPressure inlet at x=0
    ux0 = rho - (f[0,0] + f[2,0] + f[4,0] + 2*(f[3,0] + f[7,0] + f[6,0]))
    uy0 = 1.5*(f[2,0] - f[4,0])
    f[1,0] = f[3,0] + (2/3)*ux0
    f[5,0] = f[7,0] + ux0/6 + uy0/6
    f[8,0] = f[6,0] + ux0/6 - uy0/6
    for a, b in ((1, 3), (2, 4), (5, 7), (6, 8)):    # bounce-back wall
        tmp = f[a, XWALL:].copy(); f[a, XWALL:] = f[b, XWALL:]; f[b, XWALL:] = tmp
    R = M @ f                                        # MRT collision
    d, ux, uy = R[0], R[1], R[2]
    usq = (ux*ux + uy*uy)/d
    Q = eq(R, d, ux, uy, usq)
    for i in range(3, 9): R[i] = (1 - S[i])*(R[i] - Q[i])
    ux = ux*w; uy = uy*w; R[1] = ux; R[2] = uy
    usq = (ux*ux + uy*uy)/d
    Q = eq(R, d, ux, uy, usq)
    for i in range(3, 9): R[i] += Q[i]
    f = M.T @ (R / Mw[:, None])
    if (it + 1) % EVERY == 0:
        tau = it + 1
        with open(os.path.join(outdir, f"lbm_VTK_P00_{tau:08d}.vti"), "wb") as fh:
            np.save(fh, f.sum(0).astype(np.float32))
    if (it + 1) % 10000 == 0:
        print(f"step {it+1}  {time.time()-t0:.0f}s  max|h-H| = {np.abs(f.sum(0)-H).max():.4f}", flush=True)
print("done", time.time() - t0)
