import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import run as B
from run import ACC, CTL, INK, PRP, UNC, run_one
from dynamics import RADIUS

# Pitcher release dispersion.
SIGMA_ANGLE_DEG = 0.25
SIGMA_MPH = 1.5
SIGMA_RPM = 100.0
SIGMA_RELEASE_X = 0.1 # m
SIGMA_RELEASE_Z = 0.1 # m

N_DEFAULT = 20

def run_mc(n):
    out = []
    r = np.random.RandomState(4000)
    for i in range(n):
        dy, dp = r.normal(0, SIGMA_ANGLE_DEG, 2)
        mph = r.normal(B.PITCH_MPH, SIGMA_MPH)
        rpm = r.normal(B.PITCH_RPM, SIGMA_RPM)
        dx = r.normal(0.0, SIGMA_RELEASE_X)
        dz = r.normal(0.0, SIGMA_RELEASE_Z)
        nom, sol, gnc, tgt = run_one(dyaw=dy, dpitch=dp, seed=i,
                                     mph=mph, rpm=rpm, log=True,
                                     dx=dx, dz=dz)
        out.append(dict(nom=nom, sol=sol, gnc=gnc, target=tgt,
                        disp=dict(dyaw=dy, dpitch=dp, mph=mph, rpm=rpm,
                                  dx=dx, dz=dz)))
        if (i + 1) % 10 == 0:
            print(f'  {i+1}/{n}')
    return out


def summarise(res):
    zu = np.array([r['nom'].y[2, -1] for r in res])
    zc = np.array([r['sol'].y[2, -1] for r in res])
    xu = np.array([r['nom'].y[0, -1] for r in res])
    xc = np.array([r['sol'].y[0, -1] for r in res])
    su = np.array([B.in_zone(x, z) for x, z in zip(xu, zu)])
    sc = np.array([B.in_zone(x, z) for x, z in zip(xc, zc)])
    mu = np.linalg.norm([(zu - B.TARGET_Z), (xu - B.TARGET_X)], axis=0)
    mc = np.linalg.norm([(zc - B.TARGET_Z), (xc - B.TARGET_X)], axis=0)
    n = len(res)
    
    vu, vc = np.abs(zu - B.TARGET_Z), np.abs(zc - B.TARGET_Z)
    lu, lc = np.abs(xu - B.TARGET_X), np.abs(xc - B.TARGET_X)
    print(f'\n{n} pitches')
    print(f'  combined miss   median {np.median(mu):.3f} -> {np.median(mc):.3f} m'
          f'   p90 {np.quantile(mu,0.9):.3f} -> {np.quantile(mc,0.9):.3f} m')
    print(f'  vertical |dz|   median {np.median(vu):.3f} -> {np.median(vc):.3f} m'
          f'   p90 {np.quantile(vu,0.9):.3f} -> {np.quantile(vc,0.9):.3f} m')
    print(f'  lateral  |dx|   median {np.median(lu):.3f} -> {np.median(lc):.3f} m'
          f'   p90 {np.quantile(lu,0.9):.3f} -> {np.quantile(lc,0.9):.3f} m')
    print(f'  spread (std)    z {zu.std():.3f} -> {zc.std():.3f} m'
          f'   x {xu.std():.3f} -> {xc.std():.3f} m')
    print(f'  strikes         {100*su.mean():.0f}% -> {100*sc.mean():.0f}%')
    sat = np.array([np.abs(np.array(r['gnc'].log['tau'])).max()
                    for r in res])
    print(f'  peak |torque|   median {np.median(sat):.3f} of '
          f'{B.WHEEL_TAU_LIMIT} N m limit')
    pe = [r['gnc'].log['pos_err'][-1] for r in res if r['gnc'].log['pos_err']]
    if pe:
        print(f'  EKF final position error  median {np.median(pe):.3f} m')
    else:
        print('  EKF final position error  n/a (perfect_nav)')
    w_err = np.array([np.linalg.norm(np.array(r['gnc'].log['w_setpoint'][-1])
                                     - np.array(r['gnc'].log['w_b'][-1]))
                      for r in res if r['gnc'].log['w_setpoint']])
    if w_err.size:
        print(f'  final |w_setpoint - w_b|  median {np.median(w_err):.3f} rad/s')
    return xu, zu, xc, zc, mu, mc


def plot(res, outdir='.'):
    xu, zu, xc, zc, mu, mc = summarise(res)
    fig, ax = plt.subplots(2, 4, figsize=(22, 9.5))
    fig.delaxes(ax[1, 3])

    # Home plate
    a = ax[0, 0]
    a.add_patch(plt.Rectangle((-B.ZONE_W/2, B.ZONE_BOT), B.ZONE_W, B.ZONE_H,
                              fill=False, ec=INK, lw=2, zorder=2))
    a.plot(0.0, B.TARGET_Z, marker='+', ms=18, color=ACC, mec='red',
           mew=0.8, zorder=2, label='target')
    a.add_patch(plt.Circle((0.0, B.TARGET_Z), RADIUS, fill=False, ec=ACC,
                           ls='--', lw=1, zorder=2))
    for r in res:
        a.plot(r['nom'].y[0], r['nom'].y[2], color=UNC, lw=0.7, alpha=0.3,
               zorder=1, ls='--')
        a.plot(r['sol'].y[0], r['sol'].y[2], color=CTL, lw=0.7, alpha=0.35,
               zorder=1, ls='--')
    a.scatter(xu, zu, s=26, c=UNC, alpha=0.75, lw=0, label='uncontrolled', zorder=3)
    a.scatter(xc, zc, s=26, c=CTL, alpha=0.8, lw=0, label='controlled', zorder=3)
    
    a.set_xlim(-0.6, 0.6); a.set_ylim(0.0, 2.0); a.set_aspect('equal')
    a.set_xlabel('x [m]'); a.set_ylabel('z [m]')
    a.set_title('Catcher view', color=INK)
    a.legend(fontsize=8, loc='lower center'); a.grid(True, alpha=0.18)

    # Trajectories
    a = ax[0, 1]
    for r in res:
        a.plot(r['nom'].y[1], r['nom'].y[2], color=UNC, lw=0.8, alpha=0.45)
        a.plot(r['sol'].y[1], r['sol'].y[2], color=CTL, lw=0.8, alpha=0.55)
    a.axhline(B.ZONE_TOP, color=INK, ls=':', lw=1.2)
    a.axhline(B.ZONE_BOT, color=INK, ls=':', lw=1.2)
    a.axhline(B.TARGET_Z, color=ACC, ls='--', lw=1.2)
    a.invert_xaxis()
    a.set_xlabel('distance to plate [m]'); a.set_ylabel('height z [m]')
    a.set_title('Trajectory', color=INK)
    a.legend(handles=[Line2D([], [], color=UNC, lw=2, label='uncontrolled'),
                      Line2D([], [], color=CTL, lw=2, label='controlled'),
                      Line2D([], [], color=ACC, ls='--', lw=2, label='target')],
             fontsize=8, loc='lower left')
    a.grid(True, alpha=0.18)

    # Miss distribution
    a = ax[0, 2]
    for v, c, lab in [(mu, UNC, 'uncontrolled'), (mc, CTL, 'controlled')]:
        sv = np.sort(v)
        a.plot(sv, np.arange(1, len(sv)+1)/len(sv), color=c, lw=2.2, label=lab)
    a.axvline(B.ZONE_H/2, color=ACC, ls='--', lw=1.5,
              label='zone half-height')
    a.axvline(RADIUS, color=INK, ls=':', lw=1.5, label='ball radius')
    a.set_xlabel('miss [m]')
    a.set_ylabel('fraction of pitches')
    a.set_title('Miss distribution', color=INK)
    a.legend(fontsize=8, loc='lower right'); a.grid(True, alpha=0.18)

    # Guidance
    a = ax[0, 3]
    for r in res:
        t = np.array(r['gnc'].log['w_setpoint_t'])
        w_set = np.array(r['gnc'].log['w_setpoint'])
        w_b = np.array(r['gnc'].log['w_b'])
        if t.size == 0:
            continue
        a.plot(t, np.linalg.norm(w_set, axis=1), color=ACC, lw=0.9, alpha=0.5)
        a.plot(t, np.linalg.norm(w_b, axis=1), color=CTL, lw=0.9, alpha=0.55)
    a.set_xlabel('time [s]'); a.set_ylabel('|w| [rad/s]')
    a.set_title('Body rate: setpoint vs achieved', color=INK)
    a.legend(handles=[Line2D([], [], color=ACC, lw=2, label='w_setpoint'),
                      Line2D([], [], color=CTL, lw=2, label='w_b (achieved)')],
             fontsize=8); a.grid(True, alpha=0.18)

    # Control
    a = ax[1, 0]
    n_wheels = np.array(res[0]['gnc'].log['tau']).shape[1]
    styles = ['-', '--', ':']
    for r in res:
        t = np.array(r['gnc'].log['t'])
        tau_log = np.array(r['gnc'].log['tau'])
        for i in range(n_wheels):
            a.plot(t, tau_log[:, i], color=CTL, lw=0.9, alpha=0.5,
                  ls=styles[i % len(styles)])
    a.axhline(B.WHEEL_TAU_LIMIT, color=ACC, ls='--', lw=1.5, label='limit')
    a.axhline(-B.WHEEL_TAU_LIMIT, color=ACC, ls='--', lw=1.5)
    a.set_xlabel('time [s]'); a.set_ylabel('wheel torque [N m]')
    a.set_title('Wheel torques', color=INK)
    a.legend(handles=[Line2D([], [], color=CTL, lw=2, ls=styles[i % len(styles)],
                             label=f'wheel {i+1}')
                      for i in range(n_wheels)]
             + [Line2D([], [], color=ACC, ls='--', lw=2, label='limit')],
             fontsize=8); a.grid(True, alpha=0.18)

    # Nav error
    a = ax[1, 1]
    any_nav = False
    for r in res:
        pe = r['gnc'].log['pos_err']
        if not pe:
            continue
        any_nav = True
        a.plot(np.array(r['gnc'].log['t'])[:len(pe)], pe,
               color=PRP, lw=0.9, alpha=0.55)
    a.set_xlabel('time [s]'); a.set_ylabel('EKF position error [m]')
    if any_nav:
        a.set_title('Position error', color=INK)
    else:
        a.set_title('Position error -- n/a, perfect_nav', color=INK)
        a.text(0.5, 0.5, 'perfect_nav=True\nno estimator in the loop',
               ha='center', va='center', transform=a.transAxes,
               fontsize=11, color=UNC)
    a.grid(True, alpha=0.18)

    a = ax[1, 2]
    for r in res:
        ae = r['gnc'].log['att_err']
        if not ae:
            continue
        a.plot(np.array(r['gnc'].log['t'])[:len(ae)], ae,
               color=PRP, lw=0.9, alpha=0.55)
    a.set_xlabel('time [s]'); a.set_ylabel('EKF attitude error [deg]')
    if any_nav:
        a.set_title('Attitude error', color=INK)
    else:
        a.set_title('Attitude error -- n/a, perfect_nav', color=INK)
    a.grid(True, alpha=0.18)

    fig.suptitle('Baseball GNC Monte Carlo (kinematic law): '
                 f'{len(res)} pitches -- release angle '
                 f'{SIGMA_ANGLE_DEG} deg, speed {SIGMA_MPH} mph, spin '
                 f'{SIGMA_RPM:.0f} rpm, release point '
                 f'{SIGMA_RELEASE_X}/{SIGMA_RELEASE_Z} m (x/z), '
                 f'tau_limit {B.WHEEL_TAU_LIMIT} N m', fontsize=13)
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, 'run_mc.png')
    fig.savefig(path, dpi=200)
    print(f'saved {path}')
    if os.environ.get('SHOW_PLOTS'):
        plt.show()


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_DEFAULT
    print(f'running {n} pitches with the kinematic law...')
    plot(run_mc(n))
