import os
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp

from dynamics import (BallParams, I_BALL, MASS, RADIUS, aero_wrapper,
                      normalize, ode_wrapper, pack)
from gnc import make_kinematic_gnc

# Scenario
RELEASE_Y, RELEASE_Z = 16.8, 1.8
PITCH_MPH, PITCH_RPM = 90.0, 2200.0

ZONE_W, ZONE_H = 0.432, 0.55
ZONE_Z_MID = 0.85
ZONE_TOP = ZONE_Z_MID + ZONE_H / 2
ZONE_BOT = ZONE_Z_MID - ZONE_H / 2
TARGET_Z = ZONE_Z_MID          
TARGET_X = 0

WHEEL_I = 3e-6
WHEEL_TAU_LIMIT = 5.0

INNER_RATE = 500.0 # Hz, EKF and inner loop
OUTER_RATE = 50.0  # Hz, guidance

# Plot colours
UNC = 'gray'      # uncontrolled baseline
CTL = 'blue'      # controlled / achieved
ACC = 'red'       # commanded / target / limits
PRP = 'darkorange'  # navigation error
INK = 'black'

A_GUID = 'blue'     # guidance acceleration

AXES = ('x', 'y', 'z')


def make_ball(tau_limit=WHEEL_TAU_LIMIT):
    return BallParams(m=MASS, r=RADIUS, I=I_BALL, g=9.81,
                      A=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
                      I_w=np.full(3, WHEEL_I), tau_limit=tau_limit)


def release_state(p, dyaw_deg=0.0, dpitch_deg=0.0, mph=PITCH_MPH,
                  rpm=PITCH_RPM, dx=0.0, dz=0.0):
    """dx and dz move the release point, dyaw and dpitch aim it."""
    v = mph * 0.44704
    w = rpm * 2 * np.pi / 60
    yaw, pit = np.radians(dyaw_deg), np.radians(dpitch_deg)
    v_hat = np.array([np.sin(yaw) * np.cos(pit), -np.cos(yaw) * np.cos(pit),
                      np.sin(pit)])
    axis = np.array([-1.0, 0.0, 0.0])
    return pack([dx, RELEASE_Y, RELEASE_Z + dz], v * v_hat, [1, 0, 0, 0],
                w * axis, np.zeros(p.n_wheels))


def in_zone(x, z):
    return abs(x) < ZONE_W / 2 and ZONE_BOT < z < ZONE_TOP


# Simulation
def torque_law(tau_vec):
    def tau(t, x):
        return tau_vec
    return tau


def torqueless_law(p):
    def tau(t, x):
        return np.zeros(p.n_wheels)
    return tau


def terminal_events():
    """Target-plane crossing (y=0) and ground impact (z=0), both terminal."""
    def hit_plane(t, x):
        return x[1]
    hit_plane.terminal = True
    hit_plane.direction = -1

    def hit_ground(t, x):
        return x[2]
    hit_ground.terminal = True
    hit_ground.direction = -1

    return [hit_plane, hit_ground]


def simulate_with_torque(p, x0, tau, aero_fns, t_max=3.0,
                         rtol=1e-9, atol=1e-11, max_step=0.01):
    F_aero, T_aero = aero_fns
    ode = ode_wrapper(p, tau_w=tau, F_ext_i=F_aero, T_ext_b=T_aero)
    sol = solve_ivp(ode, (0.0, t_max), x0, method="DOP853", max_step=max_step,
                    rtol=rtol, atol=atol, events=terminal_events())
    assert sol.success, sol.message
    return sol


def nominal_trajectory(p, x0, aero_fns, t_max=3.0):
    F_aero, T_aero = aero_fns
    ode = ode_wrapper(p, tau_w=torqueless_law(p), F_ext_i=F_aero, T_ext_b=T_aero)
    sol = solve_ivp(ode, (0.0, t_max), x0, method="DOP853", max_step=0.005,
                    rtol=1e-9, atol=1e-11, events=terminal_events(),
                    dense_output=True)
    assert sol.success, sol.message
    return sol


def simulate_gnc(p, x0, aero_fns, gnc, t_max=3.0, rate=100.0,
                 rtol=1e-8, atol=1e-10):
    
    F_aero, T_aero = aero_fns
    events = terminal_events()
    dt = 1.0 / rate

    t = 0.0
    x = np.asarray(x0, dtype=float)
    t_all, y_all, tau_history = [np.array([0.0])], [x.reshape(-1, 1)], []
    tau = np.zeros(p.n_wheels)

    while t < t_max - 1e-9:
        t_next = min(t + dt, t_max)
        tau_fn = torque_law(tau)
        ode = ode_wrapper(p, tau_w=tau_fn, F_ext_i=F_aero, T_ext_b=T_aero)
        sol = solve_ivp(ode, (t, t_next), x, method="DOP853", max_step=dt,
                        rtol=rtol, atol=atol, events=events)
        assert sol.success, sol.message

        tau_history.append((t, tau.copy()))
        t_all.append(sol.t[1:])
        y_all.append(sol.y[:, 1:])
        x = normalize(sol.y[:, -1])
        t = sol.t[-1]

        if sol.t_events[0].size or sol.t_events[1].size:
            break

        tau = gnc(t, x)

    return SimpleNamespace(t=np.concatenate(t_all),
                           y=np.concatenate(y_all, axis=1),
                           tau_history=tau_history)


def run_one(dyaw=0.0, dpitch=0.0, seed=42, mph=PITCH_MPH, rpm=PITCH_RPM,
            perfect_nav=False, log=True, dx=0.0, dz=0.0):
    """One pitch: uncontrolled baseline vs IMU-only EKF + kinematic guidance.

    dyaw, dpitch, dx and dz are the release dispersion the Monte Carlo draws.
    """
    p = make_ball()
    fns = aero_wrapper(p)
    x0 = release_state(p, dyaw, dpitch, mph, rpm, dx=dx, dz=dz)

    nom = nominal_trajectory(p, x0, fns, t_max=2.0)
    target = np.array([0.0, TARGET_Z])

    gnc = make_kinematic_gnc(p, fns, target, x0, seed=seed, t_max=2.0,
                             flight_time=nom.t[-1], log=log,
                             perfect_nav=perfect_nav, inner_rate=INNER_RATE,
                             outer_rate=OUTER_RATE)
    sol = simulate_gnc(p, x0, fns, gnc, t_max=2.0, rate=INNER_RATE)
    return nom, sol, gnc, target


def summarise(nom, sol, gnc):
    z_u, z_c = nom.y[2, -1], sol.y[2, -1]
    x_u, x_c = nom.y[0, -1], sol.y[0, -1]
    miss_u = np.hypot(x_u - TARGET_X, z_u - TARGET_Z)
    miss_c = np.hypot(x_c - TARGET_X, z_c - TARGET_Z)
    print(f'flight {nom.t[-1]:.3f} s, target (x,z) = '
          f'({TARGET_X:.3f}, {TARGET_Z:.3f}) m\n')
    print(f'{"uncontrolled":22s} x={x_u:+.3f} z={z_u:.3f}  miss {miss_u:.3f} m'
          f'  {"strike" if in_zone(x_u, z_u) else "ball"}')
    print(f'{"EKF + kinematic":22s} x={x_c:+.3f} z={z_c:.3f}  miss {miss_c:.3f} m'
          f'  {"strike" if in_zone(x_c, z_c) else "ball"}')

    tau = np.array(gnc.log['tau'])
    lim = WHEEL_TAU_LIMIT
    print(f'\npeak |torque|  {np.abs(tau).max():.3f} of {lim} N m limit'
          f'   saturated {100*float((np.abs(tau) >= lim - 1e-9).mean()):.0f}%'
          f' of wheel-samples')

    w_set = np.array(gnc.log['w_setpoint'])
    w_b = np.array(gnc.log['w_b'])
    if w_set.size:
        err = np.linalg.norm(w_set - w_b, axis=1)
        print(f'|w_setpoint - w_b|  median {np.median(err):.3f}'
              f'  final {err[-1]:.3f} rad/s')
    print(f'EKF final position error  {gnc.log["pos_err"][-1]:.3f} m'
          ' (dead reckoned, unobservable from an IMU)')


def predicted_crossings(gnc):
    """Where each guidance solve says the ball will cross the plate.

    r + v*t_go + a_guid*t_go^2/2, the same constant-accel solve the guidance
    does, evaluated at t_go. a_guid is only the part spin can deliver, so the
    prediction sits short of the target by the amount Magnus cannot cover.
    """
    t = np.asarray(gnc.log['w_setpoint_t'])
    a = np.asarray(gnc.log['a_guid'])
    r = np.asarray(gnc.log['r_hat'])
    v = np.asarray(gnc.log['v_hat'])
    tgo = np.asarray(gnc.log['t_go'])
    if t.size == 0:
        return np.zeros(0), np.zeros((0, 3))
    ok = tgo > 0
    s = tgo[ok][:, None]
    return t[ok], r[ok] + v[ok] * s + 0.5 * a[ok] * s**2


def plot(nom, sol, gnc, target, outdir='.'):
    t = np.asarray(gnc.log['t'])
    tau = np.asarray(gnc.log['tau'])
    tw = np.asarray(gnc.log['w_setpoint_t'])
    w_set = np.asarray(gnc.log['w_setpoint'])
    w_b = np.asarray(gnc.log['w_b'])

    fig, ax = plt.subplots(3, 5, figsize=(26, 13))
    fig.delaxes(ax[2, 4])

    
    for k in range(3):
        a = ax[0, k]
        a.plot(sol.t, sol.y[10 + k], color=INK, lw=1.0, ls=':', alpha=0.8,
               label='truth')
        if w_set.size:
            a.plot(tw, w_b[:, k], color=CTL, lw=1.6, label='achieved')
            a.plot(tw, w_set[:, k], color=ACC, lw=1.6, label='commanded')
        a.set_xlabel('time [s]')
        a.set_ylabel(f'w_{AXES[k]} [rad/s]')
        a.set_title(f'Body rate {AXES[k]}: setpoint vs achieved',
                    color=INK)
        a.legend(handles=[Line2D([], [], color=ACC, lw=2, label='commanded'),
                          Line2D([], [], color=CTL, lw=2, label='achieved'),
                          Line2D([], [], color=INK, ls=':', lw=2, label='truth')],
                 fontsize=8)
        a.grid(True, alpha=0.18)

    a = ax[0, 3]
    styles = ['-', '--', ':', '-.']
    for i in range(tau.shape[1]):
        a.plot(t, tau[:, i], color=CTL, lw=1.4, ls=styles[i % len(styles)],
               label=f'wheel {i+1}')
    a.axhline(WHEEL_TAU_LIMIT, color=ACC, ls='--', lw=1.4, label='limit')
    a.axhline(-WHEEL_TAU_LIMIT, color=ACC, ls='--', lw=1.4)
    a.set_xlabel('time [s]'); a.set_ylabel('wheel torque [N m]')
    a.set_title('Wheel torques', color=INK)
    a.legend(fontsize=8); a.grid(True, alpha=0.18)

    a = ax[0, 4]
    t_go = np.asarray(gnc.log['t_go'])
    if t_go.size:
        a.plot(tw, t_go, color=CTL, lw=1.8, label='t_go (from EKF)')
        a.plot(tw, nom.t[-1] - tw, color=INK, lw=1.2, ls='--',
               label='ideal (t_flight - t)')
    a.axhline(0.0, color=ACC, lw=1.2)
    a.set_xlabel('time [s]'); a.set_ylabel('t_go [s]')
    a.set_title('Time to go', color=INK)
    a.legend(fontsize=8); a.grid(True, alpha=0.18)

    for k in range(3):
        a = ax[1, k]
        a.plot(nom.t, nom.y[3 + k], color=UNC, lw=1.6, label='uncontrolled')
        a.plot(sol.t, sol.y[3 + k], color=CTL, lw=1.6, label='controlled')
        a.set_xlabel('time [s]')
        a.set_ylabel(f'v_{AXES[k]} [m/s]')
        a.set_title(f'Velocity {AXES[k]}', color=INK)
        a.legend(fontsize=8); a.grid(True, alpha=0.18)

    a = ax[1, 3]
    a.plot(t, gnc.log['pos_err'], color=PRP, lw=1.6)
    a.set_xlabel('time [s]'); a.set_ylabel('EKF position error [m]')
    a.set_title('Position error',
                color=INK)
    a.grid(True, alpha=0.18)

    a = ax[1, 4]
    a.plot(t, gnc.log['att_err'], color=PRP, lw=1.6)
    a.set_xlabel('time [s]'); a.set_ylabel('EKF attitude error [deg]')
    a.set_title('Attitude error', color=INK)
    a.grid(True, alpha=0.18)

    a = ax[2, 0]
    a.plot(nom.y[1], nom.y[2], color=UNC, lw=2, label='uncontrolled')
    a.plot(sol.y[1], sol.y[2], color=CTL, lw=2, label='controlled')
    a.axhline(ZONE_TOP, color=INK, ls=':', lw=1.2)
    a.axhline(ZONE_BOT, color=INK, ls=':', lw=1.2)
    a.axhline(TARGET_Z, color=ACC, ls='--', lw=1.2, label='target z')
    a.invert_xaxis()
    a.set_xlabel('distance to plate y [m]'); a.set_ylabel('height z [m]')
    a.set_title('Trajectory', color=INK)
    a.legend(fontsize=8); a.grid(True, alpha=0.18)

    a = ax[2, 1]
    a.add_patch(plt.Rectangle((-ZONE_W/2, ZONE_BOT), ZONE_W, ZONE_H,
                              fill=False, ec=INK, lw=2, zorder=2))
    a.plot(target[0], TARGET_Z, marker='+', ms=13, color=ACC, mec='red',
           zorder=2, label='target')
    a.add_patch(plt.Circle((target[0], TARGET_Z), RADIUS, fill=False,
                           ec=ACC, ls='--', lw=1, zorder=2))
    a.plot(nom.y[0], nom.y[2], color=UNC, lw=1.4, label='uncontrolled', zorder=3)
    a.plot(sol.y[0], sol.y[2], color=CTL, lw=1.4, label='controlled', zorder=3)
    a.plot(sol.y[0, -1], sol.y[2, -1], marker='o', ms=6, color=CTL)
    a.set_xlim(-0.45, 0.45); a.set_ylim(0.0, 2.0); a.set_aspect('equal')
    a.set_xlabel('x [m]'); a.set_ylabel('z [m]')
    a.set_title('Catcher view', color=INK)
    a.legend(fontsize=8, loc='lower center'); a.grid(True, alpha=0.18)

    a = ax[2, 2]
    tc, pred = predicted_crossings(gnc)
    a.add_patch(plt.Rectangle((-ZONE_W/2, ZONE_BOT), ZONE_W, ZONE_H,
                              fill=False, ec=INK, lw=2, zorder=2))
    a.plot(TARGET_X, TARGET_Z, marker='+', ms=13, color=ACC, zorder=2,
           label='target')
    a.add_patch(plt.Circle((TARGET_X, TARGET_Z), RADIUS, fill=False, ec=ACC,
                           ls='--', lw=1, zorder=2))
    if tc.size:
        a.plot(pred[:, 0], pred[:, 2], color=A_GUID, lw=0.8, alpha=0.35,
               zorder=3)
        sc = a.scatter(pred[:, 0], pred[:, 2], s=22, c=tc,
                       cmap='plasma', zorder=4, label='predicted crossing')
        fig.colorbar(sc, ax=a, fraction=0.046, pad=0.03, label='time [s]')
    a.plot(sol.y[0, -1], sol.y[2, -1], marker='o', ms=11, mfc='none', mec=INK,
           mew=1.8, zorder=5, label='flown')
    a.set_aspect('equal')
    a.set_xlabel('x [m]'); a.set_ylabel('z [m]')
    a.set_title('Guidance solution convergence', color=INK)
    a.legend(fontsize=8, loc='lower center'); a.grid(True, alpha=0.18)
    a.set_xlim(-0.6, 0.6); a.set_ylim(0.0, 2.0); a.set_aspect('equal')

    a = ax[2, 3]
    a_guid = np.asarray(gnc.log['a_guid'])
    if a_guid.size:
        for k in range(3):
            a.plot(tw, a_guid[:, k], color=A_GUID, lw=1.6,
                   ls=styles[k % len(styles)], label=AXES[k])
    a.set_xlabel('time [s]'); a.set_ylabel('acceleration [m/s^2]')
    a.set_title('a_guid components', color=INK)
    a.legend(fontsize=8); a.grid(True, alpha=0.18)

    fig.suptitle('Baseball GNC, single nominal pitch (kinematic law)',
                 fontsize=15)
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, 'run.png')
    fig.savefig(path, dpi=200)
    print(f'\nsaved {path}')
    if os.environ.get('SHOW_PLOTS'):
        plt.show()


if __name__ == '__main__':
    nom, sol, gnc, target = run_one(dyaw=0.5, dpitch=0.0)
    summarise(nom, sol, gnc)
    plot(nom, sol, gnc, target)
