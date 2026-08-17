import numpy as np

from dynamics import aero_force, q2dcm, q_normalize, unpack
from imu import IMU


# Navigation
GYRO_NOISE, ACCEL_NOISE = 0.001, 0.001  # rad/s, m/s^2
GYRO_BIAS, ACCEL_BIAS = 0.0001, 0.001  # rad/s, m/s^2

CAL_BIAS_RESIDUAL = 0.001    # m/s^2 
CAL_GYRO_RESIDUAL = 1e-5    # rad/s
LAUNCH_VEL_SIGMA = 0.01     # m/s per axis


def launch_attitude_sigma(bias_residual=CAL_BIAS_RESIDUAL):
    tilt = np.arcsin(min(bias_residual * np.sqrt(2) / 9.81, 1.0))
    return tilt / 2


def noise():
    Qn = np.diag([1.0]*3 + [1e-9]*4 + [1e-6]*3 + [1e-6])
    Rn = np.diag([GYRO_NOISE**2]*3 + [(10 * ACCEL_NOISE)**2]*3)
    P0 = np.diag([0.5]*3 + [1e-4]*4 + [1e-2]*3 + [0.1**2])
    return Qn, Rn, P0


V, Q, W, K = slice(0, 3), slice(3, 7), slice(7, 10), 10
NX = 11
ROT_SUBSTEPS = 4


def q_mult(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2])


def dq_from_rate(w, dt):
    n = np.linalg.norm(w)
    if n * dt < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    th = n * dt
    return np.concatenate([[np.cos(th/2)], (w/n) * np.sin(th/2)])


def skew(a):
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])


def omega_mat(w):
    """q_dot = 1/2 Omega(w) q, the same kinematics as dynamics.q_kinematics."""
    wx, wy, wz = w
    return np.array([[0, -wx, -wy, -wz],
                     [wx, 0, wz, -wy],
                     [wy, -wz, 0, wx],
                     [wz, wy, -wx, 0]])


def xi_mat(q):
    """q_dot = 1/2 Xi(q) w."""
    a, b, c, d = q
    return np.array([[-b, -c, -d], [a, -d, c], [d, a, -b], [-c, b, a]])


def aero(x, p, scale=1.0):
    X = np.concatenate([np.zeros(3), x[V], x[Q], x[W], np.zeros(p.n_wheels)])
    return scale * aero_force(X, p)


def partials(x, p, scale=1.0, eps=1e-6):
    F0 = aero(x, p, scale)
    h0 = q2dcm(x[Q]).T @ (x[K] * F0 / p.m)
    dF = np.zeros((3, 10))
    dacc = np.zeros((3, 10))
    for i in range(10):
        xp = x.copy()
        step = eps * max(1.0, abs(x[i]))
        xp[i] += step
        Fp = aero(xp, p, scale)
        dF[:, i] = (Fp - F0) / step
        dacc[:, i] = (q2dcm(xp[Q]).T @ (x[K] * Fp / p.m) - h0) / step
    return F0, h0, dF, dacc


def rot_rate(q, w, eta, tau, p):
    L_b = p.J_eff @ w + p.A @ eta
    return (0.5 * xi_mat(q) @ w,
            p.J_eff_inv @ (-p.A @ tau - np.cross(w, L_b)))


def rotate(q, w, eta, tau, dt, p, n=ROT_SUBSTEPS):
    h = dt / n
    for i in range(n):
        e0 = eta
        k1q, k1w = rot_rate(q, w, e0, tau, p)
        k2q, k2w = rot_rate(q + h/2*k1q, w + h/2*k1w, e0 + tau*h/2, tau, p)
        k3q, k3w = rot_rate(q + h/2*k2q, w + h/2*k2w, e0 + tau*h/2, tau, p)
        k4q, k4w = rot_rate(q + h*k3q, w + h*k3w, e0 + tau*h, tau, p)
        q = q_normalize(q + h/6*(k1q + 2*k2q + 2*k3q + k4q))
        w = w + h/6*(k1w + 2*k2w + 2*k3w + k4w)
        eta = e0 + tau * h
    return q, w, eta


def predict(x, P, eta, tau, dt, p, Qn, scale=1.0):
    v, q, w, k = x[V], x[Q], x[W], x[K]
    tau = np.clip(tau, -p.tau_limit, p.tau_limit)

    F0, _, dF, _ = partials(x, p, scale)
    a = np.array([0.0, 0.0, -p.g]) + k * F0 / p.m

    q_n, w_n, eta_n = rotate(q, w, eta, tau, dt, p)
    x_n = np.concatenate([v + a * dt, q_n, w_n, [k]])

    # Jacobian: closed form everywhere except the aero block
    L_b = p.J_eff @ w + p.A @ eta
    A = np.zeros((NX, NX))
    A[V, 0:10] = k * dF / p.m
    A[V, K] = F0 / p.m
    A[Q, Q] = 0.5 * omega_mat(w)
    A[Q, W] = 0.5 * xi_mat(q)
    A[W, W] = p.J_eff_inv @ (skew(L_b) - skew(w) @ p.J_eff)

    Ad = np.eye(NX) + A * dt
    return x_n, Ad @ P @ Ad.T + Qn * dt, eta_n, v * dt + 0.5 * a * dt**2


def update(x, P, z, p, Rn, scale=1.0):
    _, h0, _, dacc = partials(x, p, scale)
    h = np.concatenate([x[W], h0])

    H = np.zeros((6, NX))
    H[0:3, W] = np.eye(3) # gyro sees body rate directly
    H[3:6, 0:10] = dacc
    H[3:6, K] = h0 / x[K]

    S = H @ P @ H.T + Rn
    Kg = P @ H.T @ np.linalg.solve(S, np.eye(6))
    x_n = x + Kg @ (z - h)
    x_n[Q] = q_normalize(x_n[Q])
    I_KH = np.eye(NX) - Kg @ H
    return x_n, I_KH @ P @ I_KH.T + Kg @ Rn @ Kg.T


def full_state(r, x, eta, p):
    Om = eta / p.I_w - p.A.T @ x[W]
    return np.concatenate([r, x[V], x[Q], x[W], Om])


# Guidance
def w_setpoint(p, X, target, g=9.81, rho=1.22):
    r_i, v_i, q, w_b, Om = unpack(X)

    # t_go along y-axis (direction of travel)
    t_go = (target[1] - r_i[1]) / v_i[1]
    if t_go < 0.09:
       return np.zeros(3), np.zeros(3), np.zeros(3)

    # solve for a using kinematic equation
    a = 2 * (target - r_i - v_i * t_go) / (t_go**2)

    # remove non commanded a (gravity and drag)
    a_g = np.array([0,0,-g])
    C_D = 0.45 # hardcoded
    v_norm = np.linalg.norm(v_i)
    v_norm2 = v_norm**2
    q_dyn = 0.5 * rho * v_norm2
    v_i_hat = v_i / v_norm

    a_d = -q_dyn * p.area * C_D * v_i_hat / p.m
    a_cmd = a - a_g - a_d

    # orthogonal acceleration
    a_guid = a_cmd - v_i * np.dot(v_i, a_cmd) / v_norm2

    C_L_max = 0.319
    C_L_req = np.linalg.norm(a_guid) * p.m / (q_dyn * p.area)
    C_L_req = min(C_L_req, C_L_max*0.99)

    # needed body angular velocity
    w_cmd_i = np.cross(v_i, a_guid) / v_norm2
    w_cmd_i_norm = w_cmd_i / np.linalg.norm(w_cmd_i)
    w_cmd_mag = -np.log(1 - C_L_req/C_L_max) / 2.48E-3
    w_cmd_i = w_cmd_i_norm * w_cmd_mag

    return w_cmd_i, a_cmd, a_guid


# Control
def inner_loop(p, X, w_cmd_b, t_settle):
    # P controller on wheels
    _, _, _, w_b, Om = unpack(X)

    L_b = p.J @ w_b + p.A @ (p.I_w * Om)
    tau = -p.A.T @ np.cross(w_b, L_b)

    w_dot_des = (w_cmd_b - w_b) / t_settle
    tau_fb = -p.A.T @ (p.J_eff @ w_dot_des)

    return np.clip(tau + tau_fb, -p.tau_limit, p.tau_limit)


def inner_loop_old(p, X, w_cmd_b, k_p):
    # P controller on wheels
    _, _, _, w_b, Om = unpack(X)
    w_err_b = w_cmd_b - w_b

    A = p.A
    tau = np.zeros(p.n_wheels)
    for i in range(p.n_wheels):
        axis = A[:, i]
        tau[i] = -k_p * np.dot(axis, w_err_b)
        
    return np.clip(tau, -p.tau_limit, p.tau_limit)


# The whole loop as one tau(t, x) callable
def make_kinematic_gnc(p, aero_fns, target, x0_hat, inner_rate=500.0,
                       outer_rate=50.0, seed=42, t_max=3.0,
                       perfect_nav=False, log=True, flight_time=None,
                       aero_scale=1.0):
    """Build the kinematic guidance law as a tau(t, x_true) callable.

    Navigation runs every inner tick, guidance every outer tick. aero_scale
    makes the onboard aero model deliberately wrong, which is what the k state
    is there to absorb.
    """
    if len(target) < 3:
        target = np.array([target[0], 0, target[1]])

    F_aero, T_aero = aero_fns
    Qn, Rn, P0 = noise()

    imu = IMU(gyro_noise_std=GYRO_NOISE, accel_noise_std=ACCEL_NOISE,
              gyro_bias_std=GYRO_BIAS, accel_bias_std=ACCEL_BIAS, seed=seed)

    rng = np.random.RandomState(seed + 57)
    x0 = np.array(x0_hat, dtype=float)
    v0 = x0[3:6] + rng.normal(0, LAUNCH_VEL_SIGMA, 3)
    q0 = q_normalize(x0[6:10] + rng.normal(0, launch_attitude_sigma(), 4))
    b_g0 = imu.gyro_bias + rng.normal(0, CAL_GYRO_RESIDUAL, 3)
    b_a0 = imu.accel_bias + rng.normal(0, CAL_BIAS_RESIDUAL, 3)

    dt = 1.0 / inner_rate
    outer_dt = 1.0 / outer_rate
    state = {'x': np.concatenate([v0, q0, x0[10:13], [1.0]]), 'P': P0,
             'r': x0[0:3].copy(),
             'eta': p.I_w * (x0[13:] + p.A.T @ x0[10:13]),
             'tau': np.zeros(p.n_wheels), 't_outer': -1e9,
             'w_cmd_i': np.zeros(3)}
    rec = {'t': [], 'tau': [], 'w_setpoint': [], 'w_setpoint_t': [],
           'w_b': [], 'pos_err': [], 'att_err': [], 'k': [],
           # per-outer-tick guidance internals, for diagnosing the solve
           'a_cmd': [], 'a_guid': [], 'r_hat': [], 'v_hat': [], 't_go': []}

    def gnc(t, x_true):
        # navigation
        if perfect_nav:
            x_fb = x_true[:13 + p.n_wheels]
        else:
            state['x'], state['P'], state['eta'], dr = predict(
                state['x'], state['P'], state['eta'], state['tau'], dt, p, Qn,
                aero_scale)
            state['r'] = state['r'] + dr
            gyro, accel = imu.measure(t, x_true, p, F_aero(t, x_true))
            z = np.concatenate([gyro - b_g0, accel - b_a0])
            state['x'], state['P'] = update(state['x'], state['P'], z, p, Rn,
                                            aero_scale)
            x_fb = full_state(state['r'], state['x'], state['eta'], p)
            if log:
                rec['pos_err'].append(
                    float(np.linalg.norm(x_fb[0:3] - x_true[0:3])))
                # angle between the estimated and true attitude
                d = abs(float(np.dot(q_normalize(x_fb[6:10]),
                                     q_normalize(x_true[6:10]))))
                rec['att_err'].append(
                    float(np.degrees(2 * np.arccos(min(d, 1.0)))))
                rec['k'].append(float(state['x'][K]))

        # guidance and control
        due = (t - state['t_outer']) >= outer_dt
        if due:
            state['t_outer'] = t
            w_cmd_i, a_cmd, a_guid = w_setpoint(p, x_fb, target, g=9.81,
                                                rho=1.22)
            state['w_cmd_i'] = w_cmd_i

            if log:
                rec['w_setpoint'].append((q2dcm(x_fb[6:10]).T @ w_cmd_i).copy())
                rec['w_setpoint_t'].append(t)
                rec['w_b'].append(x_fb[10:13].copy())
                rec['a_cmd'].append(a_cmd.copy())
                rec['a_guid'].append(a_guid.copy())
                rec['r_hat'].append(x_fb[0:3].copy())
                rec['v_hat'].append(x_fb[3:6].copy())
                t_go = (target[1] - x_fb[1]) / x_fb[4]
                rec['t_go'].append(float(t_go))

        # rotate to body frame
        w_cmd_b = q2dcm(x_fb[6:10]).T @ state['w_cmd_i']
        state['tau'] = inner_loop(p, x_fb, w_cmd_b, 0.02)
        tau = state['tau']
        if log:
            rec['t'].append(t)
            rec['tau'].append(np.asarray(tau).copy())
        return tau

    gnc.log = rec
    gnc.imu = imu
    gnc.nav = state
    return gnc
