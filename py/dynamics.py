import numpy as np

# Constants

# Standard baseball dimensions (metric)
MASS = 0.145       
DIAMETER = 0.074  
RADIUS = DIAMETER / 2.0
AREA = np.pi * (DIAMETER ** 2) / 4.0
I_BALL = 0.4 * MASS * RADIUS**2 # Solid sphere

RHO = 1.22
g =  9.81
C_D = 0.45


class BallParams:
    def __init__(self, m=MASS, r=DIAMETER/2, I=I_BALL, g=g, A=None, I_w=None, tau_limit=None):
        # Constants
        self.m = m
        self.r = r
        self.I = I
        self.g = g
        self.area = np.pi * self.r**2


        # Reaction wheels
        if A is None:
            A = np.array([[0.0], [0.0], [1.0]])
        if I_w is None:
            I_w = np.array([1E-8])
        self.I_w = I_w
        if tau_limit is None:
            tau_limit = 10
        self.tau_limit = tau_limit

        # Verify shapes
        if A.shape[0] != 3:
            raise ValueError(f'A must be at least 3 x n, got {A.shape} instead.')
        norm = np.linalg.norm(A, axis=0)
        if np.any(norm < 1e-12):
            raise ValueError(f'Wheel axis with 0 length')
        self.A = A / norm
        if I_w.shape != (self.A.shape[1],):
            raise ValueError(f'I_w must be have {self.A.shape[1]} entries, got {I_w.shape} instead.')
        self.n_wheels = self.A.shape[1]


        # Inertia (assumes fly wheels have 0 mass)
        Aw = self.A * self.I_w
        self.J = self.I * np.eye(3) + Aw @ self.A.T
        self.J_eff = self.J - Aw @ self.A.T
        self.J_eff_inv = np.linalg.inv(self.J_eff)


    def change(self, **changes):
        """Create new parameters with changes, useful for MC sweeps."""
        kwargs = dict(m=self.m, r=self.r, I=self.I, g=self.g, A=self.A,
                      I_w=self.I_w, tau_limit=self.tau_limit)
        for key in changes:
            if key not in kwargs:
                raise ValueError(f'Unknown Parameter {key}')
        kwargs.update(changes)
        return BallParams(**kwargs)


# State
def pack(r_i, v_i, q_ib, w_b, Om):
    """ Pack state vector
    13 States
    1:3   position (inertial)
    4:6   velocity (inertial)
    7:10  quaternion (inertial to body)
    11:13 angular velocity (body)
    13:   angular velocity (each reaction wheel)
    """
    return np.concatenate([
        np.array(r_i),
        np.array(v_i),
        q_normalize(q_ib),
        np.array(w_b),
        np.array(Om)
    ])

def unpack(X):
    """ Unpack state vector
    13 States
    1:3   position (inertial)
    4:6   velocity (inertial)
    7:10  quaternion (inertial to body)
    11:13 angular velocity (body)
    13:   angular velocity (each reaction wheel)
    """
    X = np.asarray(X)
    return X[0:3], X[3:6], X[6:10], X[10:13], X[13:]

def normalize(X):
    X = np.array(X)
    X[6:10] = q_normalize(X[6:10])
    return X

# Quaternions
def q_normalize(q):
    n = np.linalg.norm(q, axis=0)
    return q / n

def q2dcm(q):
    a, b, c, d = q_normalize(q)
    return np.array([
        [1 - 2 * (c * c + d * d), 2 * (b * c - d * a),     2 * (b * d + c * a)],
        [2 * (b * c + d * a),     1 - 2 * (b * b + d * d), 2 * (c * d - b * a)],
        [2 * (b * d - c * a),     2 * (c * d + b * a),     1 - 2 * (b * b + c * c)],
    ])

def q_kinematics(q, w_b):
    """q_dot = 1/2 xi(q) omega"""
    a, b, c, d = q
    xi = np.array([
        [-b, -c, -d],
        [ a, -d,  c],
        [ d,  a, -b],
        [-c,  b,  a],
    ])
    return 0.5 * xi @ w_b


# Angular Momentum
def L_body(X, p):
    _, _, _, w_b, Om = unpack(X)
    return p.J @ w_b + p.A @ (p.I_w * Om)

# Derivative
def derivative(X, p, tau_w=None, F_ext_i=None, T_ext_b=None):
    """Time derivative of the state, X_dot
    X        state vector, length 13 + n_wheels
    p        BallParams
    tau_w    motor torques applied to reaction wheels about their axes [N m], (n,)
    F_ext_i  external force (inertial) excluding gravity [N], (3,)
    T_ext_b  external torque (body) [N m], (3,)
    """

    r_i, v_i, q, w_b, Om = unpack(X)
    n = p.n_wheels

    if tau_w is None:
        tau_w = np.zeros(n)
    elif tau_w.shape != (n,):
        raise ValueError(f'tau_w must have {n} entries')

    # Saturation
    tau_w = np.clip(tau_w, -p.tau_limit, p.tau_limit)

    if F_ext_i is None:
        F_ext_i = np.zeros(3)
    if T_ext_b is None:
        T_ext_b = np.zeros(3)


    # Translational acceleration
    g_i = np.array([0, 0, -p.g])
    a_i = g_i + F_ext_i / p.m

    # Attitude kinematics
    q_dot = q_kinematics(q, w_b)

    # Rotation
    L_b = L_body(X, p)
    w_dot = p.J_eff_inv @ (T_ext_b - p.A @ tau_w - np.cross(w_b, L_b))

    # Reaction wheels
    Om_dot = tau_w / p.I_w - p.A.T @ w_dot

    return np.concatenate([v_i, a_i, q_dot, w_dot, Om_dot])


def ode_wrapper(p, tau_w=None, F_ext_i=None, T_ext_b=None):
    """Ode wrapper into form f(t, x)"""
    def ode(t, x):
        return derivative(
            x, p,
            tau_w=None if tau_w is None else tau_w(t, x),
            F_ext_i=None if F_ext_i is None else F_ext_i(t, x),
            T_ext_b=None if T_ext_b is None else T_ext_b(t, x)
            )
    return ode


def lift_coefficient(omega_mag):
    """
    Calculate the dimensionless lift coefficient (C_L) based on rotational speed.
    Derived from equation (10) for golf balls and baseballs.
    """
    return 3.19e-1 * (1.0 - np.exp(-2.48e-3 * omega_mag))


# Aerodynamics
def aero_state(X, p):
    """Speed, unit velocity, inertial spin vector and spin magnitude."""
    _, v_i, q, w_b, _ = unpack(X)
    v = np.linalg.norm(v_i)
    if v < 1e-8:
        return 0.0, np.zeros(3), np.zeros(3), 0.0
    v_hat = v_i / v
    w_i = q2dcm(q) @ w_b
    omega_mag = np.linalg.norm(w_i)
    return v, v_hat, w_i, omega_mag


def aero_force(X, p):
    v, v_hat, w_i, omega_mag = aero_state(X, p)
    if v < 1e-8:
        return np.zeros(3)

    q_dyn = 0.5 * RHO * v**2

    # Drag
    Fd = -q_dyn * C_D * p.area * v_hat

    # Lift (Magnus), C_L on the spin across the velocity
    lift_dir = np.cross(w_i, v_hat)
    w_perp = np.linalg.norm(lift_dir)
    if w_perp > 1e-8:
        C_L = lift_coefficient(w_perp)
        Fl = q_dyn * C_L * p.area * lift_dir / w_perp
    else:
        Fl = np.zeros(3)

    return Fd + Fl


def aero_torque(X, p):
    """Aerodynamic spin decay. Neglected: over a single pitch's flight time
    the spin-down rate is small enough not to matter."""
    return np.zeros(3)


def aero_wrapper(p, air=None):
    """Aero wrapper into form f(t, x)"""
    global RHO
    if air is not None:
        RHO = air.rho

    def F(t, x):
        return aero_force(x, p)

    def T(t, x):
        return aero_torque(x, p)

    return F, T


if __name__ == '__main__':
    from scipy.integrate import solve_ivp

    RELEASE_Y, RELEASE_Z = 16.8, 1.8

    def flight(v_mph, spin_rpm, spin_axis):
        p = BallParams(m=MASS, r=RADIUS, I=I_BALL, g=9.81)
        F, T = aero_wrapper(p)

        v = v_mph * 0.44704
        w_mag = spin_rpm * 2 * np.pi / 60
        w_b = w_mag * np.asarray(spin_axis, float)

        # Ball travels along -y
        x0 = pack([0.0, RELEASE_Y, RELEASE_Z], [0.0, -v, 0.0], [1, 0, 0, 0],
                  w_b, np.zeros(p.n_wheels))

        def plate(t, x): return x[1]
        plate.terminal = True
        plate.direction = -1

        def ground(t, x): return x[2]
        ground.terminal = True
        ground.direction = -1

        s = solve_ivp(ode_wrapper(p, F_ext_i=F, T_ext_b=T), (0, 2.0), x0,
                      method='DOP853', max_step=0.002, rtol=1e-10, atol=1e-12,
                      events=[plate, ground])

        assert s.success
        return s.y[0, -1], s.y[2, -1], s.t[-1], bool(s.t_events[0].size)

    print('Validation against Robinson (2013) Aerodynamic Model\n')

    print(f'ball: m={MASS} kg, r={RADIUS * 1000:.1f} mm, '
          f'release {RELEASE_Y} m from plate\n')

    print(f'{"pitch":28s} {"t [s]":>6s} {"horiz [in]":>11s} {"vert [in]":>10s}'
          f'   published')

    cases = [
        ('4-seam fastball 95/2400', 95, 2400, (-1, 0, 0), 'rise  ~15-17 in'),
        ('curveball 80/2500', 80, 2500, (1, 0, 0), 'drop  ~10-15 in'),
        ('slider 85/2400', 85, 2400, (0, 0, 1), 'horiz ~6-15 in'),
        ('gyro spin 85/2400 (control)', 85, 2400, (0, 1, 0), 'no movement'),
    ]

    for lab, vm, rpm, axis, expect in cases:
        x, z, t, ok = flight(vm, rpm, axis)
        x0_, z0_, _, _ = flight(vm, 0.0, axis)   # spinless reference
        print(f'{lab:28s} {t:6.3f} {(x-x0_)/0.0254:11.1f} '
              f'{(z-z0_)/0.0254:10.1f}   {expect}'
              f'{"" if ok else "  [hit ground]"}')
