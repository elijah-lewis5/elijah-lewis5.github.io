import numpy as np
from dynamics import unpack, q2dcm


class IMU:
    """6-DOF IMU with gyroscope and accelerometer."""

    def __init__(self, gyro_noise_std=0.01, accel_noise_std=0.05,
                 gyro_bias_std=0.001, accel_bias_std=0.1,
                 seed=None):
        self.gyro_noise_std = gyro_noise_std
        self.accel_noise_std = accel_noise_std
        self.gyro_bias_std = gyro_bias_std
        self.accel_bias_std = accel_bias_std

        self.rng = np.random.RandomState(seed)

        # Constant bias, drawn once per flight
        self.gyro_bias = self.rng.normal(0, gyro_bias_std, 3)
        self.accel_bias = self.rng.normal(0, accel_bias_std, 3)

    def measure(self, t, x, p, F_ext_i):
        """Noisy gyro and accelerometer readings."""
        _, _, q, w_b, _ = unpack(x)

        R_i2b = q2dcm(q).T

        specific_force_b = R_i2b @ (F_ext_i / p.m)

        gyro_noise = self.rng.normal(0, self.gyro_noise_std, 3)
        accel_noise = self.rng.normal(0, self.accel_noise_std, 3)

        gyro_meas = w_b + self.gyro_bias + gyro_noise
        accel_meas = specific_force_b + self.accel_bias + accel_noise

        return gyro_meas, accel_meas

    def set_seed(self, seed):
        """Change the random seed and reinitialize RNG."""
        self.rng = np.random.RandomState(seed)
        self.reset_bias(self.gyro_bias_std, self.accel_bias_std)

    def reset_bias(self, gyro_bias_std=None, accel_bias_std=None, seed=None):
        """Resample the biases for a new flight, stored values if None."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        if gyro_bias_std is None:
            gyro_bias_std = self.gyro_bias_std
        if accel_bias_std is None:
            accel_bias_std = self.accel_bias_std

        self.gyro_bias = self.rng.normal(0, gyro_bias_std, 3)
        self.accel_bias = self.rng.normal(0, accel_bias_std, 3)
