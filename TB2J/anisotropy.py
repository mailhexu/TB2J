import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import matrix_rank
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import curve_fit


@dataclass
class Anisotropy:
    T: np.ndarray = None
    direction: np.ndarray = None
    amplitude: float = None
    isotropic_part: float = None

    @classmethod
    def from_T6(cls, T6):
        T = T6_to_T(T6)
        isotropic_part = np.trace(T) / 3
        T -= isotropic_part * np.eye(3)
        direction, amplitude = anisotropy_tensor_to_vector(T)
        return cls(
            direction=direction, amplitude=amplitude, isotropic_part=isotropic_part, T=T
        )

    @classmethod
    def from_direction_amplitude(cls, direction, amplitude, isotropic_part=0.0):
        T = anisotropy_vector_to_tensor(direction, amplitude)
        return cls(
            T=T, direction=direction, amplitude=amplitude, isotropic_part=isotropic_part
        )

    @classmethod
    def from_tensor(cls, T):
        isotropic_part = np.trace(T) / 3 * np.eye(3)
        T -= np.trace(T) / 3 * np.eye(3)
        direction, amplitude = anisotropy_tensor_to_vector(T)
        return cls(
            T=T, direction=direction, amplitude=amplitude, isotropic_part=isotropic_part
        )

    def __post_init__(self):
        if self.isotropic_part is None:
            self.isotropic_part = 0
        if self.T is None:
            self.T = (
                anisotropy_vector_to_tensor(self.direction, self.amplitude)
                + self.isotropic_part
            )
        elif self.direction is None or self.amplitude is None:
            self.isotropic_part = np.trace(self.T) / 3 * np.eye(3)
            # print(f'anisotropic tensor = {self.anisotropic_part}')
            self.direction, self.amplitude = anisotropy_tensor_to_vector(
                self.anisotropic_part
            )
            self.isotropic_part = np.trace(self.T) / 3
        if self.T is None or self.direction is None or self.amplitude is None:
            raise ValueError(
                "The input does not have enough information to create the anisotropy object."
            )

    def is_rank_one(self):
        """
        single-axis anisotropy should be rank-one.
        """
        print(f"rank = {matrix_rank(self.anisotropic_part)}")
        print(f"anisotropic_part = {self.anisotropic_part}")
        return matrix_rank(self.anisotropic_part) == 1

    def tensor(self):
        return self.T

    @property
    def anisotropic_part(self):
        return self.T - self.isotropic_part * np.eye(3)

    @property
    def axis(self):
        return self.direction

    @property
    def axis_type(self):
        if self.is_easy_axis():
            return "easy"
        else:
            return "hard"

    def axis_angle(self, unit="rad"):
        theta = np.arccos(self.direction[2])
        phi = np.arctan2(self.direction[1], self.direction[0])
        if unit.startswith("deg"):
            theta = theta * 180 / np.pi
            phi = phi * 180 / np.pi
        return theta, phi

    def amplitude(self):
        return self.amplitude

    def is_easy_axis(self):
        return self.amplitude > 0

    def is_hard_axis(self):
        return self.amplitude < 0

    def energy_vector_form(self, S=None, angle=None, include_isotropic=False):
        if S is None:
            S = sphere_to_cartesian(angle)
        print(f"S shape = {S.shape}")
        # return anisotropy_energy_vector_form(S, *angles, self.amplitude, self.isotropic_part)
        print(f"direction shape = {self.direction.shape}")
        print(f"amplitude shape = {self.amplitude.shape}")
        print(f"iso shape = {self.isotropic_part.shape}")
        E = -self.amplitude * (S @ self.direction) ** 2
        if include_isotropic:
            E = E + self.isotropic_part
        return E

    def energy_tensor_form(self, S=None, angle=None, include_isotropic=False):
        # return anisotropy_energy_tensor_form(self.T, S)
        if S is None:
            S = sphere_to_cartesian(angle)
        if include_isotropic:
            return -S.T @ self.T @ S
        else:
            return -S.T @ self.anisotropic_part @ S

    @classmethod
    def fit_from_data(cls, thetas, phis, values, test=False, units="rad"):
        """
        Fit the anisotropic tensor to the data
        parameters:
           thetas: the polar angle in degree
           phis: the azimuthal angle in degree
           values: the anisotropic value
        Return:
           the anisotropic object fitted from the data
        """
        angles = np.vstack([thetas, phis])
        if units.lower().startswith("deg"):
            angles = np.deg2rad(angles)
        params, cov = curve_fit(anisotropy_energy, angles, values)
        fitted_values = anisotropy_energy(angles, *params)

        delta = fitted_values - values

        # print(f'Max value = {np.max(values)}, Min value = {np.min(values)}')
        if np.abs(delta).max() > 1e-4:
            print("Warning: The fitting is not consistent with the data.")
            print(f"Max-min = {np.max(values) - np.min(values)}")
            print(f"delta = {np.max(np.abs(delta))}")
        T = T6_to_T(params)
        obj = cls(T=T)

        if test:
            values2 = []
            for i in range(len(thetas)):
                E = obj.energy_tensor_form(
                    angle=[thetas[i], phis[i]], include_isotropic=True
                )
                values2.append(E)
            # delta2 = np.array(values2) - values
            # print(delta2)

            ax = plot_3D_scatter(angles, values - np.min(values), color="r")
            plot_3D_scatter(angles, values2 - np.min(values), ax=ax, color="b")
            plt.show()

        return obj

    @classmethod
    def fit_from_data_vector_form(cls, thetas, phis, values):
        """
        Fit the anisotropic tensor to the data
        parameters:
           thetas: the polar angle in degree
           phis: the azimuthal angle in degree
           values: the anisotropic value
        Return:
           the anisotropic object fitted from the data
        """
        angles = np.vstack([thetas, phis])
        params, cov = curve_fit(anisotropy_energy_vector_form, angles, values)
        fitted_values = anisotropy_energy_vector_form(angles, *params)
        delta = fitted_values - values
        print(f"Max value = {np.max(values)}, Min value = {np.min(values)}")
        print(f"Max-min = {np.max(values) - np.min(values)}")
        print(f"delta = {delta}")
        theta_a, phi_a, amplitude, isotropic_part = params
        direction = sphere_to_cartesian([theta_a, phi_a])
        return cls.from_direction_amplitude(
            direction=direction, amplitude=amplitude, isotropic_part=isotropic_part
        )

    @classmethod
    def fit_from_data_file(cls, fname, method="tensor"):
        """
        Fit the anisotropic tensor to the data
        parameters:
           fname: the file name of the data
        Return:
           anisotropy: the anisotropic object
        """
        data = np.loadtxt(fname)
        theta, phi, value = data[:, 0], data[:, 1], data[:, 2]
        if method == "tensor":
            return cls.fit_from_data(theta, phi, value)
        elif method == "vector":
            return cls.fit_from_data_vector_form(theta, phi, value)
        else:
            raise ValueError(f"Unknown method {method}")

    @classmethod
    def fit_from_xyz_data_file(cls, fname, method="tensor"):
        """
        Fit the anisotropic tensor to the data with x y z val form
        parameters:
           fname: the file name of the data
        Return:
              anisotropy: the anisotropic object
        """
        data = np.loadtxt(fname)
        xyz, value = data[:, 0:3], data[:, 3]
        theta, phi = np.array([cartesian_to_sphere(t) for t in xyz]).T
        if method == "tensor":
            return cls.fit_from_data(theta, phi, value)
        elif method == "vector":
            return cls.fit_from_data_vector_form(theta, phi, value)
        else:
            raise ValueError(f"Unknown method {method}")

    def plot_3d(self, ax=None, figname=None, show=True, surface=True):
        """
        plot the anisotropic energy in all directions in 3D
        S is the spin unit vector
        """
        # theta, phi = np.meshgrid(theta, phi)
        # value = self.energy_tensor_form(sphere_to_cartesian([theta, phi]))
        # x, y, z = sphere_to_cartesian(theta, phi, )

        if surface:
            thetas = np.arange(0, 181, 1)
            phis = np.arange(0, 362, 1)

            X, Y = np.meshgrid(thetas, phis)
            Z = np.zeros(X.shape)
            for i in range(len(thetas)):
                for j in range(len(phis)):
                    # S = sphere_to_cartesian([thetas[i], phis[j]])
                    # E = self.energy_vector_form(angle=[thetas[i], phis[j]])
                    E = self.energy_tensor_form(angle=[thetas[i], phis[j]])
                    Z[j, i] = E
            Z = Z - np.min(Z)
            imax = np.argmax(Z)
            X_max, Y_max, Z_max = (
                X.flatten()[imax],
                Y.flatten()[imax],
                Z.flatten()[imax],
            )
            imin = np.argmin(Z)
            X_min, Y_min, Z_min = (
                X.flatten()[imin],
                Y.flatten()[imin],
                Z.flatten()[imin],
            )

            X, Y, Z = sphere_to_cartesian([X, Y], Z)
            X_max, Y_max, Z_max = sphere_to_cartesian([X_max, Y_max], r=Z_max)
            X_min, Y_min, Z_min = sphere_to_cartesian([X_min, Y_min], r=Z_min)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.5)
            # scatter the minimal and maximal points
            ax.scatter(X_max, Y_max, Z_max, color="r", marker="o")
            ax.scatter(X_min, Y_min, Z_min, color="b", marker="o")

            # draw the easy axis
            if self.is_easy_axis():
                color = "r"
            else:
                color = "b"
            d = self.direction
            d = d / np.sign(d[0])
            d *= np.abs(self.amplitude) * 2.5
            ax.quiver(
                -d[0],
                -d[1],
                -d[2],
                2 * d[0],
                2 * d[1],
                2 * d[2],
                color=color,
                arrow_length_ratio=0.03,
            )

        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            thetas = np.arange(0, 181, 5)
            phis = np.arange(0, 360, 5)
            Es = []
            angles = []
            for t in thetas:
                for p in phis:
                    # S = sphere_to_cartesian([t, p])
                    angles.append([t, p])
                    Es.append(
                        self.energy_tensor_form(angle=[t, p], include_isotropic=False)
                    )
                    # x, y, z = sphere_to_cartesian([t, p], r=E)
                    # ax.scatter(x,y, z, cmap='viridis')
            angles = np.array(angles)
            # plot_3D_scatter(angles.T, Es-np.min(Es), ax=None)
            plot_3D_scatter(angles.T, Es, ax=None)
        if figname is not None:
            plt.savefig(figname)
            plt.close()
        if show:
            plt.show()

    def plot_contourf(self, ax=None, figname=None, show=False):
        if ax is None:
            fig, ax = plt.subplots()
        X, Y = np.meshgrid(np.arange(0, 180, 1), np.arange(-180, 180, 1))
        Z = np.zeros(X.shape)
        ntheta, nphi = X.shape
        for i in range(ntheta):
            for j in range(nphi):
                E = self.energy_tensor_form(angle=[X[i, j], Y[i, j]])
                Z[i, j] = E
        # find the X, Y for min and max of Z
        X_max, Y_max = np.unravel_index(np.argmax(Z), Z.shape)
        X_min, Y_min = np.unravel_index(np.argmin(Z), Z.shape)
        X_max, Y_max = X[X_max, Y_max], Y[X_max, Y_max]
        X_min, Y_min = X[X_min, Y_min], Y[X_min, Y_min]
        c = ax.contourf(X, Y, Z, cmap="viridis", levels=200)
        # print(X_max, Y_max, X_min, Y_min)
        # ax.scatter(X_max, Y_max, color="r", marker="o")
        # ax.scatter(X_min, Y_min, color="b", marker="o")
        ax.set_xlabel("$\theta$ (degree)")
        ax.set_ylabel("$\phi$ degree")
        # ax.scatter(X_max, Y_max, color="r", marker="o")
        # ax.scatter(X_min, Y_min, color="r", marker="o")

        # colorbar
        _cbar = plt.colorbar(c, ax=ax)

        if figname is not None:
            plt.savefig(figname)
            plt.close()
        if show:
            print(f"Max = {X_max}, {Y_max}, Min = {X_min}, {Y_min}")
            plt.show()
        return ax

    def tensor_strings(self, include_isotropic=False, multiplier=1):
        """
        convert the energy tensor to strings for easy printing
        parameters:
        include_isotropic: if include the isotropic part
        multiplier: the multiplier for the tensor. Use for scaling the tensor by units.
        """
        if include_isotropic:
            T = self.T
        else:
            T = self.T
        strings = np.array2string(T * multiplier, precision=5, separator=" ")
        return strings


def plot_3D_scatter(angles, values, ax=None, **kwargs):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    thetas, phis = angles
    for i in range(len(thetas)):
        E = values[i]
        t, p = thetas[i], phis[i]
        x, y, z = sphere_to_cartesian([t, p], r=E)
        ax.scatter(x, y, z, **kwargs)
    return ax


def plot_3D_surface(fname, figname=None, show=True):
    data = np.loadtxt(fname)
    theta, phi, value = data[:, 0], data[:, 1], data[:, 2]
    value = value - np.min(value)

    imax = np.argmax(value)
    angle_max = theta[imax], phi[imax]
    imin = np.argmin(value)
    angle_min = theta[imin], phi[imin]
    amplitude = np.max(value) - np.min(value)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    x, y, z = sphere_to_cartesian([theta, phi], r=value)
    # ax.plot_surface(x, y, z, triangles=None, cmap='viridis')
    ax.scatter(x, y, z, cmap="viridis")
    # ax.scatter(theta,phi, value, cmap='viridis')
    # draw a surface plot with a color map.
    if figname is not None:
        plt.savefig(figname)
    if show:
        plt.show()
    plt.close()
    return angle_max, angle_min, amplitude


def plot_3D_surface_interpolation(fname, figname=None, show=True):
    data = np.loadtxt(fname)
    theta, phi, value = data[:, 0], data[:, 1], data[:, 2]

    imax = np.argmax(value)
    angle_max = theta[imax], phi[imax]
    imin = np.argmin(value)
    angle_min = theta[imin], phi[imin]
    amplitude = np.max(value) - np.min(value)

    value = value - np.min(value)
    # interploate the data
    interp = LinearNDInterpolator((theta, phi), value)
    thetas = np.arange(0, 181, 1)
    phis = np.arange(0, 340, 1)

    X, Y = np.meshgrid(thetas, phis)
    Z = interp(X, Y)
    print(Z)
    # print(np.max(Z), np.min(Z))
    Z = Z - np.min(Z)
    X, Y, Z = sphere_to_cartesian([X, Y], Z)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis")
    if figname is not None:
        plt.savefig(figname)
    if show:
        plt.show()
    plt.close()
    return angle_max, angle_min, amplitude


def sphere_to_cartesian(angles, r=1):
    """
    Transform the spherical coordinates to the cartesian coordinates
    parameters:
    angles: the polar and azimuthal angle in degree
    r: the radius
    Return:
    x, y, z: the cartesian coordinates
    """
    # print(angles)
    theta, phi = angles
    theta = theta * np.pi / 180
    phi = phi * np.pi / 180
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])


def cartesian_to_sphere(xyz, unit="deg"):
    """
    Transform the cartesian coordinates to the spherical coordinates
    parameters:
    xyz: the cartesian coordinates
    Return:
    theta, phi: the polar and azimuthal angle in degree
    """
    x, y, z = xyz
    r = np.linalg.norm(xyz)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    if unit.lower().startswith("deg"):
        theta = theta * 180 / np.pi
        phi = phi * 180 / np.pi
    elif unit.lower.startswith("rad"):
        pass
    else:
        raise ValueError("unit must be 'deg' or 'rad'")
    return np.array([theta, phi])


def anisotropy_energy(angle, Txx, Tyy, Tzz, Tyz, Tzx, Txy):
    """
    Calculate the anisotropic energy
    parameters:
    angle: the polar and azimuthal angle in degree
    Txx, Tyy, Tzz, Tyz, Tzx, Txy: the anisotropic tensor
    Return:
    E: the anisotropic energy
    """
    # transform the tensor to the matrix form
    T = np.array([[Txx, Txy, Tzx], [Txy, Tyy, Tyz], [Tzx, Tyz, Tzz]])

    S = sphere_to_cartesian(angle)
    E = -np.diag(S.T @ T @ S)
    return E


def anisotropy_energy_vector_form(S, k_theta, k_phi, amplitude, isotropic_part=0):
    """
    Calculate the anisotropic energy from the vector form
    parameters:
        S: the spin vector
        kx, ky, kz: the direction of the anisotropy
    """
    Scart = sphere_to_cartesian(S)
    k = sphere_to_cartesian([k_theta, k_phi])
    print(f"k shape = {k.shape}")
    print(f"Scart shape = {Scart.shape}")
    E = [-amplitude * (Si @ k) ** 2 + isotropic_part for Si in Scart.T]
    return E


def T6_to_T(T6):
    """
    Transform the anisotropic tensor to the matrix form
    """
    Txx, Tyy, Tzz, Tyz, Tzx, Txy = T6
    T = np.array([[Txx, Txy, Tzx], [Txy, Tyy, Tyz], [Tzx, Tyz, Tzz]])
    return T


def is_rank_one(T):
    """
    Check if the tensor is rank one
    """
    return matrix_rank(T) == 1


def fit_anisotropy(thetas, phis, values):
    """
    Fit the anisotropic tensor to the data
    parameters:
    theta: the polar angle in degree
    phi: the azimuthal angle in degree
    value: the anisotropic value
    Return:
     T: the anisotropic tensor
    """
    # transform the data to the cartesian coordinates
    # fit the data to the anisotropic energy
    angles = np.vstack([thetas, phis])
    params, cov = curve_fit(anisotropy_energy, angles, values)
    # check if the fitting is consistent with the data
    T = T6_to_T(params)
    direction, amp = anisotropy_tensor_to_vector(T)
    return direction, amp


def anisotropy_energy_vector_form2(direction, amplitude, S):
    """
    Calculate the anisotropic energy
    parameters:
    direction: the easy axis/ hard axis of the anisotropy
    amplitude: the amplitude of the anisotropy
    S: the spin vector
    Return:
    E: the anisotropic energy, E = - amplitude * (S @ direction)**2
    """
    # normalize the direction
    direction = direction / np.linalg.norm(direction)
    print(f"direction shape = {direction.shape}")
    E = -amplitude * (S @ direction) ** 2
    return E


def anisotropy_energy_tensor_form(T, S):
    """
    Calculate the anisotropic energy
    parameters:
    T: the anisotropic tensor
    S: the spin vector
    Return:
    E: the anisotropic energy, E = - S.T @ T @ S
    """
    E = -S.T @ T @ S
    return E


def anisotropy_vector_to_tensor(direction, amplitude):
    """
    Transform the anisotropic vector to the anisotropic tensor
    parameters:
    direction: the easy axis/ hard axis of the anisotropy (direction is normalized)
    amplitude: the amplitude of the anisotropy
    Return:
    T: the anisotropic tensor
    """
    direction = direction / np.linalg.norm(direction)
    T = amplitude * np.outer(direction, direction)
    return T


def anisotropy_tensor_to_vector(T):
    """
    Transform the anisotropic tensor to the anisotropic vector
    parameters:
    T: the anisotropic tensor
    Return:
    direction: the easy axis/ hard axis of the anisotropy
    amplitude: the amplitude of the anisotropy, if the anisotropy is positive, the easy axis is the easy axis, otherwise, the hard axis is the easy axis
    """
    w, v = np.linalg.eig(T)
    if not is_rank_one(T):
        # print("Warning: The anisotropy  tensor is not rank one. The tensor cannot be transformed to the vector form.")
        # print(f"The eigenvalues are {w}.")
        pass
    index = np.argmax(np.abs(w))
    direction = v[:, index]
    direction = direction / np.sign(direction[0])
    amplitude = w[index]
    return direction, amplitude


def test_anisotorpy_vector_to_tensor():
    direction = np.random.rand(3)
    direction = direction / np.linalg.norm(direction)
    amplitude = random.uniform(-1, 1)
    S = np.random.rand(3)
    S = S / np.linalg.norm(S)
    T = anisotropy_vector_to_tensor(direction, amplitude)
    E_vector = anisotropy_energy_vector_form(direction, amplitude, S)
    E_tensor = anisotropy_energy_tensor_form(T, S)
    diff = E_vector - E_tensor
    print(f"diff = {diff}")
    assert np.abs(diff) < 1e-10

    # test if the inverse transformation get the direction and amplitude back
    dir2, amp2 = anisotropy_tensor_to_vector(T)

    # set the first element of the direction to be positive
    if direction[0] * dir2[0] < 0:
        dir2 = -dir2
    diff = np.linalg.norm(dir2 - direction)
    # print(f'direction = {direction}, amplitude = {amplitude}')
    # print(f'dir2 = {dir2}, amp2 = {amp2}')
    assert diff < 1e-10
    assert np.abs(amp2 - amplitude) < 1e-10


def test_anisotropy_tensor_to_vector():
    T = np.random.rand(3, 3)
    T = T + T.T
    T = T - np.trace(T) / 3
    direction, amplitude = anisotropy_tensor_to_vector(T)
    T2 = anisotropy_vector_to_tensor(direction, amplitude)
    print(f"T = {T}")
    print(f"T2 = {T2}")
    diff = np.linalg.norm(T - T2)
    print(f"diff = {diff}")
    assert diff < 1e-10


def test_fit_anisotropy():
    data = np.loadtxt("anisotropy.dat")
    theta, phi, value = data[:, 0], data[:, 1], data[:, 2]
    angles = np.vstack([theta, phi])
    T6 = fit_anisotropy(theta, phi, value)
    T = T6_to_T(T6)
    if not is_rank_one(T):
        print("Warning: The anisotropy  tensor is not rank one. ")
    fitted_values = anisotropy_energy(angles, *T6)
    delta = fitted_values - value
    print(f"Max value = {np.max(value)}, Min value = {np.min(value)}")
    print(
        f"Max fitted value = {np.max(fitted_values)}, Min fitted value = {np.min(fitted_values)}"
    )
    for i in range(len(theta)):
        print(
            f"theta = {theta[i]}, phi = {phi[i]}, value = {value[i]}, fitted value = {fitted_values[i]}, delta = {delta[i]}"
        )


def view_anisotropy_strain():
    strains1 = [(x, 0.0) for x in np.arange(0.000, 0.021, 0.002)]
    strains2 = [(0.02, y) for y in np.arange(0.000, 0.021, 0.002)]
    strains = strains1 + strains2
    path = Path("anisotropy_strain_from_tensor")
    path.mkdir(exist_ok=True)
    fname = "a.dat"
    fh = open(fname, "w")
    fh.write("# s0  s1  axis   amplitude   axis_type angle_111 \n")
    for strain in strains:
        s0, s1 = strain
        ani = Anisotropy.fit_from_data_file(f"a{s0:.3f}_b{s1:.3f}_plusU.dat")
        dx, dy, dz = ani.direction
        angle_from_111 = (
            np.arccos(ani.direction @ np.array([1, 1, 1]) / np.sqrt(3)) * 180 / np.pi
        )
        fh.write(
            f"{s0:.3f}  {s1:.3f} ( {dx:.3f}  {dy:.3f}  {dz:.3f} ) {ani.amplitude:.5f} {ani.axis_type} {angle_from_111} \n"
        )
        ani.plot_3d(
            surface=True, figname=path / f"a{s0:.3f}_b{s1:.3f}_plusU.png", show=False
        )
    fh.close()


def view_anisotropy_strain_raw():
    strains1 = [(x, 0.0) for x in np.arange(0.000, 0.021, 0.002)]
    strains2 = [(0.02, y) for y in np.arange(0.000, 0.021, 0.002)]
    strains = strains1 + strains2
    path = Path("anisotropy_strain_raw")
    path.mkdir(exist_ok=True)
    fname = "a_raw.dat"
    fh = open(fname, "w")
    fh.write("# s0  s1  direction(max) direction(min)  amplitude\n")
    for strain in strains:
        print(f"strain = {strain}")
        s0, s1 = strain
        # plot_3D_surface(f"a{s0:.3f}_b{s1:.3f}_plusU.dat")
        angle_max, angle_min, amplitude = plot_3D_surface(
            f"a{s0:.3f}_b{s1:.3f}_plusU.dat",
            figname=path / f"a{s0:.3f}_b{s1:.3f}_plusU.png",
        )
        dmax = sphere_to_cartesian(angle_max)
        dmin = sphere_to_cartesian(angle_min)
        fh.write(
            f"{s0:.3f}  {s1:.3f} ( {dmax[0]:.3f}  {dmax[1]:.3f}  {dmax[2]:.3f} ) ({dmin[0]:.3f}  {dmin[1]:.3f}  {dmin[2]:.3f}) {amplitude:.5f} \n"
        )

        plt.close()


if __name__ == "__main__":
    # test_fit_anisotropy()
    # test_anisotorpy_vector_to_tensor()
    # test_anisotropy_tensor_to_vector()
    # ani = Anisotropy.fit_from_data("anisotropy.dat")
    # ani = Anisotropy.fit_from_data_file("a0.002_b0.000_plusU.dat")
    # ani = Anisotropy.fit_from_data_file("a0.002_b0.000_plusU.dat", method='tensor')
    # print(f'direction = {ani.direction}')
    # ani.plot_3d()
    # plot_3D_surface("a0.002_b0.000_plusU.dat")
    # plot_3D_surface_interpolation("a0.020_b0.020_plusU.dat")
    # plot_3D_surface("a0.020_b0.020_plusU.dat")
    # plot_3D_surface("a0.020_b0.000_plusU.dat")
    view_anisotropy_strain()
    # view_anisotropy_strain_raw()
    # s0=0.000
    # s1=0.000
    # plot_3D_surface_interpolation(f"a{s0:.3f}_b{s1:.3f}_plusU.dat", figname=None)
