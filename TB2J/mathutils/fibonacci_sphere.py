# -*- coding: utf-8 -*-
"""
scan the spheres in the space of (theta, phi), so that the points are uniformly distributed on a sphere.
The algorithm is based on fibonacci spiral sampling method.
Reference:
https://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/

But note that the convention of theta and phi are different from the one in the reference.
Here, cos(theta) = z, and phi is the azimuthal angle in the x-y plane.
In the reference, theta is the azimuthal angle in the x-y plane, and phi is the polar angle.
"""

import numpy as np
from numpy import pi


def fibonacci_sphere(samples=100):
    """
    Fibonacci Sphere Sampling Method
    Parameters:
        samples (int): number of points to sample on the sphere
    Returns:
        theta and phi: numpy arrays with shape (samples,) containing the angles of the sampled points.
    """
    # Initialize the golden ratio and angles
    goldenRatio = (1 + np.sqrt(5)) / 2
    phi = np.arange(samples) * (2 * pi / goldenRatio)
    theta = np.arccos(1 - 2 * np.arange(samples) / samples)
    return theta, phi


def fibonacci_semisphere(samples=100):
    """
    Fibonacci Sphere Sampling Method for the upper hemisphere of a sphere.

    Parameters:
        samples (int): number of points to sample on the sphere

    Returns:
        theta and phi: numpy arrays with shape (samples,) containing the angles of the sampled points.
    """
    # Initialize the golden ratio and angles
    goldenRatio = (1 + np.sqrt(5)) / 2
    phi = np.arange(samples) * (2 * pi / goldenRatio)
    theta = np.arccos(np.linspace(0, 1, samples))
    return theta, phi


def test_fibonacci_sphere():
    import matplotlib.pyplot as plt

    # Generate points on the sphere
    samples = 20000
    theta, phi = fibonacci_sphere(samples)
    # theta, phi = fibonacci_semisphere(samples)

    # Convert spherical coordinates to Cartesian coordinates
    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(theta)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, s=13.5)
    # ax.plot(x, y, z)

    # Set aspect to 'equal' for equal scaling in all directions
    ax.set_aspect("equal")

    plt.show()


if __name__ == "__main__":
    test_fibonacci_sphere()
