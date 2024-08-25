from ase.dft.kpoints import kpoint_convert
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math, random

# How to use the converter to go from scaled to cartesian
# ct = kpoint_convert(self.bs.path.cell, skpts_kc=sc)


def stencil(
    pt_sc,
    pt_ct,
    path,
    density,
    extent=0.5,
    plot_stencil=False,
    stencil_type="xyz",
):
    """
    Parameters:
    -----------
    pt_sc: ndarray
        kpoint to center the stencil around in scaled coordinates
    pt_ct: ndarray
        kpoint to center the stencil around in cartesian coordinates
    path: ase.dft.kpoints.BandPath
        Bandpath object used for sc->ct conversion
    density: float
        Desired kpt density in # of points per recpirocal lattice length (1/A),
        therefore units are in #*A
    extent: float
        Extent (in scaled coordinates) from pt_sc that the stencil should move,
        e.g. pt_sc = [0,.5,0], extent = 0.4 will lead to (for y-direction) the
        path going from [0,.1,0]->[0,.9,0]
    plot_stencil: bool
        Determines whether or not the visualization of the stencil is to be shown,
        can be useful for checking that it looks as expected.
    Returns:
    --------
    new_path_kpts: ndarray
        numpy array of the new path dictated by the stencil. This is returned
        in a format immediately usable by GPAW through the kpts keyword.
    """
    # XYZ Directions
    zmax = pt_sc + np.array([0, 0, 0.5])
    zmin = pt_sc - np.array([0, 0, 0.5])
    ymax = pt_sc + np.array([0, 0.5, 0])
    ymin = pt_sc - np.array([0, 0.5, 0])
    xmax = pt_sc + np.array([0.5, 0, 0])
    xmin = pt_sc - np.array([0.5, 0, 0])

    # Corner points
    c1 = np.array([-0.5, 0.5, 0.5])
    c2 = np.array([-0.5, -0.5, 0.5])
    c3 = np.array([-0.5, 0.5, -0.5])
    c4 = np.array([-0.5, -0.5, -0.5])

    if stencil_type == "xyz":
        new_paths = [[zmin, zmax], [ymin, ymax], [xmin, xmax]]
    elif stencil_type == "xyz_corners" or stencil_type == "xyz+corners":
        new_paths = [
            [zmin, zmax],
            [ymin, ymax],
            [xmin, xmax],
            [pt_sc - c1, pt_sc + c1],
            [pt_sc - c2, pt_sc + c2],
            [pt_sc - c3, pt_sc + c3],
            [pt_sc - c4, pt_sc + c4],
        ]
    else:
        raise ValueError("Stencil type must be 'xyz', or 'xyz_corners'")

    full_new_path = []
    individual_paths = []
    path_indices = []

    if plot_stencil:
        font = {
            "family": "sans-serif",
            "sans-serif": "Comic Sans MS",
            "size": 14,
        }
        matplotlib.rc("font", **font)
        fig = plt.figure(random.randint(0, 100), figsize=(12, 6))
        ax = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122, projection="3d")

    s_idx = 0
    for new_path in new_paths:
        # Get cartesian distance b/w endpoints
        path_min_ct = kpoint_convert(path.cell, skpts_kc=new_path[0])
        path_max_ct = kpoint_convert(path.cell, skpts_kc=new_path[1])

        path_distance = np.linalg.norm(path_max_ct - path_min_ct, ord=2)

        # Determine number of points as dictated by the density
        npts = math.ceil(path_distance * density)
        if npts % 2 == 0:
            npts += 1

        # Interpolate path with pt_sc as the center.
        new_path_kpts = np.linspace(new_path[0], new_path[1], npts)

        assert (
            new_path_kpts.shape[0] % 2 == 1
        ), "New path should be of odd length."
        assert np.allclose(
            new_path_kpts[int(new_path_kpts.shape[0] / 2)], pt_sc
        ), "midpoint of this path should be = to the center points, pt_sc"

        if plot_stencil:
            # Scaled plot.
            ax.scatter(
                new_path_kpts[:, 0],
                new_path_kpts[:, 1],
                new_path_kpts[:, 2],
                s=20,
            )
            ax.scatter(pt_sc[0], pt_sc[1], pt_sc[2], s=500, marker="*")
            # Cartesian plot
            new_path_kpts_ct = kpoint_convert(path.cell, skpts_kc=new_path_kpts)
            ax2.scatter(
                new_path_kpts_ct[:, 0],
                new_path_kpts_ct[:, 1],
                new_path_kpts_ct[:, 2],
                s=20,
            )
            ax2.scatter(pt_ct[0], pt_ct[1], pt_ct[2], s=500, marker="*")

        # Append to the full kpt list
        full_new_path.extend([list(x) for x in new_path_kpts])
        path_indices.append([s_idx, s_idx + new_path_kpts.shape[0] - 1])
        s_idx += new_path_kpts.shape[0]

    if plot_stencil:
        ax.set_title("New refined bandpath (Scaled)")
        ax.set_xlabel("$X$")
        ax.set_ylabel("$Y$")
        ax.set_zlabel("$Z$")
        ax2.set_title("New refined bandpath (Cartesian)")
        ax2.set_xlabel("$X [Å^{-1}]$")
        ax2.set_ylabel("$Y [Å^{-1}]$")
        ax2.set_zlabel("$Z [Å^{-1}]$")
        plt.savefig("a.pdf")
        plt.show()

    return (full_new_path, path_indices)


def spin_texture_grid(kpt_center, npts=10, n_hat="z", extent=0.5, plot=False):
    """
    Creates the kpt grid on which the spin texture can be computed. This will
    be centered around the k-point defined by kpt_center.

    In the future n_hat can be an arbitrary 3-vector specifying the orientation
    of the plane to generate a 2d grid of kpts for the spin texture. Currently
    only cartesian directions are accepted as strings ('x', 'y', or 'z').


    For kpt_center = G, npts = 5, n_hat = z, you'd get something like this:

    (T-.5,T+.5) x   x   x   x   x (T+.5,T+.5)

                x   x   x   x   x

                x   x   G   x   x

                x   x   x   x   x

    (T-.5,T-.5) x   x   x   x   x (T+.5,T-.5)

    The orientation of this plane will be along n_hat, which for now only
    accepts cartesian directions. This really doesn't matter that much as
    most frequently people are looking at Rashba systems in 2D/Quasi-2D systems.




    !Note this will be generated from a numpy meshgrid! Plot/analyze accordingly
    when analyzing your DFT data.

    !Note you should turn kpoint symmetrization OFF for this calculation (as
    you should for the band structure calculation), else you will miss any such
    spin anisotropy induced by the effects of SOC. Also your results will be
    very annoying to plot.

    """

    if type(kpt_center) == list:
        kpt_center = np.array(kpt_center)

    assert kpt_center.shape == (1, 3) or kpt_center.shape == (3,), (
        "Input kpt should be a 1,3 " "numpy array (or list of len(3)"
    )

    if n_hat == "z":

        # Linspace along first direction
        kx_min = kpt_center[0] - 0.5
        kx_max = kpt_center[0] + 0.5
        kx = np.linspace(kx_min, kx_max, npts)

        # Linspace along second direction
        ky_min = kpt_center[1] - 0.5
        ky_max = kpt_center[1] + 0.5
        ky = np.linspace(ky_min, ky_max, npts)

        # Make meshgrid
        kxx, kyy = np.meshgrid(kx, ky)

        # Grab the unchanging point.
        kz = kpt_center[2]

        kpts = []
        for i in range(npts):
            for j in range(npts):
                kpts.append([kxx[i, j], kyy[i, j], kz])
        kpts = np.array(kpts)

    elif n_hat == "y":

        # Linspace along first direction
        kx_min = kpt_center[0] - 0.5
        kx_max = kpt_center[0] + 0.5
        kx = np.linspace(kx_min, kx_max, npts)

        # Linspace along second direction
        kz_min = kpt_center[2] - 0.5
        kz_max = kpt_center[2] + 0.5
        kz = np.linspace(kz_min, kz_max, npts)

        # Make meshgrid
        kxx, kzz = np.meshgrid(kx, kz)

        # Grab the unchanging point.
        ky = kpt_center[1]

        kpts = []
        for i in range(npts):
            for j in range(npts):
                kpts.append([kxx[i, j], ky, kzz[i, j]])
        kpts = np.array(kpts)

    elif n_hat == "x":
        ky_min = kpt_center[1] - 0.5
        ky_max = kpt_center[1] + 0.5
        ky = np.linspace(ky_min, ky_max, npts)

        kz_min = kpt_center[2] - 0.5
        kz_max = kpt_center[2] + 0.5
        kz = np.linspace(kz_min, kz_max, npts)

        # Make meshgrid
        kyy, kzz = np.meshgrid(ky, kz)

        # Grab the unchanging point.
        kx = kpt_center[0]

        kpts = []
        for i in range(npts):
            for j in range(npts):
                kpts.append([kx, kyy[i, j], kzz[i, j]])
        kpts = np.array(kpts)

    else:
        raise ValueError(
            "Arbitrary spin-texture plane orientation not yet"
            'implemented, use "x", "y", or default: "z"'
        )

    if plot:

        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection="3d")
        ax.scatter(kpts[:, 0], kpts[:, 1], kpts[:, 2])
        plt.show()

    return kpts
