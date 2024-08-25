import matplotlib.pyplot as plt, matplotlib
import pickle, warnings, json
import numpy as np, os
from scipy.interpolate import interp1d
from ase.dft.kpoints import kpoint_convert
from ase.parallel import paropen
from ase.spectrum.band_structure import BandStructure
from gpaw.spinorbit import soc_eigenstates
from gpaw import GPAW


def read_bs_json(
    prefix,
    override_bs=None,
    override_e_kn=None,
    override_s_knv=None,
    show=False,
):
    """
    Reads band structure json file and returns the BandStructure object.
    Inputs:
        prefix (str): Prefix for the three files which must be in the form:
            prefix_bs.json, prefix_e_kn.npy, prefix_s_knv.npy
    Outputs:
        bs (BandStructure): The BandStructure object
    """

    if override_bs is not None:
        bs = BandStructure.read(override_bs)
    else:
        bs = BandStructure.read(prefix + "_bs.json")

    if override_e_kn is not None:
        e_kn = np.load(override_e_kn)
    else:
        e_kn = np.load(prefix + "_e_kn.npy")

    if override_s_knv is not None:
        s_knv = np.load(override_s_knv)
    else:
        s_knv = np.load(prefix + "_s_knv.npy")

    s_knx = s_knv[:, :, 0].T
    s_kny = s_knv[:, :, 1].T
    s_knz = s_knv[:, :, 2].T
    # print(type(bs.path))

    x, X, labels = bs.path.get_linear_kpoint_axis()

    visualize_soc_bs(e_kn, s_knz, x, X, labels, fignum=None, show=show)

    return None


def gpw_to_data_files(
    gpw_file_name, prefix, mode="unrefined", spin_texture=False
):
    """
    Reads the gpaw gpw file, computes soc/non-soc band structure, trims to
    the relevant bands, and saves prefix_bs.json, prefix_e_kn.npy, prefix_s_knv.npy

    Inputs:
        FIXME get rid of the gpw file name, this should be determined directly
        from the prefix, as well as whether or not the calc is for spin_texture.

        gpw_file_name (str): The gpw file name from the non-selfconsistent band
        structure calculation
        prefix (str): The desired prefix for the bs.json, e_kn.npy, and s_knv.npy
        mode (str
    Outputs:
        Nothing.
    """

    # Load the initial GPAW bs calculation (without SOC) and pull some relevant
    # information from this needed for later.
    calc = GPAW(gpw_file_name)
    ef = calc.get_fermi_level()

    # Prefix files accordingly to normal or refined mode.
    if not spin_texture:
        bs = calc.band_structure()
        if mode == "refined":
            bs.write(prefix + "_bs_refined.json")
        else:
            bs.write(prefix + "_bs.json")

    n_elect = calc.get_number_of_electrons()

    # FIXME This should be deleted at some point.
    nonsoc_bands = calc.get_number_of_bands()

    # Require spin-paired ground state calculation, and determine the soc_vbm index.
    assert (
        calc.get_number_of_spins() == 1
    ), "Collinear spin not implemented, spinless non-SOC calc only."
    if n_elect % 1 > 0.01:
        warnings.warn(
            "Non integer electron number, beware, n_elect is cast to an int."
        )
    n_elect = int(n_elect)
    if n_elect % 2 != 0:
        warnings.warn("Odd number of electrons in a spin paired calculation.")

    soc_vbm = n_elect - 1  # Really is n_elect/2 then *2 because SOC.

    # Compute the spin-orbit band structure.
    soc = soc_eigenstates(calc)
    e_kn = soc.eigenvalues()

    # Not necessary/doesn't change results, but often gives
    # better look to the plot with 0 within the gap.
    e_kn -= ef

    # Compute <i|\sigma|i> for \sigma_x,y,z
    s_knv = soc.spin_projections()

    # Remove bands >4 above and <4 below CBM and VBM respectively.
    bandana = 4
    # print(soc_vbm, bandana)
    e_kn = e_kn[:, soc_vbm - bandana + 1 : soc_vbm + bandana + 1]
    s_knv = s_knv[:, soc_vbm - bandana + 1 : soc_vbm + bandana + 1, :]

    if spin_texture:
        np.save("spin_texture_" + prefix + "_e_kn.npy", e_kn)
        np.save("spin_texture_" + prefix + "_s_knv.npy", s_knv)
    else:
        if mode == "refined":
            np.save(prefix + "_e_kn_refined.npy", e_kn)
            np.save(prefix + "_s_knv_refined.npy", s_knv)
        else:
            np.save(prefix + "_e_kn.npy", e_kn)
            np.save(prefix + "_s_knv.npy", s_knv)


def visualize_soc_bs(e_kn, s_nki, x, X, labels, fignum=None, show=False):
    """
    Visualize the SOC included band structure.

    Input:
        e_kn (np.array):    Energy eigenvalues indexed by E_kn = e_kn[k,n], where k denotes k-idx, n band index.
        s_kni (np.array):   Spin projections along the i'th cartesian direction (prepared
                                by user). s_kni[k,n] indexes the k'th kpt, n'th band, and
                                gives the i'th cartesian proejection.
        x, X, labels:       Obtain from ase.dft.kpoints.BandPath.get_linear_kpoint_axis(),
                                see this function for documentation.
        fignum (int):       Figure number to plot to. If None, will create a new figure.
        show (bool):        If True, will plt.show() at the end of the function.

    Output:
        SOC Band structure.
        Known bug: Interpolation is strange after a discontinuous jump in the bandpath.
    """
    if fignum is not None:
        ...
    else:
        fig = plt.figure(200, figsize=(18, 9))
    font = {"family": "sans-serif", "sans-serif": "Comic Sans MS", "size": 12}
    matplotlib.rc("font", **font)

    for i in range(len(X) - 1):
        if abs(X[i] - X[i + 1]) < 1e-5:
            labels[i + 1] = labels[i] + "," + labels[i + 1]
    plt.xticks(X, labels)

    for i in range(len(X))[1:-1]:
        plt.axvline(X[i], color="black", linewidth=".1")

    # Interpolate and give line plot for better visuals.
    interp = True
    interp_x = x.copy()
    if interp:
        paths = []
        for i in range(len(X) - 1):
            if abs(X[i] - X[i + 1]) > 1e-5:
                if i > 0:
                    if abs(X[i] - X[i - 1]) < 1e-5:
                        idx1 = np.where(np.isclose(x, X[i]))[0][1]
                        idx2 = np.where(np.isclose(x, X[i + 1]))[0][0]
                    else:
                        idx1 = np.where(np.isclose(x, X[i]))[0][0]
                        idx2 = np.where(np.isclose(x, X[i + 1]))[0][0]
                else:
                    idx1 = np.where(np.isclose(x, X[i]))[0][0]
                    idx2 = np.where(np.isclose(x, X[i + 1]))[0][0]
                paths.append([idx1, idx2 + 1])

        # Add a small value so there are no overlaps in the interpolation.
        for i in range(len(interp_x) - 1):
            if abs(interp_x[i] - interp_x[i + 1]) < 1e-5:
                interp_x[i + 1] += 0.005

    for n in range(e_kn.shape[1]):
        if interp:
            for path in paths:
                idxs = np.arange(len(x[path[0] : path[1]]))
                # if energies.shape[0] >= 3:
                # print(e_kn[path[0]:path[1], n].shape)
                curr_path = e_kn[path[0] : path[1], n]
                if len(curr_path) > 3:
                    e_idx = interp1d(idxs, curr_path, kind="quadratic")
                else:
                    e_idx = interp1d(idxs, curr_path, kind="linear")

                x_idx = interp1d(idxs, x[path[0] : path[1]], kind="linear")

                idxs_fine = np.linspace(idxs[0], idxs[-1], len(idxs) * 15)

                xs_plot = x_idx(idxs_fine)
                es_plot = e_idx(idxs_fine)
                plt.plot(xs_plot, es_plot, linewidth=0.4, color="gray")

        else:
            plt.plot(x, e_kn[:, n], color="gray", linewidth=0.1)

    things = plt.scatter(
        np.tile(x, len(e_kn.T)),
        e_kn.T.reshape(-1),
        c=s_nki.reshape(-1),
        s=40,
        cmap="coolwarm",
        edgecolor = 'k',
        zorder=10,
    )

    plt.colorbar(things)
    plt.ylabel("Energies [eV]")
    plt.axis([0, x[-1], -4.5, 4.5])
    plt.ylim(np.min(e_kn) - 0.1, np.max(e_kn) + 0.1)
    plt.tight_layout()

    if show:
        plt.show()


def interp_path(energies, kpts, x, bs, n, mult=5, plot=False):
    """
    Interpolate the band structure along the path.

    Parameters:
    ----------
    energies: np.array
        Energies along the slice of the bandpath to be interpolated
    kpts: np.array
        K-points along the slice of the bandpath to be interpolated
    bs: BandStructure
        BandStructure object from which the bandpath was obtained
    x: np.array
        linear_kpoint_axis along the slice of the bandpath to be interpolated
    mult (optional): int (default = 5)
        Number of points to interpolate between each pair of input points.
        E.g. multiplier = 5 will interpolate to 5x as many x-values as the input
    plot (optional): bool (default = False)
        If True, will plot the interpolated band segment

    Returns:
    -------

    """

    # Set up the indexing for later interp1d's, this is easier than trying to
    # interpolate the kpts and energies and x's w.r.t one another.
    index_list = np.arange(energies.shape[0])
    assert (
        energies.shape[0] == kpts.shape[0] == x.shape[0]
    ), "energies, kpts, and x must have the same length"

    # Quadratic intep for eigenvalues as we should be in the parabolic regime
    # at the band extrema
    if energies.shape[0] >= 3:
        E_dense = interp1d(index_list, energies, kind="quadratic")
    else:
        E_dense = interp1d(index_list, energies, kind="linear")

    # Linear interp for kpts because we are just interpolating between
    # points on a line
    kx_dense = interp1d(index_list, kpts[:, 0], kind="linear")
    ky_dense = interp1d(index_list, kpts[:, 1], kind="linear")
    kz_dense = interp1d(index_list, kpts[:, 2], kind="linear")

    # Linear interp for x because we are just interpolating between
    # points on a line
    x_dense = interp1d(index_list, x, kind="linear")

    dense_indices = np.linspace(
        np.min(index_list), np.max(index_list), mult * len(index_list)
    )

    ks = np.array(
        [
            kx_dense(dense_indices),
            ky_dense(dense_indices),
            kz_dense(dense_indices),
        ]
    ).T

    xs = x_dense(dense_indices)
    Es = E_dense(dense_indices)
    if n == 4:  # CBM Index
        E_edge = np.min(Es)
    elif n == 3:  # VBM Index
        E_edge = np.max(Es)
    else:
        raise ValueError("n must be 3 or 4")

    rel_idx = np.min(np.where(np.isclose(Es, E_edge)))

    # Plot the interpolated path in 3D.
    # fig = plt.figure(44, figsize = (12, 6))
    # ax = fig.add_subplot(121, projection='3d')
    # ax.scatter(kpts[:,0], kpts[:,1], kpts[:,2], s=8)
    # ax.scatter(ks[:,0], ks[:,1], ks[:,2], s=2, color = 'black')
    # ax.scatter(ks[rel_idx,0], ks[rel_idx,1], ks[rel_idx,2], s=20, color = 'red')
    # plt.show()

    edge_kpt_sc = ks[rel_idx, :]
    edge_kpt_ct = kpoint_convert(bs.path.cell, skpts_kc=edge_kpt_sc)

    if plot:
        debug = False
        if debug:
            plt.plot(xs, Es, label="interpolation")
            plt.scatter(x, energies, s=2, label="DFT Data")
            plt.scatter(
                xs[rel_idx], Es[rel_idx], marker="o", label="Interp Band Edge"
            )
            plt.legend()
        else:
            plt.plot(xs, Es, color="black")
            plt.scatter(
                xs[rel_idx], Es[rel_idx], marker="o", color="green", s=40
            )

    if not os.path.isdir("tmp"):
        os.system("mkdir tmp")

    file_idx = 0
    while os.path.exists(f"tmp/{file_idx}_tmp_x.txt"):
        file_idx += 1
    np.savetxt(f"tmp/{file_idx}_tmp_x.txt", xs)
    np.savetxt(f"tmp/{file_idx}_tmp_E.txt", Es)
    np.savetxt(
        f"tmp/{file_idx}_edge_marker.txt", np.array([xs[rel_idx], Es[rel_idx]])
    )

    txt_x = xs[rel_idx]
    if n == 4:  # CBM Index
        txt_y = Es[rel_idx] - 0.1
    elif n == 3:  # VBM Index
        txt_y = Es[rel_idx] + 0.1

    return E_edge, edge_kpt_ct, txt_x, txt_y
