from ase.spectrum.band_structure import BandStructure
from scipy.interpolate import interp1d
from ase.dft.kpoints import kpoint_convert
from rashbapy.utils import visualize_soc_bs, gpw_to_data_files, interp_path
from gpaw.spinorbit import soc_eigenstates
from rashbapy.stencil import stencil
import matplotlib.pyplot as plt
from gpaw import GPAW
import matplotlib
import numpy as np
import json, sys, warnings, os

FONT_SIZE = 12

class Rashba:
    def __init__(
        self,
        prefix,
        e_kn_file=None,
        s_knv_file=None,
        bs_file=None,
        cbm_idx=4,
        vbm_idx=3,
        interp=True,
        spin_texture=False,
    ):
        """
        Class for analyzing Rashba splitting in a band structure.
        Parameters
        ----------
        prefix : str
            Prefix of the data file names.
        e_kn_file : str, optional
            File name of the eigenvalues. The default is None.
        s_knv_file : str, optional
            File name of the spin projections. The default is None.
        bs_file : str, optional
            File name of json'ed band structure. The default is None.
        cbm_idx : int, optional
            Index of the conduction band max. The default is 4. Correct if the
            pp functions were used to post-process DFT data.
        vbm_idx : int, optional
            Index of the valence band max. The default is 3. Correct if the
            pp functions were used to post-process DFT data.
        Returns
        -------
        None.
        """

        if spin_texture:
            ...
        else:
            if e_kn_file is not None:
                self.e_kn = np.load(e_kn_file)
            else:
                self.e_kn = np.load(f"{prefix}_e_kn.npy")

            if s_knv_file is not None:
                self.s_knv = np.load(s_knv_file)
            else:
                self.s_knv = np.load(f"{prefix}_s_knv.npy")

        # Extract cartesian component of the spin projections.
        self.s_knz = self.s_knv[:, :, 2].T
        self.s_kny = self.s_knv[:, :, 1].T
        self.s_knx = self.s_knv[:, :, 0].T

        self.interp = interp

        if bs_file is not None:
            self.bs = BandStructure.read(bs_file)
        else:
            self.bs = BandStructure.read(f"{prefix}_bs.json")

        self.prefix = prefix

        # These are the default if e_kn, s_knv, and bs were made with pp.py
        self.n_vbm = vbm_idx
        self.n_cbm = cbm_idx

        self.E_vbm = None
        self.E_cbm = None

        self.vbm_k_idx = None
        self.cbm_k_idx = None

        self.vbm_kpt_sc = None
        self.cbm_kpt_sc = None

        self.vbm_kpt_ct = None
        self.cbm_kpt_ct = None

        self.vbm_nn, self.vbm_nn_ct = None, None
        self.vbm_nn_sc, self.vbm_nn_eig = None, None
        self.cbm_nn, self.cbm_nn_ct = None, None
        self.cbm_nn_sc, self.cbm_nn_eig = None, None
        self.rashba_vbm, self.rashba_cbm = None, None
        self.dE_vbm, self.dE_cbm = None, None
        self.dk_vbm, self.dk_cbm = None, None

        self.refined_path_npts = None

        # These are to contain the data for plot regeneration.
        # Format is as follows:
        # txt_x: list of floats containing x-location of txt.
        # txt_y: list of floats containing y-location of txt.
        # txt_v: list of strings containing the txt to be displayed.
        # line_y: list of numpy arrays
        self.plot_dict = {"txt_x": [], "txt_y": [], "txt_v": [], "line_x": []}

    def summarize(self):
        print("=====================================================")
        print(f"VBM band index: {self.n_vbm}\nCBM band index: {self.n_cbm}")
        print(f"E_VBM: {self.E_vbm}\nE_CBM: {self.E_cbm}")
        print(f"VBM kpt index: {self.vbm_k_idx}")
        print(f"CBM kpt index: {self.cbm_k_idx}")
        print(f"VBM kpt (scaled): \t {self.vbm_kpt_sc}")
        print(f"CBM kpt (scaled): \t {self.cbm_kpt_sc}")
        print(f"VBM kpt (cartesian):\t {self.vbm_kpt_ct}")
        print(f"CBM kpt (cartesian): \t{self.cbm_kpt_ct}")
        print("=====================================================")
        print(f"Nearest high sym point to VBM: {self.vbm_nn}")
        print(f"Cartesian coord: {self.vbm_nn_ct}")
        print(f"Scaled coord: \t {self.vbm_nn_sc}")
        print(f"Eigenvalue: {self.vbm_nn_eig}")
        print("=====================================================")
        print(f"Nearest high sym point to CBM: {self.cbm_nn}")
        print(f"Cartesian coord: {self.cbm_nn_ct}")
        print(f"Scaled coord: \t {self.cbm_nn_sc}")
        print(f"Eigenvalue: {self.cbm_nn_eig}")
        print("=====================================================")
        if self.rashba_vbm is not None and self.rashba_cbm is not None:
            print(f"VBM Rashba Coefficent: {self.rashba_vbm:.3f}")
            print(f"CBM Rashba Coefficent: {self.rashba_cbm:.3f}")
        else:
            print(f"VBM Rashba Coefficent: {self.rashba_vbm}")
            print(f"CBM Rashba Coefficent: {self.rashba_cbm}")
        if self.dE_vbm is not None and self.dE_cbm is not None:
            print(f"VBM |dE|: {self.dE_vbm:.3f} [eV]")
            print(f"VBM |dk|: {self.dk_vbm:.3f} [1/Å]")
            print(f"CBM |dE|: {self.dE_cbm:.3f} [eV]")
            print(f"CBM |dk|: {self.dk_cbm:.3f} [1/Å]")
        else:
            print(f"VBM |dE|: {self.dE_vbm} [eV]")
            print(f"VBM |dk|: {self.dk_vbm} [1/Å]")
            print(f"CBM |dE|: {self.dE_cbm} [eV]")
            print(f"CBM |dk|: {self.dk_cbm} [1/Å]")

        print("=====================================================")
        print(f"Number of points in refined bandpath: {self.refined_path_npts}")

    @staticmethod
    def post_process_gpaw(prefix, mode="unrefined"):
        """
        This part of the code serves as a postprocessing step of DFT (or in 
        general, any electronic structure codes' results.)

        1. Reads the GPAW gpw file.
        2. Computes SOC/Non-SOC Band Structure
        3. Trims to only consider relevant bands around CB, VB
        4. Saves:
            prefix_bs.json   -> BandStructure
            prefix_e_kn.npy  -> Band eigenvalues {kpt, band}
            prefix_s_knv.npy -> Spin projections {kpt, band, cartesian}

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
        calc = GPAW(f"bs_{prefix}.gpw")
        ef = calc.get_fermi_level()

        # Prefix files accordingly to normal or refined mode.
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

        if mode == "refined":
            np.save(prefix + "_e_kn_refined.npy", e_kn)
            np.save(prefix + "_s_knv_refined.npy", s_knv)
        else:
            np.save(prefix + "_e_kn.npy", e_kn)
            np.save(prefix + "_s_knv.npy", s_knv)

    def extract_segment_data(self, pmin, pmax, typ, mult=20, plot=False):
        """
        Find the VBM and CBM, rashba splitting, and nearest high sym point
        in a given segment of the band structure.

        Parameters:
        -----------
        pmin: int
            Index of the starting point of the segment
        pmax: int
            Index of the ending point of the segment
        typ: str
            Type of the band structure to be analyzed. Can be either
            'unrefined' or 'refined'
        mult: int
            Multiplication factor for the number of points in the interpolated
            band segment.
        plot: bool
            If True, return plotting info for the band structure segment.
            This should be collected for all segments and plotted together.

        Returns:
        --------

        """
        pmax += 1

        s_e_kn = np.load(f"{self.prefix}_e_kn_refined.npy")
        s_s_knv = np.load(f"{self.prefix}_s_knv_refined.npy")
        s_bs = BandStructure.read(f"{self.prefix}_bs_refined.json")
        x, X, labels = s_bs.path.get_linear_kpoint_axis()

        s_s_knz = s_s_knv[:, :, 2].T
        s_s_kny = s_s_knv[:, :, 1].T
        s_s_knx = s_s_knv[:, :, 0].T

        if typ == "vbm":

            # Find the band edge, and kpt info
            s_E_vbm, s_vbm_kpt_ct, txt_x, txt_y = interp_path(
                s_e_kn[pmin:pmax, self.n_vbm],
                s_bs.path.kpts[pmin:pmax, :],
                x[pmin:pmax],
                s_bs,
                self.n_vbm,
                mult=mult,
                plot=plot,
            )

            # Find info about the nearest high sym point
            x, X, labels = s_bs.path.get_linear_kpoint_axis()
            nn_info = self.find_nearest_high_sym(
                s_vbm_kpt_ct, mode="refined", rbs=s_bs
            )
            s_nn_name = nn_info[0]  # Name of the nearest high sym point
            s_nn_ct = nn_info[
                1
            ]  # Cartesian coord of the nearest high sym point
            s_nn_sc = nn_info[2]  # Scaled coord of the nearest high sym point
            s_nn_idx = nn_info[3]  # Index of the nearest high sym point
            s_nn_E = s_e_kn[
                s_nn_idx, self.n_vbm
            ]  # Energy of the nearest high sym point

            # Calculate the Rashba splitting
            dE = s_E_vbm - s_nn_E
            dk = np.linalg.norm(s_vbm_kpt_ct - s_nn_ct, ord=2)
            if abs(dk - 0.0) < 1e-2:
                alpha = 0.0
            else:
                alpha = abs(2 * dE / dk)
                print(alpha)

            if plot:
                plt.text(
                    txt_x, txt_y, f"{alpha:.2f}", fontsize=FONT_SIZE, color="k"
                )

            return (alpha, s_nn_name)

        elif typ == "cbm":
            # Find the band edge, and kpt info
            s_E_cbm, s_cbm_kpt_ct, txt_x, txt_y = interp_path(
                s_e_kn[pmin:pmax, self.n_cbm],
                s_bs.path.kpts[pmin:pmax, :],
                x[pmin:pmax],
                s_bs,
                self.n_cbm,
                mult=mult,
                plot=plot,
            )

            x, X, labels = s_bs.path.get_linear_kpoint_axis()
            nn_info = self.find_nearest_high_sym(
                s_cbm_kpt_ct, mode="refined", rbs=s_bs
            )
            s_nn_name = nn_info[0]
            s_nn_ct = nn_info[1]
            s_nn_sc = nn_info[2]
            s_nn_idx = nn_info[3]
            s_nn_E = s_e_kn[s_nn_idx, self.n_cbm]

            dE = s_nn_E - s_E_cbm
            dk = np.linalg.norm(s_cbm_kpt_ct - s_nn_ct, ord=2)
            if abs(dk - 0.0) < 1e-2:
                alpha = 0.0
            else:
                alpha = abs(2 * dE / dk)
                print(alpha)

            if plot:
                plt.text(
                    txt_x, txt_y, f"{alpha:.2f}", fontsize=FONT_SIZE, color="k"
                )

            return (alpha, s_nn_name)

        else:
            raise ValueError("typ must be 'cbm' or 'vbm'")
            sys.exit(1)

    def collect_segment_data(self, plot=False, fignum=404, show=False):
        """
        Collect data from each segment of the refined band structure path.

        Parameters:
        -----------
        plot: bool
            If True, plot the band structure and segment data.
        fignum: int
            Figure number for the plot.
        Returns:
        --------
        None
        """

        s_e_kn = np.load(f"{self.prefix}_e_kn_refined.npy")
        s_s_knv = np.load(f"{self.prefix}_s_knv_refined.npy")
        s_bs = BandStructure.read(f"{self.prefix}_bs_refined.json")

        s_s_knz = s_s_knv[:, :, 2].T
        s_s_kny = s_s_knv[:, :, 1].T
        s_s_knx = s_s_knv[:, :, 0].T

        vbm_paths = np.loadtxt(
            self.prefix + "_vbm_path_indices.csv", delimiter=","
        )
        cbm_paths = np.loadtxt(
            self.prefix + "_cbm_path_indices.csv", delimiter=","
        )

        # Collect data for the segment VBMs
        vbm_alphas = []
        vbm_nns = []

        # Plot the band structure and segment data
        if plot:
            plt.figure(fignum, figsize=(18, 9))
            x, X, labels = s_bs.path.get_linear_kpoint_axis()
            font = {
                "family": "sans-serif",
                "sans-serif": "Comic Sans MS",
                "size": FONT_SIZE,
            }
            matplotlib.rc("font", **font)

            # Clean up the Kpt labels which don't correspond to a high
            # symmetry point
            for i in range(len(X)):
                if "Kpt" in labels[i]:
                    labels[i] = "-"

            # Add a comma to the labels which contain discontinuous
            # jumps in the kpt path
            for i in range(len(X) - 1):
                if abs(X[i] - X[i + 1]) < 1e-5:
                    labels[i + 1] = labels[i] + "," + labels[i + 1]
            plt.xticks(X, labels)

            # Vertical lines at high symmetry points/break points
            # Vertical lines at high symmetry points/break points
            for i in range(len(X))[1:-1]:
                plt.axvline(X[i], color="black", linewidth=".1")

        for seg in range(vbm_paths.shape[0]):
            pmin = int(vbm_paths[seg, 0])
            pmax = int(vbm_paths[seg, 1])
            alpha, nn_name = self.extract_segment_data(
                pmin, pmax, "vbm", plot=plot
            )
            vbm_alphas.append(alpha)
            vbm_nns.append(nn_name)

        # Collect data for the segment CBMs
        cbm_alphas = []
        cbm_nns = []

        for seg in range(cbm_paths.shape[0]):
            pmin = int(cbm_paths[seg, 0])
            pmax = int(cbm_paths[seg, 1])
            alpha, nn_name = self.extract_segment_data(
                pmin, pmax, "cbm", plot=plot
            )
            cbm_alphas.append(alpha)
            cbm_nns.append(nn_name)

        if plot:
            # Plot the colored SOC band structure.
            main_plot = plt.scatter(
                np.tile(x, len(s_e_kn.T)),
                s_e_kn.T.reshape(-1),
                c=s_s_knz.reshape(-1),
                s=8,
                cmap="coolwarm",
                zorder=10,
            )
            plt.colorbar(main_plot)
            plt.ylabel("Energies [eV]")
            plt.axis([0, x[-1], -4.5, 4.5])
            plt.ylim(np.min(s_e_kn) - 0.1, np.max(s_e_kn) + 0.1)
            plt.tight_layout()
            if show:
                plt.show()

        with open(self.prefix + "_refined_out.txt", "w") as fil:
            max_alpha_v = max(vbm_alphas)
            max_alpha_c = max(cbm_alphas)
            print(cbm_alphas, vbm_alphas)
            alpha_idx_c = cbm_alphas.index(max_alpha_c)
            alpha_idx_v = vbm_alphas.index(max_alpha_v)

            alpha_nn_v = vbm_nns[alpha_idx_v]
            alpha_nn_c = cbm_nns[alpha_idx_c]

            fil.write(f"VBM alpha: {max_alpha_v}\n")
            fil.write(f"Max VBM splitting around: {alpha_nn_v}\n")
            fil.write(
                f"Max VBM splitting path idxs: {vbm_paths[alpha_idx_v]}\n"
            )
            fil.write(f"CBM alpha: {max_alpha_c}\n")
            fil.write(f"Max CBM splitting around: {alpha_nn_c}\n")
            fil.write(
                f"Max CBM splitting path idxs: {cbm_paths[alpha_idx_c]}\n"
            )

    def find_vbm_cbm(self, plot=False, mult=20):
        """
        Find the VBM and CBM of the unrefined band structure.

        Parameters:
        -----------
        plot: bool
            If True, plot the band structure.
        interp: bool
            If True, interpolate the band structure to a finer mesh for the
            computation of VBM and CBM. Note this method is STRONGLY preferred.
            Additionally, certian fields will not be filled out with this
            approach, as they're not necessary (vbm/cbm indices, scaled kpt,
            etc.)
        mult: int
            The number of points to interpolate to. Only used if interp = True.

        Returns:
        --------
        None, updates instance variables:
            self.E_vbm
            self.E_cbm
            self.vbm_kpt_ct
            self.cbm_kpt_ct
            And some other unnecessary ones if interp=False.
        """
        if self.interp:
            x, X, labels = self.bs.path.get_linear_kpoint_axis()

            paths = []
            # Extract the subsets of the Setyawan-Curtarolo path with care to
            # address the segments with discontinuous jumps.
            for i in range(len(X) - 1):

                # Only compute for full paths, e.g. for GMK,LH, this if would
                # hit G-M, M-K, L-H, and successfully avoid K,L.
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

                    # Append the band indices to the path list.
                    # This gives in the aboe example e.g. indices that pick out
                    # [G, M], [M,K], [L-H]
                    paths.append([idx1, idx2 + 1])

            # Extract VBM, CBM and corresponding kpt for the full S-C bandpath.
            E_vbm = -1e10
            E_cbm = 1e10
            for path in paths:
                this_vbm, this_vbm_k_ct, t_x, _ = interp_path(
                    self.e_kn[path[0] : path[1], self.n_vbm],
                    self.bs.path.kpts[path[0] : path[1], :],
                    x[path[0] : path[1]],
                    self.bs,
                    self.n_vbm,
                    mult=mult,
                    plot=plot,
                )
                if this_vbm > E_vbm:
                    E_vbm = this_vbm
                    vbm_k_ct = this_vbm_k_ct
                    v_x = t_x

                this_cbm, this_cbm_k_ct, t_x, _ = interp_path(
                    self.e_kn[path[0] : path[1], self.n_cbm],
                    self.bs.path.kpts[path[0] : path[1], :],
                    x[path[0] : path[1]],
                    self.bs,
                    self.n_cbm,
                    mult=mult,
                    plot=plot,
                )
                if this_cbm < E_cbm:
                    E_cbm = this_cbm
                    cbm_k_ct = this_cbm_k_ct
                    c_x = t_x

            self.E_vbm = E_vbm
            self.E_cbm = E_cbm
            self.vbm_kpt_ct = vbm_k_ct
            self.cbm_kpt_ct = cbm_k_ct
            self.v_x = v_x
            self.c_x = c_x

            if plot:
                # plt.figure(1)
                x, X, labels = self.bs.path.get_linear_kpoint_axis()
                plt.scatter(v_x, self.E_vbm, marker="x", label="VBM", s=200)
                plt.scatter(c_x, self.E_cbm, marker="x", label="CBM", s=200)
                plt.legend()
                visualize_soc_bs(
                    self.e_kn, self.s_knz, x, X, labels, fignum=100
                )

        else:
            # The non-interpolated means of finding the VBM/CBM is still
            # implemented, but MUCH less reliable. It is not recommended.
            self.E_vbm = np.max(self.e_kn[:, self.n_vbm])
            self.E_cbm = np.min(self.e_kn[:, self.n_cbm])

            self.vbm_k_idx = np.min(
                np.where(np.isclose(self.e_kn[:, self.n_vbm], self.E_vbm))
            )
            self.cbm_k_idx = np.min(
                np.where(np.isclose(self.e_kn[:, self.n_cbm], self.E_cbm))
            )
            # Extract the scaled kpt coordinate of band edges.
            self.vbm_kpt_sc = self.bs.path.kpts[self.vbm_k_idx]
            self.cbm_kpt_sc = self.bs.path.kpts[self.cbm_k_idx]

            # Extract the cartesian kpt coordinate of the band edges.

            # Method 1: ba method
            # self.vbm_kpt_ct = self.bs.path.cartesian_kpts()[self.vbm_k_idx]
            # self.cbm_kpt_ct = self.bs.path.cartesian_kpts()[self.cbm_k_idx]

            # Method 2: tpiba method, this is the one consistent with the codebase.
            self.vbm_kpt_ct = kpoint_convert(
                self.bs.path.cell, skpts_kc=self.vbm_kpt_sc
            )
            self.cbm_kpt_ct = kpoint_convert(
                self.bs.path.cell, skpts_kc=self.cbm_kpt_sc
            )

            if plot:
                plt.figure(1)
                x, X, labels = self.bs.path.get_linear_kpoint_axis()
                plt.scatter(
                    x[self.vbm_k_idx],
                    self.E_vbm,
                    marker="x",
                    label="VBM",
                    s=300,
                )
                plt.scatter(
                    x[self.cbm_k_idx],
                    self.E_cbm,
                    marker="x",
                    label="CBM",
                    s=300,
                )
                plt.legend()
                visualize_soc_bs(self.e_kn, self.s_knz, x, X, labels, fignum=1)

    def find_vbm_cbm_nn(self, plot=False):
        """
        This function takes the CBM, and VBM kpoints, and finds the nearest
        high symmetry point. E.g. for a conduction band:

        |\            /|
        | \          / |
        |  \----    /  | This example would find "M" as the CBM's nearest high
        |       \  /   | symmetry point.
        K        \/    M

        This function stores the relevant information in the Rashba object e.g.:
        for CBM and VBM: nn kpt, nn kpt cartesian coordinates, nn kpt scaled
        coordinates, and the nn kpt's index as well as corresponding eigenvalue.

        Parameters
        ----------
        plot: bool
            Passes whether the relevant information is to be plotted or not to
            corresponding helper functions.

        Returns
        -------
        None
        """
        if self.vbm_kpt_ct is None or self.cbm_kpt_ct is None:
            self.find_vbm_cbm(plot=plot)

        vbm_info = self.find_nearest_high_sym(self.vbm_kpt_ct)
        self.vbm_nn = vbm_info[0]
        self.vbm_nn_ct = vbm_info[1]
        self.vbm_nn_sc = vbm_info[2]
        self.vbm_nn_k_idx = vbm_info[3]
        self.vbm_nn_eig = self.e_kn[self.vbm_nn_k_idx, self.n_vbm]

        cbm_info = self.find_nearest_high_sym(self.cbm_kpt_ct)
        self.cbm_nn = cbm_info[0]
        self.cbm_nn_ct = cbm_info[1]
        self.cbm_nn_sc = cbm_info[2]
        self.cbm_nn_k_idx = cbm_info[3]
        self.cbm_nn_eig = self.e_kn[self.cbm_nn_k_idx, self.n_cbm]

    def refine_bandpath(
        self,
        density=30,
        plot_stencil=False,
        stencil_type="xyz",
        plot_path=False,
        debug=False,
    ):
        """
        Parameters
        ----------
        density: int/float
            Desired linear kpt density in points per reciprocal lattice length.
        plot_stencil: bool
            Whether or not to plot the output individual bandpath stencils
        stencil_type: str
            'xyz' or 'xyz+corners', dictating the type of stencil to be made.
        plot_path: bool
            Whether or not to plot the full set of resultant kpoints.

        Output:
        -------
        """
        if self.vbm_nn is None or self.cbm_nn is None:
            self.find_vbm_cbm_nn()

        # If the VBM and CBM are at the same point, then we can reduce the
        # computational cost by just having one stencil.
        if self.vbm_nn == self.cbm_nn:
            refined_path_vbm, vbm_path_indices = stencil(
                self.vbm_nn_sc,
                self.vbm_nn_ct,
                self.bs.path,
                density=density,
                extent=0.5,
                plot_stencil=plot_stencil,
                stencil_type=stencil_type,
            )
            refined_path = refined_path_vbm
            cbm_path_indices = vbm_path_indices
            self.refined_path_npts = len(refined_path)

        # If the VBM and CBM are at different points, unlucky.
        # We have to stencil around both points.
        else:
            # VBM
            refined_path_vbm, vbm_path_indices = stencil(
                self.vbm_nn_sc,
                self.vbm_nn_ct,
                self.bs.path,
                density=density,
                extent=0.5,
                plot_stencil=plot_stencil,
                stencil_type=stencil_type,
            )

            refined_path_cbm, cbm_path_indices = stencil(
                self.cbm_nn_sc,
                self.cbm_nn_ct,
                self.bs.path,
                density=density,
                extent=0.5,
                plot_stencil=plot_stencil,
                stencil_type=stencil_type,
            )

            refined_path_vbm.extend(refined_path_cbm)
            cbm_idx_shift = max(vbm_path_indices[-1]) + 1
            for i in range(len(cbm_path_indices)):
                for j in [0, 1]:
                    cbm_path_indices[i][j] += cbm_idx_shift

            refined_path = refined_path_vbm
            self.refined_path_npts = len(refined_path)

        if plot_path:
            fig = plt.figure(44, figsize=(12, 6))
            ax = fig.add_subplot(121, projection="3d")
            ax.scatter(
                refined_path[:, 0], refined_path[:, 1], refined_path[:, 2], s=20
            )
            plt.show()

        # np.save(self.prefix+"_refined_path.npy", refined_path)
        if debug:
            print(len(refined_path))
            print(vbm_path_indices)
            print(cbm_path_indices)

        refined_path = np.array(refined_path)
        vbm_path_indices = np.array(vbm_path_indices, dtype="int")
        cbm_path_indices = np.array(cbm_path_indices, dtype="int")

        np.savetxt(
            self.prefix + "_refined_path.csv", refined_path, delimiter=","
        )
        np.savetxt(
            self.prefix + "_vbm_path_indices.csv",
            vbm_path_indices,
            delimiter=",",
        )
        np.savetxt(
            self.prefix + "_cbm_path_indices.csv",
            cbm_path_indices,
            delimiter=",",
        )

    def compute_rashba_splitting(self, plot=False):
        if self.vbm_nn is None or self.cbm_nn is None:
            self.find_vbm_cbm_nn(plot=plot)

        vbm_delta_k = np.linalg.norm(self.vbm_kpt_ct - self.vbm_nn_ct, ord=2)
        cbm_delta_k = np.linalg.norm(self.cbm_kpt_ct - self.cbm_nn_ct, ord=2)

        vbm_delta_E = self.E_vbm - self.vbm_nn_eig
        cbm_delta_E = self.cbm_nn_eig - self.E_cbm

        self.dE_vbm = vbm_delta_E
        self.dE_cbm = cbm_delta_E

        self.dk_vbm = vbm_delta_k
        self.dk_cbm = cbm_delta_k

        # If there is no spin-splitting, avoid the division by zero with this
        if abs(vbm_delta_k) < 1e-5:
            self.rashba_vbm = 0
        else:
            self.rashba_vbm = 2 * vbm_delta_E / vbm_delta_k
        if abs(cbm_delta_k) < 1e-5:
            self.rashba_cbm = 0
        else:
            self.rashba_cbm = 2 * cbm_delta_E / cbm_delta_k

        if plot:
            if abs(self.E_vbm - self.E_cbm) < 0.4:
                plt.text(self.v_x, self.E_vbm - 0.1, f"{self.rashba_vbm:.2f}")
                plt.text(self.c_x, self.E_cbm + 0.1, f"{self.rashba_cbm:.2f}")
            else:
                plt.text(self.v_x, self.E_vbm + 0.1, f"{self.rashba_vbm:.2f}")
                plt.text(self.c_x, self.E_cbm - 0.1, f"{self.rashba_cbm:.2f}")

        with open(self.prefix + "_out.txt", "w") as fil:
            fil.write(f"VBM alpha: {self.rashba_vbm}\n")
            fil.write(f"Max VBM splitting around: {self.vbm_nn}\n")
            fil.write(f"CBM alpha: {self.rashba_cbm}\n")
            fil.write(f"Max CBM splitting around: {self.cbm_nn}\n")

    def find_nearest_high_sym(
        self, kpt, cartesian=True, mode="unrefined", rbs=None
    ):
        """
        Add docs later

        USER MUST BE RESPONSIBLE FOR INPUTTING KPT IN CORRECT COORDINATE FORMAT
        """

        # Clean up the code of this function a bit with this.
        if mode == "refined":
            bs = rbs
            path = rbs.path
        else:
            path = self.bs.path
            bs = self.bs

        if cartesian:
            sp_point_names = list(bs.path.special_points.keys())
            sp_point_values_sc = list(bs.path.special_points.values())
            # METHOD 1
            # Converts the special points from scaled coords to cartesian.
            sp_point_values = list(
                kpoint_convert(path.cell, skpts_kc=np.array(sp_point_values_sc))
            )
            sp_point_values_np = np.array(sp_point_values)

        else:
            print("NO longer supporting scaled coordinate NN detection.")
            sys.exit(1)
            sp_point_names = list(bs.path.special_points.keys())
            sp_point_values = list(bs.path.special_points.values())
            sp_point_values_np = np.array(sp_point_values)

        # Computes norm between target KPT and high sym points.
        norms = np.linalg.norm(sp_point_values_np - kpt, axis=1, ord=2)

        # Argmin of the norm, giving the nearest neighbor kpt value.
        amin = np.argmin(norms)

        debug = False
        if debug:
            unsorted_nearest = list(
                zip(list(norms), sp_point_names, list(sp_point_values))
            )

            sorted_nearest = sorted(unsorted_nearest, key=lambda x: x[0])
            print("Special point names")
            for i in sp_point_names:
                print(i)
            print("Norms")
            for i in norms:
                print(i)
            print("Unsorted, (Norm, Point Name)")
            for i in unsorted_nearest:
                print(i)
            print(f"Ranked closest points, (Norm, Point Name)")
            for i in sorted_nearest:
                print(i)
            print(sp_point_names[amin])

        # Prints out NN name, NN value (cartesian), NN value (scaled).
        # print(sp_point_names[amin])
        # print(sp_point_values[amin])
        # print(sp_point_values_sc[amin])
        all_norms = np.linalg.norm(
            path.kpts - sp_point_values_sc[amin], axis=1, ord=2
        )
        nn_k_idx = np.min(np.where(all_norms == 0))

        return [
            sp_point_names[amin],
            sp_point_values[amin],
            sp_point_values_sc[amin],
            nn_k_idx,
        ]

    @staticmethod
    def plot_spin_texture(prefix):
        """
        Add docs later. This function is responsible for plotting the spin 
        texture from the dft data in either:

        prefix_vbm_st.gpw + prefix_cbm_st.gpw
        OR
        prefix_st.gpw

        The first case is for the scenario where splittingin the VBM and CBM 
        occur at different kpoints, the second scneario is for where the 
        splitting occurs at the same kpoint.

        Spin projections are computed from a non-self-consistent SOC calculation
        and the spin projections are plotted with their component along the 
        z-direction as the color, and the x-y direction within the plane. The
        spin projections are normalzied such that the in-plane magnitude is 1.


        """

        # Splitting at same KPT
        vgpw = os.path.exists(f"{prefix}_vbm_st.gpw")
        cgpw = os.path.exists(f"{prefix}_cbm_st.gpw")

        # Splitting at same KPT
        bgpw = os.path.exists(f"{prefix}_st.gpw")

        if vgpw and cgpw:
                # VBM
                calc = GPAW(f'{prefix}_vbm_st.gpw')

                # Compute relevant number of electrons.
                n_elect = calc.get_number_of_electrons()
                n_elect = int(n_elect)
                soc_vbm = n_elect - 1
                soc_cbm = soc_vbm + 1

                # Do the calc
                soc = soc_eigenstates(calc)
                e_kn = soc.eigenvalues()
                s_knv = soc.spin_projections()

                # Extract vbm data
                vbm_kpts = calc.get_ibz_k_points()
                e_vbm = e_kn[:,soc_vbm]
                s_vbm = s_knv[:,soc_vbm, :]

                # CBM
                calc = GPAW(f'{prefix}_cbm_st.gpw')

                # Do the calc
                soc = soc_eigenstates(calc)
                e_kn = soc.eigenvalues()
                s_knv = soc.spin_projections()

                # Extract cbm data
                cbm_kpts = calc.get_ibz_k_points()
                e_cbm = e_kn[:,soc_cbm]
                s_cbm = s_knv[:,soc_cbm, :]

                assert len(cbm_kpts) == len(vbm_kpts) , "Kpoint meshes should match!"
                npts = np.sqrt(len(cbm_kpts))
                
        elif bgpw:
                calc = GPAW(f'{prefix}_st.gpw')

                soc = soc_eigenstates(calc)
                e_kn = soc.eigenvalues()
                s_knv = soc.spin_projections()
                kpts = calc.get_ibz_k_points()
                npts = np.sqrt(len(kpts))

                cbm_kpts = kpts
                vbm_kpts = kpts

                n_elect = calc.get_number_of_electrons()
                n_elect = int(n_elect)

                soc_vbm = n_elect - 1
                soc_cbm = soc_vbm + 1

                e_vbm = e_kn[:,soc_vbm]
                s_vbm = s_knv[:,soc_vbm, :]

                e_cbm = e_kn[:,soc_cbm]
                s_cbm = s_knv[:,soc_cbm, :]
        else:
                raise ValueError("The gpw files are missing in both cases")
        del calc

        # Normalize s.t. in plane arrow lengths are constant.
        # The in/out of plane projection is decided by the color of the arrows.
        N_vbm = np.sqrt(s_vbm[:,0]**2+s_vbm[:,1]**2)
        N_cbm = np.sqrt(s_cbm[:,0]**2+s_cbm[:,1]**2)

        # Only allow square k-meshes for now.
        assert npts.is_integer() , f"Should have a square input k-mesh along a "\
                "given plane!, nkpts currently = {len(kpts)}"
        npts = int(npts)

        # Reshape the energies 
        plottable_e_vbm = np.reshape(e_vbm, (npts, npts))
        plottable_e_cbm = np.reshape(e_cbm, (npts, npts))

        # Basic plot setup.
        plt.subplots(1,2, figsize = (12,8))
        # emap = 'Spectral'
        hue_neg, hue_pos = 250, 15
        emap = sns.color_palette(palette = "Spectral", as_cmap=True)

        # VBM setup
        plt.subplot(1,2,1)
        plt.xlabel("$k_x$")
        plt.ylabel("$k_y$")
        plt.title("VBM Spin Texture")

        # Heatmap
        plt.imshow(plottable_e_vbm,
                   interpolation = 'bicubic',
                   cmap = emap,
                   extent = (min(vbm_kpts[:,0]), max(vbm_kpts[:,0]),
                             min(vbm_kpts[:,1]), max(vbm_kpts[:,1]))
                   )

        # Spin texture
        plt.quiver(vbm_kpts[:,0],
                   vbm_kpts[:,1],
                   s_vbm[:,0]/N_vbm,
                   s_vbm[:,1]/N_vbm,
                   s_vbm[:,2],
                   cmap = 'binary')

        # CBM setup
        plt.subplot(1,2,2)
        plt.title("CBM Spin Texture")
        plt.xlabel("$k_x$")
        plt.ylabel("$k_y$")

        # Energy heatmap
        plt.imshow(plottable_e_cbm,
                   interpolation = 'bicubic',
                   cmap = emap,
                   extent = (min(cbm_kpts[:,0]), max(cbm_kpts[:,0]),
                             min(cbm_kpts[:,1]), max(cbm_kpts[:,1]))
                   )

        # Spin texture
        plt.quiver(cbm_kpts[:,0],
                   cbm_kpts[:,1],
                   s_cbm[:,0]/N_cbm,
                   s_cbm[:,1]/N_cbm,
                   s_cbm[:,2],
                   cmap = 'binary')

        plt.tight_layout()

        plt.savefig(f"{prefix}_texture.pdf", dpi = 800)
