#!/usr/bin/env python
# coding: utf-8

import uproot
from pathlib import Path
import awkward as ak
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from enum import auto, Enum
import tqdm
mpl.rcParams['font.size'] = 11

SUMMARIZE = False
REFRESH = False
STEP_SIZE = "10 GB"
TNAME = "trackingNtuple/tree"

# REGIONS = ["Inclusive", "BarrelFlat", "BarrelTilt", "Endcap"]
REGIONS = ["BarrelFlat"]
LAYERS = [1, 2, 3, 4, 5, 6]
MODULE_TYPES = {
    1: [23, 24],
    2: [23, 24],
    3: [23, 24],
    4: [25],
    5: [25],
    6: [25],
}
IS_UPPERS = {
    23: [False],
    24: [True],
    25: [False, True],
}
MIN_PT = 3 # 10 # 3 # 0.6
PITCH_UM = 90
PITCH_CM = PITCH_UM / 1000.0 / 10.0


class Zaxis(Enum):
    NORMAL = 1
    ONE_MINUS = 2
    ZOOMED = 3


class ModuleType(Enum):
    # NB: `auto()` starts at 1
    # UNKNOWN = auto()
    PXB = auto()
    PXF = auto()
    IB1 = auto()
    IB2 = auto()
    OB1 = auto()
    OB2 = auto()
    W1A = auto()
    W2A = auto()
    W3A = auto()
    W1B = auto()
    W2B = auto()
    W3B = auto()
    W4 = auto()
    W5 = auto()
    W6 = auto()
    W7 = auto()
    Ph1PXB = auto()
    Ph1PXF = auto()
    Ph2PXB = auto()
    Ph2PXF = auto()
    Ph2PXB3D = auto()
    Ph2PXF3D = auto()
    Ph2PSP = auto()
    Ph2PSS = auto()
    Ph2SS = auto()


def main():
    # fname = Path("/Users/alexandertuna/Downloads/cms/lst_playing/data/trackingNtuple.2025_03_15_03h02m56s.root")
    # fname = Path("/Users/alexandertuna/Downloads/cms/lst_playing/data/trackingNtuple.2025_03_21_11h59m00s.10muon_0p5gev_1p5gev.root")
    # fname = Path("/Users/alexandertuna/Downloads/cms/lst_playing/data/trackingNtuple.2025_04_01_10h06m00s.root")
    # fname = Path("/Users/alexandertuna/Downloads/cms/lst_playing/data/trackingNtuple.2025_04_02_12h00m00s.1muon_0p7gev.root")
    # fname = Path("/Users/alexandertuna/Downloads/cms/lst_playing/data/trackingNtuple.2025_04_08_13h00m00s.1muon_1p5gev.root")
    # fname = Path("/Users/alexandertuna/Downloads/cms/lst_playing/data/trackingNtuple.2025_04_10_00h00m00s.10muon_0p5_5p0.root")
    # fname = Path("/Users/alexandertuna/Downloads/cms/lst_playing/data/trackingNtuple.2025_04_12_00h00m00s.10muon_0p5_5p0.root")
    # fname = Path("/ceph/users/atuna/CMSSW_15_1_0_pre2/src/ttbar/n10/trackingNtuple.root")
    # fname = Path("/ceph/users/atuna/CMSSW_15_1_0_pre2/src/ttbar/n100/trackingNtuple.2025_05_09_10h00m00s.ttbar_PU200.n100.root")
    # fname = Path("/ceph/users/atuna/CMSSW_15_1_0_pre2/src/ttbar/n1000/trackingNtuple.2025_05_09_00h00m00s.ttbar_PU200.n1000.root")
    # fname = Path("/ceph/users/atuna/CMSSW_15_1_0_pre2/src/muonGun/n1e4/trackingNtuple.2025_05_23_00h00m00s.10muon_3p0_3p1.root")
    # fname = Path("/ceph/users/atuna/CMSSW_15_1_0_pre2/src/muonGun/n1e4/trackingNtuple.2025_05_23_00h00m00s.10muon_10p0_11p0.root")
    # fname = Path("/ceph/users/atuna/CMSSW_15_1_0_pre2/src/muonGun/n1e5/trackingNtuple.2025_05_29_00h00m00s.10muon_3p0_3p5.root")
    fname = Path("/ceph/users/atuna/CMSSW_15_1_0_pre2/src/ttbar/n1e4/output/")
    # fname = Path("/ceph/users/atuna/CMSSW_15_1_0_pre2/src/qcd/trackingNtuple.2025_06_03_00h00m00s.qcd.root")
    title = get_title(fname)
    data = Data(fname).data
    plotter = Plotter(data)
    plotter.plot(title, "cluster_size.pdf")

    fname = {}
    fname["QCD"] = Path("/ceph/users/atuna/CMSSW_15_1_0_pre2/src/qcd/trackingNtuple.2025_06_03_00h00m00s.qcd.root")
    fname["ttbar PU200"] = Path("/ceph/users/atuna/CMSSW_15_1_0_pre2/src/ttbar/n1e4/output/")
    fname["muonGun"] = Path("/ceph/users/atuna/CMSSW_15_1_0_pre2/src/muonGun/n1e5/trackingNtuple.2025_05_29_00h00m00s.10muon_3p0_3p5.root")
    # data = {name: Data(fn).data for name, fn in fname.items()}
    # plotter = PlotterOverlaid(data)
    # plotter.plot(title, "cluster_size_overlaid.pdf")


def get_title(fname: Path) -> str:
    if "2025_04_12_00h00m00s.10muon_0p5_5p0" in fname.stem:
        return "10 muons, $p_{T}$ range [0.5, 5.0], 50k events"
    if "2025_05_23_00h00m00s.10muon_3p0_3p1" in fname.stem:
        return "10 muons, $p_{T}$ range [3.0, 3.1], 10k events"
    if "2025_05_29_00h00m00s.10muon_3p0_3p5" in fname.stem:
        return "10 muons, $p_{T}$ range [3.0, 3.5], 100k events"
    if "2025_05_23_00h00m00s.10muon_10p0_11p0" in fname.stem:
        return "10 muons, $p_{T}$ range [10, 11], 10k events"
    if "ttbar/n10/trackingNtuple.root" in str(fname):
        return "ttbar PU200, 10 events"
    if "2025_05_09_10h00m00s.ttbar_PU200.n100" in fname.stem:
        return "ttbar PU200, 100 events"
    if "2025_05_09_00h00m00s.ttbar_PU200.n1000" in fname.stem:
        return "ttbar PU200, 1000 events"
    if "ttbar/n1e4" in str(fname):
        return "ttbar PU200, 10k events"
    if "2025_06_03_00h00m00s.qcd" in str(fname):
        return "RelValQCD_Pt_1800_2400, 700 events"
    raise Exception(f"Unknown file name: {fname}")


def title_blurb(layer: int, mtype: int, is_upper: bool) -> str:
    lname = layer_name(layer)
    mname = module_name(mtype)
    uname = upper_name(mtype, is_upper)
    if uname:
        return f"{lname}, {mname} ({uname})"
    return f"{lname}, {mname}"


def region_name(region: str) -> str:
    if region == "Inclusive":
        return "inclusive"
    elif region == "BarrelFlat":
        return "barrel (flat)"
    elif region == "BarrelTilt":
        return "barrel (tilt)"
    elif region == "Endcap":
        return "endcap"
    raise Exception


def layer_name(layer: int) -> str:
    if layer == 0:
        return "all layers"
    return f"layer {layer}"

def module_name(mtype: int) -> str:
    if mtype == 23:
        return "PS pixel"
    if mtype == 24:
        return "PS strip"
    if mtype == 25:
        return "2S"
    raise Exception(f"Unknown module type: {mtype}")

def upper_name(mtype: int, is_upper: bool) -> str:
    if mtype in [23, 24]:
        return ""
    if mtype == 25:
        return "upper" if is_upper else "lower"
    raise Exception(f"Unknown module type: {mtype} and upper: {is_upper}")


class PlotterOverlaid:

    def __init__(self, data: dict[ak.Array]) -> None:
        self.data = data


    def plot(self, title: str, pdfname: str) -> None:
        self.title = title
        print(f"Writing plots at {pdfname} ...")
        with PdfPages(pdfname) as pdf:
            self.plot_cluster_size(pdf)
            self.plot_cluster_size_cdf(pdf)


    def plot_cluster_size(self, pdf: PdfPages) -> None:
        bins = np.arange(-0.5, 24.5, 1)
        color = {"muonGun": "black", "ttbar PU200": "red", "QCD": "blue"}
        for region in REGIONS:
            reg = region_name(region)

            presel = {}
            for name, data in self.data.items():
                presel[name] = \
                    (data["ph2_simhit_pt"] > MIN_PT) & \
                    (data["ph2_simhit_p"] > 0.5 * data["ph2_simtrk_p"]) & \
                    (data[f"ph2_is{region}"])

            for layer in LAYERS:
                print(f"Cluster size plots for {layer=}")
                for mtype in MODULE_TYPES[layer]:
                    for is_upper in IS_UPPERS[mtype]:
                        mask, total = {}, {}
                        for name, data in self.data.items():
                            mask[name] = presel[name] \
                                & ((layer == 0) | (data["ph2_layer"] == layer)) \
                                & (data["ph2_moduleType"] == mtype) \
                                & (data["ph2_isUpper"] == is_upper)
                            total[name] = ak.sum(mask[name])

                        blurb = title_blurb(layer, mtype, is_upper)
                        fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
                        for row, (name, data) in enumerate(self.data.items()):
                            for ax in axs:
                                ax.hist(ak.flatten(data["ph2_clustSize"][mask[name]]), bins=bins, density=True, color=color[name], histtype="step", linewidth=1.5)
                                ax.set_title(f"All hits, {reg}, {blurb}")
                                ax.set_xlabel("Cluster size")
                                ax.set_ylabel("Hits, normalized (ph2_*)")
                                ax.tick_params(right=True, top=True)
                                mean = ak.mean(data["ph2_clustSize"][mask[name]])
                                ax.text(0.63, 0.9 - 0.07*row, f"{name} mean = {mean:.2f}", transform=ax.transAxes, fontsize=8, color=color[name])
                            if total[name] > 0:
                                axs[1].semilogy()

                        fig.subplots_adjust(wspace=0.30, right=0.95, left=0.10, bottom=0.15)
                        pdf.savefig()
                        plt.close()


    def plot_cluster_size_cdf(self, pdf: PdfPages) -> None:
        bins = np.arange(-0.5, 15.5, 1)
        color = {"muonGun": "black", "ttbar PU200": "red", "QCD": "blue"}
        for region in REGIONS:
            reg = region_name(region)

            presel = {}
            for name, data in self.data.items():
                presel[name] = \
                    (data["ph2_simhit_pt"] > MIN_PT) & \
                    (data["ph2_simhit_p"] > 0.5 * data["ph2_simtrk_p"]) & \
                    (data[f"ph2_is{region}"])

            for layer in LAYERS:
                print(f"Cluster size plots for {layer=}")
                for mtype in MODULE_TYPES[layer]:
                    for is_upper in IS_UPPERS[mtype]:
                        mask, total = {}, {}
                        hist, cdf = {}, {}
                        for name, data in self.data.items():
                            mask[name] = presel[name] \
                                & ((layer == 0) | (data["ph2_layer"] == layer)) \
                                & (data["ph2_moduleType"] == mtype) \
                                & (data["ph2_isUpper"] == is_upper)
                            total[name] = ak.sum(mask[name])
                            hist[name], bin_edges = np.histogram(ak.flatten(data["ph2_clustSize"][mask[name]]), bins=bins, density=(total[name] > 0))
                            cdf[name] = np.cumsum(hist[name] * np.diff(bin_edges))
                            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

                        fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
                        blurb = title_blurb(layer, mtype, is_upper)
                        for row, (name, data) in enumerate(self.data.items()):
                            for ax in axs:
                                ax.plot(bin_centers, cdf[name], marker=".", color=color[name])
                                ax.set_xlabel("Cluster size")
                                ax.set_ylabel("CDF")
                                ax.set_title(f"{reg} hits, {blurb},  sim. track > {MIN_PT} GeV", fontsize=10)
                                ax.tick_params(right=True, top=True)
                                ax.grid()
                                ax.text(0.60, 0.5 - 0.07*row, f"{name}", transform=ax.transAxes, fontsize=16, color=color[name])
                        axs[1].set_ylim([0.98, 1.0])

                        fig.subplots_adjust(wspace=0.30, right=0.95, left=0.10, bottom=0.15)
                        pdf.savefig()
                        plt.close()



class Plotter:

    def __init__(self, data: ak.Array) -> None:
        self.data = data


    def plot(self, title: str, pdfname: str) -> None:
        self.title = title
        print(f"Writing plots at {pdfname} ...")
        with PdfPages(pdfname) as pdf:

            if SUMMARIZE:
                # self.plot_title("10 muons, pt range [0.5, 5.0], 50k events", pdf)
                self.plot_title(title, pdf)
                self.plot_title("CDF with z-axis range [0, 1]", pdf)
                self.plot_cluster_size_vs_pt(pdf, cdf=True, zaxis=Zaxis.NORMAL)
                self.plot_title("CDF with z-axis range [0.99, 1]", pdf)
                self.plot_cluster_size_vs_pt(pdf, cdf=True, zaxis=Zaxis.ZOOMED)
                self.plot_title("1-CDF with z-axis range [1e-4, 1]", pdf)
                self.plot_cluster_size_vs_pt(pdf, cdf=True, zaxis=Zaxis.ONE_MINUS)
            
            else:
                self.plot_title(title, pdf)
                # self.plot_pt_eta_phi(pdf)
                # self.plot_tof(pdf)
                # self.plot_tof_vs_cosphi(pdf)
                # self.plot_cluster_size(pdf)
                # self.plot_cluster_size_cdf(pdf)
                self.plot_cluster_size_grouped(pdf)
                # self.plot_cluster_size_vs_rdphi(pdf)
                # self.plot_cluster_size_vs_rdphi(pdf, cdf=True)
                # self.plot_cluster_size_vs_cosphi(pdf)
                # self.plot_cluster_size_vs_cosphi(pdf, cdf=True)
                # self.plot_cluster_size_vs_pt(pdf)

                # self.plot_cluster_size_vs_rdphi(pdf, cosphi=[0.3, 0.5])
                # # self.plot_cluster_size_vs_rdphi(pdf, cosphi=[0.2, 0.3])
                # # self.plot_cluster_size_vs_rdphi(pdf, cosphi=[0.3, 0.4])
                # # self.plot_cluster_size_vs_rdphi(pdf, cosphi=[0.4, 0.5])
                # # self.plot_cluster_size_vs_rdphi(pdf, cosphi=[0.5, 0.6])
                # # self.plot_cluster_size_vs_rdphi(pdf, cosphi=[0.6, 0.7])
                # # self.plot_cluster_size_vs_rdphi(pdf, cosphi=[0.7, 0.8])
                # self.plot_cluster_size(pdf, cosphi=[0.3, 0.5])
                # self.plot_pt_vs_cosphi(pdf)
                # self.plot_simhit_dphi(pdf)
                # self.plot_simtrk_vs_simhit(pdf)
                # self.plot_simhit_pt_and_p(pdf)
                # self.plot_simhit_cosphi(pdf)
                # self.plot_order_and_side(pdf)
                # self.plot_nsimhit(pdf)
                # ### self.plot_pdgid(pdf)
                # self.plot_event_displays(pdf)
                self.dump_event_info(pdf)


    def plot_title(self, title: str, pdf: PdfPages) -> None:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.text(0.5, 0.5, title, fontsize=16, ha="center")
        ax.axis("off")
        pdf.savefig()
        plt.close()


    def plot_pt_eta_phi(self, pdf: PdfPages) -> None:
        # bins_pt = np.arange(0, 6, 0.05)
        bins_pt = np.arange(0, 25, 0.05)
        bins_etaphi = [
            np.arange(-3, 3, 0.1),
            np.arange(-3.2, 3.25, 0.1)
        ]
        _mask = (self.data.ph2_simhit_pt > MIN_PT)
        for region in REGIONS:
            reg = region_name(region)
            mask = _mask & self.data[f"ph2_is{region}"]
            fig, axs = plt.subplots(ncols=3, figsize=(14, 4))
            fig.subplots_adjust(left=0.09, right=1.09, wspace=0.5)
            # fig.subplots_adjust(right=0.95, wspace=0.5)
            # fig.subplots_adjust(wspace=0.5)
            for ax in axs:
                ax.tick_params(right=True, top=True)

            axs[0].hist(ak.flatten(self.data.ph2_simhit_pt[mask]), bins=bins_pt)
            axs[0].set_xlabel("Sim. hit $p_T$ [GeV]")
            axs[0].set_ylabel("Hits (ph2_*)")
            axs[0].set_title(f"{reg} hits, sim. track $p_T$ > {MIN_PT} GeV")

            axs[1].hist(ak.flatten(self.data.ph2_simtrk_pt[mask]), bins=bins_pt)
            axs[1].set_xlabel("Sim. track $p_T$ [GeV]")
            axs[1].set_ylabel("Hits (ph2_*)")
            axs[1].set_title(f"{reg} hits, sim. track $p_T$ > {MIN_PT} GeV")

            _, _, _, im = axs[2].hist2d(ak.flatten(self.data.ph2_eta[mask]).to_numpy(),
                                        ak.flatten(self.data.ph2_phi[mask]).to_numpy(),
                                        bins=bins_etaphi,
                                        cmin=0.5,
                          )
            axs[2].set_xlabel("Hit eta")
            axs[2].set_ylabel("Hit phi")
            axs[2].set_title(f"{reg} hits, sim. track $p_T$ > {MIN_PT} GeV")
            cbar = fig.colorbar(im, ax=axs)
            cbar.set_label("Hits (ph2_*)")

            pdf.savefig()
            plt.close()


    def plot_tof(self, pdf: PdfPages) -> None:
        bins = np.arange(-0.5, 21.5, 0.1)
        for region in REGIONS:
            rname = region_name(region)
            _mask = self.data[f"ph2_is{region}"]
            for layer in LAYERS:
                for mtype in MODULE_TYPES[layer]:
                    for is_upper in IS_UPPERS[mtype]:
                        mask = _mask \
                            & ((layer == 0) | (self.data.ph2_layer == layer)) \
                            & (self.data["ph2_moduleType"] == mtype) \
                            & (self.data["ph2_isUpper"] == is_upper)
                        total = ak.sum(mask)
                        blurb = title_blurb(layer, mtype, is_upper)
                        fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
                        for ax in axs:
                            ax.hist(ak.flatten(self.data.ph2_simhit_tof[mask]), bins=bins)
                            ax.set_title(f"All hits, {rname}, {blurb}")
                            ax.set_xlabel("Time-of-flight (TOF) [ns]")
                            ax.set_ylabel("Hits (ph2_*)")
                            ax.tick_params(right=True, top=True)
                        if total > 0:
                            axs[1].semilogy()
                        pdf.savefig()
                        plt.close()


    def plot_tof_vs_cosphi(self, pdf: PdfPages) -> None:
        bins_cosphi = np.arange(-1, 1.01, 0.02)
        bins_tof = np.arange(-0.5, 21.5, 0.1)
        for region in REGIONS:
            reg = region_name(region)
            _mask = self.data[f"ph2_is{region}"]
            for layer in LAYERS:
                mask = _mask & ((layer == 0) | (self.data.ph2_layer == layer))
                total = ak.sum(mask)
                lay = layer_name(layer)
                fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
                for it, ax in enumerate(axs):
                    _, _, _, im = ax.hist2d(ak.flatten(self.data.ph2_simhit_cosphi[mask]).to_numpy(),
                                            ak.flatten(self.data.ph2_simhit_tof[mask]).to_numpy(),
                                            bins=[bins_cosphi, bins_tof],
                                            cmin=0.5,
                                            cmap="hot",
                                            norm=mpl.colors.LogNorm() if (total > 0 and it == 1)  else None,
                                            )
                    ax.set_title(f"All hits, {reg}, {lay}")
                    ax.set_xlabel("cos($\\phi$) of $p_{T, sim}$ and $x_{T, sim}$")
                    ax.set_ylabel("Time-of-flight (TOF) [ns]")
                    fig.colorbar(im, ax=ax, label="Hits (ph2_*)")
                    fig.subplots_adjust(wspace=0.30, right=0.95, left=0.10, bottom=0.15)
                    ax.tick_params(right=True, top=True)
                pdf.savefig()
                plt.close()



    def plot_cluster_size(self, pdf: PdfPages, cosphi=[-1, 1]) -> None:
        cosphi_min, cosphi_max = cosphi
        bins = np.arange(-0.5, 24.5, 1)
        for region in REGIONS:
            reg = region_name(region)
            _mask = \
                (self.data.ph2_simhit_pt > MIN_PT) & \
                (self.data.ph2_simhit_p > 0.5 * self.data.ph2_simtrk_p) & \
                (self.data.ph2_simhit_cosphi >= cosphi_min) & \
                (self.data.ph2_simhit_cosphi <= cosphi_max) & \
                self.data[f"ph2_is{region}"]
            for layer in LAYERS:
                for mtype in MODULE_TYPES[layer]:
                    for is_upper in IS_UPPERS[mtype]:
                        mask = _mask \
                            & ((layer == 0) | (self.data.ph2_layer == layer)) \
                            & (self.data["ph2_moduleType"] == mtype) \
                            & (self.data["ph2_isUpper"] == is_upper)
                        total = ak.sum(mask)
                        blurb = title_blurb(layer, mtype, is_upper)
                        fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
                        for ax in axs:
                            ax.hist(ak.flatten(self.data.ph2_clustSize[mask]), bins=bins)
                            ax.set_title(f"All hits, {reg}, {blurb}")
                            ax.set_xlabel("Cluster size")
                            ax.set_ylabel("Hits (ph2_*)")
                            ax.tick_params(right=True, top=True)
                            ax.text(0.7, 0.9, f"Mean = {ak.mean(self.data.ph2_clustSize[mask]):.2f}", transform=ax.transAxes, fontsize=8)
                            if cosphi != [-1, 1]:
                                ax.text(0.5, 0.9, f"cos($\\phi$) in [{cosphi_min}, {cosphi_max}]", transform=ax.transAxes, ha="center")
                        if total > 0:
                            axs[1].semilogy()
                        pdf.savefig()
                        plt.close()


    def plot_cluster_size_cdf(self, pdf: PdfPages) -> None:

        bins = np.arange(-0.5, 17.5, 1)
        mask_basic = \
            (self.data.ph2_simhit_pt > MIN_PT) & \
            (self.data.ph2_simhit_p > 0.5 * self.data.ph2_simtrk_p) & \
            (self.data.ph2_simhit_cosphi > 0.15)

        for region in REGIONS:

            reg = region_name(region)
            mask_region = mask_basic & self.data[f"ph2_is{region}"]

            for layer in LAYERS:
                for mtype in MODULE_TYPES[layer]:
                    for is_upper in IS_UPPERS[mtype]:

                        mask = mask_region \
                            & ((layer == 0) | (self.data.ph2_layer == layer)) \
                            & (self.data["ph2_moduleType"] == mtype) \
                            & (self.data["ph2_isUpper"] == is_upper)
                        total = ak.sum(mask)

                        hist, bin_edges = np.histogram(ak.flatten(self.data.ph2_clustSize[mask]), bins=bins, density=(total > 0))
                        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

                        cdf = np.cumsum(hist * np.diff(bin_edges))

                        blurb = title_blurb(layer, mtype, is_upper)
                        fig, axs = plt.subplots(ncols=2, figsize=(12, 4))
                        fig.subplots_adjust(wspace=0.25)
                        for ax in axs:
                            ax.plot(bin_centers, cdf, marker=".")
                            ax.set_xlabel("Cluster size")
                            ax.set_ylabel("CDF")
                            ax.set_title(f"{reg} hits, {blurb},  sim. track > {MIN_PT} GeV", fontsize=10)
                            ax.tick_params(right=True, top=True)
                            ax.grid()
                        axs[1].set_ylim([0.98, 1.0])

                        pdf.savefig()
                        plt.close()
        

    def plot_cluster_size_grouped(self, pdf: PdfPages) -> None:

        bins = np.arange(-0.5, 24.5, 1)

        for region in REGIONS:

            reg = region_name(region)

            mask_all = (self.data[f"ph2_is{region}"])
            mask_hipt = mask_all & (self.data["ph2_simhit_pt"] > MIN_PT) & (self.data["ph2_simhit_p"] > 0.5 * self.data["ph2_simtrk_p"])
            mask_lopt = mask_all & (self.data["ph2_simhit_pt"] < MIN_PT) & (self.data["ph2_simhit_p"] > 0.5 * self.data["ph2_simtrk_p"])
            mask_rest = mask_all & (self.data["ph2_simhit_p"] < 0.5 * self.data["ph2_simtrk_p"])
            total = ak.sum(mask_all)

            for layer in LAYERS:

                for mtype in MODULE_TYPES[layer]:

                    for is_upper in IS_UPPERS[mtype]:

                        here = ((layer == 0) | (self.data["ph2_layer"] == layer)) \
                            & (self.data["ph2_moduleType"] == mtype) \
                            & (self.data["ph2_isUpper"] == is_upper)
                        blurb = title_blurb(layer, mtype, is_upper)

                        color = {
                            "all": "black",
                            "hipt": "red",
                            "lopt": "blue",
                            "rest": "green",
                        }

                        mean = {
                            "all": ak.mean(self.data["ph2_clustSize"][here & mask_all]),
                            "hipt": ak.mean(self.data["ph2_clustSize"][here & mask_hipt]),
                            "lopt": ak.mean(self.data["ph2_clustSize"][here & mask_lopt]),
                            "rest": ak.mean(self.data["ph2_clustSize"][here & mask_rest]),
                        }

                        simhit_pt = "$p_T^\\text{sim hit}$"
                        simhit_p = "$p^\\text{sim hit}$"
                        simtrk_p = "$p^\\text{sim track}$"
                        label = {
                            "all": f"All clusters",
                            "hipt": f"{simhit_pt} > {MIN_PT} GeV, {simhit_p} > 0.5 * {simtrk_p}",
                            "lopt": f"{simhit_pt} < {MIN_PT} GeV, {simhit_p} > 0.5 * {simtrk_p}",
                            "rest": f"{simhit_p} < 0.5 * {simtrk_p}",
                        }

                        fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
                        for ax in axs:
                            args = {"bins": bins, "histtype": "step", "linewidth": 1.5}
                            ax.hist(ak.flatten(self.data["ph2_clustSize"][here & mask_all]), color=color["all"], **args)
                            ax.hist(ak.flatten(self.data["ph2_clustSize"][here & mask_hipt]), color=color["hipt"], **args)
                            ax.hist(ak.flatten(self.data["ph2_clustSize"][here & mask_lopt]), color=color["lopt"], **args)
                            ax.hist(ak.flatten(self.data["ph2_clustSize"][here & mask_rest]), color=color["rest"], **args)
                            ax.set_title(f"Hits, {reg}, {blurb}")
                            ax.set_xlabel("Cluster size")
                            ax.set_ylabel("Hits (ph2_*)")
                            ax.tick_params(right=True, top=True)
                            args = {"transform": ax.transAxes, "fontsize": 10}
                            xtext, ytext, dx, dy = 0.3, 0.9, 0.5, 0.06
                            ax.text(xtext, ytext - 0*dy, label["all"], color=color["all"], **args)
                            ax.text(xtext, ytext - 1*dy, label["hipt"], color=color["hipt"], **args)
                            ax.text(xtext, ytext - 2*dy, label["lopt"], color=color["lopt"], **args)
                            ax.text(xtext, ytext - 3*dy, label["rest"], color=color["rest"], **args)
                            ax.text(xtext+dx, ytext - 4*dy, "Mean:", color=color["all"], **args)
                            ax.text(xtext+dx, ytext - 5*dy, f'{mean["all"]:.2f}', color=color["all"], **args)
                            ax.text(xtext+dx, ytext - 6*dy, f'{mean["hipt"]:.2f}', color=color["hipt"], **args)
                            ax.text(xtext+dx, ytext - 7*dy, f'{mean["lopt"]:.2f}', color=color["lopt"], **args)
                            ax.text(xtext+dx, ytext - 8*dy, f'{mean["rest"]:.2f}', color=color["rest"], **args)
                        if total > 0:
                            axs[1].semilogy()

                        fig.subplots_adjust(wspace=0.30, right=0.95, left=0.10, bottom=0.15)
                        pdf.savefig()
                        plt.close()


    def plot_cluster_size_vs_rdphi(self, pdf: PdfPages, cosphi=[-1, 1], cdf=False) -> None:
        bins = [
            np.arange(-0.001, 0.10, 0.002),
            np.arange(-0.5, 17.5, 1),
        ]
        cosphi_min, cosphi_max = cosphi
        mask_basic = \
            (self.data.ph2_simhit_pt > MIN_PT) & \
            (self.data.ph2_simhit_p > 0.5 * self.data.ph2_simtrk_p) & \
            (self.data.ph2_simhit_cosphi >= cosphi_min) & \
            (self.data.ph2_simhit_cosphi <= cosphi_max) & \
            (self.data.ph2_simhit_cosphi > 0.15)

        for region in REGIONS:

            reg = region_name(region)
            mask_region = mask_basic & self.data[f"ph2_is{region}"]

            for layer in LAYERS:

                mask = mask_region & ((layer == 0) | (self.data.ph2_layer == layer))
                total = ak.sum(mask)

                if cdf:
                    counts, xedges, yedges = np.histogram2d(ak.flatten(self.data.ph2_simhit_rdphi[mask]).to_numpy(),
                                                            ak.flatten(self.data.ph2_clustSize[mask]).to_numpy(),
                                                            bins=bins)
                    cumsum = np.cumsum(counts, axis=1)
                    column_sums = counts.sum(axis=1)
                    column_sums[column_sums == 0] = 1
                    cdf_array = np.transpose(np.divide(
                        cumsum.T,
                        column_sums,
                    ))
                    vmin = 5e-4
                    fig, ax = plt.subplots(figsize=(8, 8))
                    mesh = ax.pcolormesh(
                        xedges,
                        yedges,
                        1 - cdf_array.T + vmin,
                        cmap="RdYlGn",
                        norm=mpl.colors.LogNorm(vmin=vmin) if total > 0 else None,
                    )
                    fig.colorbar(mesh, ax=ax, label="1 - per-column CDF")

                else:
                    fig, ax = plt.subplots(figsize=(8, 8))
                    _, _, _, im = ax.hist2d(ak.flatten(self.data.ph2_simhit_rdphi[mask]).to_numpy(),
                                            ak.flatten(self.data.ph2_clustSize[mask]).to_numpy(),
                                            norm=mpl.colors.LogNorm() if total > 0 else None,
                                            cmin=0.5,
                                            bins=bins,
                                            )
                    fig.colorbar(im, ax=ax, label="Hits (ph2_*)")

                # styling
                ax.set_xlabel("r * dphi(hit, sim. hit) [cm]")
                ax.set_ylabel("Cluster size")
                ax.set_title(f"{reg} hits, layer {layer or 'inclusive'}, with sim. track > {MIN_PT} GeV")
                if cosphi != [-1, 1]:
                    ax.text(0.5, 0.9, f"cos($\\phi$) in [{cosphi_min}, {cosphi_max}]", transform=ax.transAxes, ha="center")
                ax.tick_params(right=True, top=True)

                pdf.savefig()
                plt.close()


    def plot_cluster_size_vs_cosphi(self, pdf: PdfPages, cdf=False) -> None:
        bins = [
            np.arange(0.2, 1.01, 0.02),
            np.arange(-0.5, 17.5, 1),
        ]
        mask_basic = (self.data.ph2_simhit_pt > MIN_PT) & (self.data.ph2_simhit_p > 0.5 * self.data.ph2_simtrk_p) & (self.data.ph2_simhit_cosphi > 0.15)

        for region in REGIONS:

            reg = region_name(region)
            mask_region = mask_basic & self.data[f"ph2_is{region}"]

            for layer in LAYERS:

                mask = mask_region & ((layer == 0) | (self.data.ph2_layer == layer))
                total = ak.sum(mask)

                if cdf:
                    counts, xedges, yedges = np.histogram2d(ak.flatten(self.data.ph2_simhit_cosphi[mask]).to_numpy(),
                                                            ak.flatten(self.data.ph2_clustSize[mask]).to_numpy(),
                                                            bins=bins)
                    cumsum = np.cumsum(counts, axis=1)
                    column_sums = counts.sum(axis=1)
                    column_sums[column_sums == 0] = 1
                    cdf_array = np.transpose(np.divide(
                        cumsum.T,
                        column_sums,
                    ))
                    vmin = 5e-4
                    fig, ax = plt.subplots(figsize=(8, 8))
                    mesh = ax.pcolormesh(
                        xedges,
                        yedges,
                        1 - cdf_array.T + vmin,
                        cmap="RdYlGn",
                        norm=mpl.colors.LogNorm(vmin=vmin) if total > 0 else None,
                    )
                    fig.colorbar(mesh, ax=ax, label="1 - per-column CDF")

                else:
                    fig, ax = plt.subplots(figsize=(8, 8))
                    _, _, _, im = ax.hist2d(ak.flatten(self.data.ph2_simhit_cosphi[mask]).to_numpy(),
                                            ak.flatten(self.data.ph2_clustSize[mask]).to_numpy(),
                                            norm=mpl.colors.LogNorm() if total > 0 else None,
                                            cmin=0.5,
                                            bins=bins,
                                )
                    fig.colorbar(im, ax=ax, label="Hits (ph2_*)")

                # styling
                ax.tick_params(right=True, top=True)
                ax.set_title(f"{reg} hits, layer {layer or 'inclusive'}, with sim. track > {MIN_PT} GeV")
                ax.set_xlabel("cos($\\phi$) of $p_{T, sim}$ and $x_{T, sim}$")
                ax.set_ylabel("Cluster size")
                pdf.savefig()
                plt.close()


    def plot_cluster_size_vs_pt(self, pdf: PdfPages, cdf=False, zaxis=Zaxis.NORMAL) -> None:
        bins = [
            # np.arange(0.6, 5.1, 0.2),
            np.arange(0.6, 15.1, 0.2),
            np.arange(-0.5, 17.5, 1),
        ]
        if zaxis not in Zaxis:
            raise Exception("Invalid zaxis")

        mask_basic = (self.data.ph2_simhit_pt > MIN_PT) & (self.data.ph2_simhit_p > 0.5 * self.data.ph2_simtrk_p) & (self.data.ph2_simhit_cosphi > 0.15)

        for region in REGIONS:

            reg = region_name(region)
            mask_region = mask_basic & self.data[f"ph2_is{region}"]

            for layer in LAYERS:

                mask = mask_region & ((layer == 0) | (self.data.ph2_layer == layer))
                total = ak.sum(mask)

                if cdf:
                    counts, xedges, yedges = np.histogram2d(ak.flatten(self.data.ph2_simhit_pt[mask]).to_numpy(),
                                                            ak.flatten(self.data.ph2_clustSize[mask]).to_numpy(),
                                                            bins=bins,
                                                            )
                    cumsum = np.cumsum(counts, axis=1)
                    column_sums = counts.sum(axis=1)
                    column_sums[column_sums == 0] = 1
                    cdf_array = np.transpose(np.divide(
                        cumsum.T,
                        column_sums,
                    ))

                    # plotting options for different z-axes
                    if zaxis == Zaxis.NORMAL:
                        cmap = "RdYlGn"
                        norm = None
                        contents = cdf_array.T
                        label = "per-column CDF"
                    elif zaxis == Zaxis.ONE_MINUS:
                        cmap = "RdYlGn_r"
                        vmin = 5e-4
                        norm = mpl.colors.LogNorm(vmin=vmin) if total > 0 else None
                        contents = 1 - cdf_array.T + vmin
                        label = "1 - per-column CDF"
                    elif zaxis == Zaxis.ZOOMED:
                        cmap = "RdYlGn"
                        vmin = 0.99
                        contents = cdf_array.T
                        contents[contents < vmin] = vmin
                        norm = None
                        label = "per-column CDF"

                    # the plot
                    fig, ax = plt.subplots(figsize=(8, 8))
                    mesh = ax.pcolormesh(xedges, yedges, contents, cmap=cmap, norm=norm)
                    fig.colorbar(mesh, ax=ax, label=label)

                else:
                    fig, ax = plt.subplots(figsize=(8, 8))
                    _, _, _, im = ax.hist2d(ak.flatten(self.data.ph2_simhit_pt[mask]).to_numpy(),
                                            ak.flatten(self.data.ph2_clustSize[mask]).to_numpy(),
                                            norm=mpl.colors.LogNorm() if total > 0 else None,
                                            cmin=0.5,
                                            bins=bins,
                                )
                    fig.colorbar(im, ax=ax, label="Hits (ph2_*)")

                # styling
                ax.tick_params(right=True, top=True)
                ax.set_title(f"{reg} hits, layer {layer or 'inclusive'}, with sim. track > {MIN_PT} GeV")
                ax.set_xlabel("$p_{T, sim. hit}$ [GeV]")
                ax.set_ylabel("Cluster size")
                pdf.savefig()
                plt.close()


    def plot_pt_vs_cosphi(self, pdf: PdfPages) -> None:
        bins = [
            np.arange(0.2, 1.01, 0.01),
            # np.arange(0.5, 2.0, 0.01),
            np.arange(0.5, 20.0, 0.01),
        ]
        mask_basic = \
            (self.data.ph2_simhit_pt > MIN_PT) & \
            (self.data.ph2_simhit_p > 0.5 * self.data.ph2_simtrk_p) & \
            (self.data.ph2_simhit_cosphi > 0.15)

        for region in REGIONS:

            reg = region_name(region)
            mask_region = mask_basic & self.data[f"ph2_is{region}"]

            for layer in LAYERS:

                mask = mask_region & ((layer == 0) | (self.data.ph2_layer == layer))
                total = ak.sum(mask)

                fig, ax = plt.subplots(figsize=(8, 8))
                _, _, _, im = ax.hist2d(ak.flatten(self.data.ph2_simhit_cosphi[mask]).to_numpy(),
                                        ak.flatten(self.data.ph2_simhit_pt[mask]).to_numpy(),
                                        norm=mpl.colors.LogNorm() if total > 0 else None,
                                        cmin=0.5,
                                        bins=bins,
                              )
                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label("Hits (ph2_*)")
                ax.set_xlabel("cos($\\phi$) of $p_{T, sim}$ and $x_{T, sim}$")
                ax.set_ylabel("Sim. hit $p_T$ [GeV]")
                ax.set_title(f"{reg} hits, layer {layer or 'inclusive'}, with sim. track > {MIN_PT} GeV")
                ax.tick_params(right=True, top=True)

                pdf.savefig()
                plt.close()


    def plot_simtrk_vs_simhit(self, pdf: PdfPages) -> None:
        bins = np.arange(200)
        fig, axs = plt.subplots(ncols=2, figsize=(14, 6))
        fig.subplots_adjust(wspace=0.25)
        for ax in axs:
            ax.tick_params(right=True, top=True)            
        cbars = [None, None]
        mask = (self.data.ph2_simhit_pt > 0.6) & (self.data.ph2_simtrk_pt > 0.6)
        print(ak.sum(mask))

        _, _, _, im = axs[0].hist2d(ak.flatten(self.data.ph2_simtrk_p[mask]).to_numpy(),
                                    ak.flatten(self.data.ph2_simhit_p[mask]).to_numpy(),
                                    bins=[bins, bins],
                                    cmin=0.5,
                                    )
        cbars[0] = fig.colorbar(im, ax=axs[0])
        cbars[0].set_label("Hits (ph2_*)")
        axs[0].set_xlabel("Sim. track $p$ [GeV]")
        axs[0].set_ylabel("Sim. hit $p$ [GeV]")

        _, _, _, im = axs[1].hist2d(ak.flatten(self.data.ph2_simtrk_pt[mask]).to_numpy(),
                                    ak.flatten(self.data.ph2_simhit_pt[mask]).to_numpy(),
                                    bins=[bins, bins],
                                    cmin=0.5,
                                    )
        cbars[1] = fig.colorbar(im, ax=axs[1])
        cbars[1].set_label("Hits (ph2_*)")
        axs[1].set_xlabel("Sim. track $p_{T}$ [GeV]")
        axs[1].set_ylabel("Sim. hit $p_{T}$ [GeV]")

        pdf.savefig()
        plt.close()


    def plot_simhit_dphi(self, pdf: PdfPages) -> None:
        bins = np.arange(-0.1, 3.2, 0.05)
        _mask = self.data.ph2_simhit_pt > 0.6
        for region in REGIONS:
            reg = region_name(region)
            mask = _mask & self.data[f"ph2_is{region}"]
            fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
            for ax in axs:
                ax.hist(ak.flatten(self.data.ph2_simhit_dphi[mask]), bins=bins)
                ax.set_xlabel("dphi(hit, sim. hit) [rad]")
                ax.set_ylabel("Hits (ph2_*)")
                ax.set_title(f"{reg}, sim. track $p_T$ > 0.6 GeV")
                ax.tick_params(right=True, top=True)
            axs[1].semilogy()
            pdf.savefig()
            plt.close()


    def plot_simhit_pt_and_p(self, pdf: PdfPages) -> None:
        bins = np.arange(0, 5, 0.1)
        _mask = (self.data.ph2_simhit_pt > 0.6)
        for region in REGIONS:
            reg = region_name(region)
            mask = _mask & self.data[f"ph2_is{region}"]
            fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
            fig.subplots_adjust(wspace=0.25)
            axs[0].hist(ak.flatten(self.data.ph2_simhit_pt[mask]), bins=bins)
            axs[1].hist(ak.flatten(self.data.ph2_simhit_p[mask]), bins=bins)
            for ax in axs:
                ax.set_ylabel("Hits (ph2_*)")
                ax.tick_params(right=True, top=True)
            axs[0].set_xlabel("$p_{T}$ [GeV]")
            axs[1].set_xlabel("$p$ [GeV]")
            axs[0].set_title(f"{reg} hits with sim. track $p_T$ > 0.6 GeV")
            axs[1].set_title(f"{reg} hits with sim. track $p_T$ > 0.6 GeV")
            pdf.savefig()
            plt.close()


    def plot_simhit_cosphi(self, pdf: PdfPages) -> None:
        bins = np.arange(-1, 1.01, 0.02)
        _mask = self.data.ph2_simhit_pt > 0.6
        for region in REGIONS:
            reg = region_name(region)
            mask_ = _mask & self.data[f"ph2_is{region}"]
            for layer in LAYERS:
                mask = mask_ & ((layer == 0) | (self.data.ph2_layer == layer))
                total = ak.sum(mask)
                lay = layer_name(layer)
                fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
                for ax in axs:
                    ax.hist(ak.flatten(self.data.ph2_simhit_cosphi[mask]), bins=bins)
                    ax.set_xlabel("cos($\\phi$) of $p_{T, sim}$ and $x_{T, sim}$")
                    ax.set_ylabel("Hits (ph2_*)")
                    ax.set_title(f"{reg} hits, layer {lay}, sim. $p_T$ > 0.6 GeV")
                    ax.tick_params(right=True, top=True)
                if total > 0:
                    axs[1].semilogy()
                pdf.savefig()
                plt.close()



    def plot_order_and_side(self, pdf: PdfPages) -> None:
        fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
        fig.subplots_adjust(wspace=0.35)
        axs[0].hist(ak.flatten(self.data.ph2_order), edgecolor="black", bins=np.arange(-1.5, 3, 1))
        axs[1].hist(ak.flatten(self.data.ph2_side), edgecolor="black", bins=np.arange(-0.5, 5, 1))
        axs[0].set_xlabel("ph2_order")
        axs[1].set_xlabel("ph2_size")
        for ax in axs:
            ax.set_ylabel("Hits (ph2_*)")
            ax.tick_params(right=True, top=True)
        pdf.savefig()
        plt.close()


    def plot_nsimhit(self, pdf: PdfPages) -> None:
        bins = np.arange(-0.5, 9.5, 1)

        for region in REGIONS:

            reg = region_name(region)

            mask_1 = (self.data.ph2_nsimhit == 1) & (self.data.ph2_simtrk_pt > 0.6) & (self.data[f"ph2_is{region}"])
            mask_2 = (self.data.ph2_nsimhit == 2) & (self.data.ph2_simtrk_pt > 0.6) & (self.data[f"ph2_is{region}"])
            mask_3 = (self.data.ph2_nsimhit == 3) & (self.data.ph2_simtrk_pt > 0.6) & (self.data[f"ph2_is{region}"])
            mask_4 = (self.data.ph2_nsimhit >= 4) & (self.data.ph2_simtrk_pt > 0.6) & (self.data[f"ph2_is{region}"])

            fig, axs = plt.subplots(ncols=2, figsize=(10, 4))

            axs[0].hist(ak.flatten(self.data.ph2_nsimhit[mask_1]), bins=bins, label="1")
            axs[0].hist(ak.flatten(self.data.ph2_nsimhit[mask_2]), bins=bins, label="2")
            axs[0].hist(ak.flatten(self.data.ph2_nsimhit[mask_3]), bins=bins, label="3")
            axs[0].hist(ak.flatten(self.data.ph2_nsimhit[mask_4]), bins=bins, label="4+")
            axs[0].set_xlabel("Number of sim. hits, {reg}")
            axs[1].set_ylabel("Hits (ph2_*)")
            axs[0].set_title(f"{reg} hits with sim. track $p_T$ > 0.6 GeV")
            axs[0].semilogy()
            axs[0].set_ylim([0.9, None])
            axs[0].tick_params(right=True, top=True)

            axs[1].hist(ak.flatten(self.data.ph2_clustSize[mask_1]), bins=bins, linewidth=2, histtype="step", density=True, label="N(sim.) = 1")
            axs[1].hist(ak.flatten(self.data.ph2_clustSize[mask_2]), bins=bins, linewidth=2, histtype="step", density=True, label="2")
            axs[1].hist(ak.flatten(self.data.ph2_clustSize[mask_3]), bins=bins, linewidth=2, histtype="step", density=True, label="3")
            axs[1].hist(ak.flatten(self.data.ph2_clustSize[mask_4]), bins=bins, linewidth=2, histtype="step", density=True, label="4+")
            axs[1].set_xlabel("Cluster size")
            axs[1].set_ylabel("Hits (ph2_*) normalized to 1")
            axs[1].legend()
            axs[1].tick_params(right=True, top=True)

            pdf.savefig()
            plt.close()


    def plot_pdgid(self, pdf: PdfPages) -> None:
        bins = np.arange(-220, 220)
        fig, ax = plt.subplots()
        ax.hist(ak.flatten(self.data.sim_pdgId), bins=bins)
        ax.set_xlabel("Sim. PDG ID")
        ax.set_ylabel("Number of sim. particles")
        ax.tick_params(right=True, top=True)
        ax.semilogy()
        pdf.savefig()
        plt.close()


    def dump_event_info(self, pdf: PdfPages, num: int = 5) -> None:

        #
        # get events with hits in barrel (flat) layer 6
        #
        mask = \
            (self.data.ph2_isBarrelFlat) & \
            (self.data.ph2_simhit_pt > MIN_PT) & \
            (self.data.ph2_simhit_p > 0.5 * self.data.ph2_simtrk_p) & \
            (self.data.ph2_simhit_cosphi > 0.15) & \
            (self.data.ph2_layer == 6)
        n_hits_l6 = ak.sum(mask, axis=-1) # [2, 9, ...]
        events_of_interest = ak.where(n_hits_l6 > 0)[0]
        print("N(hits, L6):", n_hits_l6, n_hits_l6.type)
        print("Events of interest:", events_of_interest.type, events_of_interest)

        #
        # describe these events
        #
        for it, ev in enumerate(events_of_interest):
            break
            if it >= num:
                break
            ph2_n = len(self.data.ph2_x[ev])
            end = " "
            print(f"Event {ev}, {ph2_n=}")
            for it in range(ph2_n):
                if self.data.ph2_layer[ev][it] != 6:
                    continue
                print(f"Event {ev}", end=end)
                print(f"ph2 hit {it:02}", end=end)
                print(f"barrelFlat {int(self.data.ph2_isBarrelFlat[ev][it])}", end=end)
                print(f"layer {self.data.ph2_layer[ev][it]}", end=end)
                print(f"isUpper {self.data.ph2_isUpper[ev][it]}", end=end)
                print(f"rod {self.data.ph2_rod[ev][it]:02}", end=end)
                print(f"module {self.data.ph2_module[ev][it]:02}", end=end)
                print(f"r {self.data.ph2_rt[ev][it]:5.1f}", end=end)
                print(f"phi {self.data.ph2_phi[ev][it]:7.4f}", end=end)
                print(f"z {self.data.ph2_z[ev][it]:7.4f}", end=end)
                print(f"TOF {self.data.ph2_simhit_tof[ev][it]:7.4f}", end=end)
                print(f"cos(dphi) {self.data.ph2_simhit_cosphi[ev][it]:6.3f}", end=end)
                print(f"clustSize {self.data.ph2_clustSize[ev][it]}", end=end)
                print(f"r*dphi {self.data.ph2_simhit_rdphi[ev][it]:.4f}", end=end)
                print(f"simHitIdx {self.data.ph2_simHitIdx[ev][it]}", end=end)
                print(f"ph2_simHitIdxFirst {self.data.ph2_simHitIdxFirst[ev][it]}", end=end)
                print(f"simTrkIdx {self.data.ph2_simhit_simTrkIdx[ev][it]}", end=end)
                print(f"simhit_pt {self.data.ph2_simhit_pt[ev][it]:.1f}", end=end)
                print("")
            print("")



    def plot_event_displays(self, pdf: PdfPages, num: int = 5) -> None:
        cosphi_min, cosphi_max = 0.3, 0.5
        mask = \
            (self.data.ph2_isBarrelFlat) & \
            (self.data.ph2_simhit_pt > MIN_PT) & \
            (self.data.ph2_simhit_p > 0.5 * self.data.ph2_simtrk_p) & \
            (self.data.ph2_simhit_cosphi > 0.15) & \
            (self.data.ph2_simhit_cosphi >= cosphi_min) & \
            (self.data.ph2_simhit_cosphi <= cosphi_max) & \
            (self.data.ph2_layer == 6)

        track_of_interest = ak.firsts(self.data.ph2_simhit_simTrkIdx[mask], axis=-1)
        n_wonky = ak.sum(mask, axis=-1) # [2, 9, ...]
        wonky = n_wonky > 0
        where_wonky = ak.where(wonky)[0]
        print("N wonky:", n_wonky, n_wonky.type)
        print("N wonky:", n_wonky[n_wonky > 0], n_wonky[n_wonky > 0].type)
        print("wonky:", wonky)
        print("where_wonky:", where_wonky)
        print("where_wonky:", where_wonky.type)
        print("TOI:", track_of_interest)
        # print(ak.sum(mask, axis=-1))
        #print(ak.sum(mask, axis=-1).type)
        # print(self.data.event[2:3])
        #print(mask)
        #print(mask.type)
        #print(mask[2].to_list())

        # https://cms-tklayout.web.cern.ch/cms-tklayout/layouts/cmssw-models/ZA_OT800_IT704/layout.html
        # 249.438, 371.678, 522.700, 687.000, 860.000, 1083.000
        radii = [23, 36, 51, 68, 86, 108.5]

        for it, ev in enumerate(where_wonky):
            if it >= num:
                break
            print(ev)

            fig, axs = plt.subplots(ncols=2, figsize=(16, 6))
            circles = [plt.Circle(xy=(0, 0), radius=rad, edgecolor='gray', fill=False) for rad in radii]
            for circle in circles:
                axs[0].add_patch(circle)
            barrel = self.data.ph2_isBarrelFlat[ev]
            trackmask = self.data.ph2_simhit_simTrkIdx[ev] == track_of_interest[ev]

            # xy
            axs[0].scatter(self.data.ph2_x[ev][barrel & trackmask],
                           self.data.ph2_y[ev][barrel & trackmask],
                           c="blue",
                           )
            axs[0].scatter(self.data.ph2_x[ev][barrel & trackmask & mask[ev]],
                           self.data.ph2_y[ev][barrel & trackmask & mask[ev]],
                           c="red",
                           )

            axs[0].set_xlim([-120, 120])
            axs[0].set_ylim([-120, 120])
            axs[0].set_xlabel("x [cm]")
            axs[0].set_ylabel("y [cm]")
            axs[0].set_title(f"Event {ev}")

            # rz
            axs[1].scatter(self.data.ph2_z[ev][barrel & trackmask],
                           self.data.ph2_rt[ev][barrel & trackmask],
                           c="blue",
                           )
            axs[1].scatter(self.data.ph2_z[ev][barrel & trackmask & mask[ev]],
                          self.data.ph2_rt[ev][barrel & trackmask & mask[ev]],
                          c="red",
                          )
            axs[1].set_xlim([-120, 120])
            axs[1].set_ylim([0, 120])
            axs[1].set_xlabel("z [cm]")
            axs[1].set_ylabel("r [cm]")
            axs[1].set_title(f"Event {ev}")
            pdf.savefig()
            plt.close()



        ##############################
        # check: ph2/sim matching
        ##############################
        #print("*"*10)
        #print(self.data.event[2:3])
        #print(self.data.ph2_nsimhit[2].to_list())
        #print(self.data.ph2_simHitIdx[2].to_list())
        #print("*"*10)
        #print(self.data.simhit_simTrkIdx[2])
        #print(self.data.simhit_simTrkIdx[2][315])
        #print(self.data.simhit_simTrkIdx[2][347])
        # track = ph2_simhit_simTrkIdx
        #print("*"*10)
        #print(self.data.ph2_simhit_simTrkIdx[2].to_list())
        #print(self.data.ph2_simhit_simTrkIdx[2][mask[2]])
        #print(ak.firsts(self.data.ph2_simhit_simTrkIdx[0:3][mask[0:3]], axis=-1))
        #print("*"*10)
        #print(ak.max(self.data.ph2_nsimhit[2]))
        #print(self.data.ph2_nsimhit[2:3])
        #print(self.data.ph2_simHitIdx[2:3])
        #print("*"*10)


        ##############################
        # check: radii
        ##############################
        # fig, ax = plt.subplots()
        # ax.hist(ak.flatten(self.data.ph2_rt[self.data.ph2_isBarrelFlat]), bins=np.arange(120))
        # ax.grid()
        # pdf.savefig()
        # plt.close()



class Data:

    def __init__(self, fname: Path) -> None:
        if not fname.exists():
            raise Exception("shit")        

        self.fname = fname
        self.dname = fname.with_suffix(".dir")

        if REFRESH or not self.dname.exists():
            self.dname.mkdir(parents=True, exist_ok=True)
            self.root_to_parquet()

        print(f"Loading data from {self.dname} ...")
        self.data = ak.from_parquet(self.dname)

    def root_to_parquet(self) -> None:
        print(f"Converting {self.fname} to parquet ...")
        fields = [
            'event',
            'ph2_isUpper', 'ph2_order', 'ph2_rod',
            'ph2_layer', 'ph2_subdet', 'ph2_side',
            'ph2_module', 'ph2_moduleType', 'ph2_simHitIdx',
            'ph2_x', 'ph2_y', 'ph2_z', 'ph2_clustSize',
            'simhit_x', 'simhit_y', 'simhit_z',
            'simhit_px', 'simhit_py', 'simhit_pz',
            'simhit_tof', 'simhit_simTrkIdx', 'simhit_particle',
            'sim_px', 'sim_py', 'sim_pz', 'sim_pt',
        ]
        if self.fname.is_file():
            files = f"{self.fname}:{TNAME}"
        elif self.fname.is_dir():
            files = f"{self.fname.resolve()}/*.root:{TNAME}"
        else:
            raise Exception(f"Invalid file or directory: {self.fname}")

        iterator = uproot.iterate(files,
                                  expressions=fields,
                                  step_size=STEP_SIZE)
        for it, data in enumerate(iterator):
            self.root_to_parquet_step(data, it)

    def root_to_parquet_step(self, data: ak.Array, it: int) -> None:
        print(f"Processing step {it} ...")
        data = self.decorate_array(data)
        data = self.drop_unused(data)
        data = self.filter_entries(data)
        pname = self.dname / f"{it:04}.parquet"
        print(f"Saving {pname} ...")
        ak.to_parquet(data, pname)

        # if REFRESH or not self.pname.exists():
        #     self.data = self.get_array()
        #     self.data = self.decorate_array(self.data)
        #     self.drop_unused()
        #     self.filter_entries()
        #     print(f"Saving {self.pname} ...")
        #     ak.to_parquet(self.data, self.pname)
        # else:
        #     print(f"Loading {self.pname}")
        #     self.data = ak.from_parquet(self.pname)


    def get_array(self) -> ak.Array:
        print(f"Opening {self.fname} ...")
        tree = uproot.open(f"{self.fname}:trackingNtuple/tree")
        print("Loading branches ...")
        data = tree.arrays([
            'event',
            'ph2_isUpper', 'ph2_order', 'ph2_rod',
            'ph2_layer', 'ph2_subdet', 'ph2_side',
            'ph2_module', 'ph2_simHitIdx',
            'ph2_x', 'ph2_y', 'ph2_z', 'ph2_clustSize',
            'simhit_x', 'simhit_y', 'simhit_z',
            'simhit_px', 'simhit_py', 'simhit_pz',
            'simhit_tof', 'simhit_simTrkIdx', 'simhit_particle',
            'sim_px', 'sim_py', 'sim_pz', 'sim_pt',
        ])
        return data
    

    def decorate_array(self, data: ak.Array) -> ak.Array:
        print("Decorating array ...")
        data["simhit_pt"] = np.sqrt(data.simhit_px**2 + data.simhit_py**2)
        data["simhit_p"] = np.sqrt(data.simhit_px**2 + data.simhit_py**2 + data.simhit_pz**2)
        data["simhit_rt"] = np.sqrt(data.simhit_x**2 + data.simhit_y**2)
        data["simhit_cosphi"] = ((data.simhit_x * data.simhit_px) + (data.simhit_y * data.simhit_py)) / (data.simhit_pt * data.simhit_rt)
        data["simhit_phi"] = np.atan2(data.simhit_y, data.simhit_x)
        data["sim_p"] = np.sqrt(data.sim_px**2 + data.sim_py**2 + data.sim_pz**2)
        data["ph2_eta"] = eta(data.ph2_x, data.ph2_y, data.ph2_z)
        data["ph2_phi"] = phi(data.ph2_x, data.ph2_y)
        data["ph2_rt"] = np.sqrt(data.ph2_x**2 + data.ph2_y**2)
        data["ph2_isBarrelFlat"] = (data.ph2_order == 0) & (data.ph2_side == 3)
        data["ph2_isBarrelTilt"] = (data.ph2_order == 0) & (data.ph2_side != 3)
        data["ph2_isEndcap"] = (data.ph2_order != 0)
        data["ph2_isInclusive"] = data.ph2_isBarrelFlat | data.ph2_isBarrelTilt | data.ph2_isEndcap
        data["simhit_simtrk_pt"] = data.sim_pt[data.simhit_simTrkIdx]
        data["simhit_simtrk_p"] = data.sim_p[data.simhit_simTrkIdx]
        data["ph2_nsimhit"] = ak.num(data.ph2_simHitIdx, axis=-1)
        data["ph2_simHitIdxFirst"] = ak.firsts(data.ph2_simHitIdx, axis=-1)
        data["ph2_simhit_p"]       = data.simhit_p[data.ph2_simHitIdxFirst]
        data["ph2_simhit_pt"]      = data.simhit_pt[data.ph2_simHitIdxFirst]
        data["ph2_simhit_rt"]      = data.simhit_rt[data.ph2_simHitIdxFirst]
        data["ph2_simhit_phi"]     = data.simhit_phi[data.ph2_simHitIdxFirst]
        data["ph2_simhit_tof"]     = data.simhit_tof[data.ph2_simHitIdxFirst]
        data["ph2_simhit_cosphi"]  = data.simhit_cosphi[data.ph2_simHitIdxFirst]
        data["ph2_simhit_simTrkIdx"] = data.simhit_simTrkIdx[data.ph2_simHitIdxFirst]
        data["ph2_simhit_dphi"] = dphi(data.ph2_phi, data.ph2_simhit_phi)
        dne = np.float32(0)
        data["ph2_simhit_p"]      = ak.fill_none(data.ph2_simhit_p, dne)
        data["ph2_simhit_pt"]     = ak.fill_none(data.ph2_simhit_pt, dne)
        data["ph2_simhit_phi"]    = ak.fill_none(data.ph2_simhit_phi, dne)
        data["ph2_simhit_tof"]    = ak.fill_none(data.ph2_simhit_tof, dne)
        data["ph2_simhit_dphi"]   = ak.fill_none(data.ph2_simhit_dphi, dne)
        data["ph2_simhit_cosphi"] = ak.fill_none(data.ph2_simhit_cosphi, dne)
        data["ph2_simhit_simTrkIdx"] = ak.fill_none(data.ph2_simhit_simTrkIdx, -1)
        data["ph2_simtrk_p"]  = data.simhit_simtrk_p[data.ph2_simHitIdxFirst]
        data["ph2_simtrk_pt"] = data.simhit_simtrk_pt[data.ph2_simHitIdxFirst]
        data["ph2_simtrk_p"]  = ak.fill_none(data["ph2_simtrk_p"], dne)
        data["ph2_simtrk_pt"] = ak.fill_none(data["ph2_simtrk_pt"], dne)
        data["ph2_simhit_rdphi"]  = data["ph2_simhit_rt"] * data["ph2_simhit_dphi"]
        data["ph2_simhit_rdphi"] = ak.fill_none(data.ph2_simhit_rdphi, -1)
        return data


    def drop_unused(self, data: ak.Array) -> ak.Array:
        print("Removing branches which arent necessary after decoration ...")
        fields = [field for field in data.fields if field.startswith("ph2_")]
        return data[fields]


    def filter_entries(self, data: ak.Array) -> ak.Array:
        print(f"Removing entries below pt = {MIN_PT} ...")
        mask = data["ph2_simhit_pt"] > MIN_PT
        return data[mask]


def eta(x, y, z):
    r_perp = np.sqrt(x**2 + y**2)
    theta = np.atan2(r_perp, z)
    return -np.log(np.tan(theta / 2.0))


def phi(x, y):
    return np.atan2(y, x)


def dphi(a, b):
    return np.abs(((a - b) + np.pi) % (2 * np.pi) - np.pi)



if __name__ == "__main__":
    main()
