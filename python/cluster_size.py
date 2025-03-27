#!/usr/bin/env python
# coding: utf-8

import math
import uproot
from pathlib import Path
import awkward as ak
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import tqdm
mpl.rcParams['font.size'] = 11

# REGIONS = ["Inclusive", "BarrelFlat", "BarrelTilt", "Endcap"]
REGIONS = ["BarrelFlat"]
LAYERS = [0, 1, 2, 3, 4, 5, 6]
MIN_PT = 0.6


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


def main():
    # title = "TenMuExtendedE_0_200 (p 0-200 GeV)"
    title = "DoubleMuPt1Extended ($p_{T}$ 0.5-1.5 GeV)"
    if "200" in title:
        fname = Path("/Users/alexandertuna/Downloads/cms/lst_playing/data/trackingNtuple.2025_03_15_03h02m56s.root")
    else:
        fname = Path("/Users/alexandertuna/Downloads/cms/lst_playing/data/trackingNtuple.2025_03_21_11h59m00s.root")
    data = Data(fname).data
    plotter = Plotter(data)
    plotter.plot(title, "cluster_size.pdf")



class Plotter:

    def __init__(self, data: ak.Array) -> None:
        self.data = data


    def plot(self, title: str, pdfname: str) -> None:
        self.title = title
        with PdfPages(pdfname) as pdf:
            self.plot_title(title, pdf)
            # self.plot_pt_eta_phi(pdf)
            # self.plot_cluster_size(pdf)
            # self.plot_cluster_size_cdf(pdf)
            self.plot_cluster_size_vs_rdphi(pdf)
            self.plot_cluster_size_vs_rdphi(pdf, cdf=True)
            self.plot_cluster_size_vs_cosphi(pdf)
            self.plot_cluster_size_vs_cosphi(pdf, cdf=True)
            #self.plot_cluster_size_vs_rdphi(pdf, cosphi=[0.3, 0.5])
            #self.plot_cluster_size(pdf, cosphi=[0.3, 0.5])
            #self.plot_pt_vs_cosphi(pdf)
            # self.plot_simhit_dphi(pdf)
            # self.plot_simtrk_vs_simhit(pdf)
            # self.plot_simhit_pt_and_p(pdf)
            # self.plot_simhit_cosphi(pdf)
            # self.plot_order_and_side(pdf)
            # self.plot_nsimhit(pdf)
            #### self.plot_pdgid(pdf)
            ##### self.plot_event_displays(pdf)


    def plot_title(self, title: str, pdf: PdfPages) -> None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, title, fontsize=20, ha="center")
        ax.axis("off")
        pdf.savefig()
        plt.close()


    def plot_pt_eta_phi(self, pdf: PdfPages) -> None:
        bins_pt = np.arange(0, 5, 0.05)
        bins_etaphi = [
            np.arange(-3, 3, 0.1),
            np.arange(-3.2, 3.25, 0.1)
        ]
        _mask = (self.data.ph2_simhit_pt > MIN_PT)
        for region in REGIONS:
            reg = region_name(region)
            mask = _mask & self.data[f"ph2_is{region}"]
            fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
            fig.subplots_adjust(right=0.95, wspace=0.5)
            # fig.subplots_adjust(wspace=0.5)
            for ax in axs:
                ax.tick_params(right=True, top=True)

            axs[0].hist(ak.flatten(self.data.ph2_simhit_pt[mask]), bins=bins_pt)
            axs[0].set_xlabel("Sim. hit $p_T$ [GeV]")
            axs[0].set_ylabel("Hits (ph2_*)")
            axs[0].set_title(f"{reg} hits, sim. track $p_T$ > {MIN_PT} GeV")

            _, _, _, im = axs[1].hist2d(ak.flatten(self.data.ph2_eta[mask]).to_numpy(),
                                        ak.flatten(self.data.ph2_phi[mask]).to_numpy(),
                                        bins=bins_etaphi,
                                        cmin=0.5,
                          )
            axs[1].set_xlabel("Hit eta")
            axs[1].set_ylabel("Hit phi")
            axs[1].set_title(f"{reg} hits, sim. track $p_T$ > {MIN_PT} GeV")
            cbar = fig.colorbar(im, ax=axs)
            cbar.set_label("Hits (ph2_*)")

            pdf.savefig()
            plt.close()

    def plot_cluster_size(self, pdf: PdfPages, cosphi=[-1, 1]) -> None:
        cosphi_min, cosphi_max = cosphi
        bins = np.arange(-0.5, 34.5, 1)
        for region in REGIONS:
            reg = region_name(region)
            _mask = \
                (self.data.ph2_simhit_pt > MIN_PT) & \
                (self.data.ph2_simhit_p > 0.5 * self.data.ph2_simtrk_p) & \
                (self.data.ph2_simhit_cosphi >= cosphi_min) & \
                (self.data.ph2_simhit_cosphi <= cosphi_max) & \
                self.data[f"ph2_is{region}"]
            for layer in LAYERS:
                mask = _mask & ((layer == 0) | (self.data.ph2_layer == layer))
                total = ak.sum(mask)
                lay = layer_name(layer)
                fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
                for ax in axs:
                    ax.hist(ak.flatten(self.data.ph2_clustSize[mask]), bins=bins)
                    ax.set_title(f"All hits, {reg}, {lay}")
                    ax.set_xlabel("Cluster size")
                    ax.set_ylabel("Hits (ph2_*)")
                    ax.tick_params(right=True, top=True)
                    if cosphi != [-1, 1]:
                        ax.text(0.5, 0.9, f"cos($\\phi$) in [{cosphi_min}, {cosphi_max}]", transform=ax.transAxes, ha="center")
                if total > 0:
                    axs[1].semilogy()
                pdf.savefig()
                plt.close()


    def plot_cluster_size_cdf(self, pdf: PdfPages) -> None:

        bins = np.arange(-0.5, 19.5, 1)
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
                lay = layer_name(layer)

                hist, bin_edges = np.histogram(ak.flatten(self.data.ph2_clustSize[mask]), bins=bins, density=(total > 0))
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

                cdf = np.cumsum(hist * np.diff(bin_edges))

                fig, axs = plt.subplots(ncols=2, figsize=(12, 4))
                fig.subplots_adjust(wspace=0.25)
                for ax in axs:
                    ax.plot(bin_centers, cdf, marker=".")
                    ax.set_xlabel("Cluster size")
                    ax.set_ylabel("CDF")
                    ax.set_title(f"{reg} hits, {lay},  sim. track > {MIN_PT} GeV")
                    ax.tick_params(right=True, top=True)
                    ax.grid()
                axs[1].set_ylim([0.99, 1.0])

                pdf.savefig()
                plt.close()
        


    def plot_cluster_size_vs_rdphi(self, pdf: PdfPages, cosphi=[-1, 1], cdf=False) -> None:
        bins = [
            np.arange(-0.001, 0.2, 0.005),
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


    def plot_pt_vs_cosphi(self, pdf: PdfPages) -> None:
        bins = [
            np.arange(0.2, 1.01, 0.01),
            np.arange(0.5, 2.0, 0.01),
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


    def plot_event_displays(self, pdf: PdfPages, num: int = 20) -> None:
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
        print("N wonky:", n_wonky)
        print("wonky:", wonky)
        print("where_wonky:", where_wonky)
        print("where_wonky:", where_wonky.type)
        print("TOI:", track_of_interest)
        print(ak.sum(mask, axis=-1))
        #print(ak.sum(mask, axis=-1).type)
        print(self.data.event[2:3])
        #print(mask)
        #print(mask.type)
        #print(mask[2].to_list())

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
        fig, ax = plt.subplots()
        ax.hist(ak.flatten(self.data.ph2_rt[self.data.ph2_isBarrelFlat]), bins=np.arange(120))
        ax.grid()
        pdf.savefig()
        plt.close()



class Data:

    def __init__(self, fname: Path) -> None:
        if not fname.exists():
            raise Exception("shit")        
        self.fname = fname
        self.data = self.get_array()
        self.decorate_array()


    def get_array(self) -> ak.Array:
        tree = uproot.open(f"{self.fname}:trackingNtuple/tree")
        print(f"Got TTree: {tree}")
        data = tree.arrays([
            'event',
            'trk_pt', 'trk_eta', 'trk_phi',
            'ph2_isBarrel', 'ph2_isLower', 'ph2_isUpper', 'ph2_isStack', 
            'ph2_order', 'ph2_ring', 'ph2_rod', 'ph2_detId', 
            'ph2_subdet', 'ph2_layer', 'ph2_side', 'ph2_module', 
            'ph2_moduleType', 'ph2_trkIdx', 'ph2_onTrk_x', 'ph2_onTrk_y', 
            'ph2_onTrk_z', 'ph2_onTrk_xx', 'ph2_onTrk_xy', 'ph2_onTrk_yy', 
            'ph2_onTrk_yz', 'ph2_onTrk_zz', 'ph2_onTrk_zx', 'ph2_tcandIdx', 
            'ph2_seeIdx', 'ph2_simHitIdx', 'ph2_simType', 'ph2_x', 'ph2_y', 
            'ph2_z', 'ph2_xx', 'ph2_xy', 'ph2_yy', 
            'ph2_yz', 'ph2_zz', 'ph2_zx', 'ph2_radL', 
            'ph2_bbxi', 'ph2_usedMask', 'ph2_clustSize',
            'simhit_x', 'simhit_y', 'simhit_z',
            'simhit_px', 'simhit_py', 'simhit_pz',
            'simhit_tof', 'simhit_particle', 'simhit_simTrkIdx', 
            'sim_event', 'sim_bunchCrossing', 'sim_pdgId',
            'sim_genPdgIds', 'sim_isFromBHadron', 
            'sim_px', 'sim_py', 'sim_pz', 
            'sim_pt', 'sim_eta', 'sim_phi',
            'sim_parentVtxIdx', 'sim_decayVtxIdx', 'sim_simHitIdx',
        ])
        return data
    

    def decorate_array(self) -> None:
        print("Decorating array")
        self.data["simhit_pt"] = np.sqrt(self.data.simhit_px**2 + self.data.simhit_py**2)
        self.data["simhit_p"] = np.sqrt(self.data.simhit_px**2 + self.data.simhit_py**2 + self.data.simhit_pz**2)
        self.data["simhit_rt"] = np.sqrt(self.data.simhit_x**2 + self.data.simhit_y**2)
        self.data["simhit_cosphi"] = ((self.data.simhit_x * self.data.simhit_px) + (self.data.simhit_y * self.data.simhit_py)) / (self.data.simhit_pt * self.data.simhit_rt)
        self.data["simhit_phi"] = np.atan2(self.data.simhit_y, self.data.simhit_x)
        self.data["sim_p"] = np.sqrt(self.data.sim_px**2 + self.data.sim_py**2 + self.data.sim_pz**2)
        self.data["ph2_eta"] = eta(self.data.ph2_x, self.data.ph2_y, self.data.ph2_z)
        self.data["ph2_phi"] = phi(self.data.ph2_x, self.data.ph2_y)
        # self.data["ph2_phi"] = np.atan2(self.data.ph2_y, self.data.ph2_x)
        self.data["ph2_rt"] = np.sqrt(self.data.ph2_x**2 + self.data.ph2_y**2)
        self.data["ph2_isBarrelFlat"] = (self.data.ph2_order == 0) & (self.data.ph2_side == 3)
        self.data["ph2_isBarrelTilt"] = (self.data.ph2_order == 0) & (self.data.ph2_side != 3)
        self.data["ph2_isEndcap"] = (self.data.ph2_order != 0)
        self.data["ph2_isInclusive"] = self.data.ph2_isBarrelFlat | self.data.ph2_isBarrelTilt | self.data.ph2_isEndcap
        self.data["simhit_simtrk_pt"] = self.data.sim_pt[self.data.simhit_simTrkIdx]
        self.data["simhit_simtrk_p"] = self.data.sim_p[self.data.simhit_simTrkIdx]
        self.data["ph2_nsimhit"] = ak.num(self.data.ph2_simHitIdx, axis=-1)
        self.data["ph2_simHitIdxFirst"] = ak.firsts(self.data.ph2_simHitIdx, axis=-1)
        self.data["ph2_simhit_p"]       = self.data.simhit_p[self.data.ph2_simHitIdxFirst]
        self.data["ph2_simhit_pt"]      = self.data.simhit_pt[self.data.ph2_simHitIdxFirst]
        self.data["ph2_simhit_rt"]      = self.data.simhit_rt[self.data.ph2_simHitIdxFirst]
        self.data["ph2_simhit_phi"]     = self.data.simhit_phi[self.data.ph2_simHitIdxFirst]
        self.data["ph2_simhit_cosphi"]  = self.data.simhit_cosphi[self.data.ph2_simHitIdxFirst]
        self.data["ph2_simhit_simTrkIdx"] = self.data.simhit_simTrkIdx[self.data.ph2_simHitIdxFirst]
        self.data["ph2_simhit_dphi"] = dphi(self.data.ph2_phi, self.data.ph2_simhit_phi)
        dne = np.float32(0)
        self.data["ph2_simhit_p"]      = ak.fill_none(self.data.ph2_simhit_p, dne)
        self.data["ph2_simhit_pt"]     = ak.fill_none(self.data.ph2_simhit_pt, dne)
        self.data["ph2_simhit_phi"]    = ak.fill_none(self.data.ph2_simhit_phi, dne)
        self.data["ph2_simhit_dphi"]   = ak.fill_none(self.data.ph2_simhit_dphi, dne)
        self.data["ph2_simhit_cosphi"] = ak.fill_none(self.data.ph2_simhit_cosphi, dne)
        self.data["ph2_simhit_simTrkIdx"] = ak.fill_none(self.data.ph2_simhit_simTrkIdx, -1)
        self.data["ph2_simhit_rdphi"]  = self.data["ph2_simhit_rt"] * self.data["ph2_simhit_dphi"]
        self.data["ph2_simtrk_p"]  = self.data.simhit_simtrk_p[self.data.ph2_simHitIdxFirst]
        self.data["ph2_simtrk_pt"] = self.data.simhit_simtrk_pt[self.data.ph2_simHitIdxFirst]
        self.data["ph2_simtrk_p"]  = ak.fill_none(self.data["ph2_simtrk_p"], dne)
        self.data["ph2_simtrk_pt"] = ak.fill_none(self.data["ph2_simtrk_pt"], dne)


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
