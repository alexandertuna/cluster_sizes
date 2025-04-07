#!/usr/bin/env python
# coding: utf-8

import uproot
from pathlib import Path
import awkward as ak
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
mpl.rcParams['font.size'] = 11

# REGIONS = ["Inclusive", "BarrelFlat", "BarrelTilt", "Endcap"]
REGIONS = ["BarrelFlat"]
# LAYERS = [0, 1, 2, 3, 4, 5, 6]
LAYERS = [6]
MIN_PT = 0.6
PITCH_UM = 90
PITCH_CM = PITCH_UM / 1000.0 / 10.0


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
    fname = Path("/Users/alexandertuna/Downloads/cms/lst_playing/data/trackingNtuple.2025_04_02_12h00m00s.root")
    data = Data(fname).data
    plotter = Plotter(data)
    plotter.plot("cluster_size_displays.pdf")


class Plotter:

    def __init__(self, data: ak.Array) -> None:
        self.data = data


    def plot(self, pdfname: str) -> None:
        with PdfPages(pdfname) as pdf:
            self.plot_local_hits(pdf)


    def plot_local_hits(self, pdf: PdfPages, num: int = 10_000) -> None:

        #
        # get events with good sim. hits in barrel (flat) layer 6
        #
        mask = \
            (self.data.simhit_isBarrelFlat) & \
            (self.data.simhit_pt > MIN_PT) & \
            (self.data.simhit_p > 0.5 * self.data.simhit_simtrk_p) & \
            (self.data.simhit_cosphi > 0.15) & \
            (self.data.simhit_layer == 6)
        n_hits_l6 = ak.sum(mask, axis=-1) # [2, 9, ...]
        events_of_interest = ak.where(n_hits_l6 > 0)[0]
        print("Events of interest:", events_of_interest.type, events_of_interest)


        #
        # keep track of n(ph2 hits) for each sim hit
        #
        n_ph2_hits = []


        #
        # plot events in their local coordinates (attempt)
        #
        for it, ev in enumerate(events_of_interest):

            if it >= num:
                break

            # good sim hits on layer 6
            mask = \
                (self.data[ev].simhit_isBarrelFlat) & \
                (self.data[ev].simhit_layer == 6) & \
                (self.data[ev].simhit_tof < 10) & \
                (self.data[ev].simhit_pt > MIN_PT) & \
                (self.data[ev].simhit_p > 0.5 * self.data[ev].simhit_simtrk_p) & \
                (self.data[ev].simhit_cosphi > 0.15)
            simhits = np.flatnonzero(mask)

            # one event display per sim hit
            for simhit in np.unique(simhits):

                isUpper = "isUpper" if bool(self.data[ev].simhit_isUpper[simhit]) else "isLower"

                hits = np.flatnonzero(self.data[ev].ph2_simHitIdxFirst == simhit)
                n_ph2_hits.append(len(hits))
                if len(hits) == 0:
                    continue

                if it > 10:
                    if it % 200 == 0:
                        print(f"Skipping event {ev}, simhit {simhit}")
                    continue

                # get the reco hits
                xs = self.data.ph2_x[ev][hits]
                ys = self.data.ph2_y[ev][hits]
                clustSizes = self.data.ph2_clustSize[ev][hits]
                rdphis = self.data.ph2_simhit_rdphi[ev][hits]

                # get the sim hits
                simhit_x = self.data.simhit_x[ev][simhit]
                simhit_y = self.data.simhit_y[ev][simhit]

                # get the angle of the hits' surface
                all_xs = ak.concatenate([xs, simhit_x])
                all_ys = ak.concatenate([ys, simhit_y])
                angle = get_angle(all_xs, all_ys)

                # rotate the hits
                xps, yps = rotate(xs, ys, -angle)
                simhit_xp, simhit_yp = rotate(simhit_x, simhit_y, -angle)
                ypavg = np.mean(yps)

                # some drawing parameters
                delta = 0.06
                xmin, xmax = min(xps) - delta, max(xps) + delta
                ymin, ymax = min(yps) - delta, max(yps) + delta

                # the canvas
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.set_xlim([xmin, xmax])
                ax.set_ylim([ymin, ymax])

                # annotating with text
                ax.text(0.1, 0.86, "Sim. hit", transform=ax.transAxes, fontsize=20, color="red")
                ax.text(0.1, 0.80, "ph2 hits", transform=ax.transAxes, fontsize=20, color="black")
                ax.text(0.1, 0.74, "Strip edges", transform=ax.transAxes, fontsize=20, color="gray")
                ha, va = "center", "bottom"
                for txt in range(len(xps)):
                    ax.text(xps[txt], yps[txt] + 0.32*delta, "Size", fontsize=10, ha=ha, va=va)
                    ax.text(xps[txt], yps[txt] + 0.26*delta, clustSizes[txt], fontsize=10, ha=ha, va=va)
                    ax.text(xps[txt], yps[txt] - 0.26*delta, r"$r*d\phi$", fontsize=10, ha=ha, va=va)
                    ax.text(xps[txt], yps[txt] - 0.32*delta, f"{int(rdphis[txt] * 10 * 1e3)} um", fontsize=10, ha=ha, va=va)

                # draw hits
                ax.scatter(simhit_xp, simhit_yp, c="red", zorder=999)
                ax.scatter(xps, yps, c="black", zorder=99)
                ax.tick_params(right=True, top=True)
                ax.set_xlabel("local x [cm]")
                ax.set_ylabel("local y [cm]")
                ax.set_title(f"Event {ev}, layer 6, simhit={int(simhit)}, {isUpper}")

                # draw cluster sizes
                for txt in range(len(xps)):
                    corner_x = xps[txt] - 0.5 * clustSizes[txt] * PITCH_CM
                    corner_y = yps[txt] - 0.01
                    height = 2 * 0.01
                    width = clustSizes[txt] * PITCH_CM
                    rect = patches.Rectangle((corner_x, corner_y), width, height,
                                             edgecolor="gray", facecolor="lightgray")
                    ax.add_patch(rect)

                # draw strip boundaries
                xlines = get_x_edges(xps, clustSizes, xmin, xmax)
                for xline in xlines:
                    ax.plot([xline, xline], [ypavg - 0.01, ypavg + 0.01], color="gray", zorder=1)

                # save
                pdf.savefig()
                plt.close()


        # summary histogram
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.hist(n_ph2_hits, bins=np.arange(-0.5, 8.5, 1), color="blue", edgecolor="black")
        ax.set_xlabel("Number of ph2 hits")
        ax.set_ylabel("Sim. hits")
        pdf.savefig()
        plt.close()


def get_angle(xs, ys):
    if min(xs) != max(xs):
        slope, intercept = np.polyfit(xs, ys, 1)
        angle = np.arctan(slope)
    else:
        angle = np.pi / 2
    return angle


def rotate(xs, ys, angle):
    x = xs * np.cos(angle) - ys * np.sin(angle)
    y = xs * np.sin(angle) + ys * np.cos(angle)
    return x, y


def get_x_edges(xs, sizes, xmin, xmax):
    if len(sizes) == 0:
        return []

    # find the starting point
    xstart = None
    for x, clustSize in zip(xs, sizes):
        if clustSize % 2 == 0:
            xstart = x
    if xstart is None:
        for x, clustSize in zip(xs, sizes):
            xstart = x + 0.5 * PITCH_CM
    if xstart is None:
        raise Exception("What the fuck")

    # walk left and walk right
    xlines = []
    xstart_l, xstart_r = xstart, xstart
    while xstart_l > xmin:
        xlines.append(xstart_l)
        xstart_l -= PITCH_CM
    while xstart_r < xmax:
        xlines.append(xstart_r)
        xstart_r += PITCH_CM

    return sorted(list(set(xlines)))


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
            'simhit_isUpper', 'simhit_isLower', 'simhit_layer',
            'simhit_module', 'simhit_rod', 'simhit_order', 'simhit_side',
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
        self.data["simhit_isBarrelFlat"] = (self.data.simhit_order == 0) & (self.data.simhit_side == 3)
        self.data["simhit_simtrk_pt"] = self.data.sim_pt[self.data.simhit_simTrkIdx]
        self.data["simhit_simtrk_p"] = self.data.sim_p[self.data.simhit_simTrkIdx]
        self.data["ph2_nsimhit"] = ak.num(self.data.ph2_simHitIdx, axis=-1)
        self.data["ph2_simHitIdxFirst"] = ak.firsts(self.data.ph2_simHitIdx, axis=-1)
        self.data["ph2_simhit_p"]       = self.data.simhit_p[self.data.ph2_simHitIdxFirst]
        self.data["ph2_simhit_pt"]      = self.data.simhit_pt[self.data.ph2_simHitIdxFirst]
        self.data["ph2_simhit_rt"]      = self.data.simhit_rt[self.data.ph2_simHitIdxFirst]
        self.data["ph2_simhit_phi"]     = self.data.simhit_phi[self.data.ph2_simHitIdxFirst]
        self.data["ph2_simhit_tof"]     = self.data.simhit_tof[self.data.ph2_simHitIdxFirst]
        self.data["ph2_simhit_cosphi"]  = self.data.simhit_cosphi[self.data.ph2_simHitIdxFirst]
        self.data["ph2_simhit_simTrkIdx"] = self.data.simhit_simTrkIdx[self.data.ph2_simHitIdxFirst]
        self.data["ph2_simhit_dphi"] = dphi(self.data.ph2_phi, self.data.ph2_simhit_phi)
        dne = np.float32(0)
        self.data["ph2_simhit_p"]      = ak.fill_none(self.data.ph2_simhit_p, dne)
        self.data["ph2_simhit_pt"]     = ak.fill_none(self.data.ph2_simhit_pt, dne)
        self.data["ph2_simhit_phi"]    = ak.fill_none(self.data.ph2_simhit_phi, dne)
        self.data["ph2_simhit_tof"]    = ak.fill_none(self.data.ph2_simhit_tof, dne)
        self.data["ph2_simhit_dphi"]   = ak.fill_none(self.data.ph2_simhit_dphi, dne)
        self.data["ph2_simhit_cosphi"] = ak.fill_none(self.data.ph2_simhit_cosphi, dne)
        self.data["ph2_simhit_simTrkIdx"] = ak.fill_none(self.data.ph2_simhit_simTrkIdx, -1)
        self.data["ph2_simtrk_p"]  = self.data.simhit_simtrk_p[self.data.ph2_simHitIdxFirst]
        self.data["ph2_simtrk_pt"] = self.data.simhit_simtrk_pt[self.data.ph2_simHitIdxFirst]
        self.data["ph2_simtrk_p"]  = ak.fill_none(self.data["ph2_simtrk_p"], dne)
        self.data["ph2_simtrk_pt"] = ak.fill_none(self.data["ph2_simtrk_pt"], dne)
        self.data["ph2_simhit_rdphi"]  = self.data["ph2_simhit_rt"] * self.data["ph2_simhit_dphi"]
        self.data["ph2_simhit_rdphi"] = ak.fill_none(self.data.ph2_simhit_rdphi, -1)


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
