"""
Inspired by PChang:
http://uaf-3.t2.ucsd.edu/~phchang/talks/PhilipChang20190330_ModuleStructure.pdf
"""

import os
import uproot # type: ignore
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap, BoundaryNorm
plt.rcParams.update({"font.size": 16})

FNAME = Path("/ceph/users/atuna/trackingNtuple_10mu_10k_pt_0p5_50_5cm_cube.root")
BNAME = os.path.basename(FNAME)
TNAME = Path("trackingNtuple/tree")
LIBRARY = "np"
BRANCHES = [
    "ph2_x",
    "ph2_y",
    "ph2_z",
    "ph2_layer",
    "ph2_side",
    "ph2_order",
    "ph2_rod",
    "ph2_ring",
    "ph2_module",
]


def main():
    num = 100_000
    data = Data(num=num).data
    print(data)
    plot = Plotter(data, scatter=True)
    plot.plot()


class Plotter:

    def __init__(self, data: pd.DataFrame, pdfname: Path = Path("tmp.pdf"), scatter: bool = False):
        self.data = data
        self.pdfname = pdfname
        self.scatter = scatter


    def plot(self) -> None:
        with PdfPages(self.pdfname) as pdf:
            #self.plot_rz_order(pdf)
            #self.plot_rz_side(pdf)
            #self.plot_rz_layer(pdf)
            #self.plot_rod_xy_1to4(pdf)
            #self.plot_rod_xy_all(pdf)
            #self.plot_rod_xy_number(pdf)
            #self.plot_rod_rz(pdf)
            #self.plot_module_rz(pdf)
            #self.plot_module_rz_number(pdf)
            self.plot_rod_rz_tilted(pdf)


    def plot_rz_order(self, pdf: PdfPages) -> None:
        colors = {
            0: "r",
            1: "b",
        }
        fig, ax = plt.subplots(figsize=(8, 6))
        for order in colors:
            subset = self.data[self.data["ph2_order"] == order]
            ax.scatter(subset["ph2_z"], subset["ph2_r"], s=1, c=colors[order], marker=".", edgecolors='none')
        ax.set_xlabel("z [cm]")
        ax.set_ylabel("r [cm]")
        ax.tick_params(right=True, top=True)
        ax.text(0.00, 1.02, BNAME, transform=ax.transAxes, fontsize=8)
        ax.text(0.03, 0.05, f"order=0", color=colors[0], transform=ax.transAxes, fontsize=20)
        ax.text(0.79, 0.05, f"order=1", color=colors[1], transform=ax.transAxes, fontsize=20)
        fig.subplots_adjust(bottom=0.12, left=0.18, right=0.96, top=0.95)
        pdf.savefig()
        plt.close()


    def plot_rz_side(self, pdf: PdfPages) -> None:
        colors = {
            1: "blue",
            2: "red",
            3: "black",
        }
        fig, ax = plt.subplots(figsize=(8, 6))
        for key in colors:
            subset = self.data[self.data["ph2_side"] == key]
            ax.scatter(subset["ph2_z"], subset["ph2_r"], s=1, c=colors[key], marker=".", edgecolors='none')
        ax.set_xlabel("z [cm]")
        ax.set_ylabel("r [cm]")
        ax.tick_params(right=True, top=True)
        ax.text(0.00, 1.02, BNAME, transform=ax.transAxes, fontsize=8)
        ax.text(0.03, 0.01, f"side=1", color=colors[1], transform=ax.transAxes, fontsize=16)
        ax.text(0.03, 0.06, f"side=2", color=colors[2], transform=ax.transAxes, fontsize=16)
        ax.text(0.03, 0.11, f"side=3", color=colors[3], transform=ax.transAxes, fontsize=16)
        fig.subplots_adjust(bottom=0.12, left=0.18, right=0.96, top=0.95)
        pdf.savefig()
        plt.close()


    def plot_rz_layer(self, pdf: PdfPages) -> None:
        name = "ph2_layer"
        colors = {
            1: "tab:blue",
            2: "tab:orange",
            3: "tab:green",
            4: "tab:red",
            5: "tab:purple",
            6: "tab:brown",
        }
        fig, ax = plt.subplots(figsize=(8, 6))
        for key in colors:
            subset = self.data[self.data[name] == key]
            ax.scatter(subset["ph2_z"], subset["ph2_r"], s=1, c=colors[key], marker=".", edgecolors='none')
        ax.set_xlabel("z [cm]")
        ax.set_ylabel("r [cm]")
        ax.tick_params(right=True, top=True)
        ax.text(0.00, 1.02, BNAME, transform=ax.transAxes, fontsize=8)
        ax.text(0.03, 0.01, f"layer=1", color=colors[1], transform=ax.transAxes, fontsize=16)
        ax.text(0.03, 0.06, f"layer=2", color=colors[2], transform=ax.transAxes, fontsize=16)
        ax.text(0.03, 0.11, f"layer=3", color=colors[3], transform=ax.transAxes, fontsize=16)
        ax.text(0.83, 0.01, f"layer=4", color=colors[4], transform=ax.transAxes, fontsize=16)
        ax.text(0.83, 0.06, f"layer=5", color=colors[5], transform=ax.transAxes, fontsize=16)
        ax.text(0.83, 0.11, f"layer=6", color=colors[6], transform=ax.transAxes, fontsize=16)
        fig.subplots_adjust(bottom=0.12, left=0.18, right=0.96, top=0.95)
        pdf.savefig()
        plt.close()

    def plot_rod_xy_1to4(self, pdf: PdfPages) -> None:
        name = "ph2_rod"
        side = 3
        colors = {
            1: "tab:blue",
            2: "tab:orange",
            3: "tab:green",
            4: "tab:red",
        }
        fig, ax = plt.subplots(figsize=(6, 6))
        for key in colors:
            subset = self.data[(self.data[name] == key) & (self.data["ph2_side"] == side)]
            ax.scatter(subset["ph2_x"], subset["ph2_y"], s=1, c=colors[key], marker=".", edgecolors='none')
        ax.set_xlabel("x [cm]")
        ax.set_ylabel("y [cm]")
        ax.set_xlim(-130, 130)
        ax.set_ylim(-130, 130)
        ax.tick_params(right=True, top=True)
        ax.text(0.00, 1.02, BNAME, transform=ax.transAxes, fontsize=8)
        ax.text(0.80, 1.02, f"ph2_side = {side}", transform=ax.transAxes, fontsize=8)
        ax.text(0.03, 0.01, f"rod=1", color=colors[1], transform=ax.transAxes, fontsize=16)
        ax.text(0.03, 0.06, f"rod=2", color=colors[2], transform=ax.transAxes, fontsize=16)
        ax.text(0.03, 0.11, f"rod=3", color=colors[3], transform=ax.transAxes, fontsize=16)
        ax.text(0.03, 0.16, f"rod=4", color=colors[4], transform=ax.transAxes, fontsize=16)
        fig.subplots_adjust(bottom=0.12, left=0.18, right=0.96, top=0.95)
        pdf.savefig()
        plt.close()

    def plot_rod_xy_all(self, pdf: PdfPages) -> None:
        name = "ph2_rod"
        side = 3
        colors = {
            1: "tab:blue",
            2: "tab:orange",
            3: "tab:green",
            4: "tab:red",
        }
        fig, ax = plt.subplots(figsize=(6, 6))
        for key in colors:
            subset = self.data[(self.data[name] % len(colors) == (key % len(colors))) & (self.data["ph2_side"] == side)]
            ax.scatter(subset["ph2_x"], subset["ph2_y"], s=1, c=colors[key], marker=".", edgecolors='none')
        ax.set_xlabel("x [cm]")
        ax.set_ylabel("y [cm]")
        ax.set_xlim(-130, 130)
        ax.set_ylim(-130, 130)
        ax.tick_params(right=True, top=True)
        ax.text(0.00, 1.02, BNAME, transform=ax.transAxes, fontsize=8)
        ax.text(0.85, 1.02, f"ph2_side = {side}", transform=ax.transAxes, fontsize=8)
        ax.text(0.03, 0.01, f"rod%4=1", color=colors[1], transform=ax.transAxes, fontsize=16)
        ax.text(0.03, 0.06, f"rod%4=2", color=colors[2], transform=ax.transAxes, fontsize=16)
        ax.text(0.03, 0.11, f"rod%4=3", color=colors[3], transform=ax.transAxes, fontsize=16)
        ax.text(0.03, 0.16, f"rod%4=0", color=colors[4], transform=ax.transAxes, fontsize=16)
        fig.subplots_adjust(bottom=0.12, left=0.18, right=0.96, top=0.95)
        pdf.savefig()
        plt.close()

    def plot_rod_xy_number(self, pdf: PdfPages) -> None:
        name = "ph2_rod"
        side = 3
        colors = {
            1: "tab:blue",
            2: "tab:orange",
            3: "tab:green",
            0: "tab:red",
        }
        fig, ax = plt.subplots(figsize=(6, 6))
        layer_min = self.data["ph2_layer"].min()
        layer_max = self.data["ph2_layer"].max()
        for layer in range(layer_min, layer_max+1):
            subset_layer = self.data[(self.data["ph2_layer"] == layer) & (self.data["ph2_side"] == side)]
            rod_min = subset_layer[name].min()
            rod_max = subset_layer[name].max()
            for rod in range(rod_min, rod_max+1):
                subset = subset_layer[subset_layer[name] == rod]
                ax.text(subset["ph2_x"].mean(), subset["ph2_y"].mean(), f"{rod}", fontsize=8, color=colors[rod % len(colors)], ha="center", va="center")
        ax.set_xlabel("x [cm]")
        ax.set_ylabel("y [cm]")
        ax.set_xlim(-130, 130)
        ax.set_ylim(-130, 130)
        ax.tick_params(right=True, top=True)
        ax.text(0.00, 1.02, BNAME, transform=ax.transAxes, fontsize=8)
        ax.text(0.85, 1.02, f"ph2_side = {side}", transform=ax.transAxes, fontsize=8)
        ax.text(0.03, 0.04, f"rod number", transform=ax.transAxes, fontsize=20)
        fig.subplots_adjust(bottom=0.12, left=0.18, right=0.96, top=0.95)
        pdf.savefig()
        plt.close()


    def plot_rod_rz(self, pdf: PdfPages) -> None:
        name = "ph2_rod"
        side = 3
        colors = {
            1: "tab:blue",
            2: "tab:orange",
        }
        fig, ax = plt.subplots(figsize=(8, 6))
        for key in colors:
            subset = self.data[(self.data[name] % len(colors) == (key % len(colors))) & (self.data["ph2_side"] == side)]
            ax.scatter(subset["ph2_z"], subset["ph2_r"], s=1, c=colors[key], marker=".", edgecolors='none')
        ax.set_xlabel("z [cm]")
        ax.set_ylabel("r [cm]")
        ax.set_xlim(-130, 130)
        ax.set_ylim(15, 120)
        ax.tick_params(right=True, top=True)
        ax.text(0.00, 1.02, BNAME, transform=ax.transAxes, fontsize=8)
        ax.text(0.85, 1.02, f"ph2_side = {side}", transform=ax.transAxes, fontsize=8)
        ax.text(0.03, 0.01, f"rod%2=1", color=colors[1], transform=ax.transAxes, fontsize=16)
        ax.text(0.03, 0.06, f"rod%2=0", color=colors[2], transform=ax.transAxes, fontsize=16)
        fig.subplots_adjust(bottom=0.12, left=0.18, right=0.96, top=0.95)
        pdf.savefig()
        plt.close()

    def plot_module_rz(self, pdf: PdfPages) -> None:
        name = "ph2_module"
        side = 3
        layer = 1
        colors = {
            1: "tab:blue",
            2: "tab:orange",
            3: "tab:green",
            4: "tab:red",
            5: "tab:purple",
            6: "tab:brown",
            7: "tab:pink",
        }
        fig, ax = plt.subplots(figsize=(8, 6))
        for key in colors:
            subset = self.data[(self.data[name] == key) & (self.data["ph2_layer"] == layer) & (self.data["ph2_side"] == side)]
            ax.scatter(subset["ph2_z"], subset["ph2_r"], s=1, c=colors[key], marker=".", edgecolors='none')
        ax.set_xlabel("z [cm]")
        ax.set_ylabel("r [cm]")
        ax.set_xlim(-18, 18)
        ax.set_ylim(21, 26)
        ax.tick_params(right=True, top=True)
        ax.text(0.00, 1.02, BNAME, transform=ax.transAxes, fontsize=8)
        ax.text(0.80, 1.02, f"side = {side}, layer = {layer}", transform=ax.transAxes, fontsize=8)
        ax.text(0.00, 0.53, f"module", color="black", transform=ax.transAxes, fontsize=24)
        ax.text(0.10, 0.47, f"1", color=colors[1], transform=ax.transAxes, fontsize=24)
        ax.text(0.23, 0.47, f"2", color=colors[2], transform=ax.transAxes, fontsize=24)
        ax.text(0.35, 0.47, f"3", color=colors[3], transform=ax.transAxes, fontsize=24)
        ax.text(0.48, 0.47, f"4", color=colors[4], transform=ax.transAxes, fontsize=24)
        ax.text(0.60, 0.47, f"5", color=colors[5], transform=ax.transAxes, fontsize=24)
        ax.text(0.73, 0.47, f"6", color=colors[6], transform=ax.transAxes, fontsize=24)
        ax.text(0.85, 0.47, f"7", color=colors[7], transform=ax.transAxes, fontsize=24)
        fig.subplots_adjust(bottom=0.12, left=0.18, right=0.96, top=0.95)
        pdf.savefig()
        plt.close()


    def plot_module_rz_number(self, pdf: PdfPages) -> None:
        name = "ph2_module"
        side = 3
        colors = {
            1: "tab:blue",
            2: "tab:orange",
            3: "tab:green",
            4: "tab:red",
            5: "tab:purple",
            6: "tab:brown",
            0: "tab:pink",
        }
        fig, ax = plt.subplots(figsize=(8, 6))
        layer_min = self.data["ph2_layer"].min()
        layer_max = self.data["ph2_layer"].max()
        for layer in range(layer_min, layer_max+1):
            subset_layer = self.data[(self.data["ph2_layer"] == layer) & (self.data["ph2_side"] == side)]
            mod_min = subset_layer[name].min()
            mod_max = subset_layer[name].max()
            for mod in range(mod_min, mod_max+1):
                subset = subset_layer[subset_layer[name] == mod]
                fontsize = 10 if layer > 3 else 6
                ax.text(subset["ph2_z"].mean(), subset["ph2_r"].mean(), f"{mod}", fontsize=fontsize, color=colors[mod % len(colors)], ha="center", va="center")
        ax.set_xlabel("z [cm]")
        ax.set_ylabel("r [cm]")
        ax.set_xlim(-130, 130)
        ax.set_ylim(10, 120)
        ax.tick_params(right=True, top=True)
        ax.text(0.00, 1.02, BNAME, transform=ax.transAxes, fontsize=8)
        ax.text(0.85, 1.02, f"ph2_side = {side}", transform=ax.transAxes, fontsize=8)
        ax.text(0.03, 0.04, f"module number", transform=ax.transAxes, fontsize=20)
        fig.subplots_adjust(bottom=0.12, left=0.18, right=0.96, top=0.95)
        pdf.savefig()
        plt.close()


    def plot_rod_rz_tilted(self, pdf: PdfPages) -> None:
        name = "ph2_rod"
        sides = (1, 2)
        order = 0
        colors = {
            1: "tab:blue",
            2: "tab:orange",
            3: "tab:green",
            4: "tab:red",
            5: "tab:purple",
            6: "tab:brown",
            7: "tab:pink",
            8: "tab:gray",
            9: "tab:olive",
            10: "tab:cyan",
            11: "lime",
            12: "black",
        }
        fig, ax = plt.subplots(figsize=(8, 6))
        mask = np.isin(self.data["ph2_side"], sides) & (self.data["ph2_order"] == order)
        for key in colors:
            subset = self.data[mask & (self.data[name] == key)]
            ax.scatter(subset["ph2_z"], subset["ph2_r"], s=1, c=colors[key], marker=".", edgecolors='none')

            xpos_l = subset["ph2_z"][(subset["ph2_layer"] == 2) & (subset["ph2_z"] < 0)].mean()
            xpos_r = subset["ph2_z"][(subset["ph2_layer"] == 2) & (subset["ph2_z"] > 0)].mean()
            ypos = 42 + (1.5 if key % 2 == 1 else 0)
            ax.text(xpos_l, ypos, key, color=colors[key], fontsize=14, ha="center")
            ax.text(xpos_r, ypos, key, color=colors[key], fontsize=14, ha="center")

        ax.set_xlabel("z [cm]")
        ax.set_ylabel("r [cm]")
        ax.set_xlim(-130, 130)
        ax.set_ylim(20, 60)
        ax.tick_params(right=True, top=True)
        ax.text(0.00, 1.02, BNAME, transform=ax.transAxes, fontsize=8)
        ax.text(0.75, 1.02, f"side = {sides}, order = {order}", transform=ax.transAxes, fontsize=8)
        ax.text(0.00, 0.63, f"module", color="black", transform=ax.transAxes, fontsize=20)
        fig.subplots_adjust(bottom=0.12, left=0.18, right=0.96, top=0.95)
        pdf.savefig()
        plt.close()


# how to make a binary histogram
# cmap = ListedColormap(['white', 'black'])
# bounds = [-0.5, 0.5, 1e10]    # or some large number instead of 1e10
# norm = BoundaryNorm(bounds, cmap.N)
# ax.hist2d(self.data["ph2_z"], self.data["ph2_r"], bins=(500, 500), cmap=cmap, norm=norm)


class Data:
    def __init__(self, num: int):
        self.data = self.load()
        # self.round_to_cm()
        self.keep_subset(num)
        self.remove_duplicates()
        self.add_branches()

    def load(self) -> pd.DataFrame:
        with uproot.open(f"{FNAME}:{TNAME}") as tree:
            branches = tree.arrays(BRANCHES, library=LIBRARY)
            return pd.DataFrame({br: np.concatenate(branches[br]) for br in BRANCHES})

    def keep_subset(self, subset: int) -> None:
        self.data = self.data.head(subset)

    def round_to_cm(self) -> None:
        self.data = self.data.round({"ph2_x": 0, "ph2_y": 0, "ph2_z": 0})

    def remove_duplicates(self) -> None:
        print(f"With duplicates: {len(self.data)}")
        self.data.drop_duplicates(inplace=True)
        print(f"Without duplicates: {len(self.data)}")
            
    def add_branches(self) -> None:
        self.data["ph2_r"] = np.sqrt(self.data["ph2_x"]**2 + self.data["ph2_y"]**2)
        self.data["ph2_phi"] = np.arctan2(self.data["ph2_y"], self.data["ph2_x"])

if __name__ == "__main__":
    main()
