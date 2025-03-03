"""
Inspired by PChang:
http://uaf-3.t2.ucsd.edu/~phchang/talks/PhilipChang20190330_ModuleStructure.pdf
"""

import os
import awkward as ak # type: ignore
import uproot # type: ignore
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
plt.rcParams.update({"font.size": 16})

FNAME = Path("/ceph/users/atuna/data/trackingNtuple_10mu_10k_pt_0p5_50_5cm_cube.root")
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
    num = 1_109_000 # 130 # 2_000_000
    data = Data(num=num).data
    # print(data)
    plot = Plotter(data)
    plot.plot()


class Plotter:

    def __init__(self, data: pd.DataFrame, pdfname: Path = Path("tmp.pdf"), scatter: bool = False):
        self.data = data
        self.pdfname = pdfname
        self.scatter = scatter


    def plot(self) -> None:

        mask = (self.data["ph2_order"] == 0) \
             & (self.data["ph2_side"] == 3) \
             & (self.data["ph2_layer"] == 1) \
             & (self.data["ph2_rod"] == 5) \
             & (self.data["ph2_module"] == 4)
        subset = self.data[mask]

        with PdfPages(self.pdfname) as pdf:
            for it, (event, group) in enumerate(subset.groupby("event")):
                print(f"Event {event}")
                self.plot_event(event, group, pdf)
                if it > 5:
                    break

        # with PdfPages(self.pdfname) as pdf:
        #     self.plot_xy_all(pdf)
        #     #self.plot_xy(pdf)
        #     #self.plot_xy_grid(pdf)

    def plot_event(self, event: int, df: pd.DataFrame, pdf: PdfPages) -> None:
        if len(df) != 2:
            print(f"Event {event} has {len(df)} hits, skipping")
            return
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df["ph2_x"], df["ph2_y"])

        bot_x, bot_y = df.iloc[0]["ph2_x"], df.iloc[0]["ph2_y"]
        top_x, top_y = df.iloc[1]["ph2_x"], df.iloc[1]["ph2_y"]
        #ax.plot([0, bot_x], [0, bot_y], color="black")
        #ax.plot([0, top_x], [0, top_y], color="black")
        ax.plot([0, top_x+bot_x], [0, top_y+bot_y], color="black", linestyle="--")
        ax.plot([bot_x, top_x], [bot_y, top_y], color="blue")

        min_x, max_x = min(bot_x, top_x) - 0.3, max(bot_x, top_x) + 0.3
        min_y, max_y = min(bot_y, top_y) - 0.8, max(bot_y, top_y) + 0.8

        print(np.round(min_x, 1), np.round(max_x, 1))
        x_ticks = np.arange(np.round(min_x, 1), np.round(max_x, 1), 0.1)
        ax.set_xticks(x_ticks)

        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.tick_params(right=True, top=True)
        ax.text(0.00, 1.02, BNAME, transform=ax.transAxes, fontsize=8)
        ax.text(0.80, 1.02, f"Event {event}", transform=ax.transAxes, fontsize=8)
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.grid()
        fig.subplots_adjust(bottom=0.12, left=0.18, right=0.96, top=0.95)
        pdf.savefig()
        plt.close()

    def plot_xy_all(self, pdf: PdfPages) -> None:
        print("plot_xy_all")
        mask = (self.data["ph2_order"] == 0) \
             & (self.data["ph2_layer"] == 1) \
             & (self.data["ph2_side"] == 3)
        subset = self.data[mask]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(subset["ph2_x"], subset["ph2_y"])
        ax.set_xlabel("x [cm]")
        ax.set_ylabel("y [cm]")
        ax.tick_params(right=True, top=True)
        ax.text(0.00, 1.02, BNAME, transform=ax.transAxes, fontsize=8)
        ax.set_xlim(-28, 28)
        ax.set_ylim(-28, 28)
        fig.subplots_adjust(bottom=0.12, left=0.18, right=0.96, top=0.95)
        pdf.savefig()
        plt.close()

    def plot_xy(self, pdf: PdfPages) -> None:
        print("plot_xy")
        fig, ax = plt.subplots(figsize=(8, 6))
        mask = (self.data["ph2_order"] == 0) \
             & (self.data["ph2_layer"] == 1) \
             & (self.data["ph2_side"] == 3) \
             & (self.data["ph2_rod"] == 5) \
             & (self.data["ph2_module"] == 4)
        subset = self.data[mask]
        print(subset.head())
        # print(subset)
        print(10 * np.sort(subset["ph2_x"]))
        ax.scatter(subset["ph2_x"], subset["ph2_y"])
        ax.set_xlabel("x [cm]")
        ax.set_ylabel("y [cm]")
        ax.tick_params(right=True, top=True)
        ax.text(0.00, 1.02, BNAME, transform=ax.transAxes, fontsize=8)
        ax.set_xlim(-7, 7)
        ax.set_ylim(24.2, 25.5)
        fig.subplots_adjust(bottom=0.12, left=0.18, right=0.96, top=0.95)
        pdf.savefig()
        plt.close()


    def plot_xy_grid(self, pdf: PdfPages) -> None:
        print("plot_xy_grid")
        fig, ax = plt.subplots(figsize=(8, 6))
        mask = (self.data["ph2_order"] == 0) \
             & (self.data["ph2_layer"] == 1) \
             & (self.data["ph2_side"] == 3) \
             & (self.data["ph2_rod"] == 5) \
             & (self.data["ph2_module"] == 4)
        subset = self.data[mask]
        ax.scatter(subset["ph2_x"], subset["ph2_y"], s=0.1, marker="|")
        ax.set_xlabel("x [cm]")
        ax.set_ylabel("y [cm]")
        ax.tick_params(right=True, top=True)
        ax.text(0.00, 1.02, BNAME, transform=ax.transAxes, fontsize=8)
        ax.set_xlim(-7, 7)
        ax.set_ylim(24.2, 25.5)
        fig.subplots_adjust(bottom=0.12, left=0.18, right=0.96, top=0.95)
        pdf.savefig()
        plt.close()


class Data:
    def __init__(self, num: int):
        self.data = self.load()
        self.keep_subset(num)
        self.remove_duplicates()
        self.add_branches()

    def load(self) -> pd.DataFrame:
        with uproot.open(f"{FNAME}:{TNAME}") as tree:
            event = tree.arrays(["event"], library="ak")["event"]
            branches = tree.arrays(BRANCHES, library="ak")
            df = pd.DataFrame({br: np.concatenate(branches[br]) for br in BRANCHES})
            df["event"] = ak.flatten(ak.broadcast_arrays(event, branches["ph2_x"])[0])
            df["ph2_x"] *= 10
            df["ph2_y"] *= 10
            return df

    def keep_subset(self, subset: int) -> None:
        self.data = self.data.head(subset)

    def remove_duplicates(self) -> None:
        print(f"With duplicates: {len(self.data)}")
        self.data.drop_duplicates(inplace=True)
        print(f"Without duplicates: {len(self.data)}")
            
    def add_branches(self) -> None:
        self.data["ph2_r"] = np.sqrt(self.data["ph2_x"]**2 + self.data["ph2_y"]**2)
        self.data["ph2_phi"] = np.arctan2(self.data["ph2_y"], self.data["ph2_x"])

if __name__ == "__main__":
    main()
