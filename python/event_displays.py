import os
import awkward as ak # type: ignore
import numpy as np
import uproot # type: ignore
from collections.abc import Sequence
from typing import List
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
plt.rcParams.update({"font.size": 16})

# FNAME = Path("/ceph/users/atuna/trackingNtuple_10mu_10k_pt_0p5_50_5cm_cube.root")
FNAME = Path("/Users/alexandertuna/Downloads/cms/lst_playing/data/trackingNtuple.2025_04_02_12h00m00s.1muon_0p7gev.root")
TNAME = Path("trackingNtuple/tree")
LIBRARY = "np"
BRANCHES = [
    "event",
    "simhit_x",
    "simhit_y",
    "simhit_z",
    "simhit_px",
    "simhit_py",
]
MAX_EVENTS = 40
BARREL_RADII = [23, 36, 51, 68, 86, 108.5]

def main() -> None:
    with uproot.open(f"{FNAME}:{TNAME}") as tree:
        branches = tree.arrays(BRANCHES, library=LIBRARY)

        extra = 1.1
        bounds = [extra * np.concatenate(branches["simhit_x"]).min(),
                  extra * np.concatenate(branches["simhit_x"]).max(),
                  extra * np.concatenate(branches["simhit_y"]).min(),
                  extra * np.concatenate(branches["simhit_y"]).max(),
                  ]

        with PdfPages("tmp.pdf") as pdf:
            for index in range(MAX_EVENTS):
                mask = ((branches["simhit_px"][index]**2 + branches["simhit_py"][index]**2) ** 0.5) > 0.58
                draw(index,
                     branches["simhit_x"][index][mask],
                     branches["simhit_y"][index][mask],
                     branches["simhit_z"][index][mask],
                     bounds,
                     pdf)

                
def draw(
    index: int,
    x: Sequence[float],
    y: Sequence[float],
    z: Sequence[float],
    bounds: List[float],
    pdf: PdfPages) -> None:
    xmin, xmax, ymin, ymax = bounds
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Sim. x [cm]")
    ax.set_ylabel("Sim. y [cm]")
    ax.tick_params(right=True, top=True)
    # ax.grid(True, linestyle='-', linewidth=0.1, color='black')
    ax.set_axisbelow(True)
    # ax.text(0.00, 1.02, f"{os.path.basename(FNAME)}", transform=ax.transAxes, fontsize=10)
    ax.text(0.85, 1.02, f"Event {index}", transform=ax.transAxes, fontsize=10)
    circles = [plt.Circle(xy=(0, 0), radius=rad, edgecolor='gray', fill=False) for rad in BARREL_RADII]
    for circle in circles:
        ax.add_patch(circle)
    fig.subplots_adjust(bottom=0.12, left=0.18, right=0.96, top=0.95)
    pdf.savefig()
    plt.close()

if __name__ == "__main__":
    main()
