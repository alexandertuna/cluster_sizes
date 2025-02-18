import os
import numpy as np
import uproot
from collections.abc import Iterable
from typing import List
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
plt.rcParams.update({"font.size": 16})

FNAME = Path("/ceph/users/atuna/trackingNtuple_10mu_10k_pt_0p5_50_5cm_cube.root")
TNAME = Path("trackingNtuple/tree")
LIBRARY = "np"
BRANCHES = [
    "event",
    "simhit_x",
    "simhit_y",
    "simhit_z",
]
MAX_EVENTS = 10

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
                draw(branches["simhit_x"][index],
                     branches["simhit_y"][index],
                     branches["simhit_z"][index],
                     bounds,
                     pdf)

                
def draw(
    x: Iterable[float],
    y: Iterable[float],
    z: Iterable[float],
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
    ax.grid(True, linestyle='-', linewidth=0.1, color='black')
    ax.set_axisbelow(True)
    ax.text(0.02, 1.02, f"{os.path.basename(FNAME)}", transform=ax.transAxes, fontsize=10)
    fig.subplots_adjust(bottom=0.12, left=0.18, right=0.96, top=0.95)
    pdf.savefig()
    plt.close()

if __name__ == "__main__":
    main()
