import numpy as np
import uproot

# FNAME = "/ceph/users/atuna/trackingNtuple_ttbar_PU200.root"
FNAME = "/ceph/users/atuna/trackingNtuple_10mu_10k_pt_0p5_50_5cm_cube.root"
TNAME = "trackingNtuple/tree"
LIBRARY = "np" # "ak"
BRANCHES = [
    "event",
    "simhit_x",
    "simhit_y",
    "simhit_z",
]

def main():

    print(f"Opening {FNAME}")

    # Print file info
    with uproot.open(FNAME) as fi:
        print(f" file: {fi}")
        print(f" keys: {fi.keys()}")
        print(f" classnames: {fi.classnames()}")
        print(fi["trackingNtuple"]["tree"])

    # Print tree/branch info
    with uproot.open(f"{FNAME}:{TNAME}") as tree:
        print(tree)
        tree.show()
        branches = tree.arrays(BRANCHES, library=LIBRARY)
        if LIBRARY == "ak":
            print(len(branches["event"]))
            print(len(branches["simhit_x"]))
            print(len(np.mean(branches["simhit_x"], axis=1)))
        elif LIBRARY == "np":
            print(branches["event"].shape)
            print(branches["simhit_x"].shape)
            print(branches["simhit_x"][0].shape)

if __name__ == "__main__":
    main()
