import uproot
import awkward as ak
import numpy as np
from pathlib import Path
import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams
rcParams["font.size"] = 16

# FNAME_TRACKING = Path("../data/trackingNtuple.2025_04_23_00h00m00s.10muon_0p5_3p0.root")
# FNAME_REF = Path("../data/muonGun/LSTNtuple.cutNA.root")
# FNAME_NEW = Path("../data/muonGun/LSTNtuple.cut04.root")
FNAME_TRACKING = Path("/ceph/users/atuna/CMSSW_15_1_0_pre2/src/ttbar/n1000/trackingNtuple.2025_05_09_00h00m00s.ttbar_PU200.n1000.root")
FNAME_REF = Path("/ceph/users/atuna/CMSSW_15_1_0_pre2/src/RecoTracker/LSTCore/standalone/clustSize/ttbarPU200_n1e3/LSTNtuple.cutNA.root")
FNAME_NEW = Path("/ceph/users/atuna/CMSSW_15_1_0_pre2/src/RecoTracker/LSTCore/standalone/clustSize/ttbarPU200_n1e3/LSTNtuple.cut04.root")
BRANCHES = [
    "sim_trkNtupIdx",
    "sim_event",
    "sim_pt",
    "sim_eta",
    "sim_phi",
    "sim_pdgId",
    "tc_pt",
    "tc_eta",
    "tc_phi",
    "tc_type",
    "tc_matched_simIdx",
]
TC_TYPE = 7 # PT5
BARREL_RADII = [23, 36, 51, 68, 86, 108.5]
DEBUG = True

def main():
    evts, idxs = find_missing_tracks()
    with PdfPages("missing_tracks.pdf") as pdf:
        inspect_hits(evts, idxs, pdf)


def find_missing_tracks() -> tuple[list, list]:
    data_ref = get_data(FNAME_REF)
    data_new = get_data(FNAME_NEW)
    mask_ref = data_ref["tc_type"] == TC_TYPE
    mask_new = data_new["tc_type"] == TC_TYPE

    ntracks_ref = ak.sum(mask_ref, axis=1)
    ntracks_new = ak.sum(mask_new, axis=1)
    track_missing = ntracks_ref > ntracks_new
    event_idx_of_interest, sim_idxs_of_interest = [], []
    for idx, missing in enumerate(track_missing):
        if not missing:
            continue
        sim_idxs = compare(data_ref, data_new, idx)
        event_idx_of_interest.append(idx)
        sim_idxs_of_interest.append(sim_idxs)
        if len(event_idx_of_interest) > 0:
            break

    print("Finished finding missing tracks")
    return event_idx_of_interest, sim_idxs_of_interest


def inspect_hits(evts: list, sim_idxs: list, pdf: PdfPages):

    data = get_tracking_ntuple()
    for evt, sims in zip(evts, sim_idxs):
        for sim in sims:
            sim_pt, sim_eta, sim_phi = [data["sim_pt"][evt][sim],
                                        data["sim_eta"][evt][sim],
                                        data["sim_phi"][evt][sim]]
            mask_matched = ak.fill_none(data["ph2_simIdx"][evt] == sim, False)
            mask_deltaR = ak.fill_none(dr(data["ph2_eta"][evt],
                                          data["ph2_phi"][evt],
                                          sim_eta,
                                          sim_phi) < 0.2, False)
            mask = mask_matched | mask_deltaR

            # convert mask to sparse array, sorted by clustSize
            idxs = np.flatnonzero(mask)
            masked_clustSize = data["ph2_clustSize"][evt][idxs]
            sort_order = np.argsort(masked_clustSize)
            idxs = idxs[sort_order]

            print(f"Event {evt}, simIdx {sim}, sim eta {sim_eta:.3f}, sim phi {sim_phi:.3f}")
            print("mask", mask)
            print("idxs", idxs)
            print('data["ph2_clustSize"]', data["ph2_clustSize"][evt][idxs])
            print('data["ph2_layer"]', data["ph2_layer"][evt][idxs])
            print("")

            # plotting
            fig, ax = plt.subplots(figsize=(8, 8))
            sc = ax.scatter(data["ph2_x"][evt][idxs],
                       data["ph2_y"][evt][idxs],
                       c=data["ph2_clustSize"][evt][idxs],
                       cmap="Set1",
                       edgecolors="black",
                       vmin=1,
                       vmax=19,
                       s=50)
            fig.colorbar(sc, ax=ax, label="Cluster size")          
            ax.set_xlim([np.min(data["ph2_x"][evt][idxs]) - 10,
                         np.max(data["ph2_x"][evt][idxs]) + 10])
            ax.set_ylim([np.min(data["ph2_y"][evt][idxs]) - 10,
                         np.max(data["ph2_y"][evt][idxs]) + 10])
            ax.set_xlabel("x [cm]")
            ax.set_ylabel("y [cm]")
            ax.set_title(f"Event {evt}, pt={sim_pt:.2f}, eta={sim_eta:.2f}, phi={sim_phi:.2f}")
            circles = [plt.Circle(xy=(0, 0), radius=rad, edgecolor='gray', fill=False) for rad in BARREL_RADII]
            for circle in circles:
                pass # ax.add_patch(circle)
            pdf.savefig()
            plt.close()


def compare(ref: ak.Array, new: ak.Array, evt: int):

    # PT5 mask
    mask_ref = ref["tc_type"][evt] == TC_TYPE
    mask_new = new["tc_type"][evt] == TC_TYPE

    # Finding sim indices of PT5s
    simIdx_ref = ak.firsts(ref["tc_matched_simIdx"][evt][mask_ref])
    simIdx_new = ak.firsts(new["tc_matched_simIdx"][evt][mask_new])
    simIdx_missing = simIdx_ref[~np.isin(simIdx_ref, simIdx_new)]
    if DEBUG:
        print("Missing simIdx of tracks:", type(simIdx_missing), simIdx_missing)

    # Finding the track(s) in the new event
    simIdx_new = ak.firsts(new["tc_matched_simIdx"][evt])
    mask_of_interest = np.isin(simIdx_new, simIdx_missing)
    if DEBUG:
        print("Mask of missing tracks in new event:", mask_of_interest)


    if DEBUG:
        idxs = range(len(ref["sim_pt"][evt]))
        for idx in idxs:
            pt, eta, phi = [ref["sim_pt"][evt][idx],
                            ref["sim_eta"][evt][idx],
                            ref["sim_phi"][evt][idx]]
            print(idx, "Sim:", format_params(pt, eta, phi))

        idxs = range(len(ref["tc_pt"][evt]))
        for idx in idxs:
            pt, eta, phi, tc = [ref["tc_pt"][evt][idx],
                                ref["tc_eta"][evt][idx],
                                ref["tc_phi"][evt][idx],
                                ref["tc_type"][evt][idx]]
            simIdx = ref["tc_matched_simIdx"][evt][idx][0] if len(ref["tc_matched_simIdx"][evt][idx]) > 0 else -1
            tag = "<--" if tc == TC_TYPE else ""
            print(idx, "Ref:", format_params(pt, eta, phi), tc, simIdx, tag)

        idxs = range(len(new["tc_pt"][evt]))
        for idx in idxs:
            pt, eta, phi, tc = [new["tc_pt"][evt][idx],
                                new["tc_eta"][evt][idx],
                                new["tc_phi"][evt][idx],
                                new["tc_type"][evt][idx]]
            simIdx = new["tc_matched_simIdx"][evt][idx][0] if len(new["tc_matched_simIdx"][evt][idx]) > 0 else -1
            tag = "<--" if tc == TC_TYPE else ""
            tag = "<=====" if simIdx in simIdx_missing else tag
            print(idx, "New:", format_params(pt, eta, phi), tc, simIdx, tag)

    return simIdx_new[mask_of_interest]


def format_params(pt, eta, phi):
    return f"{pt=:.3f}, {eta=:6.3f}, {phi=:6.3f}"


def get_data(fname: Path) -> dict:
    with uproot.open(fname) as fi:
        tree = fi["tree"]
        # if DEBUG and "NA" in fname.stem:
        #     for key in tree.keys():
        #         print(key)
        return tree.arrays(BRANCHES)


def get_tracking_ntuple() -> dict:
    with uproot.open(FNAME_TRACKING) as fi:
        tree = fi["trackingNtuple/tree"]
        # for key in tr.keys():
        #     print(key)
        BRS = [
            "sim_pt",
            "sim_eta",
            "sim_phi",
            "sim_pdgId",
            "sim_simHitIdx",
            "ph2_simHitIdx",
            "ph2_layer",
            "ph2_side",
            "ph2_x",
            "ph2_y",
            "ph2_z",
            "ph2_clustSize",
        ]
        print("Reading tracking ntuple")
        data = tree.arrays(BRS, entry_stop=1)
        data["ph2_nsimhit"] = ak.num(data["ph2_simHitIdx"], axis=-1)
        data["ph2_simHitIdxFirst"] = ak.firsts(data["ph2_simHitIdx"], axis=-1)
        data["ph2_eta"] = get_eta(data.ph2_x, data.ph2_y, data.ph2_z)
        data["ph2_phi"] = get_phi(data.ph2_x, data.ph2_y)
        data["ph2_simIdx"] = get_ph2_sim_idx(data)
        print("Finished reading tracking ntuple")
        return data


def get_ph2_sim_idx(data):
    # find ph2 hits associated with sim particles
    PARQUET_NAME = Path("ph2_simIdx.parquet")
    if PARQUET_NAME.exists():
        return ak.from_parquet(PARQUET_NAME)
    ph2_simIdxs = []
    for evt in tqdm.tqdm(range(len(data["ph2_simHitIdx"]))):
        ph2_simIdx = []
        for ph2_idx in tqdm.tqdm(range(len(data["ph2_simHitIdx"][evt]))):
            ph2_simHitIdxFirst = data["ph2_simHitIdxFirst"][evt][ph2_idx]
            for i_sim, sim_simHitIdxs in enumerate(data["sim_simHitIdx"][evt]):
                if np.isin(ph2_simHitIdxFirst, sim_simHitIdxs):
                    ph2_simIdx.append(i_sim)
                    break
            else:
                ph2_simIdx.append(None)
        ph2_simIdxs.append(ph2_simIdx)
    arr = ak.Array(ph2_simIdxs)
    ak.to_parquet(arr, PARQUET_NAME)
    return arr


def get_eta(x, y, z):
    r_perp = np.sqrt(x**2 + y**2)
    theta = np.atan2(r_perp, z)
    return -np.log(np.tan(theta / 2.0))

def get_phi(x, y):
    return np.atan2(y, x)


def dphi(a, b):
    return np.abs(((a - b) + np.pi) % (2 * np.pi) - np.pi)


def deta(a, b):
    return np.abs(a - b)


def dr(eta_a, phi_a, eta_b, phi_b):
    return np.sqrt(deta(eta_a, eta_b)**2 + dphi(phi_a, phi_b)**2)



if __name__ == "__main__":
    main()
