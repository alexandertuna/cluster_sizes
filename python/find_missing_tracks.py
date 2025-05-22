import uproot
import awkward as ak
import numpy as np
from pathlib import Path

FNAME_TRACKING = Path("../data/trackingNtuple.2025_04_23_00h00m00s.10muon_0p5_3p0.root")
FNAME_REF = Path("../data/muonGun/LSTNtuple.cutNA.root")
FNAME_NEW = Path("../data/muonGun/LSTNtuple.cut04.root")
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
PT_LO = 1.9
PT_HI = 2.1
DEBUG = True

def main():
    bonk_tracking_ntuple(FNAME_TRACKING)
    return
    data_ref = get_data(FNAME_REF)
    data_new = get_data(FNAME_NEW)
    mask_ref = data_ref["tc_type"] == TC_TYPE
    mask_new = data_new["tc_type"] == TC_TYPE

    ntracks_ref = ak.sum(mask_ref, axis=1)
    ntracks_new = ak.sum(mask_new, axis=1)
    track_missing = ntracks_ref > ntracks_new
    for idx, missing in enumerate(track_missing):
        if missing:
            print("Missing:", idx)
            compare(data_ref, data_new, idx)
            break


def bonk_tracking_ntuple(fname: Path):

    with uproot.open(fname) as fi:
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

        data = tree.arrays(BRS)
        data["ph2_nsimhit"] = ak.num(data["ph2_simHitIdx"], axis=-1)
        data["ph2_simHitIdxFirst"] = ak.firsts(data["ph2_simHitIdx"], axis=-1)
        data["ph2_eta"] = get_eta(data.ph2_x, data.ph2_y, data.ph2_z)
        data["ph2_phi"] = get_phi(data.ph2_x, data.ph2_y)
        # connect ph2 to simHitIdx
        # connect simHitIdx to simIdx
        # connect ph2 to simIdx

    # return
    # these should be given as input from LSTNtuple
    evt = 10
    sim_of_interest = ak.Array([6])

    if DEBUG:
        idxs = range(len(data["sim_pt"][evt]))
        for idx in idxs:
            pt, eta, phi = [data["sim_pt"][evt][idx],
                            data["sim_eta"][evt][idx],
                            data["sim_phi"][evt][idx]]
            print(idx, "Trk:", format_params(pt, eta, phi))

    # find ph2 hits associated with sim particles
    ph2_simIdxs = []
    for evt in range(len(data["ph2_simHitIdx"])):
        if evt > 20:
            break
        ph2_simIdx = []
        for ph2_idx in range(len(data["ph2_simHitIdx"][evt])):
            ph2_simHitIdxFirst = data["ph2_simHitIdxFirst"][evt][ph2_idx]
            for i_sim, sim_simHitIdxs in enumerate(data["sim_simHitIdx"][evt]):
                if np.isin(ph2_simHitIdxFirst, sim_simHitIdxs):
                    ph2_simIdx.append(i_sim)
                    break
            else:
                ph2_simIdx.append(None)
        ph2_simIdxs.append(ph2_simIdx)
    # data["ph2_simIdx"] = ak.Array(ph2_simIdxs)    
    ph2_simIdx = ak.Array(ph2_simIdxs)    

    # get the cluster sizes of the ph2 hits of interest
    evt = 10
    sims_of_interest = ak.Array([6])
    print(len(ph2_simIdxs))
    print("ph2_simIdxs", ph2_simIdxs[evt])
    for sim in sims_of_interest:
        mask_evt = ph2_simIdxs[evt] == sim
        print("mask_evt", mask_evt)
        print('data["ph2_eta"]', data["ph2_eta"][evt][mask_evt])
        print('data["ph2_phi"]', data["ph2_phi"][evt][mask_evt])
        print('data["ph2_clustSize"]', data["ph2_clustSize"][evt][mask_evt])
        print('data["ph2_layer"]', data["ph2_layer"][evt][mask_evt])


def compare(ref: ak.Array, new: ak.Array, evt: int):
    # Event info
    # print("ref['sim_event'][evt]", ref["sim_pt"])
    # print("ref['sim_event'][evt]", ref["sim_event"])
    # print("ref['sim_trkNtupIdx'][evt]", ref["sim_trkNtupIdx"])

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
            simIdx = ref["tc_matched_simIdx"][evt][idx][0]
            tag = "<--" if tc == TC_TYPE else ""
            print(idx, "Ref:", format_params(pt, eta, phi), tc, simIdx, tag)

        idxs = range(len(new["tc_pt"][evt]))
        for idx in idxs:
            pt, eta, phi, tc = [new["tc_pt"][evt][idx],
                                new["tc_eta"][evt][idx],
                                new["tc_phi"][evt][idx],
                                new["tc_type"][evt][idx]]
            simIdx = new["tc_matched_simIdx"][evt][idx][0]
            tag = "<--" if tc == TC_TYPE else ""
            tag = "<=====" if simIdx in simIdx_missing else tag
            print(idx, "New:", format_params(pt, eta, phi), tc, simIdx, tag)


def format_params(pt, eta, phi):
    return f"{pt=:.3f}, {eta=:6.3f}, {phi=:6.3f}"


def get_data(fname: Path) -> dict:
    with uproot.open(fname) as fi:
        tree = fi["tree"]
        # if DEBUG and "NA" in fname.stem:
        #     for key in tree.keys():
        #         print(key)
        return tree.arrays(BRANCHES)


def get_eta(x, y, z):
    r_perp = np.sqrt(x**2 + y**2)
    theta = np.atan2(r_perp, z)
    return -np.log(np.tan(theta / 2.0))

def get_phi(x, y):
    return np.atan2(y, x)

if __name__ == "__main__":
    main()
