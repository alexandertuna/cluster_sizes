#!/usr/bin/env python
# coding: utf-8

# In[196]:


import math
import uproot
from pathlib import Path
import awkward as ak
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import tqdm


# ## Getting the file and tree

# In[197]:


# fname = Path("/Users/alexandertuna/Downloads/cms/lst_playing/data/trackingNtuple.2025_03_15_03h02m56s.root")
fname = Path("/Users/alexandertuna/Downloads/cms/lst_playing/data/trackingNtuple.2025_03_21_11h59m00s.root")
if not fname.exists():
    raise Exception("shit")


# In[198]:


file = uproot.open(f"{fname}")
print(file.keys())


# In[199]:


tree = uproot.open(f"{fname}:trackingNtuple/tree")
print(tree)


# In[200]:


def get_prefixes(col):
    return sorted(list(set([obj.split("_")[0] for obj in col])))
print(get_prefixes(tree.keys()))


# ## Getting branches into a data array

# In[201]:


print(tree.keys())


# In[202]:


data = tree.arrays([
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


data["simhit_pt"] = np.sqrt(data.simhit_px**2 + data.simhit_py**2)
data["simhit_p"] = np.sqrt(data.simhit_px**2 + data.simhit_py**2 + data.simhit_pz**2)
data["simhit_rt"] = np.sqrt(data.simhit_x**2 + data.simhit_y**2)
data["simhit_cosphi"] = ((data.simhit_x * data.simhit_px) + (data.simhit_y * data.simhit_py)) / (data.simhit_pt * data.simhit_rt)
data["simhit_phi"] = np.atan2(data.simhit_y, data.simhit_x)

data["sim_p"] = np.sqrt(data.sim_px**2 + data.sim_py**2 + data.sim_pz**2)

data["ph2_phi"] = np.atan2(data.ph2_y, data.ph2_x)
data["ph2_rt"] = np.sqrt(data.ph2_x**2 + data.ph2_y**2)
data["ph2_isBarrelFlat"] = (data.ph2_order == 0) & (data.ph2_side == 3)
data["ph2_isBarrelTilt"] = (data.ph2_order == 0) & (data.ph2_side != 3)
data["ph2_isEndcap"] = (data.ph2_order != 0)


# ## Coordinate conversions

# In[203]:


def eta(x, y, z):
    r_perp = np.sqrt(x**2 + y**2)
    theta = np.atan2(r_perp, z)
    return -np.log(np.tan(theta / 2.0))

def phi(x, y):
    return np.atan2(y, x)

def dphi(a, b):
    return np.abs(((a - b) + np.pi) % (2 * np.pi) - np.pi)


# ## Creating more simhit_* arrays with simhit/sim-matching

# In[204]:


data["simhit_simtrk_pt"] = data.sim_pt[data.simhit_simTrkIdx]
data["simhit_simtrk_p"] = data.sim_p[data.simhit_simTrkIdx]


# ## Creating more `ph2_*` arrays with truth-matching (vectorized)

# In[205]:


data["ph2_nsimhit"] = ak.num(data.ph2_simHitIdx, axis=-1)
data["ph2_simHitIdxFirst"] = ak.firsts(data.ph2_simHitIdx, axis=-1)

data["ph2_simhit_p"]       = data.simhit_p[data.ph2_simHitIdxFirst]
data["ph2_simhit_pt"]      = data.simhit_pt[data.ph2_simHitIdxFirst]
data["ph2_simhit_rt"]      = data.simhit_rt[data.ph2_simHitIdxFirst]
data["ph2_simhit_phi"]     = data.simhit_phi[data.ph2_simHitIdxFirst]
data["ph2_simhit_cosphi"]  = data.simhit_cosphi[data.ph2_simHitIdxFirst]
data["ph2_simhit_dphi"] = dphi(data.ph2_phi, data.ph2_simhit_phi)

dne = np.float32(0)
data["ph2_simhit_p"]      = ak.fill_none(data.ph2_simhit_p, dne)
data["ph2_simhit_pt"]     = ak.fill_none(data.ph2_simhit_pt, dne)
data["ph2_simhit_phi"]    = ak.fill_none(data.ph2_simhit_phi, dne)
data["ph2_simhit_dphi"]   = ak.fill_none(data.ph2_simhit_dphi, dne)
data["ph2_simhit_cosphi"] = ak.fill_none(data.ph2_simhit_cosphi, dne)

data["ph2_simhit_rdphi"]  = data["ph2_simhit_rt"] * data["ph2_simhit_dphi"]

data["ph2_simtrk_p"]  = data.simhit_simtrk_p[data.ph2_simHitIdxFirst]
data["ph2_simtrk_pt"] = data.simhit_simtrk_pt[data.ph2_simHitIdxFirst]

data["ph2_simtrk_p"]  = ak.fill_none(data["ph2_simtrk_p"], dne)
data["ph2_simtrk_pt"] = ak.fill_none(data["ph2_simtrk_pt"], dne)

# print(data["ph2_simhit_p"])
# print(data["ph2_simtrk_p"])


# ## Plotting ph2_* things

# In[206]:


def plot():
    fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
    fig.subplots_adjust(wspace=0.35)
    axs[0].hist(ak.flatten(data.ph2_order), edgecolor="black", bins=np.arange(-1.5, 3, 1))
    axs[1].hist(ak.flatten(data.ph2_side), edgecolor="black", bins=np.arange(-0.5, 5, 1))
    axs[0].set_xlabel("ph2_order")
    axs[1].set_xlabel("ph2_size")
    for ax in axs:
        ax.set_ylabel("Hits (ph2_*)")
        ax.tick_params(right=True, top=True)

plot()


# In[207]:


def plot():
    bins = np.arange(-0.5, 9.5, 1)
    fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
    for ax in axs:
        ax.hist(ak.flatten(data.ph2_nsimhit), edgecolor="black", bins=bins)
        ax.set_xlabel("N(sim. hit)")
        ax.set_ylabel("Hits (ph2_*)")
        ax.tick_params(right=True, top=True)
    axs[1].semilogy()

plot()


# In[208]:


def plot():
    bins = np.arange(-1, 1.02, 0.02)
    fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
    for ax in axs:
        ax.hist(ak.flatten(data.ph2_simhit_cosphi), bins=bins)
        ax.set_xlabel("cos($\\phi$) of $p_{T, sim}$ and $x_{T, sim}$")
        ax.set_ylabel("Hits (ph2_*)")
        ax.set_title("All hits")
        ax.tick_params(right=True, top=True)
    axs[1].semilogy()

plot()


# In[209]:


def plot():
    bins = np.arange(-1, 1.02, 0.02)
    mask = data.ph2_simhit_pt > 0.6
    fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
    for ax in axs:
        ax.hist(ak.flatten(data.ph2_simhit_cosphi[mask]), bins=bins)
        ax.set_xlabel("cos($\\phi$) of $p_{T, sim}$ and $x_{T, sim}$")
        ax.set_ylabel("Hits (ph2_*)")
        ax.set_title("Hits with associated sim track")
        ax.tick_params(right=True, top=True)
    axs[1].semilogy()

plot()


# In[210]:


def plot():
    bins = np.arange(-0.1, 3.2, 0.05)
    mask = data.ph2_simhit_pt > 0.6
    fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
    for ax in axs:
        ax.hist(ak.flatten(data.ph2_simhit_dphi[mask]), bins=bins)
        # ax.hist(ak.flatten(data.ph2_simhit_dphi), bins=bins)
        ax.set_xlabel("dphi(hit, sim. hit) [rad]")
        ax.set_ylabel("Hits (ph2_*)")
        ax.set_title("Hits with associated sim track")
        # ax.set_title("All hits")
        ax.tick_params(right=True, top=True)
    axs[1].semilogy()

plot()


# In[211]:


def plot():
    bins = np.arange(-0.001, 0.25, 0.001)
    fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
    mask = data.ph2_simhit_pt > 0.6
    for ax in axs:
        ax.hist(ak.flatten(data.ph2_simhit_rdphi[mask]), bins=bins)
        ax.set_xlabel("r * dphi(hit, sim. hit) [cm]")
        ax.set_ylabel("Hits (ph2_*)")
        ax.set_title("Hits with associated sim track")
        ax.tick_params(right=True, top=True)
    axs[1].semilogy()

plot()


# In[212]:


def plot():
    bins = np.arange(-0.001, 0.03, 0.0001)
    fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
    mask = data.ph2_simhit_pt > 0.6
    for ax in axs:
        ax.hist(ak.flatten(data.ph2_simhit_rdphi[mask]), bins=bins)
        ax.set_xlabel("r * dphi(hit, sim. hit) [cm]")
        ax.set_ylabel("Hits (ph2_*)")
        ax.set_title("Hits with associated sim track")
        ax.tick_params(right=True, top=True)
    axs[1].semilogy()

plot()


# In[213]:


def plot():
    bins = np.arange(0, 5, 0.1)
    mask = (data.ph2_simhit_pt > 0.6)
    fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
    fig.subplots_adjust(wspace=0.25)
    axs[0].hist(ak.flatten(data.ph2_simhit_pt[mask]), bins=bins)
    axs[1].hist(ak.flatten(data.ph2_simhit_p[mask]), bins=bins)
    for ax in axs:
        ax.set_ylabel("Hits (ph2_*)")
        ax.tick_params(right=True, top=True)
    axs[0].set_xlabel("$p_{T}$ [GeV]")
    axs[1].set_xlabel("$p$ [GeV]")
    axs[0].set_title("Hits matched to sim. track with $p_{T}$ > 0.6 GeV")
    axs[1].set_title("Hits matched to sim. track with $p_{T}$ > 0.6 GeV")

plot()


# ## Plotting ph2_clustSize

# In[214]:


#mask = data.ph2_pt > 0.6
#bins = np.arange(-0.5, 34.5, 1)
#bin_centers = (bins[:-1] + bins[1:]) / 2.0


# In[215]:


def plot():
    bins = np.arange(-0.5, 34.5, 1)
    fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
    for ax in axs:
        ax.hist(ak.flatten(data.ph2_clustSize), bins=bins)
        ax.set_title("All hits")
        ax.set_xlabel("Cluster size")
        ax.set_ylabel("Hits (ph2_*)")
        ax.tick_params(right=True, top=True)
    axs[1].semilogy()

plot()


# In[216]:


def plot():
    bins = np.arange(-0.5, 34.5, 1)
    mask = (data.ph2_simhit_pt > 0.6)
    fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
    for ax in axs:
        ax.hist(ak.flatten(data.ph2_clustSize[mask]), bins=bins)
        ax.set_title("Hits matched to sim. track with $p_{T}$ > 0.6 GeV")
        ax.set_xlabel("Cluster size")
        ax.set_ylabel("Hits (ph2_*)")
        ax.tick_params(right=True, top=True)
    #axs[1].set_ylim([1, None])
    axs[1].semilogy()

plot()


# In[217]:


def plot():
    mask = (data.ph2_simhit_pt > 0.6)
    fig, axs = plt.subplots(ncols=3, figsize=(14, 4))
    bins = [np.arange(-0.5, 199.5, 4), np.arange(-0.5, 14.5)]
    for it, ax in enumerate(axs):
        _, _, _, im = ax.hist2d(ak.flatten(data.ph2_simhit_pt[mask]).to_numpy(),
                                ak.flatten(data.ph2_clustSize[mask]).to_numpy(),
                                norm=(mpl.colors.LogNorm() if it==2 else None),
                                bins=bins, cmin=0.5, cmap="inferno")
        ax.set_xlabel("Sim. $p_{T}$ [GeV]")
        ax.set_ylabel("Cluster size")
        ax.tick_params(right=True, top=True)
    axs[1].set_xlim([0.4, None])
    axs[1].semilogx()

plot()


# In[218]:


def plot():
    mask = (data.ph2_simhit_pt > 0.6)
    bins = np.arange(-0.5, 19.5, 1)
    hist, bin_edges = np.histogram(ak.flatten(data.ph2_clustSize[mask]), bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    cdf = np.cumsum(hist * np.diff(bin_edges))
    
    fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
    fig.subplots_adjust(wspace=0.25)
    for ax in axs:
        ax.plot(bin_centers, cdf, marker=".")
        ax.set_title("Hits matched to sim. track with $p_{T}$ > 0.6 GeV")
        ax.set_xlabel("Cluster size")
        ax.set_ylabel("CDF")
        ax.tick_params(right=True, top=True)
        ax.grid()
    axs[1].set_ylim([0.99, 1.0])

plot()


# In[219]:


def plot():
    mask_barrelFlat = (data.ph2_simhit_pt > 0.6) & (data.ph2_isBarrelFlat)
    bins = np.arange(-0.5, 19.5, 1)
    hist, bin_edges = np.histogram(ak.flatten(data.ph2_clustSize[mask_barrelFlat]), bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    cdf = np.cumsum(hist * np.diff(bin_edges))
    
    fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
    fig.subplots_adjust(wspace=0.25)
    for ax in axs:
        ax.plot(bin_centers, cdf, marker=".")
        ax.set_title("Barrel (flat) hits where $p_{T}$ > 0.6 GeV")
        ax.set_xlabel("Cluster size")
        ax.set_ylabel("CDF")
        ax.tick_params(right=True, top=True)
        ax.grid()
    axs[1].set_ylim([0.99, 1.0])

plot()


# In[220]:


def plot():
    mask_barrelTilt = (data.ph2_simhit_pt > 0.6) & (data.ph2_isBarrelTilt)
    bins = np.arange(-0.5, 19.5, 1)
    hist, bin_edges = np.histogram(ak.flatten(data.ph2_clustSize[mask_barrelTilt]), bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    cdf = np.cumsum(hist * np.diff(bin_edges))
    
    fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
    fig.subplots_adjust(wspace=0.25)
    for ax in axs:
        ax.plot(bin_centers, cdf, marker=".")
        ax.set_title("Barrel (tilt) hits where $p_{T}$ > 0.6 GeV")
        ax.set_xlabel("Cluster size")
        ax.set_ylabel("CDF")
        ax.tick_params(right=True, top=True)
        ax.grid()
    axs[1].set_ylim([0.99, 1.0])

plot()


# In[221]:


def plot():
    mask_endcap = (data.ph2_simhit_pt > 0.6) & (data.ph2_isEndcap)
    bins = np.arange(-0.5, 19.5, 1)
    hist, bin_edges = np.histogram(ak.flatten(data.ph2_clustSize[mask_endcap]), bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    cdf = np.cumsum(hist * np.diff(bin_edges))

    fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
    fig.subplots_adjust(wspace=0.25)
    for ax in axs:
        ax.plot(bin_centers, cdf, marker=".")
        ax.set_title("Endcap hits where $p_{T}$ > 0.6 GeV")
        ax.set_xlabel("Cluster size")
        ax.set_ylabel("CDF")
        ax.tick_params(right=True, top=True)
        ax.grid()
    axs[1].set_ylim([0.99, 1.0])

plot()


# In[222]:


def plot():
    bins = np.arange(-0.5, 19.5, 1)
    mask_lowpt = (data.ph2_simhit_pt > 0.6) & (data.ph2_simhit_pt < 2)
    hist, bin_edges = np.histogram(ak.flatten(data.ph2_clustSize[mask_lowpt]), bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    cdf = np.cumsum(hist * np.diff(bin_edges))

    fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
    fig.subplots_adjust(wspace=0.25)
    for ax in axs:
        ax.plot(bin_centers, cdf, marker=".")
        ax.set_title("Hits where 0.6 < $p_{T}$ < 2 GeV")
        ax.set_xlabel("Cluster size")
        ax.set_ylabel("CDF")
        ax.tick_params(right=True, top=True)
        ax.grid()
    axs[1].set_ylim([0.99, 1.0])

plot()


# In[223]:


def plot():
    mask_medpt = (data.ph2_simhit_pt > 2) & (data.ph2_simhit_pt < 5)
    bins = np.arange(-0.5, 19.5, 1)
    hist, bin_edges = np.histogram(ak.flatten(data.ph2_clustSize[mask_medpt]), bins=bins, density=True)
    cdf = np.cumsum(hist * np.diff(bin_edges))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
    fig.subplots_adjust(wspace=0.25)
    for ax in axs:
        ax.plot(bin_centers, cdf, marker=".")
        ax.set_title("Hits where 2 < $p_{T}$ < 5 GeV")
        ax.set_xlabel("Cluster size")
        ax.set_ylabel("CDF")
        ax.tick_params(right=True, top=True)
        ax.grid()
    axs[1].set_ylim([0.99, 1.0])

plot()


# In[224]:


def plot():
    mask_hipt = (data.ph2_simhit_pt > 5)
    bins = np.arange(-0.5, 19.5, 1)
    hist, bin_edges = np.histogram(ak.flatten(data.ph2_clustSize[mask_hipt]), bins=bins, density=True)
    cdf = np.cumsum(hist * np.diff(bin_edges))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
    fig.subplots_adjust(wspace=0.25)
    for ax in axs:
        ax.plot(bin_centers, cdf, marker=".")
        ax.set_title("Hits where $p_{T}$ > 5 GeV")
        ax.set_xlabel("Cluster size")
        ax.set_ylabel("CDF")
        ax.tick_params(right=True, top=True)
        ax.grid()
    axs[1].set_ylim([0.99, 1.0])

plot()


# In[225]:


def plot():
    mask = (data.ph2_simhit_pt > 0.6)
    bins = [np.arange(-0.001, 0.5, 0.005),
    np.arange(-0.5, 19.5, 1)]
    fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
    fig.subplots_adjust(wspace=0.25)
    for ax in axs:
        ax.hist2d(ak.flatten(data.ph2_simhit_rdphi[mask]).to_numpy(),
                  ak.flatten(data.ph2_clustSize[mask]).to_numpy(),
                  norm=mpl.colors.LogNorm(),
                  bins=bins,
                  )
        ax.set_xlabel("r * dphi(hit, sim. hit) [cm]")
        ax.set_ylabel("Cluster size")
        ax.set_title("Hits with associated sim track")
        ax.tick_params(right=True, top=True)
    axs[1].set_xlim([0.001, None])
    axs[1].semilogx()

plot()


# In[226]:


def plot():
    masks = [
        (data.ph2_simhit_pt > 0.6) & (data.ph2_isBarrelFlat),
        (data.ph2_simhit_pt > 0.6) & (data.ph2_isBarrelTilt),
        (data.ph2_simhit_pt > 0.6) & (data.ph2_isEndcap),
    ]
    titles = [
        "Barrel (flat)",
        "Barrel (tilt)",
        "Endcap",
    ]
    bins = [
        # np.arange(-0.001, 0.5, 0.005),
        np.arange(-0.001, 0.2, 0.005),
        np.arange(-0.5, 19.5, 1),
    ]
    fig, axs = plt.subplots(nrows=3, figsize=(8, 20))
    fig.subplots_adjust(wspace=0.25, hspace=0.2)
    for mask, title, ax in zip(masks, titles, axs):
        _, _, _, im = ax.hist2d(ak.flatten(data.ph2_simhit_rdphi[mask]).to_numpy(),
                                ak.flatten(data.ph2_clustSize[mask]).to_numpy(),
                                norm=mpl.colors.LogNorm(),
                                bins=bins,
                      )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Hits (ph2_*)")
        ax.set_xlabel("r * dphi(hit, sim. hit) [cm]")
        ax.set_ylabel("Cluster size")
        ax.set_title(f"{title} hits with sim track")
        ax.tick_params(right=True, top=True)
    #axs[1].set_xlim([0.001, None])
    #axs[1].semilogx()

plot()


# In[231]:


def plot():
    with PdfPages("cluster_size.per_layer.pdf") as pdf:

        pt = 0.6
        n_layers = 6

#        baseline_mask = (data.ph2_simhit_cosphi > 0.15)
#        baseline_mask = (data.ph2_nsimhit > 1)
#        (data.ph2_nsimhit == 1) & \

        baseline_mask = \
        (data.ph2_simhit_pt > pt) & \
        (data.ph2_simhit_p > 0.5 * data.ph2_simtrk_p) & \
        (data.ph2_simhit_cosphi > 0.15)
        masks = [
            baseline_mask & (data.ph2_isBarrelFlat),
            baseline_mask & (data.ph2_isBarrelTilt),
            baseline_mask & (data.ph2_isEndcap),
        ]
        titles = [
            "Barrel (flat)",
            "Barrel (tilt)",
            "Endcap",
        ]

        #
        # cos(dphi)
        #
        bins = np.arange(-1.1, 1.10, 0.01)
        for mask, title in zip(masks, titles):
            # continue
            for layer in range(n_layers+1):

                mask_layer = mask if layer == 0 else (mask & (data.ph2_layer == layer))
                total = ak.sum(mask_layer)

                # cos(dphi)
                fig, ax = plt.subplots(figsize=(8, 8))
                _ = ax.hist(ak.flatten(data.ph2_simhit_cosphi[mask_layer]), bins=bins)
                if total > 0:
                    ax.semilogy()
                    ax.set_ylim([0.9, None])
                ax.set_xlabel("cos($\\phi$) of $p_{T, sim}$ and $x_{T, sim}$")
                ax.set_ylabel("Hits (ph2_*)")
                ax.set_title(f"{title} hits, layer {layer or 'inclusive'}, with sim. track > {pt} GeV")
                ax.tick_params(right=True, top=True)

                pdf.savefig()
                plt.close()


        #
        # sim track p vs sim hit p
        #
        bins = [
            np.arange(-10, 210, 2),
            np.arange(-10, 210, 2),
        ]
        for mask, title in zip(masks, titles):
            # continue
            for layer in range(n_layers+1):

                mask_layer = mask if layer == 0 else (mask & (data.ph2_layer == layer))
                total = ak.sum(mask_layer)

                # cluster size vs. r*dphi
                fig, ax = plt.subplots(figsize=(8, 8))
                _, _, _, im = ax.hist2d(ak.flatten(data.ph2_simtrk_p[mask_layer]).to_numpy(),
                                        ak.flatten(data.ph2_simhit_p[mask_layer]).to_numpy(),
                                        #norm=mpl.colors.LogNorm() if total > 0 else None,
                                        bins=bins,
                                        cmin=0.5,
                              )
                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label("Hits (ph2_*)")
                ax.set_xlabel("Sim. track $p$ [GeV]")
                ax.set_ylabel("Sim. hit $p$ [GeV]")
                ax.set_title(f"{title} hits, layer {layer or 'inclusive'}, with sim. track > {pt} GeV")
                ax.tick_params(right=True, top=True)

                pdf.savefig()
                plt.close()


        
        #
        # cluster size vs. cos(dphi)
        #
        bins = [
            np.arange(0.2, 1.01, 0.01),
            np.arange(-0.5, 19.5, 1),
        ]
        for mask, title in zip(masks, titles):
            # continue
            for layer in range(n_layers+1):

                mask_layer = mask if layer == 0 else (mask & (data.ph2_layer == layer))
                total = ak.sum(mask_layer)

                fig, ax = plt.subplots(figsize=(8, 8))
                _, _, _, im = ax.hist2d(ak.flatten(data.ph2_simhit_cosphi[mask_layer]).to_numpy(),
                                        ak.flatten(data.ph2_clustSize[mask_layer]).to_numpy(),
                                        norm=mpl.colors.LogNorm() if total > 0 else None,
                                        cmin=0.5,
                                        bins=bins,
                              )
                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label("Hits (ph2_*)")
                ax.set_xlabel("cos($\\phi$) of $p_{T, sim}$ and $x_{T, sim}$")
                ax.set_ylabel("Cluster size")
                ax.set_title(f"{title} hits, layer {layer or 'inclusive'}, with sim. track > {pt} GeV")
                ax.tick_params(right=True, top=True)

                pdf.savefig()
                plt.close()

        #
        # cluster size vs. r*dphi
        #
        bins = [
            np.arange(-0.001, 0.2, 0.005),
            np.arange(-0.5, 19.5, 1),
        ]
        for mask, title in zip(masks, titles):
            # continue
            for layer in range(n_layers+1):

                mask_layer = mask if layer == 0 else (mask & (data.ph2_layer == layer))
                total = ak.sum(mask_layer)

                fig, ax = plt.subplots(figsize=(8, 8))
                _, _, _, im = ax.hist2d(ak.flatten(data.ph2_simhit_rdphi[mask_layer]).to_numpy(),
                                        ak.flatten(data.ph2_clustSize[mask_layer]).to_numpy(),
                                        norm=mpl.colors.LogNorm() if total > 0 else None,
                                        cmin=0.5,
                                        bins=bins,
                              )
                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label("Hits (ph2_*)")
                ax.set_xlabel("r * dphi(hit, sim. hit) [cm]")
                ax.set_ylabel("Cluster size")
                ax.set_title(f"{title} hits, layer {layer or 'inclusive'}, with sim. track > {pt} GeV")
                ax.tick_params(right=True, top=True)

                pdf.savefig()
                plt.close()

        #
        # cluster size CDF
        #
        bins = np.arange(-0.5, 19.5, 1)

        for mask, title in zip(masks, titles):
            # continue
            for layer in range(n_layers+1):

                mask_layer = mask if layer == 0 else (mask & (data.ph2_layer == layer))
                total = ak.sum(mask_layer)
        
                hist, bin_edges = np.histogram(ak.flatten(data.ph2_clustSize[mask_layer]), bins=bins, density=(total > 0))
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

                cdf = np.cumsum(hist * np.diff(bin_edges))

                fig, axs = plt.subplots(ncols=2, figsize=(12, 4))
                fig.subplots_adjust(wspace=0.25)
                for ax in axs:
                    ax.plot(bin_centers, cdf, marker=".")
                    ax.set_xlabel("Cluster size")
                    ax.set_ylabel("CDF")
                    ax.set_title(f"{title} hits, layer {layer or 'inclusive'}, with sim. track > {pt} GeV")
                    ax.tick_params(right=True, top=True)
                    ax.grid()
                axs[1].set_ylim([0.99, 1.0])

                pdf.savefig()
                plt.close()
        

plot()
print("boop")


# ## Comparing cluster size with number of sim hits

# In[228]:


def plot():
    bins = np.arange(-0.5, 9.5, 1)

    layer = 6
    mask_0 = (data.ph2_nsimhit == 0) & (data.ph2_simtrk_pt > 0.6) & (data.ph2_isBarrelFlat) # & (data.ph2_layer == layer)
    mask_1 = (data.ph2_nsimhit == 1) & (data.ph2_simtrk_pt > 0.6) & (data.ph2_isBarrelFlat) # & (data.ph2_layer == layer)
    mask_2 = (data.ph2_nsimhit == 2) & (data.ph2_simtrk_pt > 0.6) & (data.ph2_isBarrelFlat) # & (data.ph2_layer == layer)
    mask_3 = (data.ph2_nsimhit == 3) & (data.ph2_simtrk_pt > 0.6) & (data.ph2_isBarrelFlat) # & (data.ph2_layer == layer)
    mask_4 = (data.ph2_nsimhit >= 4) & (data.ph2_simtrk_pt > 0.6) & (data.ph2_isBarrelFlat) # & (data.ph2_layer == layer)
    
    fig, axs = plt.subplots(ncols=2, figsize=(10, 4))

    # axs[0].hist(ak.flatten(data.ph2_nsimhit[mask_0]), bins=bins, label="0")
    axs[0].hist(ak.flatten(data.ph2_nsimhit[mask_1]), bins=bins, label="1")
    axs[0].hist(ak.flatten(data.ph2_nsimhit[mask_2]), bins=bins, label="2")
    axs[0].hist(ak.flatten(data.ph2_nsimhit[mask_3]), bins=bins, label="3")
    axs[0].hist(ak.flatten(data.ph2_nsimhit[mask_4]), bins=bins, label="4+")
    axs[0].set_xlabel("Number of sim. hits")
    axs[0].semilogy()
    axs[0].set_ylim([0.9, None])
    axs[0].tick_params(right=True, top=True)

    # axs[1].hist(ak.flatten(data.ph2_clustSize[mask_0]), bins=bins, linewidth=2, histtype="step", density=True, label="0")
    axs[1].hist(ak.flatten(data.ph2_clustSize[mask_1]), bins=bins, linewidth=2, histtype="step", density=True, label="N(sim.) = 1")
    axs[1].hist(ak.flatten(data.ph2_clustSize[mask_2]), bins=bins, linewidth=2, histtype="step", density=True, label="2")
    axs[1].hist(ak.flatten(data.ph2_clustSize[mask_3]), bins=bins, linewidth=2, histtype="step", density=True, label="3")
    axs[1].hist(ak.flatten(data.ph2_clustSize[mask_4]), bins=bins, linewidth=2, histtype="step", density=True, label="4+")
    axs[1].set_xlabel("Cluster size")
    axs[1].set_ylabel("Hits (ph2_*) normalized to 1")
    axs[1].legend()
    # axs[1].semilogy()
    axs[1].tick_params(right=True, top=True)

plot()
# print(data.ph2_nsimhit)


# ## Comparing simhit with simtrk

# In[229]:


def plot():
    bins = np.arange(200)
    fig, axs = plt.subplots(ncols=2, figsize=(14, 6))
    fig.subplots_adjust(wspace=0.25)
    cbars = [None, None]
    mask = (data.ph2_simhit_pt > 0.6) & (data.ph2_simtrk_pt > 0.6)
    print(ak.sum(mask))

    _, _, _, im = axs[0].hist2d(ak.flatten(data.ph2_simtrk_p[mask]).to_numpy(),
                                ak.flatten(data.ph2_simhit_p[mask]).to_numpy(),
                                bins=[bins, bins],
                                cmin=0.5,
                                )
    cbars[0] = fig.colorbar(im, ax=axs[0])
    cbars[0].set_label("Hits (ph2_*)")
    axs[0].set_xlabel("Sim. track $p$ [GeV]")
    axs[0].set_ylabel("Sim. hit $p$ [GeV]")

    _, _, _, im = axs[1].hist2d(ak.flatten(data.ph2_simtrk_pt[mask]).to_numpy(),
                                ak.flatten(data.ph2_simhit_pt[mask]).to_numpy(),
                                bins=[bins, bins],
                                cmin=0.5,
                                )
    cbars[1] = fig.colorbar(im, ax=axs[1])
    cbars[1].set_label("Hits (ph2_*)")
    axs[1].set_xlabel("Sim. track $p_{T}$ [GeV]")
    axs[1].set_ylabel("Sim. hit $p_{T}$ [GeV]")

    for ax in axs:
        ax.tick_params(right=True, top=True)            

plot()


# ## Checking sim_pdgId

# In[230]:


def plot():
    bins = np.arange(-220, 220)
    fig, ax = plt.subplots()
    ax.hist(ak.flatten(data.sim_pdgId), bins=bins)
    ax.set_xlabel("Sim. PDG ID")
    ax.set_ylabel("Number of sim. particles")
    ax.tick_params(right=True, top=True)
    ax.semilogy()

plot()


# In[ ]:




