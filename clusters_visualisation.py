#%%  importations

from scipy.io import loadmat
from scipy.io import savemat
from scipy import sparse
import numpy as np
import mne
from mne.time_frequency import tfr_morlet
import os
import fnmatch
import copy
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

#%% collect file names
filenames = []
for filename in os.listdir('SCdataNew/'):
    if fnmatch.fnmatch(filename, '[0-9A-Z]*.edf'):
        filenames.append(filename[0:-4])

print(len(filenames))
print(filenames)

#%% load the results
#4: lots of clusters, threshold t=5
#5: threshold with t=0.5, few clusters
#6: threshold with t=1, more clusters
#7: threshold with t=25
#evoked_1: evoked without threshold
#evoked_2: evoked threshold t=0.5
#evoked_3: evoked threshold t=10
#evoked_unified_1: without threshold
#evoked_unified_2: threshold t=5
with open('cluster_tests_evoked_unified_2.pickle', 'rb') as handle:
    results = pickle.load(handle)

#%% global variables needed later on
event_id = { "Visual 'b' high":1,"Visual 'b' medium":2, "Visual 'b' low":3,
"Visual 'g' high":4, "Visual 'g' medium":5, "Visual 'g' low":6,
"Audio 'b' high":7, "Audio 'b' medium":8, "Audio 'b' low":9,
"Audio 'g' high":10, "Audio 'g' medium":11, "Audio 'g' low":12,
"AudioVisual 'b' high":13,"AudioVisual 'b' medium":14,"AudioVisual 'b' low":15,
"AudioVisual 'g' high":16,"AudioVisual 'g' medium":17,"AudioVisual 'g' low":18,
"AudioVisual fusion sync high":19, "AudioVisual fusion sync medium":20,
"AudioVisual fusion sync low":21, "AudioVisual fusion Async high":22,
"AudioVisual fusion Async medium":23, "AudioVisual fusion Async low":24}

combinations_3 = { #"fusion_vg_high" : [4,7,19], "fusion_vg_medium" : [5,8,20], "fusion_vg_low" : [6,9,21],
"av_b_high" : [1,7,13], "av_b_medium" : [2,7,14], "av_b_low" : [3,7,15],
"av_g_high" : [4,10,16], "av_g_medium" : [5,10,17], "av_g_low" : [6,10,18]
}

combinations_2 = {"visual_low" : [3,6], "audio_low": [7,10], "av_b_hm" : [13,14], "av_b_ml": [14,15],
"av_b_hl": [13,15], "av_g_hm": [16,17], "av_g_ml": [17,18], "av_g_hl": [16,18]}

freqs = np.array([4,6,8,10,12,14,16])
n_cycles = 4  
idx_ch = [0,1,2,3,4,33,36]
#%% show combin_2
for combin in combinations_2:
    print(combin)
    time = []
    freq = []
    channels = []
    c = []
    v = []
    F_obs = results[combin][0]
    clusters = results[combin][1]
    count = sum(map(lambda p : p<0.05, results[combin][2]))
    print("nr. clusters with p < 0.05: ",count,"/",len(results[combin][2]))
    count = sum(map(lambda p : p<0.01, results[combin][2]))
    print("nr. clusters with p < 0.01: ",count,"/",len(results[combin][2]))

    nr = []
    for j in range(len(results[combin][1])):
        nr.append(len(results[combin][1][j][0]))
    for i in range(len(clusters)):
        if results[combin][2][i] < 0.05:
            print("cluster nr. : ",i)
            print("data points in the cluster: ", len(results[combin][1][i][0]),"/",sum(nr),"(",len(results[combin][1][i][0])/sum(nr)*100,"%)")
            time = (results[combin][1][i][0]*(1.676+0.52)/184-0.52)*1000
            freq = results[combin][1][i][1]*2+4
            channels = results[combin][1][i][2]+1
            c = [i for j in range(len(results[combin][1][i][0]))]

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(time,freq,channels, marker="o", c=c, s=1, cmap="RdBu")
            ax.set_xlabel('time [ms]')
            ax.set_ylabel('fequency [Hz]')
            ax.set_zlabel('channels')
            plt.show()

            print("time: \t\tmin:", min(time),"; \tmax:", max(time))
            print("freq: \t\tmin:", min(freq),"; \tmax:", max(freq))
            print("channels: \tmin:", min(channels),"; \tmax:", max(channels))
    print()
    print()

#%% show combin_3
for combin in combinations_3:
    print(combin)
    time = []
    freq = []
    channels = []
    c = []
    v = []
    F_obs = results[combin][0]
    clusters = results[combin][1]
    count = sum(map(lambda p : p<0.05, results[combin][2]))
    print("nr. clusters with p < 0.05: ",count,"/",len(results[combin][2]))
    count = sum(map(lambda p : p<0.01, results[combin][2]))
    print("nr. clusters with p < 0.01: ",count,"/",len(results[combin][2]))

    for i in range(len(clusters)):
        nr = []
        for j in range(len(results[combin][1])):
            nr.append(len(results[combin][1][j][0]))
        if results[combin][2][i] < 0.05:
            print("cluster nr. : ",i)
            print("data points in the cluster: ", len(results[combin][1][i][0]),"/",sum(nr),"(",len(results[combin][1][i][0])/sum(nr)*100,"%)")
            time = (results[combin][1][i][0]*(1.676+0.52)/184-0.52)*1000
            freq = results[combin][1][i][1]*2+4
            channels = results[combin][1][i][2]+1
            c = [i for j in range(len(results[combin][1][i][0]))]

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(time,freq,channels, marker="o", c=c, s=1, cmap="RdBu")
            ax.set_xlabel('time [ms]')
            ax.set_ylabel('fequency [Hz]')
            ax.set_zlabel('channels')
            plt.show()

            print("time: \t\tmin:", min(time),"; \tmax:", max(time))
            print("freq: \t\tmin:", min(freq),"; \tmax:", max(freq))
            print("channels: \tmin:", min(channels),"; \tmax:", max(channels))
    print()
    print()
# %%
_ = plt.scatter(time,time)#, 'ro')
# %%
results[combin][1][1][0]
#%%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#x = np.array(x)
#y = np.array(y)
#z = np.array(z)

ax.scatter(time,freq,channels, marker="o", c=c, s=1, cmap="RdBu")
ax.set_xlabel('time [ms]')
ax.set_ylabel('fequency [Hz]')
ax.set_zlabel('channels')
plt.show()
# %%
print("time: ", min(time), max(time))
print("freq: ", min(freq), max(freq))
print("channels: ", min(channels), max(channels))
# %%
results[combin][2][1]
# %%
combinations_2|combinations_3
# %%
