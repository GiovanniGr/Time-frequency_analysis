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

#%% global variables needed later on

#event_id is a dictionary that stores all the event numbers
event_id = { "Visual 'b' high":1,"Visual 'b' medium":2, "Visual 'b' low":3,
"Visual 'g' high":4, "Visual 'g' medium":5, "Visual 'g' low":6,
"Audio 'b' high":7, "Audio 'b' medium":8, "Audio 'b' low":9,
"Audio 'g' high":10, "Audio 'g' medium":11, "Audio 'g' low":12,
"AudioVisual 'b' high":13,"AudioVisual 'b' medium":14,"AudioVisual 'b' low":15,
"AudioVisual 'g' high":16,"AudioVisual 'g' medium":17,"AudioVisual 'g' low":18,
"AudioVisual fusion sync high":19, "AudioVisual fusion sync medium":20,
"AudioVisual fusion sync low":21, "AudioVisual fusion Async high":22,
"AudioVisual fusion Async medium":23, "AudioVisual fusion Async low":24}

#combinations_3 stores the indexes of the signals to combine
combinations_3 = { #"fusion_vg_high" : [4,7,19], "fusion_vg_medium" : [5,8,20], "fusion_vg_low" : [6,9,21],
"av_b_high" : [1,7,13], "av_b_medium" : [2,7,14], "av_b_low" : [3,7,15],
"av_g_high" : [4,10,16], "av_g_medium" : [5,10,17], "av_g_low" : [6,10,18]
}

#combinations_2 as well but for two signals
combinations_2 = {"visual_low" : [3,6], "audio_low": [7,10], "av_b_hm" : [13,14], "av_b_ml": [14,15],
"av_b_hl": [13,15], "av_g_hm": [16,17], "av_g_ml": [17,18], "av_g_hl": [16,18]}

freqs = np.array([4,6,8,10,12,14,16])
n_cycles = 4  
idx_ch = [0,1,2,3,4,33,36]
#%% collect the power per id per person
power_per_person_per_id = {}
n=0
for filename in filenames:
    print(filename, ", ", n,"/",len(filenames))
    n+=1
    power_per_person_per_id[filename] = mne.time_frequency.read_tfrs('./SCdataPowers_Evoked_unified/'+filename+'_id-tfr.h5')
#%% save the powers not only for person but also for id
power_per_person_per_id_dict = {}
for filename in filenames:
    power_per_person_per_id_dict[filename] = {}
    for i in range(len(power_per_person_per_id[filename])):
        power_per_person_per_id_dict[filename][i+1] = power_per_person_per_id[filename][i]
#%% collect ind per id per person
comb_per_person_per_id = {}
for filename in filenames:
    print(filename)
    comb_per_person_per_id[filename] = {}
    infile = open('./SCdata_evoked_Ep_unified/'+filename+'.pkl','rb')
    induc_per_id = pickle.load(infile)
    infile.close()
    keys = list(induc_per_id.keys())
    for comb in combinations_3:
        combination = mne.combine_evoked([induc_per_id[keys[combinations_3[comb][0]-1]],induc_per_id[keys[combinations_3[comb][1]-1]]],[1,1])
        comb_per_person_per_id[filename][combinations_3[comb][0]] = tfr_morlet(combination, freqs=freqs, n_cycles=n_cycles, use_fft=True, decim=3, n_jobs=1, average=True, return_itc=False)

#%% create the channel adjacency matrix 
montage = loadmat("montage.mat")
ch_names = power_per_person_per_id[filename][0].ch_names
diction_ch_positions = {k:v for k,v in zip(ch_names,montage["montage"])}

info = mne.create_info(ch_names, 250, 'eeg')
montage = mne.channels.make_dig_montage(diction_ch_positions)
info.set_montage(montage)
adj_matrix = mne.channels.find_ch_adjacency(info,"eeg")
plt.imshow(adj_matrix[0].toarray(), cmap='gray', origin='lower',
           interpolation='nearest')
plt.xlabel('{} sensors'.format(len(ch_names)))
plt.ylabel('{} sensors'.format(len(ch_names)))
plt.title('Between-sensor adjacency')

#%%build the adjacency for each feature
chan_adjacency = adj_matrix[0]
adjacency = mne.stats.combine_adjacency(184, 7, chan_adjacency) #freq_adjacency, chan_adjacency)
print(np.swapaxes(power_per_person_per_id[filename][18].data,0,2).shape)
print(adjacency.shape)
#%% run the permutation_cluster_test for each combination of conditions
results = {}
n=1
for combin in combinations_3:
    print(n,"/",len(list(combinations_3.keys())))
    n+=1
    X1 = []
    X2 = [] 
    for filename in power_per_person_per_id_dict:
        X1.append(np.swapaxes(power_per_person_per_id_dict[filename][combinations_3[combin][2]].data,0,2))
        X2.append(np.swapaxes(comb_per_person_per_id[filename][combinations_3[combin][0]].data,0,2))
    X1 = np.array(X1)
    X2 = np.array(X2)

    F_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_test(X=[X1, X2], adjacency=adjacency, threshold=5)
    results[combin] = [F_obs, clusters, cluster_pv, H0]
n=1
for combin in combinations_2:
    print(n,"/",len(list(combinations_2.keys())))
    n+=1
    X1 = []
    X2 = [] 
    for filename in power_per_person_per_id:
        X1.append(np.swapaxes(power_per_person_per_id[filename][combinations_2[combin][0]].data,0,2))
        X2.append(np.swapaxes(power_per_person_per_id[filename][combinations_2[combin][1]].data,0,2))
    X1 = np.array(X1)
    X2 = np.array(X2)

    F_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_test(X=[X1, X2], adjacency=adjacency, threshold=5)
    results[combin] = [F_obs, clusters, cluster_pv, H0]
#%% save the results
#4: lots of clusters, threshold t=5
#5: threshold with t=0.5, few clusters
#6: threshold with t=1, more clusters
#7: threshold with t=25
#evoked_1: evoked without threshold selected
#evoked_2: evoked threshold t=0.5
#evoked_3: evoked threshold t=10
#evoked_unified_1: without threshold selected
#evoked_unified_2: threshold t=5

with open('cluster_tests_evoked_unified_2.pickle', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
#%% load the results
with open('cluster_tests_evoked_unified_1.pickle', 'rb') as handle:
    results = pickle.load(handle)

'''
#%% statistics on the clusters
combin = "fusion_vg_low"
print("dimensions F_obs:")
print("1st dim, time:\t\t",len(results[combin][0]))
print("2nd dim, freq:\t\t",len(results[combin][0][0]))
print("3rd dim, channels:\t",len(results[combin][0][0][0]))

print("\ndimensions clusters:")
print("1st dim, nr. clusters:\t",len(results[combin][1]))
print("2nd dim, x,y,z:\t\t",len(results[combin][1][0]))
print("3rd dim, points:\t",len(results[combin][1][0][0]))

print("\ndimensions cluster_pv:")
print("1st dim, nr. clusters:\t",len(results[combin][2]))

print("\ndimensions H0:")
print("1st dim, nr. iterations:",len(results[combin][3]))
'''
#%% show the results of the permutation test for combination_3
for combin in combinations_3:
    print(combin)
    count = sum(map(lambda p : p<0.05, results[combin][2]))
    print("nr. clusters with p < 0.05: ",count,"/",len(results[combin][2]))
    count = sum(map(lambda p : p<0.01, results[combin][2]))
    print("nr. clusters with p < 0.01: ",count,"/",len(results[combin][2]))
    print("nr. elements per cluster:")
    nr = []
    for i in range(len(results[combin][1])):
        nr.append(len(results[combin][1][i][0]))
    if(len(nr)>0):
        print("dim. biggest / dim. total: ",max(nr),"/",sum(nr)," = ",max(nr)/sum(nr)*100,"%")
        print("total points: ", sum(nr))
    print()
#%% show the results of the permutation test for combination_2
for combin in combinations_2:
    print(combin)
    count = sum(map(lambda p : p<0.05, results[combin][2]))
    print("nr. clusters with p < 0.05: ",count,"/",len(results[combin][2]))
    count = sum(map(lambda p : p<0.01, results[combin][2]))
    print("nr. clusters with p < 0.01: ",count,"/",len(results[combin][2]))
    nr = []
    for i in range(len(results[combin][1])):
        nr.append(len(results[combin][1][i][0]))
    if(len(nr)>0):
        print("dim. biggest / dim. total: ",max(nr),"/",sum(nr)," = ",max(nr)/sum(nr)*100,"%")
        print("total points: ", sum(nr))
    print()
