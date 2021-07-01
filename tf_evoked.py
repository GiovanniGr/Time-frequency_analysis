#%%  importations

from scipy.io import loadmat #to load the .mat files from matlab
import numpy as np
import mne #eeg analysis library
from mne.time_frequency import tfr_morlet
import os #to acquire the filenames
import fnmatch
import pickle #to save the results

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
"Audio 'b' high":7,# "Audio 'b' medium":8, "Audio 'b' low":9,
"Audio 'g' high":10,# "Audio 'g' medium":11, "Audio 'g' low":12,
"AudioVisual 'b' high":13,"AudioVisual 'b' medium":14,"AudioVisual 'b' low":15,
"AudioVisual 'g' high":16,"AudioVisual 'g' medium":17,"AudioVisual 'g' low":18,
"AudioVisual fusion sync high":19, "AudioVisual fusion sync medium":20,
"AudioVisual fusion sync low":21, "AudioVisual fusion Async high":22,
"AudioVisual fusion Async medium":23, "AudioVisual fusion Async low":24
}

#these variables are used for the power computation
freqs = np.array([4,6,8,10,12,14,16])
n_cycles = 4  
idx_ch = [0,1,2,3,4,33,36] #ROI

#%% create the info
ch_names = []
for i in range(1,10):
    ch_names.append("e0"+str(i))
for i in range(10,127):
    ch_names.append("e"+str(i))
info = mne.create_info(ch_names, 250, 'eeg')
#%% compute the powers starting from the Matlab trials

nr = 0 #counter
for filename in filenames: #for each person:
    print("Computing nr.",nr,": ",filename)
    nr=nr+1
    
    #create the numpy data. These are the epochs without the synchronization with an event
    data = loadmat('./SCdataTrials2/'+filename+'.mat')
    dataNp = np.empty((len(data["trial"][0]),126,550))
    for i in range(len(data["trial"][0])):
        dataNp[i] = data["trial"][0][i]/1000000
    
    # create the events
    trialtable = loadmat('./SCdataTrialtable/'+filename+'Trialtable.mat')
    eventsCodes = trialtable["trialtable"][:,0]
    eventsCodes[eventsCodes == 11] = 10
    eventsCodes[eventsCodes == 12] = 10
    eventsCodes[eventsCodes == 8] = 7
    eventsCodes[eventsCodes == 9] = 7
    times = np.arange(start=275,stop=(eventsCodes.size*550+275),step=550,dtype=int)
    events = times[:,None]
    events = np.concatenate((events, np.zeros(eventsCodes.shape)[:,None]), axis=1)
    events = np.concatenate((events, eventsCodes[:,None]), axis=1)

    # create the epochs
    epochs = mne.EpochsArray(data = dataNp, info = info, events= events.astype(int), tmin=-0.52, event_id = event_id, baseline=(0.48,0.58))
    ep = epochs.copy()
    print(type(ep))

    epochs_per_id = {}
    evok_per_id = {}
    ind_per_id = {}
    ave_ind_per_id = {}
    power_per_id = []
    for id in event_id:
        epochs_per_id[id] = epochs.copy()[id]
        evok_per_id[id] = epochs.copy()[id].average()

        #compute the tf
        power_per_id.append(tfr_morlet(evok_per_id[id], freqs=freqs, n_cycles=n_cycles, use_fft=True, decim=3, n_jobs=1, average=True, return_itc=False))

    #save the results
    outfile = open('./SCdata_evoked_Ep_unified/'+filename+'.pkl','wb')
    pickle.dump(evok_per_id,outfile)
    outfile.close()

    mne.time_frequency.write_tfrs('./SCdataPowers_Evoked_unified/'+filename+'_id-tfr.h5', power_per_id, overwrite=True)


#%% for visualisation purposes, combine the evoked of different users
ev_per_id = {}
for filename in filenames:
    print(filename)
    infile = open('./SCdata_evoked_Ep_unified/'+filename+'.pkl','rb')
    new_dict = pickle.load(infile)
    infile.close()
    for id in event_id:
        if id in ev_per_id:
            ev_per_id[id].append(new_dict[id])
        else:
            ev_per_id[id] = [new_dict[id]]

ev = {}
for event in event_id:
    ev[event] = mne.combine_evoked(ev_per_id[event], "equal")
#%% show the erp signals
a = [6,7] #choose the idx of the events to show
keys = [list(ev.keys())[x] for x in a]
print(keys)
a_subset = {key: value for key, value in ev.items() if key in keys}
audio = ["e01","e02","e03","e04","e05","e34","e37"]
mne.viz.plot_compare_evokeds(a_subset, combine="mean", picks=audio, vlines=[0,0.58], styles={key: {"linewidth" : 0.75} for key, value in ev.items() if key in keys})

#%%read the powers
power_per_person_per_id = {}
for filename in filenames:
    print(filename)
    power_per_person_per_id[filename] = mne.time_frequency.read_tfrs('./SCdataPowers_Evoked_unified/'+filename+'_id-tfr.h5')

#%% average over people, just for visualisation porpuses
power_per_id = []
for id in range(len(event_id)):
    print(id)
    aux = []
    for filename in power_per_person_per_id:
        aux.append(power_per_person_per_id[filename][id])
    power_per_id.append(mne.combine_evoked(aux,"equal")) 

mne.time_frequency.write_tfrs('./SCdataPowers_Evoked_unified/averagedOverPeople-tfr.h5', power_per_id, overwrite=True)
#%% plot the powers averaged for the people
power_per_id = mne.time_frequency.read_tfrs('./SCdataPowers_Evoked_unified/averagedOverPeople-tfr.h5')
for id in range(len(event_id)):
    power_per_id[id].plot(idx_ch,  mode='logratio', title="Power of "+list(event_id.keys())[id]+" in ROI", tmin=0.5, tmax=1.1)

