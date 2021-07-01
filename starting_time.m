%the code computes the distribution of the starting time for trial

names = dir("SCdataOrigin/*.mat");

%definition of the dictionaries to store the results
starting_time = containers.Map();
starting_time_per_person_per_cond = containers.Map();
num_sample_per_trial = containers.Map();
starting_time_after = containers.Map();
num_sample_per_trial_after = containers.Map();

%for each person
for i = 1:length(names)
    load("SCdataOrigin/"+names(i).name);
    len = length(data.time);
    
    for j = 1:len
        %extract the starting time for each trial
        key_start = num2str(data.time{1,j}(1)); 
        
        %and count the occurrencies for each starting time
        if isKey(starting_time,key_start)
            starting_time(key_start) = starting_time(key_start)+1; 
        else 
            starting_time(key_start)=1;
        end
        
        %repeat but taking also into account the different events
        key_s_p_c = strcat(num2str(data.time{1,j}(1)),",",names(i).name,",",num2str(trialtable(j,1)));
        if isKey(starting_time_per_person_per_cond,key_s_p_c)
            starting_time_per_person_per_cond(key_s_p_c) = starting_time_per_person_per_cond(key_s_p_c)+1;
        else 
            starting_time_per_person_per_cond(key_s_p_c)=1;
        end
        
        
        key_len = num2str(length(data.time{1,j}));
        if isKey(num_sample_per_trial,key_len)
            num_sample_per_trial(key_len) = num_sample_per_trial(key_len)+1;
        else 
            num_sample_per_trial(key_len)=1;
        end
        
        %crop the trials to align the times
        switch key_start
            case "-0.896"
                data.trial{1,j} = data.trial{1,j}(:,95:550);
                data.time{1,j} = data.time{1,j}(95:550);
            case "-0.8"
                data.trial{1,j} = data.trial{1,j}(:,71:526);
                data.time{1,j} = data.time{1,j}(71:526);
            case "-0.616"
                data.trial{1,j} = data.trial{1,j}(:,25:480);
                data.time{1,j} = data.time{1,j}(25:480);
            case "-0.52"
                data.trial{1,j} = data.trial{1,j}(:,1:456);
                data.time{1,j} = data.time{1,j}(1:456);
        end
        
        %check that everything is correct
        key_start = num2str(data.time{1,j}(1));
        if isKey(starting_time_after,key_start)
            starting_time_after(key_start) = starting_time_after(key_start)+1;
        else 
            starting_time_after(key_start)=1;
        end
        
        key_len = num2str(length(data.time{1,j}));
        if isKey(num_sample_per_trial_after,key_len)
            num_sample_per_trial_after(key_len) = num_sample_per_trial_after(key_len)+1;
        else 
            num_sample_per_trial_after(key_len)=1;
        end
    end
    %save the results
    trial = data.trial;
    save(strcat('SCdataTrials/',names(i).name), 'trial')
    
    
end

%print the results
keys(starting_time)
values(starting_time)

keys(starting_time_per_person_per_cond)
values(starting_time_per_person_per_cond)

keys(num_sample_per_trial)
values(num_sample_per_trial)

keys(starting_time_after)
values(starting_time_after)

keys(num_sample_per_trial_after)
values(num_sample_per_trial_after)