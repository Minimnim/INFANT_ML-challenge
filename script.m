% This script explains the steps to pre-process the data, generate features
% and seperates the data into training and validations sets, add the IDs and
% labels of each epoch, and make them ready to get used
% ---------------------------------------------------------------------------
% This code is used to preprocess the data (low- and high_pass
% filtering and downsampling) and generate features using the NEURAL
% package (O’Toole, J. M. & Boylan, G. B. NEURAL: quantitative features 
% for newborn EEG using Matlab. ArXiv E-Prints(2017).1704.05694.)
% ATTENTION: please update the path to the location that eeg.csv and all
% the codes are saved
P = 'C:\Users\PhysioUser\OneDrive - University College Cork\machine learning competition\CSV_format\CSV_format'; 
S = dir(fullfile(P,'*.csv'));
% ATTENTION: Please update it to the path the EEG grades are saved
path_to_grades = 'C:\Users\PhysioUser\OneDrive - University College Cork\machine learning competition\CSV_format\CSV_format\File with grades'
for k = 1:numel(S)
    F = fullfile(P,S(k).name);
    S(k).data = readtable(F);
    channel_names = S(k).data.Properties.VariableNames(:,2:10)% the current 
    % eeg.csv files had 9 channels, column 2-10; Please change it according 
    % to the number of channels in the unseen data 
    data = table2array(S(k).data)
    filtered_data = zeros(length(data),10)
    f = length(data)/3600
    %low- and high-pass filter
    for x = 2:10
        [b,a] = butter(5,[0.5 30]/(f/2),'bandpass')
        filtered_data(:,x) = filtfilt(b,a,data(:,x))
    end 
    filtered_data(:,1) = data(:,1)
    %to downsample data to 64 Hz
    downsampled = zeros(length(data)/(f/64),10)
    time_to_interp = filtered_data(1,1) : 1/64 : filtered_data(end,1)
    downsampled(:,1) = time_to_interp.'
    for x = 2:10
        y = [time_to_interp ; interp1(filtered_data(:,1),filtered_data(:,x), time_to_interp)]
        downsampled(:,x) = y(2,:).'
    end 
    %to generate the features using the NEURAL pachakge (O’Toole, J. M. & Boylan, G. B. NEURAL: 
    %quantitative features for newborn EEG using Matlab. ArXiv E-Prints
    %(2017).1704.05694.)
    bi_mont = {{'F3', 'C3'},{'T3','C3'}, {'O1', 'C3'}, {'C3', 'Cz'}, {'Cz', 'C4'}, {'F4', 'C4'}, {'T4', 'C4'}, {'O2', 'C4'}};
    [bi_sigs,bi_labels] = set_bi_montage(downsampled(:,2:10).',channel_names, bi_mont);
    a = struct
    a.eeg_data_ref = downsampled(:,2:10).'
    a.Fs = 64
    a.ch_labels_bi = bi_mont
    a.ch_labels_ref = channel_names
    a.eeg_data = bi_sigs
    a.ch_labels = bi_labels
    feat_st = generate_all_features(a)
    file_name = fullfile(path_to_grades, append('features', '_', erase(S(k).name, '.csv'), '.mat'));
    save(file_name, 'feat_st');
end
%---------------------------------------------------------------------------
% to combine all the features in a single .CSV file.
cd(path_to_grades)
P = path_to_grades
S = dir(fullfile(P,'*.mat')); 
n = 169 %number of the eeg.csv files; ATTENTION: please update it for your dataset
all = zeros(n,102)
for k = 1:numel(S)
    F = fullfile(P,S(k).name);
    load(S(k).name)
    data = struct2table(feat_st)
    data = table2array(data)
    all(k,:) = data
end
%the path to the folder that eeg_grades.csv is saved
csvwrite('all.csv',all)
% important features are saved in a file and other features are discarded
important_features = [all(:,9:12) all(:,22) all(:,31:34) all(:,47:50) all(:,55:58) all(:,75:78) all(:,91:100) all(:,102)]
csvwrite('important_features.csv',important_features)
%---------------------------------------------------------------------------
%this code separates the training and validations sets, adds the IDs and
%labels of each epoch, and make them ready for XGBoost 
data = readtable('all.csv')
data = table2array(data)
grades = readtable('eeg_grades.csv')
val_table = grades(isnan(grades{:, 4}), :) %column 4 is the column with known grades
ismember = ismember(grades{:,1},val_table{:,1})
train_idx = find(ismember == 0)
class_train = grades{train_idx,4}
class_train = array2table(class_train)
data_train = data(train_idx,:)
data_train = array2table(data_train)
ID_train = grades{train_idx,2} %column 2 is the column with IDs
ID_train = array2table(ID_train)
train_table = [ID_train, class_train, data_train]
writetable(train_table, 'training_bycode.csv', 'WriteVariableNames',0)
val_idx = find(ismember == 1)
data_val = data(val_idx,:)
data_val = array2table(data_val)
ID_val = grades{val_idx,2}
ID_val = array2table(ID_val)
val_table = [ID_val, data_val]
writetable(val_table, 'validation_bycode.csv', 'WriteVariableNames',0)
%---------------------------------------------------------------------------
%this code separates the training and validations sets, adds the IDs and
%labels of each epoch, and make them ready for XGBoost (this version is for
%important features only)
data = readtable('important_features.csv')
data = table2array(data)
grades = readtable('eeg_grades.csv')
val_table = grades(isnan(grades{:, 4}), :) %column 4 is the column with known grades
clear ismember
ismember = ismember(grades{:,1},val_table{:,1})
train_idx = find(ismember == 0)
class_train = grades{train_idx,4}
class_train = array2table(class_train)
data_train = data(train_idx,:)
data_train = array2table(data_train)
ID_train = grades{train_idx,2} %column 2 is the column with IDs
ID_train = array2table(ID_train)
train_table = [ID_train, class_train, data_train]
writetable(train_table, 'training_bycode_important_features.csv', 'WriteVariableNames',0)
val_idx = find(ismember == 1)
data_val = data(val_idx,:)
data_val = array2table(data_val)
ID_val = grades{val_idx,2}
ID_val = array2table(ID_val)
val_table = [ID_val, data_val]
writetable(val_table, 'validation_bycode_important_features.csv', 'WriteVariableNames',0)






