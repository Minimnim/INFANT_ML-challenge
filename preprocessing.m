data = readtable('ID01_epoch1.csv')
data = table2array(data)
filtered_data = zeros(length(data),10)
f = length(data)/3600
for x = 2:10
    [b,a] = butter(5,[0.5 30]/(f/2),'bandpass')
    filtered_data(:,x) = filtfilt(b,a,data(:,x))
end 
filtered_data(:,1) = data(:,1)
writematrix(filtered_data, 'filtered_ID01_epoch1.txt', 'Delimiter', 'tab')
downsampled = zeros(length(data)/(f/64),10)
time_to_interp = filtered_data(1,1) : 1/64 : filtered_data(end,1)
downsampled(:,1) = time_to_interp.'
for x = 2:10
    y = [time_to_interp ; interp1(filtered_data(:,1),filtered_data(:,x), time_to_interp)]
    downsampled(:,x) = y(2,:).'
end 
channel_names = data.Properties.VariableNames(:,2:10)
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




%y_filter = filter(b,a,c)
%[z p k] = butter(5, [0.5 30]/100,'bandpass')
%[sos, g] = zp2sos(z,p,k)
%y_filtfilt_zpk = filtfilt(sos,g,c) 
%bandpass function
%[myFilteredData, bpdf] = bandpass(c, [0.5 30], 200)
%myFilteredData2 = filtfilt(bpdf, c)
%different orders 
%[b,a] = butter(1,[0.5 30]/100,'bandpass')
%y_order1 = filtfilt(b,a,c)
%[b,a] = butter(3,[0.5 30]/100,'bandpass')
%y_order3 = filtfilt(b,a,c)
%[b,a] = butter(7,[0.5 30]/100,'bandpass')
%y_order7 = filtfilt(b,a,c)
%low pass and high pass with different orders
%[b,a] = butter(6, 30/100,'low')
%y_low = filtfilt(b,a,c)
%[b,a] = butter(1, 0.5/100,'high')
%y_high = filtfilt(b,a,y_low)
%x_filt=filter_zerophase(x,Fs,LP_fc,HP_fc,L_filt)
%dobandpass=do_bandpass_filtering(c,200,0.5,30)
%withnans =filter_butterworth_withnans(c,200,30,0.5)
%filterbutter=filter_butter(c,0.5, 30, 6)
%x_filt=filter_zerophase(x,Fs,LP_fc,HP_fc