matfile = '/home/luye/storage/2018.06.12_job-776618_pop-100_DA-control_CTX-SWA/STN_20180612_pop-100_dur-10s_job-776618.old'
load(matfile)

signals = block(1).segments{1}.analogsignals; % cell array
spiketrains = block(1).segments{1}.spiketrains; % cell array

vm_signal = 'None';
i_vm_signal = 0;
for i = 1:length(signals)
    if strcmp(signals{i}.name, 'Vm')
        vm_signal = signals{i};
        i_vm_signal = i;
    end
end

fprintf('Index of Vm signal is %d', i_vm_signal);

spike_threshold = -10; % mV
rec_dt = 1 / vm_signal.sampling_rate;

for i = 1:length(spiketrains)
   cell_vm = vm_signal.signal(:,i);
   vm_thresholded = cell_vm > spike_threshold;
   vm_diffed = diff(vm_thresholded);
   spike_indices = find(vm_diffed == 1) + 1;
   spike_times = spike_indices * rec_dt;
   block(1).segments{1}.spiketrains{i}.times = spike_times;
   fprintf('Fixed spike times for spiketrain %d', i);
end

save(strcat(matfile, '_fixed'), '-mat-binary', 'block');