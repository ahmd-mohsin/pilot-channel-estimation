%% MONTE_CARLO_UMI_SIMULATION_WITH_ROBUST_PROGRESS
% Monte Carlo simulation for 3GPP UMi outdoor scenario
%
% Features:
% - 3GPP Urban Micro (UMi) Street Canyon outdoor
% - Small-scale fading (multipath with TDL-C)
% - Large-scale fading (path loss)
% - Shadowing (log-normal)
% - Doppler shift (user mobility)
% - N Monte Carlo realizations
% - Saves H_perfect and H_estimated only
% - Robust progress: single-line in Desktop, coarse multi-line in CLI/HPC

clear; close all; clc;

fprintf('=================================================================\n');
fprintf('3GPP UMi OUTDOOR SCENARIO - MONTE CARLO SIMULATION\n');
fprintf('=================================================================\n\n');

%% Step 1: Load Configuration (user-provided helper)
fprintf('Loading UMi outdoor configuration...\n');
cfg = config_system_umi();  % <-- keep your existing function

fprintf('Configuration:\n');
fprintf('  Scenario: %s\n', cfg.largescale.scenario);
fprintf('  Carrier Frequency: %.2f GHz\n', cfg.largescale.frequency/1e9);
fprintf('  Bandwidth: %d RBs (%.2f MHz)\n', cfg.carrier.NSizeGrid, ...
    cfg.carrier.NSizeGrid * 12 * cfg.carrier.SubcarrierSpacing / 1000);
fprintf('  MIMO: %dx%d (Tx x Rx)\n', cfg.mimo.nTxAnts, cfg.mimo.nRxAnts);
fprintf('  Distance: %.1f m\n', cfg.largescale.distance);
fprintf('  BS Height: %.1f m\n', cfg.largescale.hBS);
fprintf('  UE Height: %.1f m\n', cfg.largescale.hUT);
fprintf('  Doppler: %.1f Hz (mobility)\n', cfg.channel.MaximumDopplerShift);
fprintf('  Delay Spread: %.1f ns\n', cfg.channel.DelaySpread * 1e9);
fprintf('  Path Loss: %.2f dB\n', cfg.derived.pathLoss_dB);
fprintf('  Shadowing: %.1f dB std dev\n', cfg.largescale.shadowStdDev);
fprintf('  SNR: %.1f dB\n', cfg.noise.SNR_dB);
fprintf('  Monte Carlo Runs: %d\n', cfg.montecarlo.numRealizations);

%% Step 2: Initialize Storage
numMC    = cfg.montecarlo.numRealizations;
K        = cfg.derived.K;
L        = cfg.derived.L;
numSlots = cfg.derived.numSlots;
nRx      = cfg.mimo.nRxAnts;
nTx      = cfg.mimo.nTxAnts;

fprintf('\nInitializing data storage...\n');
fprintf('  Dimensions per realization: [%d x %d x %d x %d x %d]\n', K, L, nRx, nTx, numSlots);
fprintf('  Total memory per dataset: %.2f GB\n', numMC * K * L * nRx * nTx * numSlots * 16 / 1e9);

H_perfect_all   = zeros(K, L, nRx, nTx, numSlots, numMC);
H_estimated_all = zeros(K, L, nRx, nTx, numSlots, numMC);

metadata = struct();
metadata.channelSeed       = zeros(numMC, 1);
metadata.noiseSeed         = zeros(numMC, 1);
metadata.shadowingGain_dB  = zeros(numMC, 1);
metadata.effectiveSNR_dB   = zeros(numMC, 1);

fprintf('Memory allocated successfully.\n');

%% Step 3: Create Output Directory
if ~exist(cfg.paths.dataDir, 'dir')
    try
        mkdir(cfg.paths.dataDir);
        fprintf('Output directory created: %s\n', cfg.paths.dataDir);
    catch
        warning('Could not create directory. Using current directory.');
        cfg.paths.dataDir = pwd;
    end
end

%% Step 4: Monte Carlo Loop with Robust Progress
fprintf('\n=================================================================\n');
fprintf('STARTING MONTE CARLO SIMULATIONS\n');
fprintf('=================================================================\n\n');

progressBarWidth   = 50;
startTime          = tic;
updateIntervalSec  = 1;   % time-gated updates in single-line mode
lastUpdateTime     = 0;
lastPrintedWidth   = 0;

% Detect environment: Desktop MATLAB vs CLI/HPC
hasDesktop = usejava('desktop');        % true in Desktop MATLAB
supportsSingleLine = hasDesktop;        % set false for CLI/HPC/diary logs

% In multi-line mode, print only every 'mlStride' realizations (≈1% default)
mlStride = max(1, floor(numMC/100));
if ~isfield(cfg,'sim') || ~isfield(cfg.sim,'saveInterval') || cfg.sim.saveInterval <= 0
    cfg.sim.saveInterval = max(1, floor(numMC/10));  % default checkpoint every 10%
end

% Initial single-line header (Desktop only)
if supportsSingleLine
    initLine = sprintf('Progress: [%s]  0.0%% | 0/%d | Elapsed: 0s | Remaining: -- | Total: -- | Speed: -- real/s', ...
                       repmat(' ',1,progressBarWidth), numMC);
    fprintf('%s', initLine);  % no newline
end

for mc = 1:numMC
    %% Seeds
    if cfg.montecarlo.varyChannelSeed
        cfg.channel.Seed = cfg.channel.Seed + mc - 1;
    end
    metadata.channelSeed(mc) = cfg.channel.Seed;

    if cfg.montecarlo.varyNoiseSeed
        noiseSeed = 1000 + mc;
    else
        noiseSeed = 1000;
    end
    metadata.noiseSeed(mc) = noiseSeed;

    %% Shadowing draw
    if cfg.montecarlo.varyShadowing
        shadowingGain_dB = cfg.largescale.shadowStdDev * randn();
    else
        shadowingGain_dB = 0;
    end
    metadata.shadowingGain_dB(mc) = shadowingGain_dB;

    %% Effective SNR (pathloss + shadowing)
    totalLoss_dB     = cfg.derived.pathLoss_dB - shadowingGain_dB;
    effectiveSNR_dB  = cfg.noise.SNR_dB - totalLoss_dB;
    metadata.effectiveSNR_dB(mc) = effectiveSNR_dB;

    cfg_current = cfg;
    cfg_current.noise.SNR_dB = effectiveSNR_dB;

    %% Channel creation and CSI generation (user-provided helpers)
    [channel, chInfo] = create_channel_model(cfg_current);                   % keep your existing function
    [H_perfect, ~, ~] = generate_perfect_csi(cfg_current, channel, chInfo); % dense pilots

    [channel_noisy, chInfo_noisy] = create_channel_model(cfg_current);
    [~, H_estimated, ~, ~] = generate_noisy_csi(cfg_current, channel_noisy, chInfo_noisy); % sparse + interp

    %% Store
    H_perfect_all(:, :, :, :, :, mc)   = H_perfect;
    H_estimated_all(:, :, :, :, :, mc) = H_estimated;

    %% Checkpoint
    if mod(mc, cfg.sim.saveInterval) == 0
        if supportsSingleLine, fprintf('\n'); end
        fprintf('[CHECKPOINT] Saved at %d/%d realizations\n', mc, numMC);
        save_intermediate_results(cfg, H_perfect_all, H_estimated_all, metadata, mc);
    end

    %% Progress printing
    elapsed = toc(startTime);
    percentComplete = 100 * mc / numMC;
    avgTimePer      = elapsed / mc;
    remaining       = avgTimePer * (numMC - mc);
    totalEst        = avgTimePer * numMC;
    speedRealSec    = 1/max(avgTimePer, eps);

    if supportsSingleLine
        % time-gated single-line update
        if (elapsed - lastUpdateTime >= updateIntervalSec) || (mc == numMC)
            lastUpdateTime = elapsed;
            numComplete = round(progressBarWidth * mc / numMC);
            bar = [repmat('=',1,numComplete), repmat(' ',1,progressBarWidth-numComplete)];

            line = sprintf('Progress: [%s] %5.1f%% | %d/%d | Elapsed: %s | Remaining: %s | Total: %s | Speed: %.2f real/s', ...
                           bar, percentComplete, mc, numMC, ...
                           format_time(elapsed), format_time(remaining), format_time(totalEst), speedRealSec);

            if length(line) < lastPrintedWidth
                line = [line, repmat(' ',1,lastPrintedWidth - length(line))];
            else
                lastPrintedWidth = length(line);
            end

            fprintf('\r%s', line);           % overwrite same line
            drawnow limitrate nocallbacks;   % keep UI/TTY responsive
        end
    else
        % coarse multi-line update only at stride (no spam)
        if (mod(mc, mlStride) == 0) || (mc == 1) || (mc == numMC)
            numComplete = round(progressBarWidth * mc / numMC);
            bar = [repmat('=',1,numComplete), repmat(' ',1,progressBarWidth-numComplete)];
            fprintf('Progress: [%s] %5.1f%% | %d/%d | Elapsed: %s | Remaining: %s | Total: %s | Speed: %.2f real/s\n', ...
                    bar, percentComplete, mc, numMC, ...
                    format_time(elapsed), format_time(remaining), format_time(totalEst), speedRealSec);
        end
    end
end

% finish single-line mode with a newline for clean console
if supportsSingleLine, fprintf('\n'); end

totalTime = toc(startTime);
fprintf('\n=================================================================\n');
fprintf('MONTE CARLO SIMULATION COMPLETE!\n');
fprintf('=================================================================\n');
fprintf('Total time: %s (%.2f sec per realization)\n', format_time(totalTime), totalTime/numMC);
fprintf('Average speed: %.2f realizations per minute\n', numMC / (totalTime/60));

%% Step 5: Save Final Results
fprintf('\nSaving final dataset...\n');
dataFile = fullfile(cfg.paths.dataDir, 'umi_channel_data.mat');
fprintf('  Saving: %s\n', dataFile);
fprintf('  Dataset size (approx): %.2f GB\n', (numel(H_perfect_all) + numel(H_estimated_all)) * 16 / 1e9);

description = struct();
description.scenario       = 'UMi-StreetCanyon outdoor (3GPP TR 38.901) with TDL-C';
description.frequency_GHz  = cfg.largescale.frequency / 1e9;
description.bandwidth_MHz  = cfg.carrier.NSizeGrid * 12 * cfg.carrier.SubcarrierSpacing / 1000;
description.numRBs         = cfg.carrier.NSizeGrid;
description.numMC          = numMC;
description.dimensions     = sprintf('[%d x %d x %d x %d x %d x %d] (K x L x nRx x nTx x Slots x MC)', size(H_perfect_all));
description.H_perfect      = 'Ground truth CSI from dense pilots (maximum sounding)';
description.H_estimated    = 'Estimated CSI from sparse pilots with interpolation';
description.date_generated = datestr(now);

fprintf('  Compressing and saving...\n');
save(dataFile, 'H_perfect_all', 'H_estimated_all', 'metadata', 'cfg', 'description', '-v7.3');

fileInfo = dir(dataFile);
fileSizeMB = fileInfo.bytes / 1e6;
fprintf('  Saved successfully: %.2f MB\n', fileSizeMB);

%% Save metadata separately
metadataFile = fullfile(cfg.paths.dataDir, 'umi_metadata.mat');
save(metadataFile, 'metadata', 'cfg', 'description');
fprintf('  Metadata saved: %s\n', metadataFile);

%% Save summary statistics
fprintf('\nComputing summary statistics...\n');
summary = compute_summary_statistics(H_perfect_all, H_estimated_all, metadata);
summaryFile = fullfile(cfg.paths.dataDir, 'umi_summary.mat');
save(summaryFile, 'summary');
fprintf('  Summary saved: %s\n', summaryFile);

%% Create README
create_readme_file(cfg, description, summary);

%% Final Summary
fprintf('\n=================================================================\n');
fprintf('DATASET GENERATION COMPLETE!\n');
fprintf('=================================================================\n');
fprintf('Dataset saved to: %s\n', cfg.paths.dataDir);
fprintf('\nFiles created:\n');
fprintf('  1. umi_channel_data.mat (%.2f MB) - Main dataset\n', fileSizeMB);
fprintf('  2. umi_metadata.mat - Configuration and metadata\n');
fprintf('  3. umi_summary.mat - Summary statistics\n');
fprintf('  4. README.txt - Dataset documentation\n');
fprintf('\nData variables:\n');
fprintf('  H_perfect_all:   [%d x %d x %d x %d x %d x %d]\n', size(H_perfect_all));
fprintf('  H_estimated_all: [%d x %d x %d x %d x %d x %d]\n', size(H_estimated_all));
fprintf('\nPerformance Summary:\n');
fprintf('  Average NMSE: %.2f dB\n', summary.mean_nmse_dB);
fprintf('  NMSE Std Dev: %.2f dB\n', summary.std_nmse_dB);
fprintf('  SNR range: [%.2f, %.2f] dB\n', summary.snr_range_dB);
fprintf('=================================================================\n\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Helper Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function timeStr = format_time(seconds)
% Format time in human-readable format
if seconds < 60
    timeStr = sprintf('%.0fs', seconds);
elseif seconds < 3600
    minutes = floor(seconds / 60);
    secs = floor(mod(seconds, 60));
    timeStr = sprintf('%dm %ds', minutes, secs);
else
    hours = floor(seconds / 3600);
    minutes = floor(mod(seconds, 3600) / 60);
    timeStr = sprintf('%dh %dm', hours, minutes);
end
end

function save_intermediate_results(cfg, H_perfect_all, H_estimated_all, metadata, mc)
% Save intermediate checkpoint (silent helper)
checkpointFile = fullfile(cfg.paths.dataDir, sprintf('checkpoint_%d.mat', mc));
save(checkpointFile, 'H_perfect_all', 'H_estimated_all', 'metadata', 'mc', '-v7.3');
end

function summary = compute_summary_statistics(H_perfect_all, H_estimated_all, metadata)
% Compute summary statistics across all realizations
numMC = size(H_perfect_all, 6);
summary.nmse_per_realization = zeros(numMC, 1);
for i = 1:numMC
    H_p = H_perfect_all(:,:,:,:,:,i);
    H_e = H_estimated_all(:,:,:,:,:,i);
    mse = mean(abs(H_e(:) - H_p(:)).^2);
    signal_power = mean(abs(H_p(:)).^2);
    summary.nmse_per_realization(i) = 10*log10(mse / signal_power);
end

summary.mean_nmse_dB   = mean(summary.nmse_per_realization);
summary.std_nmse_dB    = std(summary.nmse_per_realization);
summary.median_nmse_dB = median(summary.nmse_per_realization);
summary.min_nmse_dB    = min(summary.nmse_per_realization);
summary.max_nmse_dB    = max(summary.nmse_per_realization);

summary.snr_range_dB = [min(metadata.effectiveSNR_dB), max(metadata.effectiveSNR_dB)];
summary.mean_snr_dB  = mean(metadata.effectiveSNR_dB);
summary.std_snr_dB   = std(metadata.effectiveSNR_dB);

summary.shadowing_range_dB = [min(metadata.shadowingGain_dB), max(metadata.shadowingGain_dB)];
summary.mean_shadowing_dB  = mean(metadata.shadowingGain_dB);
summary.std_shadowing_dB   = std(metadata.shadowingGain_dB);
end

function create_readme_file(cfg, description, summary)
% Create README text file
readmeFile = fullfile(cfg.paths.dataDir, 'README.txt');
fid = fopen(readmeFile, 'w');

fprintf(fid, '================================================================\n');
fprintf(fid, '3GPP UMi OUTDOOR CHANNEL DATASET\n');
fprintf(fid, '================================================================\n\n');

fprintf(fid, 'Generated: %s\n\n', description.date_generated);

fprintf(fid, 'SCENARIO\n');
fprintf(fid, '----------------------------------------------------------------\n');
fprintf(fid, 'Environment: %s\n', description.scenario);
fprintf(fid, 'Frequency: %.2f GHz\n', description.frequency_GHz);
fprintf(fid, 'Bandwidth: %.2f MHz (%d RBs)\n', description.bandwidth_MHz, description.numRBs);
fprintf(fid, 'Distance: %.1f m (UE to gNB)\n', cfg.largescale.distance);
fprintf(fid, 'BS Height: %.1f m\n', cfg.largescale.hBS);
fprintf(fid, 'UE Height: %.1f m\n', cfg.largescale.hUT);
fprintf(fid, 'Doppler: %.1f Hz\n', cfg.channel.MaximumDopplerShift);
fprintf(fid, 'Delay Spread: %.1f ns\n', cfg.channel.DelaySpread * 1e9);

fprintf(fid, '\nFADING MODELS\n');
fprintf(fid, '----------------------------------------------------------------\n');
fprintf(fid, 'Small-scale: TDL-C (multipath, NLOS urban)\n');
fprintf(fid, 'Large-scale: Path loss (%.2f dB)\n', cfg.derived.pathLoss_dB);
fprintf(fid, 'Shadowing: Log-normal (%.1f dB std)\n', cfg.largescale.shadowStdDev);

fprintf(fid, '\nDATA DESCRIPTION\n');
fprintf(fid, '----------------------------------------------------------------\n');
fprintf(fid, 'Monte Carlo Realizations: %d\n', description.numMC);
fprintf(fid, 'Dimensions: %s\n', description.dimensions);
fprintf(fid, '\nVariables:\n');
fprintf(fid, '  H_perfect_all   - Ground truth CSI (dense pilots)\n');
fprintf(fid, '  H_estimated_all - Estimated CSI (sparse pilots + interpolation)\n');
fprintf(fid, '  metadata        - Per-realization metadata (seeds, SNR, shadowing)\n');
fprintf(fid, '  cfg             - Complete configuration structure\n');

fprintf(fid, '\nPERFORMANCE SUMMARY\n');
fprintf(fid, '----------------------------------------------------------------\n');
fprintf(fid, 'Average NMSE: %.2f dB\n', summary.mean_nmse_dB);
fprintf(fid, 'NMSE Std Dev: %.2f dB\n', summary.std_nmse_dB);
fprintf(fid, 'NMSE Range: [%.2f, %.2f] dB\n', summary.min_nmse_dB, summary.max_nmse_dB);
fprintf(fid, '\nEffective SNR Range: [%.2f, %.2f] dB\n', summary.snr_range_dB);
fprintf(fid, 'Mean SNR: %.2f dB\n', summary.mean_snr_dB);

fprintf(fid, '\nUSAGE\n');
fprintf(fid, '----------------------------------------------------------------\n');
fprintf(fid, 'Load data:\n');
fprintf(fid, '  load(''umi_channel_data.mat'')\n\n');
fprintf(fid, 'Access data:\n');
fprintf(fid, '  H_perfect_mc1   = H_perfect_all(:,:,:,:,:,1);   %% First realization\n');
fprintf(fid, '  H_estimated_mc1 = H_estimated_all(:,:,:,:,:,1);\n\n');
fprintf(fid, 'For ML training:\n');
fprintf(fid, '  X_train = H_estimated_all;  %% Input (noisy)\n');
fprintf(fid, '  Y_train = H_perfect_all;    %% Target (perfect)\n');

fprintf(fid, '\n================================================================\n');
fclose(fid);
end
