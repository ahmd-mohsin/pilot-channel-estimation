%% BATCH_GENERATE_DATASETS
% Generate multiple CSI datasets with varying configurations
%
% This script generates training datasets across different:
% - SNR values
% - Channel models
% - Doppler shifts
% - Pilot densities
%
% Useful for training robust ML models

clear; close all; clc;

fprintf('=================================================================\n');
fprintf('BATCH DATASET GENERATION\n');
fprintf('=================================================================\n\n');

%% Configuration Space
% Define the parameter variations to explore

% SNR range (dB)
snr_values = [0, 5, 10, 15, 20, 25];

% Channel models
channel_models = {'TDL-A', 'TDL-C'};

% Doppler shifts (Hz)
doppler_values = [5, 30, 100];

% Sparse pilot configurations [NumSymbols, Comb]
pilot_configs = {
    [1, 4],  % Very sparse
    [2, 2],  % Medium
    [4, 2]   % Dense
};

%% Select combinations to generate
% For demonstration, we'll do a subset. You can expand this.

combinations = [];
combo_idx = 1;

% Generate all combinations (or sample strategically)
for snr = snr_values
    for ch_idx = 1:length(channel_models)
        for dop = doppler_values
            for p_idx = 1:length(pilot_configs)
                combinations(combo_idx).snr = snr;
                combinations(combo_idx).channel = channel_models{ch_idx};
                combinations(combo_idx).doppler = dop;
                combinations(combo_idx).pilot_sym = pilot_configs{p_idx}(1);
                combinations(combo_idx).pilot_comb = pilot_configs{p_idx}(2);
                combo_idx = combo_idx + 1;
            end
        end
    end
end

fprintf('Total combinations: %d\n', length(combinations));
fprintf('Estimated time: ~%.1f minutes\n\n', length(combinations) * 0.5);

% Option to generate subset
generate_all = input('Generate all combinations? (1=yes, 0=sample 10): ');
if ~generate_all
    % Sample 10 random combinations
    sample_idx = randperm(length(combinations), min(10, length(combinations)));
    combinations = combinations(sample_idx);
    fprintf('Generating %d sampled combinations\n\n', length(combinations));
end

%% Generate datasets
base_cfg = config_system();
base_cfg.sim.verboseOutput = false;  % Less verbose for batch

% Create base output directory
base_output_dir = '/mnt/user-data/outputs/batch_datasets';
if ~exist(base_output_dir, 'dir')
    mkdir(base_output_dir);
end

% Storage for all results
all_results = struct();

tic;
for i = 1:length(combinations)
    fprintf('-------------------------------------------\n');
    fprintf('Combination %d/%d\n', i, length(combinations));
    fprintf('  SNR: %.1f dB\n', combinations(i).snr);
    fprintf('  Channel: %s\n', combinations(i).channel);
    fprintf('  Doppler: %.1f Hz\n', combinations(i).doppler);
    fprintf('  Pilots: %d symbols, comb %d\n', ...
        combinations(i).pilot_sym, combinations(i).pilot_comb);
    
    % Create configuration
    cfg = base_cfg;
    cfg.noise.SNR_dB = combinations(i).snr;
    cfg.channel.DelayProfile = combinations(i).channel;
    cfg.channel.MaximumDopplerShift = combinations(i).doppler;
    cfg.srs_sparse.NumSRSSymbols = combinations(i).pilot_sym;
    cfg.srs_sparse.KTC = combinations(i).pilot_comb;
    
    % Create unique output directory
    dir_name = sprintf('snr%d_%s_dop%d_p%ds%dc', ...
        round(combinations(i).snr), ...
        strrep(combinations(i).channel, '-', ''), ...
        round(combinations(i).doppler), ...
        combinations(i).pilot_sym, ...
        combinations(i).pilot_comb);
    cfg.paths.outputDir = fullfile(base_output_dir, dir_name);
    cfg.paths.figuresDir = fullfile(cfg.paths.outputDir, 'figures');
    cfg.paths.dataDir = fullfile(cfg.paths.outputDir, 'data');
    
    try
        % Create channel
        [channel, chInfo] = create_channel_model(cfg);
        
        % Generate perfect CSI
        [H_perfect, ~, ~] = generate_perfect_csi(cfg, channel, chInfo);
        
        % Reset and generate noisy CSI
        reset(channel);
        channel.Seed = cfg.channel.Seed;
        [H_noisy, H_interpolated, pilotMask, ~] = generate_noisy_csi(cfg, channel, chInfo);
        
        % Evaluate
        metrics = evaluate_channel_estimates(H_perfect, H_noisy, ...
            H_interpolated, pilotMask, cfg);
        
        % Store results
        all_results(i).config = combinations(i);
        all_results(i).metrics = metrics.summary;
        all_results(i).dir = cfg.paths.outputDir;
        
        % Export data (no figures for speed)
        export_data(H_perfect, H_noisy, H_interpolated, pilotMask, metrics, cfg);
        
        fprintf('  NMSE: %.2f dB | EVM: %.2f%%\n', ...
            metrics.nmse_interpolated_dB, metrics.evm_percent);
        fprintf('  Saved to: %s\n', dir_name);
        
    catch ME
        fprintf('  ERROR: %s\n', ME.message);
        all_results(i).error = ME.message;
    end
end
total_time = toc;

fprintf('\n=================================================================\n');
fprintf('BATCH GENERATION COMPLETE\n');
fprintf('=================================================================\n');
fprintf('Total time: %.2f minutes\n', total_time/60);
fprintf('Datasets generated: %d\n', length(combinations));
fprintf('Output directory: %s\n', base_output_dir);

%% Save summary
summary_file = fullfile(base_output_dir, 'batch_summary.mat');
save(summary_file, 'all_results', 'combinations');
fprintf('Summary saved to: batch_summary.mat\n');

%% Create comparison plots
fprintf('\nGenerating comparison plots...\n');

figure('Position', [100, 100, 1400, 800]);

% Extract metrics
snrs = [all_results.config];
snrs = [snrs.snr];
nmses = [all_results.metrics];
nmses = [nmses.nmse_interpolated_dB];
evms = [all_results.metrics];
evms = [evms.evm_percent];

% Plot NMSE vs SNR
subplot(2, 2, 1);
scatter(snrs, nmses, 100, 'filled');
xlabel('SNR (dB)'); ylabel('NMSE (dB)');
title('NMSE vs SNR');
grid on;

% Plot EVM vs SNR
subplot(2, 2, 2);
scatter(snrs, evms, 100, 'filled');
xlabel('SNR (dB)'); ylabel('EVM (%)');
title('EVM vs SNR');
grid on;

% Histogram of NMSE
subplot(2, 2, 3);
histogram(nmses, 20);
xlabel('NMSE (dB)'); ylabel('Count');
title('Distribution of NMSE');
grid on;

% Summary table
subplot(2, 2, 4);
axis off;
summary_text = {
    'Batch Generation Summary'
    ''
    sprintf('Total datasets: %d', length(combinations))
    sprintf('Total time: %.1f min', total_time/60)
    ''
    'Statistics:'
    sprintf('  NMSE range: [%.1f, %.1f] dB', min(nmses), max(nmses))
    sprintf('  NMSE mean: %.1f dB', mean(nmses))
    sprintf('  EVM range: [%.1f, %.1f]%%', min(evms), max(evms))
    sprintf('  EVM mean: %.1f%%', mean(evms))
    ''
    sprintf('Saved to:\n  %s', base_output_dir)
};
text(0.1, 0.9, summary_text, 'VerticalAlignment', 'top', ...
    'FontSize', 9, 'FontName', 'FixedWidth');

sgtitle('Batch Dataset Generation Summary');
saveas(gcf, fullfile(base_output_dir, 'batch_summary.png'));

fprintf('Summary plot saved.\n');
fprintf('\nAll done! Check %s for all datasets.\n', base_output_dir);