%% MAIN_CHANNEL_ESTIMATION
% Main script for generating perfect and noisy CSI estimates
%
% This script demonstrates:
% 1. Perfect CSI generation using dense pilot sounding (full SRS)
% 2. Noisy CSI generation using sparse pilots + interpolation
% 3. Performance evaluation and comparison
% 4. Comprehensive visualization
% 5. Data export for ML model training
%
% The perfect CSI represents the best achievable estimate using complete
% transmission information, while the noisy CSI represents practical
% estimation from sparse pilots (similar to real 5G NR systems).

clear; close all; clc;

%% Add paths
addpath(pwd);

fprintf('=================================================================\n');
fprintf('CHANNEL STATE INFORMATION ESTIMATION\n');
fprintf('Perfect CSI vs Noisy CSI with Interpolation\n');
fprintf('=================================================================\n\n');

%% Step 1: Load Configuration
fprintf('Step 1: Loading configuration...\n');
cfg = config_system();

fprintf('  Carrier: %d RBs @ %d kHz SCS\n', cfg.carrier.NSizeGrid, cfg.carrier.SubcarrierSpacing);
fprintf('  MIMO: %dx%d (Tx x Rx)\n', cfg.mimo.nTxAnts, cfg.mimo.nRxAnts);
fprintf('  Channel: %s, Doppler = %.1f Hz\n', cfg.channel.DelayProfile, cfg.channel.MaximumDopplerShift);
fprintf('  SNR: %.1f dB\n', cfg.noise.SNR_dB);
fprintf('  Simulation: %d frames (%d slots)\n', cfg.sim.numFrames, cfg.derived.numSlots);

%% Step 2: Create Channel Model
fprintf('\nStep 2: Creating channel model...\n');
[channel, chInfo] = create_channel_model(cfg);

%% Step 3: Generate Perfect CSI (Dense Pilots)
fprintf('\nStep 3: Generating Perfect CSI (ground truth from simulator)...\n');
fprintf('  Using dense SRS: %d symbols, comb %d\n', ...
    cfg.srs_dense.NumSRSSymbols, cfg.srs_dense.KTC);

tic;
[H_perfect, txGrid_dense, pathGains_perfect] = generate_perfect_csi(cfg, channel, chInfo);
time_perfect = toc;

fprintf('  Time elapsed: %.2f seconds\n', time_perfect);
fprintf('  Generated H_perfect: [%d x %d x %d x %d x %d]\n', size(H_perfect));

%% Step 4: Generate Noisy CSI (Sparse Pilots + Interpolation)
fprintf('\nStep 4: Generating Noisy CSI (sparse pilots + interpolation)...\n');
fprintf('  Using sparse SRS: %d symbols, comb %d\n', ...
    cfg.srs_sparse.NumSRSSymbols, cfg.srs_sparse.KTC);

% Create a fresh channel object with same seed to get same realization
[channel_noisy, chInfo_noisy] = create_channel_model(cfg);

tic;
[H_noisy, H_interpolated, pilotMask, nvar_est] = generate_noisy_csi(cfg, channel_noisy, chInfo_noisy);
time_noisy = toc;

fprintf('  Time elapsed: %.2f seconds\n', time_noisy);
fprintf('  Generated H_interpolated: [%d x %d x %d x %d x %d]\n', size(H_interpolated));
fprintf('  Pilot density: %.2f%%\n', 100 * sum(pilotMask(:)) / numel(pilotMask(:,:,1)));

%% Step 5: Evaluate Performance
fprintf('\nStep 5: Evaluating performance...\n');
metrics = evaluate_channel_estimates(H_perfect, H_noisy, H_interpolated, pilotMask, cfg);

%% Step 6: Visualize Results
fprintf('\nStep 6: Generating visualizations...\n');
figs = visualize_results(H_perfect, H_noisy, H_interpolated, pilotMask, metrics, cfg);

% Create figures directory and save
try
    if ~exist(cfg.paths.figuresDir, 'dir')
        mkdir(cfg.paths.figuresDir);
    end
    
    % Save figures
    for i = 1:length(figs)
        filename = fullfile(cfg.paths.figuresDir, sprintf('figure_%d.png', i));
        saveas(figs(i), filename);
    end
    fprintf('  Saved %d figures to: %s\n', length(figs), cfg.paths.figuresDir);
catch ME
    fprintf('  Warning: Could not save figures to disk (%s)\n', ME.message);
    fprintf('  Figures are still displayed and available in workspace as ''figs''.\n');
end

%% Step 7: Export Data
fprintf('\nStep 7: Exporting data...\n');
export_data(H_perfect, H_noisy, H_interpolated, pilotMask, metrics, cfg);

%% Step 8: Summary
fprintf('\n=================================================================\n');
fprintf('SUMMARY\n');
fprintf('=================================================================\n');
fprintf('Channel Estimation completed successfully!\n\n');
fprintf('Perfect CSI (Dense Pilots):\n');
fprintf('  - Generated from full sounding sequence\n');
fprintf('  - Shape: [%d x %d x %d x %d x %d]\n', size(H_perfect));
fprintf('  - This is your "ground truth" from simulator\n\n');
fprintf('Noisy CSI (Sparse Pilots + Interpolation):\n');
fprintf('  - Generated from practical sparse pilots\n');
fprintf('  - Shape: [%d x %d x %d x %d x %d]\n', size(H_interpolated));
fprintf('  - This is your "noisy input" for denoising\n\n');
fprintf('Performance Gap (to be closed by your Transformer/Diffusion model):\n');
fprintf('  - NMSE at pilots: %.2f dB\n', metrics.nmse_pilots_dB);
fprintf('  - NMSE after interpolation: %.2f dB\n', metrics.nmse_interpolated_dB);
fprintf('  - EVM: %.2f%%\n', metrics.evm_percent);
fprintf('  - Improvement potential: %.2f dB\n', ...
    metrics.nmse_interpolated_dB - metrics.nmse_pilots_dB);
fprintf('\nData saved to: %s\n', cfg.paths.outputDir);
fprintf('  - MATLAB: channel_estimates.mat\n');
fprintf('  - JSON: metrics_summary.json, configuration.json\n');
fprintf('  - CSV: Sample channel matrices and errors\n');
fprintf('  - TXT: results_report.txt\n');
fprintf('  - PNG: All figures\n');

fprintf('\n=================================================================\n');
fprintf('NEXT STEPS FOR YOUR ML MODEL\n');
fprintf('=================================================================\n');
fprintf('1. Load the data:\n');
fprintf('   >> load(''%s'')\n', fullfile(cfg.paths.dataDir, 'channel_estimates.mat'));
fprintf('\n2. Training data:\n');
fprintf('   - Input (X):  H_interpolated  (noisy estimate)\n');
fprintf('   - Target (Y): H_perfect       (ground truth)\n');
fprintf('   - Mask:       pilotMask       (pilot locations)\n');
fprintf('\n3. Your Transformer/Diffusion model should:\n');
fprintf('   - Take H_interpolated as input (noisy CSI)\n');
fprintf('   - Predict H_perfect as output (denoised CSI)\n');
fprintf('   - Use pilotMask to enforce data consistency\n');
fprintf('   - Add SNR conditioning for robustness\n');
fprintf('\n4. Key advantages of this approach:\n');
fprintf('   ✓ Standards-compatible (no pilot pattern changes)\n');
fprintf('   ✓ Drop-in post-processor (after existing estimator)\n');
fprintf('   ✓ Handles sparse pilots better than interpolation\n');
fprintf('   ✓ Can learn complex channel structure patterns\n');
fprintf('=================================================================\n\n');

fprintf('Run time: %.2f seconds\n', time_perfect + time_noisy);
fprintf('All done! Check the figures and exported data.\n\n');