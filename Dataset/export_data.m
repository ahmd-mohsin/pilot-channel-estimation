function export_data(H_perfect, H_noisy, H_interpolated, pilotMask, metrics, cfg)
% EXPORT_DATA - Save all data and results to files
%
% Inputs:
%   H_perfect      - Perfect channel estimates
%   H_noisy        - Noisy estimates
%   H_interpolated - Interpolated estimates
%   pilotMask      - Pilot mask
%   metrics        - Performance metrics
%   cfg            - Configuration

% Create output directories
try
    if ~exist(cfg.paths.dataDir, 'dir')
        mkdir(cfg.paths.dataDir);
    end
catch ME
    warning('Could not create data directory: %s. Using current directory.', ME.message);
    cfg.paths.dataDir = pwd;
end

if cfg.sim.verboseOutput
    fprintf('\nExporting data to: %s\n', cfg.paths.dataDir);
end

%% Save channel estimates
% Save as .mat file
data_file = fullfile(cfg.paths.dataDir, 'channel_estimates.mat');
save(data_file, 'H_perfect', 'H_noisy', 'H_interpolated', 'pilotMask', ...
    'metrics', 'cfg', '-v7.3');

if cfg.sim.verboseOutput
    fprintf('  Saved: channel_estimates.mat (%.2f MB)\n', ...
        dir(data_file).bytes / 1e6);
end

%% Save metrics to JSON
metrics_json = jsonencode(metrics.summary, 'PrettyPrint', true);
json_file = fullfile(cfg.paths.dataDir, 'metrics_summary.json');
fid = fopen(json_file, 'w');
fprintf(fid, '%s', metrics_json);
fclose(fid);

if cfg.sim.verboseOutput
    fprintf('  Saved: metrics_summary.json\n');
end

%% Save configuration to JSON
% Create simplified config for JSON export
cfg_export.carrier = cfg.carrier;
cfg_export.mimo = cfg.mimo;
cfg_export.channel = cfg.channel;
cfg_export.noise = cfg.noise;
cfg_export.sim = cfg.sim;
cfg_export.srs_dense = cfg.srs_dense;
cfg_export.srs_sparse = cfg.srs_sparse;

cfg_json = jsonencode(cfg_export, 'PrettyPrint', true);
cfg_file = fullfile(cfg.paths.dataDir, 'configuration.json');
fid = fopen(cfg_file, 'w');
fprintf(fid, '%s', cfg_json);
fclose(fid);

if cfg.sim.verboseOutput
    fprintf('  Saved: configuration.json\n');
end

%% Export sample data to CSV for external tools
% Export first antenna pair, first slot
slot_idx = 1;
H_perfect_sample = abs(squeeze(H_perfect(:, :, 1, 1, slot_idx)));
H_interp_sample = abs(squeeze(H_interpolated(:, :, 1, 1, slot_idx)));
error_sample = abs(H_perfect_sample - H_interp_sample);

% Save perfect channel
csv_file = fullfile(cfg.paths.dataDir, 'H_perfect_slot1_ant1.csv');
writematrix(H_perfect_sample, csv_file);

% Save interpolated channel
csv_file = fullfile(cfg.paths.dataDir, 'H_interpolated_slot1_ant1.csv');
writematrix(H_interp_sample, csv_file);

% Save error
csv_file = fullfile(cfg.paths.dataDir, 'error_slot1_ant1.csv');
writematrix(error_sample, csv_file);

if cfg.sim.verboseOutput
    fprintf('  Saved: Sample CSV files for slot 1, antenna 1\n');
end

%% Create a detailed text report
report_file = fullfile(cfg.paths.dataDir, 'results_report.txt');
fid = fopen(report_file, 'w');

fprintf(fid, '=================================================================\n');
fprintf(fid, 'CHANNEL ESTIMATION PERFORMANCE REPORT\n');
fprintf(fid, '=================================================================\n\n');

fprintf(fid, 'Generated: %s\n\n', datestr(now));

fprintf(fid, 'CONFIGURATION\n');
fprintf(fid, '-----------------------------------------------------------------\n');
fprintf(fid, 'Carrier:\n');
fprintf(fid, '  Bandwidth: %d RBs (%.2f MHz)\n', cfg.carrier.NSizeGrid, ...
    cfg.carrier.NSizeGrid * 12 * cfg.carrier.SubcarrierSpacing / 1000);
fprintf(fid, '  Subcarrier Spacing: %d kHz\n', cfg.carrier.SubcarrierSpacing);
fprintf(fid, '  Cyclic Prefix: %s\n', cfg.carrier.CyclicPrefix);
fprintf(fid, '\nMIMO:\n');
fprintf(fid, '  Tx Antennas: %d\n', cfg.mimo.nTxAnts);
fprintf(fid, '  Rx Antennas: %d\n', cfg.mimo.nRxAnts);
fprintf(fid, '  Layers: %d\n', cfg.mimo.nLayers);
fprintf(fid, '\nChannel:\n');
fprintf(fid, '  Model: %s\n', cfg.channel.DelayProfile);
fprintf(fid, '  Delay Spread: %.1f ns\n', cfg.channel.DelaySpread * 1e9);
fprintf(fid, '  Max Doppler: %.1f Hz\n', cfg.channel.MaximumDopplerShift);
fprintf(fid, '\nNoise:\n');
fprintf(fid, '  SNR: %.1f dB\n', cfg.noise.SNR_dB);
fprintf(fid, '\nSRS Configuration:\n');
fprintf(fid, '  Dense (Perfect CSI): %d symbols, comb %d\n', ...
    cfg.srs_dense.NumSRSSymbols, cfg.srs_dense.KTC);
fprintf(fid, '  Sparse (Noisy CSI): %d symbols, comb %d\n', ...
    cfg.srs_sparse.NumSRSSymbols, cfg.srs_sparse.KTC);

fprintf(fid, '\n\nPERFORMANCE METRICS\n');
fprintf(fid, '-----------------------------------------------------------------\n');
fprintf(fid, 'NMSE at Pilot Locations: %.2f dB\n', metrics.nmse_pilots_dB);
fprintf(fid, 'NMSE After Interpolation: %.2f dB\n', metrics.nmse_interpolated_dB);
fprintf(fid, 'EVM: %.2f%%\n', metrics.evm_percent);
fprintf(fid, 'Correlation: %.4f\n', metrics.correlation);
fprintf(fid, '\nPer-Slot NMSE Statistics:\n');
fprintf(fid, '  Mean: %.2f dB\n', metrics.summary.mean_nmse_per_slot_dB);
fprintf(fid, '  Std Dev: %.2f dB\n', metrics.summary.std_nmse_per_slot_dB);
fprintf(fid, '  Min: %.2f dB\n', min(metrics.nmse_per_slot));
fprintf(fid, '  Max: %.2f dB\n', max(metrics.nmse_per_slot));

fprintf(fid, '\n\nPER-ANTENNA NMSE (dB)\n');
fprintf(fid, '-----------------------------------------------------------------\n');
fprintf(fid, 'Rx\\Tx  ');
for tx = 1:cfg.mimo.nTxAnts
    fprintf(fid, '   Tx%d   ', tx);
end
fprintf(fid, '\n');
for rx = 1:cfg.mimo.nRxAnts
    fprintf(fid, 'Rx%d    ', rx);
    for tx = 1:cfg.mimo.nTxAnts
        fprintf(fid, '%7.2f  ', metrics.nmse_per_antenna(rx, tx));
    end
    fprintf(fid, '\n');
end

fprintf(fid, '\n\nDATA DIMENSIONS\n');
fprintf(fid, '-----------------------------------------------------------------\n');
fprintf(fid, 'Channel Matrix Shape: [K x L x nRx x nTx x Slots]\n');
fprintf(fid, '  K (Subcarriers): %d\n', size(H_perfect, 1));
fprintf(fid, '  L (OFDM Symbols per Slot): %d\n', size(H_perfect, 2));
fprintf(fid, '  nRx (Rx Antennas): %d\n', size(H_perfect, 3));
fprintf(fid, '  nTx (Tx Antennas): %d\n', size(H_perfect, 4));
fprintf(fid, '  Slots: %d\n', size(H_perfect, 5));
fprintf(fid, '\nTotal Elements: %d\n', numel(H_perfect));
fprintf(fid, 'Memory Size: %.2f MB (per channel matrix)\n', ...
    numel(H_perfect) * 16 / 1e6); % complex double = 16 bytes

fprintf(fid, '\n=================================================================\n');
fclose(fid);

if cfg.sim.verboseOutput
    fprintf('  Saved: results_report.txt\n');
    fprintf('\nData export complete.\n');
end

end