function figs = visualize_results(H_perfect, H_noisy, H_interpolated, pilotMask, metrics, cfg)
% VISUALIZE_RESULTS - Create comprehensive visualization of results
%
% Inputs:
%   H_perfect      - Perfect channel estimates
%   H_noisy        - Noisy estimates at pilots
%   H_interpolated - Interpolated estimates
%   pilotMask      - Pilot locations
%   metrics        - Performance metrics
%   cfg            - Configuration
%
% Outputs:
%   figs - Array of figure handles

figs = [];

% Select first Tx-Rx antenna pair and middle slot for visualization
slot_idx = ceil(size(H_perfect, 5) / 2);
rx_idx = 1;
tx_idx = 1;

%% Figure 1: Channel Magnitude Comparison
figs(end+1) = figure('Position', [100, 100, 1400, 900]);

% Get time-frequency grids for selected slot
H_perf = squeeze(abs(H_perfect(:, :, rx_idx, tx_idx, slot_idx)));
H_est = squeeze(abs(H_interpolated(:, :, rx_idx, tx_idx, slot_idx)));
H_err = abs(H_perf - H_est);
mask = pilotMask(:, :, slot_idx);

% Determine color limits for consistent scaling
c_max = max([H_perf(:); H_est(:)]);
c_min = 0;

% Subplot 1: Perfect channel
subplot(2, 2, 1);
imagesc(1:size(H_perf,2), 1:size(H_perf,1), H_perf);
xlabel('OFDM Symbol'); ylabel('Subcarrier');
title(sprintf('Perfect Channel (Slot %d, Rx%d-Tx%d)', slot_idx, rx_idx, tx_idx));
colorbar; axis xy; caxis([c_min c_max]);
colormap(gca, 'jet');

% Subplot 2: Estimated channel with pilot overlay
subplot(2, 2, 2);
imagesc(1:size(H_est,2), 1:size(H_est,1), H_est);
hold on;
% Mark pilot locations
[k_pilot, l_pilot] = find(mask);
plot(l_pilot, k_pilot, 'wo', 'MarkerSize', 3, 'LineWidth', 0.5);
hold off;
xlabel('OFDM Symbol'); ylabel('Subcarrier');
title('Interpolated Channel Estimate (○ = pilots)');
colorbar; axis xy; caxis([c_min c_max]);
colormap(gca, 'jet');

% Subplot 3: Absolute error
subplot(2, 2, 3);
imagesc(1:size(H_err,2), 1:size(H_err,1), H_err);
xlabel('OFDM Symbol'); ylabel('Subcarrier');
title('Absolute Error |H_{perfect} - H_{estimated}|');
colorbar; axis xy;
colormap(gca, 'hot');

% Subplot 4: Error with pilot overlay
subplot(2, 2, 4);
imagesc(1:size(H_err,2), 1:size(H_err,1), H_err);
hold on;
plot(l_pilot, k_pilot, 'wo', 'MarkerSize', 3, 'LineWidth', 0.5);
hold off;
xlabel('OFDM Symbol'); ylabel('Subcarrier');
title('Estimation Error (○ = pilots)');
colorbar; axis xy;
colormap(gca, 'hot');

sgtitle('Channel State Information Comparison');

%% Figure 2: Time-Frequency Error Evolution
if size(H_perfect, 5) > 1
    figs(end+1) = figure('Position', [150, 150, 1400, 600]);
    
    imagesc(metrics.error_tf);
    xlabel('OFDM Symbol (across slots)'); ylabel('Subcarrier');
    title('Estimation Error Evolution Over Time');
    colorbar;
    axis xy;
    colormap('hot');
end

%% Figure 3: Performance Metrics
figs(end+1) = figure('Position', [200, 200, 1400, 800]);

% Subplot 1: NMSE per slot
subplot(2, 2, 1);
plot(1:length(metrics.nmse_per_slot), metrics.nmse_per_slot, '-o', 'LineWidth', 2);
xlabel('Slot'); ylabel('NMSE (dB)');
title('NMSE per Slot');
grid on;
ylim([min(metrics.nmse_per_slot)-2, max(metrics.nmse_per_slot)+2]);

% Subplot 2: NMSE per antenna
subplot(2, 2, 2);
imagesc(metrics.nmse_per_antenna);
xlabel('Tx Antenna'); ylabel('Rx Antenna');
title('NMSE per Antenna Pair (dB)');
colorbar; axis xy;
xticks(1:cfg.mimo.nTxAnts); yticks(1:cfg.mimo.nRxAnts);
colormap(gca, 'jet');

% Subplot 3: Metrics summary bar chart
subplot(2, 2, 3);
metrics_vals = [metrics.nmse_pilots_dB, metrics.nmse_interpolated_dB];
bar(metrics_vals);
xticklabels({'At Pilots', 'After Interpolation'});
ylabel('NMSE (dB)');
title('NMSE Comparison');
grid on;

% Subplot 4: Text summary
subplot(2, 2, 4);
axis off;
summary_text = {
    'Performance Summary:'
    ''
    sprintf('NMSE at Pilots: %.2f dB', metrics.nmse_pilots_dB)
    sprintf('NMSE Interpolated: %.2f dB', metrics.nmse_interpolated_dB)
    sprintf('EVM: %.2f%%', metrics.evm_percent)
    sprintf('Correlation: %.4f', metrics.correlation)
    ''
    'Configuration:'
    sprintf('SNR: %.1f dB', cfg.noise.SNR_dB)
    sprintf('Channel: %s', cfg.channel.DelayProfile)
    sprintf('Doppler: %.1f Hz', cfg.channel.MaximumDopplerShift)
    sprintf('Dense SRS Symbols: %d', cfg.srs_dense.NumSRSSymbols)
    sprintf('Sparse SRS Symbols: %d', cfg.srs_sparse.NumSRSSymbols)
};
text(0.1, 0.9, summary_text, 'VerticalAlignment', 'top', ...
    'FontSize', 10, 'FontName', 'FixedWidth');

sgtitle('Performance Metrics Analysis');

%% Figure 4: Channel Frequency Response Comparison
figs(end+1) = figure('Position', [250, 250, 1400, 600]);

% Select middle OFDM symbol
sym_idx = ceil(size(H_perfect, 2) / 2);

H_perf_freq = squeeze(H_perfect(:, sym_idx, rx_idx, tx_idx, slot_idx));
H_est_freq = squeeze(H_interpolated(:, sym_idx, rx_idx, tx_idx, slot_idx));

subplot(1, 2, 1);
plot(1:length(H_perf_freq), abs(H_perf_freq), 'b-', 'LineWidth', 2); hold on;
plot(1:length(H_est_freq), abs(H_est_freq), 'r--', 'LineWidth', 1.5);
% Mark pilots
if slot_idx <= size(pilotMask, 3)
    pilot_sc = find(pilotMask(:, sym_idx, slot_idx));
    plot(pilot_sc, abs(H_perf_freq(pilot_sc)), 'ko', 'MarkerSize', 6, 'LineWidth', 2);
end
xlabel('Subcarrier'); ylabel('|H|');
title(sprintf('Channel Frequency Response (Symbol %d)', sym_idx));
legend('Perfect', 'Estimated', 'Pilot Locations', 'Location', 'best');
grid on;

subplot(1, 2, 2);
plot(1:length(H_perf_freq), angle(H_perf_freq), 'b-', 'LineWidth', 2); hold on;
plot(1:length(H_est_freq), angle(H_est_freq), 'r--', 'LineWidth', 1.5);
if slot_idx <= size(pilotMask, 3)
    plot(pilot_sc, angle(H_perf_freq(pilot_sc)), 'ko', 'MarkerSize', 6, 'LineWidth', 2);
end
xlabel('Subcarrier'); ylabel('Phase (rad)');
title('Channel Phase Response');
legend('Perfect', 'Estimated', 'Pilot Locations', 'Location', 'best');
grid on;

sgtitle('Frequency Domain Channel Response');

if cfg.sim.verboseOutput
    fprintf('\nGenerated %d figures.\n', length(figs));
end

end