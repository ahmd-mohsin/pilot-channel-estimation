function metrics = evaluate_channel_estimates(H_perfect, H_noisy, H_interpolated, pilotMask, cfg)
% EVALUATE_CHANNEL_ESTIMATES - Compute performance metrics
%
% Inputs:
%   H_perfect      - Perfect channel estimates (ground truth from simulator)
%   H_noisy        - Raw noisy estimates at pilot locations
%   H_interpolated - Interpolated noisy estimates
%   pilotMask      - Pilot location mask
%   cfg            - Configuration structure
%
% Outputs:
%   metrics - Structure containing various performance metrics

if cfg.sim.verboseOutput
    fprintf('\nEvaluating Channel Estimates...\n');
end

%% Compute NMSE (Normalized Mean Square Error)
% NMSE at pilot locations (noisy vs perfect)
pilot_locations = repmat(pilotMask, [1, 1, 1, cfg.mimo.nRxAnts, cfg.mimo.nTxAnts]);
pilot_locations = permute(pilot_locations, [1, 2, 4, 5, 3]);

H_noisy_at_pilots = H_noisy;
H_perfect_at_pilots = H_perfect;

% Mask to only pilot locations
H_noisy_at_pilots(~pilot_locations) = 0;
H_perfect_at_pilots_masked = H_perfect;
H_perfect_at_pilots_masked(~pilot_locations) = 0;

% Compute NMSE at pilots
num_pilots = sum(pilot_locations(:) > 0);
if num_pilots > 0
    mse_pilots = sum(abs(H_noisy_at_pilots(:) - H_perfect_at_pilots_masked(:)).^2) / num_pilots;
    power_perfect_pilots = sum(abs(H_perfect_at_pilots_masked(:)).^2) / num_pilots;
    metrics.nmse_pilots_dB = 10*log10(mse_pilots / power_perfect_pilots);
else
    metrics.nmse_pilots_dB = NaN;
end

% NMSE after interpolation (full grid)
mse_interp = mean(abs(H_interpolated(:) - H_perfect(:)).^2);
power_perfect = mean(abs(H_perfect(:)).^2);
metrics.nmse_interpolated_dB = 10*log10(mse_interp / power_perfect);

% NMSE per slot
numSlots = size(H_perfect, 5);
metrics.nmse_per_slot = zeros(numSlots, 1);
for s = 1:numSlots
    mse_slot = mean(abs(H_interpolated(:,:,:,:,s) - H_perfect(:,:,:,:,s)).^2, 'all');
    power_slot = mean(abs(H_perfect(:,:,:,:,s)).^2, 'all');
    metrics.nmse_per_slot(s) = 10*log10(mse_slot / power_slot);
end

%% Compute EVM (Error Vector Magnitude)
% EVM is the RMS of the error normalized by RMS of perfect signal
evm_rms = sqrt(mean(abs(H_interpolated(:) - H_perfect(:)).^2));
signal_rms = sqrt(mean(abs(H_perfect(:)).^2));
metrics.evm_percent = 100 * (evm_rms / signal_rms);

%% Compute correlation
% Correlation between perfect and estimated channels
H_perf_vec = H_perfect(:);
H_est_vec = H_interpolated(:);
correlation = abs(sum(H_perf_vec .* conj(H_est_vec))) / ...
              sqrt(sum(abs(H_perf_vec).^2) * sum(abs(H_est_vec).^2));
metrics.correlation = correlation;

%% Per-antenna metrics
metrics.nmse_per_antenna = zeros(cfg.mimo.nRxAnts, cfg.mimo.nTxAnts);
for rx = 1:cfg.mimo.nRxAnts
    for tx = 1:cfg.mimo.nTxAnts
        H_perf_ant = H_perfect(:, :, rx, tx, :);
        H_est_ant = H_interpolated(:, :, rx, tx, :);
        
        mse_ant = mean(abs(H_est_ant(:) - H_perf_ant(:)).^2);
        power_ant = mean(abs(H_perf_ant(:)).^2);
        metrics.nmse_per_antenna(rx, tx) = 10*log10(mse_ant / power_ant);
    end
end

%% Time-frequency error distribution
[K, L, ~, ~, numSlots] = size(H_perfect);
metrics.error_tf = zeros(K, L*numSlots);
for s = 1:numSlots
    idx_start = (s-1)*L + 1;
    idx_end = s*L;
    error_slot = abs(H_interpolated(:,:,1,1,s) - H_perfect(:,:,1,1,s));
    metrics.error_tf(:, idx_start:idx_end) = error_slot;
end

%% Summary statistics
metrics.summary.nmse_pilots_dB = metrics.nmse_pilots_dB;
metrics.summary.nmse_interpolated_dB = metrics.nmse_interpolated_dB;
metrics.summary.evm_percent = metrics.evm_percent;
metrics.summary.correlation = metrics.correlation;
metrics.summary.mean_nmse_per_slot_dB = mean(metrics.nmse_per_slot);
metrics.summary.std_nmse_per_slot_dB = std(metrics.nmse_per_slot);

if cfg.sim.verboseOutput
    fprintf('Evaluation complete.\n');
    fprintf('  NMSE at pilots: %.2f dB\n', metrics.nmse_pilots_dB);
    fprintf('  NMSE after interpolation: %.2f dB\n', metrics.nmse_interpolated_dB);
    fprintf('  EVM: %.2f%%\n', metrics.evm_percent);
    fprintf('  Correlation: %.4f\n', metrics.correlation);
end

end