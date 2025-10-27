function [H_noisy, H_interpolated, pilotMask, nvar_est] = generate_noisy_csi(cfg, channel, chInfo)
% GENERATE_NOISY_CSI - Generate noisy CSI using sparse pilots + interpolation
%
% This represents practical CSI estimation from sparse pilots (like actual
% DM-RS/SRS patterns in real systems), followed by time/frequency interpolation.
%
% Inputs:
%   cfg     - Configuration structure
%   channel - nrTDLChannel object (will be reset to match perfect CSI)
%   chInfo  - Channel information
%
% Outputs:
%   H_noisy        - Noisy channel estimates at pilot locations [K x L x nRx x nTx x numSlots]
%   H_interpolated - Interpolated channel estimates [K x L x nRx x nTx x numSlots]
%   pilotMask      - Binary mask indicating pilot locations [K x L x numSlots]
%   nvar_est       - Estimated noise variance per slot

% Initialize storage
numSlots = cfg.derived.numSlots;
K = cfg.derived.K;
L = cfg.derived.L;
nTxAnts = cfg.mimo.nTxAnts;
nRxAnts = cfg.mimo.nRxAnts;

% Storage for results
H_noisy = zeros(K, L, nRxAnts, nTxAnts, numSlots);
H_interpolated = zeros(K, L, nRxAnts, nTxAnts, numSlots);
pilotMask = false(K, L, numSlots);
nvar_est = zeros(numSlots, 1);

% Maximum channel delay
maxChDelay = chInfo.MaximumChannelDelay;

% Note: Channel object passed in is already fresh with correct seed

if cfg.sim.verboseOutput
    fprintf('\nGenerating Noisy CSI (Sparse Pilots + Interpolation)...\n');
end

% Create carrier copy
carrier = cfg.nrCarrier;

% For holding estimates across slots
H_prev = zeros(K, L, nRxAnts, nTxAnts);

for nSlot = 0:numSlots-1
    % Update slot
    carrier.NSlot = nSlot;
    
    % Generate sparse SRS for this slot
    [srsSymbols, srsIndices, srsConfig, srsIndInfo] = generate_srs(carrier, cfg.srs_sparse);
    
    % Check if this slot has SRS
    isSRSSlot = ~isempty(srsSymbols);
    
    % Create resource grid
    grid = nrResourceGrid(carrier, nTxAnts);
    
    if isSRSSlot
        % Map SRS to grid
        grid(srsIndices) = srsSymbols;
        
        % Create pilot mask for this slot
        mask = false(K, L);
        [k, l, ~] = ind2sub([K, L, nTxAnts], srsIndices);
        for i = 1:length(k)
            mask(k(i), l(i)) = true;
        end
        pilotMask(:, :, nSlot+1) = mask;
    end
    
    % OFDM modulation
    [txWaveform, waveformInfo] = nrOFDMModulate(carrier, grid);
    
    % Zero pad for channel delay
    txWaveform = [txWaveform; zeros(maxChDelay, size(txWaveform, 2))];
    
    % Pass through channel
    [rxWaveform, pathGains] = channel(txWaveform);
    
    % Add AWGN
    SNR = 10^(cfg.noise.SNR_dB/10);
    N0 = 1/sqrt(2.0 * nRxAnts * double(waveformInfo.Nfft) * SNR);
    noise = N0 * complex(randn(size(rxWaveform)), randn(size(rxWaveform)));
    rxWaveform = rxWaveform + noise;
    
    % Practical timing synchronization
    if isSRSSlot
        % Use correlation-based timing estimation
        offset = nrTimingEstimate(carrier, rxWaveform, srsIndices, srsSymbols);
    else
        % Use offset from previous slot
        pathFilters = getPathFilters(channel);
        offset = nrPerfectTimingEstimate(pathGains, pathFilters);
    end
    
    % OFDM demodulation - ensure we have enough samples
    rxWaveform_sync = rxWaveform(1+offset:end, :);
    
    % Make sure we have enough samples for full slot
    minSamples = waveformInfo.SymbolLengths(1) * L;  % Minimum samples needed
    if size(rxWaveform_sync, 1) < minSamples
        % Pad with zeros if needed
        rxWaveform_sync = [rxWaveform_sync; zeros(minSamples - size(rxWaveform_sync, 1), size(rxWaveform_sync, 2))];
    end
    
    rxGrid = nrOFDMDemodulate(carrier, rxWaveform_sync);
    
    % Ensure rxGrid has correct number of symbols
    if size(rxGrid, 2) < L
        % Pad with zeros if needed
        rxGrid = [rxGrid, zeros(size(rxGrid, 1), L - size(rxGrid, 2), size(rxGrid, 3))];
    elseif size(rxGrid, 2) > L
        % Truncate if too many
        rxGrid = rxGrid(:, 1:L, :);
    end
    
    % Practical channel estimation
    if isSRSSlot
        % Estimate channel at pilot locations only
        [H_est, nvar] = nrChannelEstimate(carrier, rxGrid, srsIndices, srsSymbols, ...
            'AveragingWindow', cfg.estimation.averagingWindow);
        
        % Store noise variance estimate
        nvar_est(nSlot+1) = nvar;
        
        % Update channel estimate at pilot symbols
        firstSym = srsConfig.SymbolStart + 1;
        lastSym = srsConfig.SymbolStart + srsConfig.NumSRSSymbols;
        
        % Store raw estimates
        H_noisy(:, :, :, :, nSlot+1) = H_est;
        
        % Interpolation strategy:
        % 1. Hold previous estimate before first SRS symbol
        % 2. Use current estimate at SRS symbols
        % 3. Hold last estimate after last SRS symbol
        
        H_current = H_prev;
        
        % Update with new estimates where available
        H_current(:, firstSym:lastSym, :, :) = H_est(:, firstSym:lastSym, :, :);
        
        % Hold last valid estimate to end of slot
        if lastSym < L
            H_current(:, lastSym:L, :, :) = repmat(H_est(:, lastSym, :, :), 1, L-lastSym+1);
        end
        
        % Frequency interpolation for RBs without pilots (due to comb structure)
        H_current = interpolate_frequency(H_current, pilotMask(:, :, nSlot+1), ...
            cfg.estimation.interpolation);
        
        H_interpolated(:, :, :, :, nSlot+1) = H_current;
        H_prev = H_current;
        
    else
        % No SRS in this slot - hold previous estimate
        H_interpolated(:, :, :, :, nSlot+1) = H_prev;
        nvar_est(nSlot+1) = nvar_est(max(1, nSlot)); % Use previous noise estimate
    end
    
    if cfg.sim.verboseOutput && mod(nSlot, 5) == 0
        fprintf('  Slot %d/%d processed (SRS: %d)\n', nSlot+1, numSlots, isSRSSlot);
    end
end

if cfg.sim.verboseOutput
    fprintf('Noisy CSI generation complete.\n');
    fprintf('  Shape: [%d x %d x %d x %d x %d] (K x L x nRx x nTx x Slots)\n', ...
        size(H_interpolated));
    fprintf('  Average SNR: %.2f dB\n', cfg.noise.SNR_dB);
    fprintf('  Average noise variance: %.6f\n', mean(nvar_est));
end

end

function H_interp = interpolate_frequency(H, pilotMask, method)
% INTERPOLATE_FREQUENCY - Interpolate channel estimates in frequency
%
% Interpolates across subcarriers where pilots are missing due to comb structure

[K, L, nRx, nTx] = size(H);
H_interp = H;

for l = 1:L
    for rx = 1:nRx
        for tx = 1:nTx
            % Get pilot locations in this symbol
            pilotLocs = find(pilotMask(:, l));
            
            if length(pilotLocs) > 1
                % Get channel values at pilots
                pilotVals = H(pilotLocs, l, rx, tx);
                
                % Interpolate to all subcarriers
                allLocs = (1:K)';
                
                % Use interp1 with extrapolation
                H_interp(:, l, rx, tx) = interp1(pilotLocs, pilotVals, allLocs, ...
                    method, 'extrap');
            elseif length(pilotLocs) == 1
                % Only one pilot - replicate
                H_interp(:, l, rx, tx) = H(pilotLocs(1), l, rx, tx);
            end
            % If no pilots, keep previous value (already in H_interp)
        end
    end
end

end