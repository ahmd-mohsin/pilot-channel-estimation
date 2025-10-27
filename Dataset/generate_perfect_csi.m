function [H_perfect, txGrid, pathGains_all] = generate_perfect_csi(cfg, channel, chInfo)
% GENERATE_PERFECT_CSI - Generate "perfect" CSI using dense pilot sounding
%
% This represents the best achievable CSI estimate using the full sounding
% sequence (dense pilots across entire bandwidth and time). The ground truth
% comes from the simulator's channel realization.
%
% Inputs:
%   cfg     - Configuration structure
%   channel - nrTDLChannel object
%   chInfo  - Channel information
%
% Outputs:
%   H_perfect      - Perfect channel estimates [K x L x nRx x nTx x numSlots]
%   txGrid         - Transmitted grids for all slots
%   pathGains_all  - Path gains for all slots (ground truth)

% Initialize storage
numSlots = cfg.derived.numSlots;
K = cfg.derived.K;
L = cfg.derived.L;
nTxAnts = cfg.mimo.nTxAnts;
nRxAnts = cfg.mimo.nRxAnts;

% Storage for results
H_perfect = zeros(K, L, nRxAnts, nTxAnts, numSlots);
txGrid = zeros(K, L, nTxAnts, numSlots);
pathGains_all = cell(numSlots, 1);

% Maximum channel delay for zero padding
maxChDelay = chInfo.MaximumChannelDelay;

% Reset channel
reset(channel);

if cfg.sim.verboseOutput
    fprintf('\nGenerating Perfect CSI (Dense Pilots)...\n');
end

% Create carrier copy for iteration
carrier = cfg.nrCarrier;

% Generate dense SRS configuration
[~, ~, srsDense, ~] = generate_srs(carrier, cfg.srs_dense);

for nSlot = 0:numSlots-1
    % Update slot
    carrier.NSlot = nSlot;
    
    % Generate SRS for this slot
    [srsSymbols, srsIndices, ~, ~] = generate_srs(carrier, cfg.srs_dense);
    
    % Check if this slot has SRS
    isSRSSlot = ~isempty(srsSymbols);
    
    % Create resource grid
    grid = nrResourceGrid(carrier, nTxAnts);
    
    if isSRSSlot
        % Map SRS to grid
        grid(srsIndices) = srsSymbols;
    end
    
    % Store transmitted grid
    txGrid(:, :, :, nSlot+1) = grid;
    
    % OFDM modulation
    [txWaveform, waveformInfo] = nrOFDMModulate(carrier, grid);
    
    % Zero pad for channel delay
    txWaveform = [txWaveform; zeros(maxChDelay, size(txWaveform, 2))];
    
    % Pass through channel (get path gains for ground truth)
    [rxWaveform, pathGains] = channel(txWaveform);
    
    % Store path gains (ground truth)
    pathGains_all{nSlot+1} = pathGains;
    
    % Add AWGN
    SNR = 10^(cfg.noise.SNR_dB/10);
    N0 = 1/sqrt(2.0 * nRxAnts * double(waveformInfo.Nfft) * SNR);
    noise = N0 * complex(randn(size(rxWaveform)), randn(size(rxWaveform)));
    rxWaveform = rxWaveform + noise;
    
    % Perfect timing synchronization
    pathFilters = getPathFilters(channel);
    offset = nrPerfectTimingEstimate(pathGains, pathFilters);
    
    % OFDM demodulation
    rxGrid = nrOFDMDemodulate(carrier, rxWaveform(1+offset:end, :));
    
    % Perfect channel estimation using known channel response
    H_perfect_slot = nrPerfectChannelEstimate(carrier, pathGains, pathFilters, offset);
    
    % Store (pad if necessary)
    if size(H_perfect_slot, 2) >= L
        H_perfect(:, :, :, :, nSlot+1) = H_perfect_slot(:, 1:L, :, :);
    else
        H_perfect(:, 1:size(H_perfect_slot,2), :, :, nSlot+1) = H_perfect_slot;
    end
    
    if cfg.sim.verboseOutput && mod(nSlot, 5) == 0
        fprintf('  Slot %d/%d processed\n', nSlot+1, numSlots);
    end
end

if cfg.sim.verboseOutput
    fprintf('Perfect CSI generation complete.\n');
    fprintf('  Shape: [%d x %d x %d x %d x %d] (K x L x nRx x nTx x Slots)\n', ...
        size(H_perfect));
end

end