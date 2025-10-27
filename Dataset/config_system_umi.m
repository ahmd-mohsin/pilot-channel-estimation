function cfg = config_system_umi()
% CONFIG_SYSTEM_UMI - Configuration for 3GPP Urban Micro (UMi) outdoor scenario
%
% This configuration implements:
% - 3GPP UMi-Street Canyon outdoor scenario with TDL-C channel
% - Small-scale fading (multipath)
% - Large-scale fading (path loss)
% - Shadowing (log-normal)
% - Doppler shift (user mobility)
% - Monte Carlo simulations

%% Carrier Configuration
cfg.carrier.NSizeGrid = 64;              % 64 RBs as requested
cfg.carrier.SubcarrierSpacing = 15;      % kHz: 15 for UMi outdoor
cfg.carrier.CyclicPrefix = 'Normal';     
cfg.carrier.NSlot = 0;                   
cfg.carrier.NFrame = 0;                  

%% MIMO Configuration
cfg.mimo.nTxAnts = 2;                    % UE transmit antennas
cfg.mimo.nRxAnts = 4;                    % gNB receive antennas (typical for UMi)
cfg.mimo.nLayers = min(cfg.mimo.nTxAnts, cfg.mimo.nRxAnts);

%% SRS Configuration (for "perfect" CSI with maximum pilots)
cfg.srs_dense.NumSRSSymbols = 4;         % Maximum practical symbols for sounding
cfg.srs_dense.SymbolStart = 10;          
cfg.srs_dense.NumSRSPorts = cfg.mimo.nTxAnts;
cfg.srs_dense.FrequencyStart = 0;        
cfg.srs_dense.NRRC = 0;                  
cfg.srs_dense.CSRS = 0;                  % 0 for maximum bandwidth (matches NSizeGrid)
cfg.srs_dense.BSRS = 0;                  
cfg.srs_dense.BHop = 0;                  % No hopping
cfg.srs_dense.KTC = 2;                   % Comb-2 (dense)
cfg.srs_dense.Repetition = 1;            
cfg.srs_dense.SRSPeriod = [1 0];         % Every slot
cfg.srs_dense.ResourceType = 'periodic'; 

%% SRS Configuration (for "noisy" CSI with sparse pilots)
cfg.srs_sparse = cfg.srs_dense;          
cfg.srs_sparse.NumSRSSymbols = 1;        % Sparse (realistic)
cfg.srs_sparse.KTC = 4;                  % Comb-4 (more sparse)
cfg.srs_sparse.SRSPeriod = [2 0];        % Every 2 slots

%% 3GPP UMi Channel Model Configuration (using TDL-C for compatibility)
cfg.channel.DelayProfile = 'TDL-C';      % TDL-C (NLOS urban, similar to CDL-C)
cfg.channel.DelaySpread = 251e-9;        % 251 ns for UMi (3GPP TR 38.901 Table 7.5-6)
cfg.channel.MaximumDopplerShift = 50;    % 50 Hz (moderate mobility, ~27 km/h @ 3.5 GHz)
cfg.channel.NumTransmitAntennas = cfg.mimo.nTxAnts;
cfg.channel.NumReceiveAntennas = cfg.mimo.nRxAnts;
cfg.channel.Seed = 42;                   % Will be varied in Monte Carlo

%% Large-Scale Fading Parameters (3GPP UMi)
% Based on 3GPP TR 38.901 Table 7.4.1
cfg.largescale.scenario = 'UMi-StreetCanyon';
cfg.largescale.frequency = 3.5e9;        % 3.5 GHz carrier
cfg.largescale.distance = 100;           % Distance between UE and gNB (meters)
cfg.largescale.hBS = 10;                 % BS height: 10m (UMi requirement)
cfg.largescale.hUT = 1.5;                % UE height: 1.5m (pedestrian/vehicle)
cfg.largescale.enablePathloss = true;    % Enable path loss
cfg.largescale.enableShadowing = true;   % Enable log-normal shadowing
cfg.largescale.shadowStdDev = 4;         % 4 dB for UMi NLOS (3GPP TR 38.901)

%% Noise Configuration
cfg.noise.SNR_dB = 15;                   % 15 dB SNR (challenging but realistic)

%% Monte Carlo Simulation Configuration
cfg.montecarlo.numRealizations = 1000;   % 1000 Monte Carlo runs as requested
cfg.montecarlo.varyChannelSeed = true;   % Different channel for each run
cfg.montecarlo.varyNoiseSeed = true;     % Different noise for each run
cfg.montecarlo.varyShadowing = true;     % Different shadowing for each run

%% Simulation Configuration
cfg.sim.numFrames = 1;                   % 1 frame per realization (10 slots)
cfg.sim.verboseOutput = false;           % Minimal output for Monte Carlo
cfg.sim.saveInterval = 100;              % Save progress every 100 realizations

%% Channel Estimation Configuration
cfg.estimation.method = 'LS';            
cfg.estimation.averagingWindow = [1 1];  
cfg.estimation.interpolation = 'linear'; 

%% Output paths
cfg.paths.outputDir = '/mnt/user-data/outputs';
cfg.paths.dataDir = fullfile(cfg.paths.outputDir, 'umi_montecarlo_data');

%% Derived parameters
cfg.nrCarrier = nrCarrierConfig;
cfg.nrCarrier.NSizeGrid = cfg.carrier.NSizeGrid;
cfg.nrCarrier.SubcarrierSpacing = cfg.carrier.SubcarrierSpacing;
cfg.nrCarrier.CyclicPrefix = cfg.carrier.CyclicPrefix;
cfg.nrCarrier.NSlot = cfg.carrier.NSlot;
cfg.nrCarrier.NFrame = cfg.carrier.NFrame;

cfg.derived.K = cfg.carrier.NSizeGrid * 12;  
cfg.derived.L = cfg.nrCarrier.SymbolsPerSlot; 
cfg.derived.numSlots = cfg.sim.numFrames * cfg.nrCarrier.SlotsPerFrame;
cfg.derived.ofdmInfo = nrOFDMInfo(cfg.nrCarrier);

%% UMi-specific derived parameters
% Path loss calculation (3GPP TR 38.901 equation 7.4-1)
cfg.derived.wavelength = 3e8 / cfg.largescale.frequency;
cfg.derived.pathLoss_dB = calculate_umi_pathloss(cfg);

end

function PL_dB = calculate_umi_pathloss(cfg)
% Calculate 3GPP UMi path loss (simplified)
% Based on TR 38.901 Table 7.4.1

fc_GHz = cfg.largescale.frequency / 1e9;
d = cfg.largescale.distance;
hBS = cfg.largescale.hBS;
hUT = cfg.largescale.hUT;

% UMi NLOS path loss (simplified)
% PL = 22.4 + 35.3*log10(d) + 21.3*log10(fc)
PL_dB = 22.4 + 35.3*log10(d) + 21.3*log10(fc_GHz);

% Add effective environment height correction
h_E = 1; % Effective environment height
PL_dB = PL_dB - 0.6*(hUT - 1.5);

% Ensure minimum distance
if d < 10
    PL_dB = 22.4 + 35.3*log10(10) + 21.3*log10(fc_GHz);
end

end