function [srsSymbols, srsIndices, srsConfig, srsIndInfo] = generate_srs(carrier, srsParams)
% GENERATE_SRS - Generate SRS symbols and resource element indices
%
% Inputs:
%   carrier   - nrCarrierConfig object
%   srsParams - SRS parameters structure (from config)
%
% Outputs:
%   srsSymbols - SRS symbols for transmission
%   srsIndices - Linear indices for mapping SRS to grid
%   srsConfig  - nrSRSConfig object
%   srsIndInfo - Additional indexing information

% Create SRS configuration object
srsConfig = nrSRSConfig;
srsConfig.NumSRSSymbols = srsParams.NumSRSSymbols;
srsConfig.SymbolStart = srsParams.SymbolStart;
srsConfig.NumSRSPorts = srsParams.NumSRSPorts;
srsConfig.FrequencyStart = srsParams.FrequencyStart;
srsConfig.NRRC = srsParams.NRRC;
srsConfig.CSRS = srsParams.CSRS;
srsConfig.BSRS = srsParams.BSRS;
srsConfig.BHop = srsParams.BHop;
srsConfig.KTC = srsParams.KTC;
srsConfig.Repetition = srsParams.Repetition;
srsConfig.SRSPeriod = srsParams.SRSPeriod;
srsConfig.ResourceType = srsParams.ResourceType;

% Generate SRS indices
[srsIndices, srsIndInfo] = nrSRSIndices(carrier, srsConfig);

% Generate SRS symbols
srsSymbols = nrSRS(carrier, srsConfig);

end