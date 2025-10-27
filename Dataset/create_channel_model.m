function [channel, chInfo] = create_channel_model(cfg)
% CREATE_CHANNEL_MODEL - Creates and configures TDL or CDL channel model
%
% Inputs:
%   cfg - Configuration structure from config_system()
%
% Outputs:
%   channel - nrTDLChannel or nrCDLChannel object
%   chInfo  - Channel information structure

% Determine channel type from DelayProfile
delayProfile = cfg.channel.DelayProfile;

if startsWith(delayProfile, 'TDL')
    % Create TDL channel
    channel = nrTDLChannel;
    channelType = 'TDL';
elseif startsWith(delayProfile, 'CDL')
    % Create CDL channel (for 3GPP UMi/UMa scenarios)
    channel = nrCDLChannel;
    channelType = 'CDL';
else
    error('Unknown delay profile: %s. Use TDL-X or CDL-X.', delayProfile);
end

% Set common parameters
channel.DelayProfile = delayProfile;
channel.DelaySpread = cfg.channel.DelaySpread;
channel.MaximumDopplerShift = cfg.channel.MaximumDopplerShift;
channel.Seed = cfg.channel.Seed;

% Set antenna configuration (different between TDL and CDL)
if strcmp(channelType, 'TDL')
    channel.NumTransmitAntennas = cfg.channel.NumTransmitAntennas;
    channel.NumReceiveAntennas = cfg.channel.NumReceiveAntennas;
else  % CDL
    % CDL uses antenna arrays instead of antenna counts
    % Create simple antenna arrays
    channel.TransmitAntennaArray.Size = [cfg.mimo.nTxAnts, 1, 1, 1, 1];
    channel.ReceiveAntennaArray.Size = [cfg.mimo.nRxAnts, 1, 1, 1, 1];
end

% Set sample rate based on OFDM info
channel.SampleRate = cfg.derived.ofdmInfo.SampleRate;

% Get channel info
chInfo = info(channel);

if cfg.sim.verboseOutput
    fprintf('Channel Model Created (%s):\n', channelType);
    fprintf('  Delay Profile: %s\n', cfg.channel.DelayProfile);
    fprintf('  Delay Spread: %.1f ns\n', cfg.channel.DelaySpread * 1e9);
    fprintf('  Max Doppler: %.1f Hz\n', cfg.channel.MaximumDopplerShift);
    fprintf('  Max Channel Delay: %d samples\n', chInfo.MaximumChannelDelay);
    fprintf('  Sample Rate: %.3f MHz\n', channel.SampleRate / 1e6);
end

end