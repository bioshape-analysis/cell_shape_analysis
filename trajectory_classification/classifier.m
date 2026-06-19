loaded = load("../utrack_output.mat");
allTracks = loaded.tracks;

probDim = 2;       
plotRes = 0;       
peakAlpha = 95;  

results = struct();


trackNames = fieldnames(allTracks);
for i = 1:length(trackNames)
    trackName = trackNames{i};
    tracks = allTracks.(trackName);

    [transDiffResults, errFlag] = basicTransientDiffusionAnalysisv1(tracks, probDim, plotRes, peakAlpha);
    
    if isfield(transDiffResults.segmentClass, 'momentScalingSpectrum')
        results.(trackName).momentScalingSpectrum = transDiffResults.segmentClass.momentScalingSpectrum;
    end
end



h5FileName = 'time_events_filtered.h5';
if exist(h5FileName, 'file') == 2
    delete(h5FileName); 
end

trackNames = fieldnames(results);
for i = 1:length(trackNames)
    trackName = trackNames{i};
    data = results.(trackName).momentScalingSpectrum;
    if ~isempty(data)
        h5create(h5FileName, ['/', trackName], size(data));
        h5write(h5FileName, ['/', trackName], data);
    end
end

