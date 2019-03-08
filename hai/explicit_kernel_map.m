clear all; clc;

sift_dir = 'data/Caltech101';
hist_dir = 'features/Caltech101';
codebook_dir = 'dictionary';
numImgDictPerClass = 5;
pyramid = [2,4];
conf.numSpatialX = [2 4] ;
conf.numSpatialY = [2 4] ;
folders = dir(sift_dir);
descrs = {};
conf.phowOpts = {'Step', 3} ;
conf.numK = 1024;
conf.vocabPath = fullfile(codebook_dir, 'vocab.mat');
conf.histPath = fullfile(hist_dir, 'hists.mat');
conf.databasePath = fullfile(hist_dir, 'database.mat');
conf.quantizer = 'kdtree';
% --------------------------------------------------------------------
%                                                     Train vocabulary
% --------------------------------------------------------------------

if ~exist(conf.vocabPath)
    for i=1:length(folders),
        folder = folders(i).name;
        if ~strcmp(folder, '.') & ~strcmp(folder, '..'),
            feaFiles = dir(fullfile(sift_dir, folder, '*.mat'));
            num = length(feaFiles);
            idx_rand = randperm(num);
            idx_rand = idx_rand(1:numImgDictPerClass);
            for ii=1:numImgDictPerClass,
                fea_path = fullfile(sift_dir, folder, feaFiles(idx_rand(ii)).name);
                load(fea_path);
                descrs{end+1}=feaSet.feaArr;
            end;
        end;
    end;

    descrs = vl_colsubset(cat(2,descrs{:}), 10e5);
    descrs = single(descrs);
    vocab = vl_kmeans(descrs, conf.numK, 'verbose', 'algorithm', 'elkan', 'MaxNumIterations', 50) ;
    save(conf.vocabPath, 'vocab');
    
else
    load(conf.vocabPath);
end

conf.vocab = vocab;
if strcmp(conf.quantizer, 'kdtree')
  conf.kdtree = vl_kdtreebuild(vocab) ;
end

% --------------------------------------------------------------------
%                                           Compute spatial histograms
% --------------------------------------------------------------------
database.imnum = 0; % total image number of the database
database.cname = {}; % name of each class
database.label = []; % label of each class
database.path = {}; % contain the pathes for each image of each class
database.nclass = 0;

if ~exist(conf.histPath),
    hists = {};
    for i=1:length(folders),
        folder = folders(i).name;
        fprintf('Processing folder %s\n', ...
                     folder);
        if ~strcmp(folder, '.') & ~strcmp(folder, '..'),
            database.nclass = database.nclass + 1;
            database.cname{database.nclass} = folder;
            feaFiles = dir(fullfile(sift_dir, folder, '*.mat'));
            num = length(feaFiles);
            database.imnum = database.imnum + num;
            database.label = [database.label; ones(num, 1)*database.nclass];

            for ii=1:num,
                fea_path = fullfile(sift_dir, folder, feaFiles(ii).name);
                load(fea_path);
                hists{end+1} = getImageDescriptor(conf, feaSet);
            end
        end
    end
    hists = cat(2, hists{:});
    save(conf.histPath , 'hists');
    features = vl_homkermap(hists, 1, 'kchi2', 'gamma', .5);
    database.features = features;
    save(conf.databasePath , 'database', 'v7.3');
end


% -------------------------------------------------------------------------
function hist = getImageDescriptor(conf, feaSet)
% -------------------------------------------------------------------------
    
    width = feaSet.width;
    height = feaSet.height;

    numWords = size(conf.vocab, 2) ;
    
    % get PHOW features
    % [frames, descrs] = vl_phow(im, conf.phowOpts{:}) ;
    frames = feaSet.frames;
    descrs = feaSet.feaArr;

    % quantize local descriptors into visual words
    switch conf.quantizer
      case 'vq'
        [drop, binsa] = min(vl_alldist(conf.vocab, single(descrs)), [], 1) ;
      case 'kdtree'
        binsa = double(vl_kdtreequery(conf.kdtree, conf.vocab, ...
                                      single(descrs), ...
                                      'MaxComparisons', 50)) ;
    end
    
    for i = 1:length(conf.numSpatialX)
      binsx = vl_binsearch(linspace(1,width,conf.numSpatialX(i)+1), frames(1,:)) ;
      binsy = vl_binsearch(linspace(1,height,conf.numSpatialY(i)+1), frames(2,:)) ;
    
      % combined quantization
      bins = sub2ind([conf.numSpatialY(i), conf.numSpatialX(i), numWords], ...
                     binsy,binsx,binsa) ;
      hist = zeros(conf.numSpatialY(i) * conf.numSpatialX(i) * numWords, 1) ;
      hist = vl_binsum(hist, ones(size(bins)), bins) ;
      hists{i} = single(hist / sum(hist)) ;
    end
    
    hist = cat(1,hists{:}) ;
    hist = hist / sum(hist) ;

end

