function [database, lenStat] = CalculateDenseSiftDescriptor(rt_img_dir, rt_data_dir)

disp('Extracting Dense SIFT features...');

subfolders = dir(rt_img_dir);

database = [];

database.imnum = 0; % total image number of the database
database.cname = {}; % name of each class
database.label = []; % label of each class
database.path = {}; % contain the pathes for each image of each class
database.nclass = 0;

conf.phowOpts = {'Step', 3} ;
model.phowOpts = conf.phowOpts ;

for ii = 1:length(subfolders),
    subname = subfolders(ii).name;
    
    if ~strcmp(subname, '.') & ~strcmp(subname, '..'),
        database.nclass = database.nclass + 1;
        
        database.cname{database.nclass} = subname;
        
        files = dir(fullfile(rt_img_dir, subname, '*.jpg'));
        
        c_num = length(files);           
        database.imnum = database.imnum + c_num;
        database.label = [database.label; ones(c_num, 1)*database.nclass];
        
        siftpath = fullfile(rt_data_dir, subname);        
        if ~isdir(siftpath),
            mkdir(siftpath);
        end
        
        for jj = 1:c_num,
            imgpath = fullfile(rt_img_dir, subname, files(jj).name);
            
            im = imread(imgpath);
            im = standarizeImage(im);
            [im_h, im_w] = size(im);
            
            fprintf('Processing %s: wid %d, hgt %d\n', ...
                     files(jj).name, im_w, im_h);

            % extract dense sift
            [frames, denseSiftArr] = vl_phow(im, model.phowOpts{:});
            feaSet.feaArr = denseSiftArr;
            feaSet.frames = frames;
            feaSet.width = im_w;
            feaSet.height = im_h;
            [pdir, fname] = fileparts(files(jj).name);
            fpath = fullfile(rt_data_dir, subname, [fname, '.mat']);
            save(fpath, 'feaSet');
            database.path = [database.path, fpath];
        end
    end
end
save('database.mat','database');

% -------------------------------------------------------------------------
function im = standarizeImage(im)
% -------------------------------------------------------------------------
    
im = im2single(im) ;
if size(im,1) > 480, im = imresize(im, [480 NaN]) ; end