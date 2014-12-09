function [trainlabels,trainimages,testlabels,testimages] = cnnload()

trlblid = fopen('train-labels.idx1-ubyte');
trimgid = fopen('train-images.idx3-ubyte');
tslblid = fopen('t10k-labels.idx1-ubyte');
tsimgid = fopen('t10k-images.idx3-ubyte');

% read train labels
fread(trlblid, 4);
numtrlbls = toint(fread(trlblid, 4));
trainlabels = fread(trlblid, numtrlbls);

% read train data
fread(trimgid, 4);
numtrimg = toint(fread(trimgid, 4));
trimgh = toint(fread(trimgid, 4));
trimgw = toint(fread(trimgid, 4));
trainimages = permute(reshape(fread(trimgid,trimgh*trimgw*numtrimg),trimgh,trimgw,numtrimg), [2 1 3]);

% read test labels
fread(tslblid, 4);
numtslbls = toint(fread(tslblid, 4));
testlabels = fread(tslblid, numtslbls);

% read test data
fread(tsimgid, 4);
numtsimg = toint(fread(tsimgid, 4));
tsimgh = toint(fread(tsimgid, 4));
tsimgw = toint(fread(tsimgid, 4));
testimages = permute(reshape(fread(tsimgid, tsimgh*tsimgw*numtsimg),tsimgh,tsimgw,numtsimg), [2 1 3]);

end

function [x] = toint(b)
    x = b(1)*16777216 + b(2)*65536 + b(3)*256 + b(4);
end

function [] = showimgs(img,lbl,h,w,fig)
    figure(fig)
    for i=1:h*w
        subplot(h,w,i);
        imshow(img(:,:,i));
        title(num2str(lbl(i)));
    end
end
