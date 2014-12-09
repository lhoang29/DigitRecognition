function [missimages, misslabels] = cnntrain(trainlabels,trainimages,testlabels,testimages,maxtrain,iter,eta)

fn = 4; % number of kernels for layer 1
ks = 5; % size of kernel

[h,w,n] = size(trainimages);
n = min(n,maxtrain);

% normalize data to [-1,1] range
nitrain = (trainimages / 255) * 2 - 1;
nitest = (testimages / 255) * 2 - 1;

% train with backprop
h1 = h-ks+1;
w1 = w-ks+1;
A1 = zeros(h1,w1,fn);

h2 = h1/2;
w2 = w1/2;
I2 = zeros(h2,w2,fn);
A2 = zeros(h2,w2,fn);

A3 = zeros(10,1);

% kernels for layer 1
W1 = randn(ks,ks,fn) * .01;
B1 = ones(1,fn);

% scale parameter and bias for layer 2
S2 = randn(1,fn) * .01;
B2 = ones(1,fn);

% weights and bias parameters for fully-connected output layer
W3 = randn(h2,w2,fn,10) * .01;
B3 = ones(10,1);

% true outputs
Y = eye(10)*2-1;

for it=1:iter
    err = 0;
    for im=1:n
        %------------ FORWARD PROP ------------%
        % Layer 1: convolution with bias followed by sigmoidal squashing
        for fm=1:fn
            A1(:,:,fm) = convn(nitrain(:,:,im),W1(end:-1:1,end:-1:1,fm),'valid') + B1(fm);
        end
        Z1 = tanh(A1);

        % Layer 2: average/subsample with scaling and bias
        for fm=1:fn
            I2(:,:,fm) = avgpool(Z1(:,:,fm));
            A2(:,:,fm) = I2(:,:,fm) * S2(fm) + B2(fm);
        end
        Z2 = tanh(A2);

        % Layer 3: fully connected
        for cl=1:10
            A3(cl) = convn(Z2,W3(end:-1:1,end:-1:1,end:-1:1,cl),'valid') + B3(cl);
        end
        Z3 = tanh(A3); % Final output
        err = err + .5 * norm(Z3 - Y(:,trainlabels(im)+1),2)^2;

        %------------ BACK PROP ------------%
        % Compute error at output layer
        Del3 = (1 - Z3.^2) .* (Z3 - Y(:,trainlabels(im)+1));

        % Compute error at layer 2
        Del2 = zeros(size(Z2));
        for cl=1:10
            Del2 = Del2 + Del3(cl) * W3(:,:,:,cl);
        end
        Del2 = Del2 .* (1 - Z2.^2);

        % Compute error at layer 1
        Del1 = zeros(size(Z1));
        for fm=1:fn
            Del1(:,:,fm) = (S2(fm)/4)*(1 - Z1(:,:,fm).^2);
            for ih=1:h1
                for iw=1:w1
                    Del1(ih,iw,fm) = Del1(ih,iw,fm) * Del2(floor((ih+1)/2),floor((iw+1)/2),fm);
                end
            end
        end

        % Update bias at layer 3
        DB3 = Del3; % gradient w.r.t bias
        B3 = B3 - eta*DB3;

        % Update weights at layer 3
        for cl=1:10
            DW3 = DB3(cl) * Z2; % gradient w.r.t weights
            W3(:,:,:,cl) = W3(:,:,:,cl) - eta * DW3;
        end

        % Update scale and bias parameters at layer 2
        for fm=1:fn
            DS2 = convn(Del2(:,:,fm),I2(end:-1:1,end:-1:1,fm),'valid');
            S2(fm) = S2(fm) - eta * DS2;

            DB2 = sum(sum(Del2(:,:,fm)));
            B2(fm) = B2(fm) - eta * DB2;
        end

        % Update kernel weights and bias parameters at layer 1
        for fm=1:fn
            DW1 = convn(nitrain(:,:,im),Del1(end:-1:1,end:-1:1,fm),'valid');
            W1(:,:,fm) = W1(:,:,fm) - eta * DW1;

            DB1 = sum(sum(Del1(:,:,fm)));
            B1(fm) = B1(fm) - eta * DB1;
        end
    end
    disp(['Error: ' num2str(err) ' at iteration ' num2str(it)]);
end

miss = 0;
numtest=size(testimages,3);
missimages = zeros(1,numtest);
misslabels = zeros(1,numtest);
for im=1:numtest
    for fm=1:fn
        A1(:,:,fm) = convn(nitest(:,:,im),W1(end:-1:1,end:-1:1,fm),'valid') + B1(fm);
    end
    Z1 = tanh(A1);

    % Layer 2: average/subsample with scaling and bias
    for fm=1:fn
        I2(:,:,fm) = avgpool(Z1(:,:,fm));
        A2(:,:,fm) = I2(:,:,fm) * S2(fm) + B2(fm);
    end
    Z2 = tanh(A2);

    % Layer 3: fully connected
    for cl=1:10
        A3(cl) = convn(Z2,W3(end:-1:1,end:-1:1,end:-1:1,cl),'valid') + B3(cl);
    end
    Z3 = tanh(A3); % Final output
    
    [pm,pl] = max(Z3);
    if pl ~= testlabels(im)+1
        miss = miss + 1;
        missimages(miss) = im;
        misslabels(miss) = pl - 1;
    end
end
disp(['Miss: ' num2str(miss) ' out of ' num2str(numtest)]);

end

function [pr] = avgpool(img)
    pr = zeros(size(img)/2);
    for r=1:2:size(img,1)
        for c=1:2:size(img,2)
            pr((r+1)/2,(c+1)/2) = (img(r,c)+img(r+1,c)+img(r,c+1)+img(r+1,c+1))/4;
        end
    end
end