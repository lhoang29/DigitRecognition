function [X, W1, W2] = nntrain2(D, M, K, iter, eta)

% Execution options
vis = 0; % visualization of data
normalize = 1; % normalize data
recsig = 0; % use recommended tanh function

n = 1000; % number of samples for each cluster
bias = ones(1,n);

for k=1:K
    X(:,:,k) = randn(D,n) * 0.1 + 0.2*k;
    
    if vis
        if D == 1
            if mod(k,2) == 0
                plot(X(1,:,k), zeros(1,n), 'o');
            else
                plot(X(1,:,k), zeros(1,n), 'xr');
            end
        elseif D == 2
            if mod(k,2) == 0
                plot(X(1,:,k), X(2,:,k), 'o');
            else
                plot(X(1,:,k), X(2,:,k), 'xr');
            end
        elseif D == 3
            if mod(k,2) == 0
                plot3(X(1,:,k), X(2,:,k), X(3,:,k), 'o');
            else
                plot3(X(1,:,k), X(2,:,k), X(3,:,k), 'xr');
            end
        end
        grid on;
        hold on;
    end
end

if vis
    hold off;
end

if normalize
    rx = reshape(X,D,n*K);
    rx = rx';
    omean = mean(rx);
    ovar = var(rx);
    rx = (rx-repmat(omean,n*K,1)) ./ sqrt(repmat(ovar,n*K,1));
    X = reshape(rx',D,n,K);
%     
%     rx = rx - repmat(omean,1,n*K); % normalize to mean zero
%     omax = max(rx,[],2);
%     omin = min(rx,[],2);
%     rx = (rx - repmat(omin,1,n*K)) ./ repmat(omax-omin,1,n*K); % covariance equalization
%     rx = rx * 2 - 1; % scale to [-1,1]
%     X = reshape(rx,D,n,K);
end

% initialize weights
W1 = randn(M, D + 1);
W2 = randn(K, M + 1);

if recsig
    W1 = W1 / sqrt(D + 1);
    W2 = W2 / sqrt(M + 1);
end

% correct labels
Y = eye(K) * 2 - 1;

% train with backprop and batch gradient descent
for it=1:iter
    dEW1 = zeros(M, D + 1);
    dEW2 = zeros(K, M + 1);
    E = 0;
    for i=1:n
        for k=1:K
            Y0 = [X(:,i,k); 1];
            Y1 = W1 * Y0; % output for the first layer
            if recsig
                T = tanh(2*Y1/3);
                Z = 1.7159 * T;
            else
                Z = tanh(Y1); % nonlinearity
            end
            Y2 = W2 * [Z; 1]; % final output

            del2 = Y2 - Y(:,k); % error at output layer
            E = E + .5 * sum(del2.^2);

            dEW2 = dEW2 + del2 * [Y1; 1]'; % backprop error w.r.t weights in second layer
            if recsig
                del1 = 1.7159 * (2/3) * (1 - T.^2) .* (W2(:,1:M)' * del2);
            else
                del1 = (1 - Z.^2) .* (W2(:,1:M)' * del2);
            end
            dEW1 = dEW1 + del1 * Y0';
        end
    end
    if mod(it,10) == 0
        disp(['Error at iteration ' num2str(it) ': ' num2str(E)]);
    end
    W1 = W1 - eta * dEW1;
    W2 = W2 - eta * dEW2;
end

% test
miss = 0;
for k=1:K
    t = randn(D,n) * 0.1 + 0.2*k;
    if normalize
        t = (t - repmat(omean',1,n)) ./ sqrt(repmat(ovar',1,n));
    end
    for i=1:n
        Y0 = [t(:,i); 1];
        Y1 = W1 * Y0; % output for the first layer
        if recsig
            Z = 1.7159*tanh(2*Y1/3); % nonlinearity
        else
            Z = tanh(Y1); % nonlinearity
        end
        Y2 = W2 * [Z; 1]; % final output
        [vpred, lpred] = max(Y2);
        if lpred ~= k
            miss = miss + 1;
        end
    end
end

disp(['Missed ' num2str(miss) ' out of ' num2str(n * k)]);

end