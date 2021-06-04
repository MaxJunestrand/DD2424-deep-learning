addpath Datasets/cifar-10-batches-mat/;

d = 3072;
K = 10;
%m = [d,50,50,K];                                                                                                                
%m = [d,50,30,20,20,10,10,10,10,K];                                          
m = [d,50,50,K];                                                              

noLayers = length(m)-1; 
lambda=0.005;
GDparams.n_batch = 100;                                                     
GDparams.eta = 0.001;
GDparams.etaMin = 1e-5;
GDparams.etaMax = 1e-1; 
GDparams.nt=0;                                                                  
GDparams.t = 1;

% Load data and make training and validation data
[X1, Y1, y1] = LoadBatch('data_batch_1.mat');                               
[X2, Y2, y2] = LoadBatch('data_batch_2.mat');                               
[X3, Y3, y3] = LoadBatch('data_batch_3.mat');                               
[X4, Y4, y4] = LoadBatch('data_batch_4.mat');                               
[X5, Y5, y5] = LoadBatch('data_batch_5.mat');                               
[Xt, Yt, yt] = LoadBatch('test_batch.mat');                                 

X5v = X5(:,(5001:10000));                                                   
Y5v = Y5(:,(5001:10000));                                                   
y5v = y5((5001:10000),1);                                                   

X5 = X5(:,(1:5000));                                                       
Y5 = Y5(:,(1:5000));                                                      
y5 = y5((1:5000),1);                                                      

Xtrain = [X1,X2,X3,X4,X5];                                               
Ytrain = [Y1,Y2,Y3,Y4,Y5];                                                
ytrain = [y1;y2;y3;y4;y5];

NetParams = init(noLayers, m, K, d);                            

%gradients = computeGradients(NetParams, X1(:,(1:5)), Y1(:,(1:5)), 0, true);
%Grads = ComputeGradsNumSlow(X1(:,(1:5)), Y1(:,(1:5)), NetParams, 0, 1e-4);

% eps = 0.001;

% Test Gradients
%gradients = computeGradients(NetParams, X1(:,(1:10)), Y1(:,(1:10)),0);             
%Grads = ComputeGradsNumSlow(X1(:,(1:10)), Y1(:,(1:10)), NetParams, 0, 1e-5);
% gradcheck_b1 = sum(abs(Grads.b{1} - grad_b{1})/max(eps, sum(abs(Grads.b{1}) + abs(gradients.b{1}))));
% gradcheck_W1 = sum(sum(abs(Grads.W{1} - grad_W{1})/max(eps, sum(sum(abs(Grads.W{1}) + abs(gradients.W{1}))))));
% gradcheck_b2 = sum(abs(Grads.b{2} - grad_b{2})/max(eps, sum(abs(Grads.b{2}) + abs(gradients.b{2}))));
% gradcheck_W2 = sum(sum(abs(Grads.W{2} - grad_W{2})/max(eps, sum(sum(abs(Grads.W{2}) + abs(gradients.W{2}))))));
% gradcheck_W = [gradcheck_W1, gradcheck_W2];
% gradcheck_b = [gradcheck_b1, gradcheck_b2];

% % Course Search
% accuracies = zeros(1, 8);
% lambdas = zeros(1, 8);
% l_min = -3;
% l_max = -3;
% 
% % Fine search
% for i=1:8
%     % Course Search
% %     l = l_min + (l_max - l_min)*rand(1, 1);
% %     lambda = 10^l;
%     
%     % Fine search
%     a = 0.0015;
%     b = 0.0030;
%     lambda = (b-a).*rand(1,1) + a;
%     
%     [accuracyValidation, ~, ~, ~, ~, NetParams] = training(NetParams,Xtrain,Ytrain,ytrain,X5v,Y5v,y5v,GDparams,lambda);
%     lambdas(1, i) = lambda;
%     accuracies(1, i) = accuracyValidation(1, end);
% end
%
lambda= 0.002930824; % Best lambda found

[~, ~, ~, ~, NetParams] = training(NetParams,Xtrain,Ytrain,ytrain,X5v,Y5v,y5v,GDparams,lambda);

function [accuracyValidation, accuracyTraining, lossValidation, lossTraining, NetParams] = training (NetParams,Xt,Yt,yt,Xv,Yv,yv,GDparams,lambda)
n = size(Xt,2);

GDparams.ns = (5*45)/GDparams.n_batch;                                                         
GDparams.n_epochs = 135;                                                        

lossTraining = zeros(1, GDparams.n_epochs);
lossValidation = zeros(1, GDparams.n_epochs);
accuracyTraining = zeros(1, GDparams.n_epochs);
accuracyValidation = zeros(1, GDparams.n_epochs);
for i = 1:(GDparams.n_epochs)
        %lossValidation = [0,0,0,0];
        %lossTraining = [0,0,0,0];
        accuracyValidation = [0,0,0,0];
        for j=1:(n/GDparams.n_batch)        
            j_start = (j-1)*GDparams.n_batch + 1;
            j_end = j*GDparams.n_batch;
            inds = j_start:j_end;
            Xbatch = Xt(:, inds); 
            Ybatch = Yt(:, inds);
            
            if(GDparams.t <= GDparams.ns)
                GDparams.nt = GDparams.etaMin + (GDparams.t/GDparams.ns)*(GDparams.etaMax-GDparams.etaMin);
            else
                GDparams.nt = GDparams.etaMax - ((GDparams.t-GDparams.ns)/GDparams.ns)*(GDparams.etaMax-GDparams.etaMin);
            end
            
            GDparams.t = mod((GDparams.t+1), 2*GDparams.ns);
            [NetParams, mu_exp, v_exp] = MiniBatchGD(Xbatch, Ybatch, GDparams, NetParams, lambda, true);    
        end
        [~, lossTraining(1, i)] = ComputeCost(Xt, Yt, NetParams, lambda, false, mu_exp, v_exp);
        [~, lossValidation(1, i)] = ComputeCost(Xv, Yv, NetParams, lambda, false, mu_exp, v_exp);
        %accuracyTraining(1,i) = ComputeAccuracy(Xt, yt, NetParams,false,mu_exp,v_exp);
        if i == GDparams.n_epochs
            accuracyValidation(1,i) = ComputeAccuracy(Xv, yv, NetParams,false,mu_exp,v_exp);
        end
        [Xt, Yt, yt] = ShuffleData(Xt, Yt, yt);    
end  
end


function [NetParams, mu_exp, v_exp] = MiniBatchGD(Xbatch, Ybatch, GDparams, NetParams, lambda, train, varargin)
[gradients, mu_exp, v_exp] = computeGradients(NetParams,Xbatch,Ybatch,lambda, train, varargin);               
                                      
    for i=1:size(NetParams.W,2)                                                   
        NetParams.W{i} = NetParams.W{i} - gradients.W{i}*GDparams.nt;       
        NetParams.b{i} = NetParams.b{i} - gradients.b{i}*GDparams.nt;       
    end
    
    if NetParams.use_bn 
        for i=1:(length(NetParams.beta))
            NetParams.beta{i} = NetParams.beta{i} - gradients.beta{i}*GDparams.nt; 
            NetParams.gamma{i} = NetParams.gamma{i} - gradients.gamma{i}*GDparams.nt; 
        end
    end

end

function s_hat = BatchNormalize(score, mean, var)                       
    s_hat =(diag(var+eps))^(-1/2)*(score-mean);                       
end

function [Xv,s,s_hat,P,mu,v, mu_exp, v_exp] = EvaluateClassifier(X, NetParams, train, varargin)        
Xvar = NetParams.noLayers;                                                  
Xv = cell(1,Xvar);                                                        
Xv{1} = X;                                                                                                                                                                            
s = cell(1,(NetParams.noLayers-1));                                      
s_hat = cell(1,(NetParams.noLayers-1));                                       
mu_exp = cell(1,(NetParams.noLayers-1));
v_exp = cell(1,(NetParams.noLayers-1));
mu = cell(1,(NetParams.noLayers-1));                                  
v = cell(1,(NetParams.noLayers-1));
alpha = 0.9;                                                                                                     

if not(train)                                                               
    mu = varargin{1}{1};
    v = varargin{1}{2};
end

for i=1:(NetParams.noLayers-1)
        if NetParams.use_bn                                                 
            n = size(Xv{i},2);

            s{i} = (NetParams.W{i})*Xv{i} + NetParams.b{i};                  
            if train                                                       
                mu_exp{i}=zeros(NetParams.m(i+1),1);
                v_exp{i}=zeros(NetParams.m(i+1),1);
                mu{i} = mean(s{i},2);                                        
               v{i} = var(s{i},0,2)*(n-1)/n;                                 
                
                mu_exp{i} = alpha * mu{i} + (1 - alpha) * mu{i};
                v_exp{i} = alpha * v{i} + (1 - alpha) * v{i};                  
            end    
            s_hat{i} = BatchNormalize(s{i}, mu{i}, v{i});                     
            scoreThilde = (NetParams.gamma{i}).*s_hat{i} + NetParams.beta{i}; 
            x = max(0,scoreThilde);                                    
            Xv{i+1} = x;                                                  
        else                                                                
            s{i} = (NetParams.W{i})*Xv{i} + NetParams.b{i};                  
            Xv{i+1} = max(0,s{i});                                           
        end
end                                                                         

S = NetParams.W{NetParams.noLayers}*Xv{NetParams.noLayers}+NetParams.b{NetParams.noLayers};
P = softmax(S);
end

function [gradients, mu_exp, v_exp] = computeGradients(NetParams, X, Y, lambda, train, varargin) 
% Set up
noLayers = NetParams.noLayers;
grad_W = cell(1,noLayers);
grad_b = cell(1,noLayers);
grad_gamma = cell(1,noLayers);
grad_beta = cell(1,noLayers);


[Xvec,s,s_hat,P,mu,v, mu_exp, v_exp] = EvaluateClassifier(X, NetParams, train, varargin);

G_batch = -(Y -P);
n_b = size(X, 2);

% layer k gradients
grad_W{noLayers} = 1/n_b * G_batch * Xvec{noLayers}' + 2*lambda * NetParams.W{noLayers};
grad_b{noLayers} = 1/n_b * G_batch * ones(n_b,1);
G_batch = NetParams.W{noLayers}' * G_batch;
G_batch = G_batch .* (Xvec{noLayers}>0);

grad_gamma{noLayers} = zeros(NetParams.m(end),1);
grad_beta{noLayers} = zeros(NetParams.m(end),1);

% k-1 layers gradients
for i = (noLayers-1):-1:1
    if NetParams.use_bn
        % 1. Compute gradient for the scale and offset parameters for layer l:
        grad_gamma{i} = 1/n_b * ( G_batch .* s_hat{i})*ones(n_b,1);
        grad_beta{i} = 1/n_b * G_batch * ones(n_b,1);
        
        % 2. Propagate the gradients through the scale and shift
        G_batch = G_batch .* (NetParams.gamma{i}*ones(n_b,1)');  
        
        % 3. Propagate G_batch through the batch normalization
        G_batch = BatchNormBackPass(G_batch, s{i}, mu{i},v{i});
    end        
    % 4. The gradients of J w.r.t. bias vector b_l and W_l

    grad_W{i} = 1/n_b * G_batch * (Xvec{i}') + 2*lambda * NetParams.W{i};
    grad_b{i} = 1/n_b * G_batch * ones(n_b,1);
    
    % 5. If l>1 progpagate G_{batch} to previous layer    
    if i>1
        G_batch = NetParams.W{i}'*G_batch;
        G_batch = G_batch .* (Xvec{i}>0);
    end
end

gradients.W = grad_W;
gradients.b = grad_b;
gradients.gamma = grad_gamma;
gradients.beta = grad_beta;

end

function g = BatchNormBackPass(g, s, mu, v)
[~,n_b] = size(g);    
sigma_1 = (v+eps).^(-0.5);
sigma_2 = (v+eps).^(-1.5);

g1 = g .*(sigma_1*ones(n_b,1)');
g2 = g .*(sigma_2*ones(n_b,1)');

D = s - mu * ones(n_b,1)';
c = (g2 .*D)*ones(n_b,1);

g = g1 - 1/n_b * (g1 * ones(n_b,1))*ones(n_b,1)' - 1/n_b*D .*(c*ones(n_b,1)');
end

function [cost, loss] = ComputeCost(X, Y, NetParams, lambda, train, varargin)
    [~, ~, ~, P, ~] = EvaluateClassifier(X, NetParams, train, varargin);
    loss = sum(-log(sum(Y.*P,1)))/size(X,2);
    cost = loss;
    for i=1:size(NetParams.W, 2)
        cost = cost + lambda*sum(sum(NetParams.W{i}.^2));
    end
end

function NetParams = init(noLayers, m, K, d)
    
    NetParams.use_bn = true;                                                
    NetParams.noLayers = noLayers;                                          
    NetParams.m = m;                                                        
    NetParams.K = K;                                                        
    NetParams.d = d;                                                        
    
    NetParams.W = cell(length(noLayers),1);                                 
    NetParams.b = cell(length(noLayers),1);                                 
    NetParams.gamma = cell(length(noLayers),1);
    %NetParams.W{1} = He(NetParams.m(2),NetParams.m(1));
    NetParams.W{1} = sigma(NetParams.m(2),NetParams.m(1));
    NetParams.b{1} = zeros(NetParams.m(2),1);                  
    NetParams.gamma{1} = ones(NetParams.m(2), 1);
    NetParams.beta{1} = zeros(NetParams.m(2),1); 
                           
    if noLayers > 2
        for i=2:(noLayers - 1)
            %NetParams.W{i} = He(NetParams.m(i+1),NetParams.m(i));
            NetParams.W{i} = sigma(NetParams.m(i+1),NetParams.m(i));
            NetParams.b{i} = zeros(NetParams.m(i+1),1);          
            NetParams.gamma{i} = ones(NetParams.m(i+1),1);
            NetParams.beta{i} = zeros(NetParams.m(i+1),1);                  
        end
    end
    %NetParams.W{noHiddenLayers+1} = He(NetParams.m(end),NetParams.m(end-1));
    NetParams.W{noHiddenLayers+1} = sigma(NetParams.m(end),NetParams.m(end-1));
    NetParams.b{noHiddenLayers+1} = zeros(NetParams.m(end),1);                     
    NetParams.gamma{noHiddenLayers+1} = ones(NetParams.m(end),1);                     
    NetParams.beta{noHiddenLayers+1} = zeros(NetParams.m(end),1);           
end

function W = He(r,c)  %He initialization  
standardDev = sqrt(2)/sqrt(c);
W = standardDev*randn(r,c);
end

function W = sigma(r, c)
    sig = 1e-1;
    %sig = 1e-3;
    %sig = 1e-4;
    W = normrnd(0,sig,r,c);
end

function acc = ComputeAccuracy(X, y, NetParams, train, varargin)                             % only modification compared to assignment2 is the NetParams as input
[~,~,~,P,~,~, ~,~] = EvaluateClassifier(X, NetParams, train, varargin);                      % create probability matrix                               
ncol = size(P,2);                                                           % no of pictures in the dataset
noCorrectClassifications = 0;                                               % variable
    for i = 1:ncol                                                          % for each picture
        if P(y(i),i) == max(P(:,i))                                         % if the predicted class is the same as the correct class
            noCorrectClassifications = noCorrectClassifications + 1;        % increment no of correct classifications
        end
    end
acc = noCorrectClassifications/ncol;  
end

function picNum = figcreator(picNum, scale, validationVector, Vector)
 figure(picNum)
    hold on;
    plot(validationVector);
    plot(Vector);
    legend('validationVector','trainingVector')
    ylim([0 scale]);
    hold off 
end

function [X, Y, y] = ShuffleData(X, Y, y)
n = size(X, 2);
shuffled = randperm(n);
X = X(:, shuffled);
Y = Y(:, shuffled);
y = y(shuffled, :);
end

function [X,Y,y] = LoadBatch(filename)                                      
A = load(filename);
X = double(A.data');
y = double(A.labels')+1;
Y = onehotencode(categorical(A.labels),2)';
mean_X = mean(X, 2);                                                                                          
std_X = std(X, 0, 2);                                                  

Xs = X - repmat(mean_X, [1, size(X, 2)]);                     
X = Xs ./repmat(std_X, [1, size(X,2)]);                            
end


function Grads = ComputeGradsNumSlow(X, Y, NetParams, lambda, h)

Grads.W = cell(numel(NetParams.W), 1);
Grads.b = cell(numel(NetParams.b), 1);
if NetParams.use_bn
    Grads.gamma = cell(numel(NetParams.gamma), 1);
    Grads.beta = cell(numel(NetParams.beta), 1);
end

for j=1:length(NetParams.b)
    Grads.b{j} = zeros(size(NetParams.b{j}));
    NetTry = NetParams;
    for i=1:length(NetParams.b{j})
        
        b_try = NetParams.b;
        b_try{j}(i) = b_try{j}(i) - h;
        NetTry.b = b_try;
        [c1, ~] = ComputeCost(X, Y, NetTry, lambda, true);        
        
        b_try = NetParams.b;
        b_try{j}(i) = b_try{j}(i) + h;
        NetTry.b = b_try;        
        [c2, ~] = ComputeCost(X, Y, NetTry, lambda, true);
        
        Grads.b{j}(i) = (c2-c1) / (2*h);
    end
end

for j=1:length(NetParams.W)
    Grads.W{j} = zeros(size(NetParams.W{j}));
        NetTry = NetParams;
    for i=1:numel(NetParams.W{j})
        
        W_try = NetParams.W;
        W_try{j}(i) = W_try{j}(i) - h;
        NetTry.W = W_try;        
        [c1, ~] = ComputeCost(X, Y, NetTry, lambda, true);
    
        W_try = NetParams.W;
        W_try{j}(i) = W_try{j}(i) + h;
        NetTry.W = W_try;        
        [c2, ~] = ComputeCost(X, Y, NetTry, lambda, true);
    
        Grads.W{j}(i) = (c2-c1) / (2*h);
    end
end

if NetParams.use_bn
    for j=1:length(NetParams.gamma)
        Grads.gamma{j} = zeros(size(NetParams.gamma{j}));
        NetTry = NetParams;
        for i=1:numel(NetParams.gamma{j})
            
            gamma_try = NetParams.gamma;
            gamma_try{j}(i) = gamma_try{j}(i) - h;
            NetTry.gamma = gamma_try;        
            [c1, ~] = ComputeCost(X, Y, NetTry, lambda, true);
            
            gamma_try = NetParams.gamma;
            gamma_try{j}(i) = gamma_try{j}(i) + h;
            NetTry.gamma = gamma_try;        
            [c2, ~] = ComputeCost(X, Y, NetTry, lambda, true);
            
            Grads.gamma{j}(i) = (c2-c1) / (2*h);
        end
    end
    
    for j=1:length(NetParams.beta)
        Grads.beta{j} = zeros(size(NetParams.beta{j}));
        NetTry = NetParams;
        for i=1:numel(NetParams.beta{j})
            
            beta_try = NetParams.beta;
            beta_try{j}(i) = beta_try{j}(i) - h;
            NetTry.beta = beta_try;        
            [c1, ~] = ComputeCost(X, Y, NetTry, lambda, true);
            
            beta_try = NetParams.beta;
            beta_try{j}(i) = beta_try{j}(i) + h;
            NetTry.beta = beta_try;        
            [c2, ~] = ComputeCost(X, Y, NetTry, lambda, true);
            
            Grads.beta{j}(i) = (c2-c1) / (2*h);
        end
    end    
end
end