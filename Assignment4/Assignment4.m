clear all                                                                   
close all                                                                   

% Open book and read in data
book_fname = 'goblet_book.txt';                                             
fid = fopen(book_fname,'r');                                                
book_data = fscanf(fid,'%c');                                               
fclose(fid);                                                                

book_chars = unique(book_data);                                             
K = length(book_chars);                                                     
hyperparam.m = 100;                                                         
hyperparam.seq_length = 25;                                                 
hyperparam.eta = 0.01; 
hyperparam.e = 1;                                                           
h0 = zeros(hyperparam.m,1);
sig = 0.01;                                                                  
RNN.b = zeros(hyperparam.m,1);
RNN.c = zeros(K,1);
RNN.U = randn(hyperparam.m,K)*sig;                                          
RNN.W = randn(hyperparam.m,hyperparam.m)*sig;                               
RNN.V = randn(K,hyperparam.m)*sig;                                          

% Given in assignment
char_to_ind = containers.Map('KeyType','char','ValueType','int32');           
ind_to_char = containers.Map('KeyType','int32','ValueType','char');         

ind_to_char = loadarrayToMapIndAsKey(ind_to_char,book_chars,K);             
char_to_ind = loadarrayToMapCharAsKey(char_to_ind,book_chars,K);            

% x0.n=10;                                                                   
% x0.x = '!';    
% Y = synthesizer(RNN,h0,x0, K);
% text = convertOneHotToText(Y, ind_to_char);

X_chars = book_data(1:hyperparam.seq_length);
Y_chars = book_data(2:hyperparam.seq_length+1);
X = convertDataToOneHot(X_chars,RNN,char_to_ind, K); 
Y = convertDataToOneHot(Y_chars,RNN,char_to_ind, K);
text = convertOneHotToText(X, ind_to_char);
[P,l,H] = forwardPass(RNN,h0,X,Y);

% Calculate Gradient Error rate %
% [grads,h] = backwardPass(RNN,h0,X, Y); 
% num_grads = ComputeGradsNum(X, Y, RNN, 1e-4); 
% 
% gradcheck_b = sum(abs(grads.b - num_grads.b)/max(eps, sum(abs(grads.b) + abs(num_grads.b))));
% gradcheck_c = sum(abs(grads.c - num_grads.c)/max(eps, sum(abs(grads.c) + abs(num_grads.c))));
% gradcheck_U = sum(abs(grads.U - num_grads.U)/max(eps, sum(abs(grads.U) + abs(num_grads.U))));
% gradcheck_W = sum(abs(grads.W - num_grads.W)/max(eps, sum(abs(grads.W) + abs(num_grads.W))));
% gradcheck_V = sum(abs(grads.V - num_grads.V)/max(eps, sum(abs(grads.V) + abs(num_grads.V))));

noReset = 0; 
countfourprint=1; 
bookLen = length(book_data);                                            
epoch = floor(bookLen/hyperparam.seq_length);
iterator = 100000;
%iterator = 7*epoch;                                                  
hprev = zeros(hyperparam.m,1); 
mu = 0.05;                                                                  
epsilon = 1e-8; 
smooth_loss=zeros(1,iterator); 
minloss = 1000;                                                             

for f = fieldnames(RNN)'
    m.(f{1}) = zeros(size(RNN.(f{1}))); 
end    

% Train
smooth_loss = train(smooth_loss, hyperparam, RNN, char_to_ind, K, iterator, book_data, hprev, minloss, m, mu, epsilon);
% Print loss function
% figCreatorOneVector(1,120,smooth_loss); 

function smooth_loss = train(smooth_loss, hyperparam, RNN, char_to_ind, K, iterator, book_data, hprev, minloss, m, mu, epsilon  )
    for i=1:iterator                                                           

        X_chars = book_data(hyperparam.e:((hyperparam.e)+(hyperparam.seq_length)-1));     
        Y_chars = book_data(hyperparam.e+1:hyperparam.e+hyperparam.seq_length);                    
        X = convertDataToOneHot(X_chars,RNN,char_to_ind,K);                           
        Y = convertDataToOneHot(Y_chars,RNN,char_to_ind,K);                          

        [grads,h,l] = backwardPass(RNN,hprev,X, Y);

        if l<minloss

            minloss=l;
            RNNstar=RNN;
            Xminstar=X_chars(1);
            hstar=hprev;
        end

        if i == 1
            smooth_loss(i) = l; 
        else    
            smooth_loss(i) = 0.999*smooth_loss(i-1) + 0.001*l;
        end
        if mod(i,10000) == 0 
            fprintf('smooth_loss=%f\n',smooth_loss(i));
            x0.n= 200; 
            x0.x=X_chars(1);
            syntehzisedData = synthesizer(RNN,hprev,x0,K);
            convSyntData = convertOneHotToText(syntehzisedData, ind_to_char);
            disp(convSyntData);
        end

        %AdaGrad
        for f = fieldnames(RNN)'
            %6
            m.(f{1}) = m.(f{1})+grads.(f{1}).^2;                                        
            %7
            RNN.(f{1}) = RNN.(f{1})-mu*grads.(f{1})./sqrt(m.(f{1})+epsilon);             
        end    

        hprev=h;                                                                     
        hyperparam.e=hyperparam.e+hyperparam.seq_length;                            

        if hyperparam.e > (length(book_data)-hyperparam.seq_length-1)
            hyperparam.e=1;
            noReset = noReset +1; 
            hprev = zeros(hyperparam.m,1);
        end

    end
end                                   

function map = loadarrayToMapIndAsKey(map,arr,K)                           
    for i=1:K                                                                   
        map(i)=arr(i);                                                        
    end
end

function map = loadarrayToMapCharAsKey(map,arr,K)                             
    for i=1:K                                                                   
        map(arr(i))=i;                                                         
    end
end

function [P,l,H,A] = forwardPass(RNN,h0,xChars, yChars)
    iterator = size(xChars,2); 
    l=0;
    hloop = h0;
    for i=1:iterator
        x = xChars(:,i);
        y = yChars(:,i);                                                            
        %1
        a = (RNN.W)*hloop + (RNN.U)*x + (RNN.b);                                    
        %2
        hloop = tanh(a);                                                            
        %3
        o = (RNN.V)*hloop + (RNN.c);                                                
        %4
        p = softmax(o);                                                             

        P(:,i) = p;
        H(:,i) = hloop; 
        A(:,i) = a;
        %5
        l = l - log((y')*p);                                                         
    end
end

function [grads,hret,l] = backwardPass(RNN,h0,xChars, yChars)
    [P,l,H,A] = forwardPass(RNN,h0,xChars, yChars);                            
    len = size(xChars,2);                                                       
    hret = H(:,len);

    dLdV = 0;                                                                   
    dLdW = 0;                                                                   
    dLdU = 0;                                                                   

    grads.b = 0;                                                                
    grads.c = 0;                                                                
    grads.U = 0;                                                                                 
    grads.W = 0;                                                                     
    grads.V = 0;                                                               
    % Loop backwards cause have to compute dLdht
    % Inspired by slides 
    for i=len:-1:1                                                                                                                                
        p = P(:,i);                                                            
        h = H(:,i);                                                                 
        at = A(:,i);                                                           
        if i == 1
            hMinusOne = h0;                                                     
        else
            hMinusOne = H(:,(i-1)); 
        end
        y = yChars(:,i);
        x = xChars(:,i);
        dLdOt = -(y-p)';   
        dLdV = dLdV + (dLdOt')*(h');                                            
        if i==len                                                              
            dLdht = dLdOt*(RNN.V);                                                  
        else
            dLdht = dLdOt*(RNN.V) + dLdatPlus1*(RNN.W);                             
        end
        dLdat = dLdht*diag(1-(tanh(at).^2));                                      
        dLdW = dLdW + (dLdat')*(hMinusOne');                                        
        dLdU = dLdU + (dLdat')*x';                                                  
        grads.c = grads.c + dLdOt';                                                                                                  
        grads.b = grads.b + diag(1 - h.^2)*(dLdht');                                 
        dLdatPlus1 = dLdat;                                                         
    end
    grads.W = dLdW;
    grads.U = dLdU;
    grads.V = dLdV;
    %Grad clip
    for f = fieldnames(grads)'                                                 
        grads.(f{1}) = max(min(grads.(f{1}),5),-5);                             
    end                       

end

function Y = synthesizer(RNN,h0,x0,K)                                                                                                  
    hloop=h0;
    Y = zeros(1, x0.n);
    for i=1:(x0.n)
        % 1
        a = (RNN.W)*hloop + (RNN.U)*x0.x + (RNN.b);                            
        % 2
        hloop = tanh(a);                                                       
        % 3
        o = (RNN.V)*hloop + (RNN.c);                                            
        p = softmax(o);

        cp = cumsum(p);                                                                    
        a = rand;                                                                    
        ixs = find(cp-a>0);                                                       
        ii = ixs(1);                                                                          
        onehotii = (ii==1:(K))';
        x0.x=onehotii;                                                                     
        Y(:,i) = onehotii;                                                                 
    end
end

function text = convertOneHotToText(Y,ind_to_char)
    text = zeros(size(Y,2));
    for i=1:(size(Y,2))                                                         
        for j=1:(size(Y,1))                                                    
            if Y(j,i) == 1 
                variable = j;                                                  
            end
        end 
    text(i) = ind_to_char(variable);                                  
    end
end

function oneHotMatrix = convertDataToOneHot(data,~,char_to_ind,K)

    for i=1:length(data)
        char = data(i);
        numRep=char_to_ind(char);
        oneHotMatrix(:,i) = (numRep==1:(K))';
    end
end

function l = ComputeLoss(xChars, yChars, RNN, h)            
    [~,l,~,~] = forwardPass(RNN,h,xChars, yChars);                             
end

function pic = figCreatorOneVector(pic, size, Vector)                             
 figure(pic)
    hold on;
    plot(Vector);
    legend('Vector')
    ylim([0 size]);    
    hold off 
end

function num_grads = ComputeGradsNum(X, Y, RNN, h)
    for f = fieldnames(RNN)'
        disp('Computing numerical gradient for')
        disp(['Field name: ' f{1} ]);
        num_grads.(f{1}) = ComputeGradNum(X, Y, f{1}, RNN, h);
    end
end
function grad = ComputeGradNum(X, Y, f, RNN, h)
    n = numel(RNN.(f));
    grad = zeros(size(RNN.(f)));
    hprev = zeros(size(RNN.W, 1), 1);
    for i=1:n
        RNN_try = RNN;
        RNN_try.(f)(i) = RNN.(f)(i) - h;
        l1 = ComputeLoss(X, Y, RNN_try, hprev);
        RNN_try.(f)(i) = RNN.(f)(i) + h;
        l2 = ComputeLoss(X, Y, RNN_try, hprev);
        grad(i) = (l2-l1)/(2*h);
    end
end