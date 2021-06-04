function [lossTraining, lossValidation, costTraining, costValidation, W, b] = MiniBatchGD(X, Y, y, GDparams, W, b, lambda, eta_min, eta_max, n_s, valX, valY, valy)
    % from previous
    n = size(X, 2);
    n_batch = GDparams.n_batch;
    n_epochs = GDparams.n_epochs;
    eta = eta_min;
    t = 0;
    lossTraining = zeros(1, n_epochs );
    lossValidation = zeros(1, n_epochs );
    costTraining = zeros(1, n_epochs ); 
    costValidation = zeros(1, n_epochs );
    for i = 1 : n_epochs
        for j = 1:(n_batch-1)
            % Code from assignment that gets image batches
            j_start = floor(j*(n/n_batch));
            j_end = floor((j+1)*(n/n_batch));
            inds = j_start:j_end;
            Xbatch = X(:, inds);
            Ybatch = Y(:, inds);

            % gradient
            P = EvaluateClassifier(Xbatch, W, b);
            [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, W, b, lambda);

            % weights & bias
                W{1} = W{1} - eta * grad_W{1};
                W{2} = W{2} - eta * grad_W{2};
                b{1} = b{1} - eta * grad_b{1};
                b{2} = b{2} - eta * grad_b{2};
            
            % Cyclic learning rate   
            if (t <= n_s)                    
                %eta = eta_min + ((t - 2*n_s*lambda)/n_s)*(eta_max - eta_min);
                eta = eta_min + t/n_s * (eta_max - eta_min);
            elseif (t <= 2*n_s)
                %eta = eta_max - ((t - (2*lambda+1)*n_s)/n_s)*(eta_max - eta_min);
                eta = eta_max - (t - n_s)/n_s * (eta_max - eta_min);
            end
            t = mod((t+1), (2*n_s));
           
        end
        
            Wstar = W;
            bstar = b;
            
            [trainCost, trainLoss] = ComputeCost(X, Y, W, b, lambda);
            [valCost, valLoss] = ComputeCost(valX, valY, W, b, lambda);
            
            lossTraining(1, i) = trainLoss;
            lossValidation(1, i) = valLoss;
            
            costTraining(1, i) = trainCost;
            costValidation(1, i) = valCost;
    end
end


