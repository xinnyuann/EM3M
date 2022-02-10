function [alpha,p1,A,z,LL] = mixMarkovEM(x,D,K,varargin)
%   [alpha,p1,A,z,LL]=mixMarkovEM(x,D,K,varargin)
%
% Inputs:
%   x : cell array of sequences
%   D : dimension of the state space.
%   K : number of mixture components. (>1)
%
% Optional Inputs:
%   'maxIter': Maximum EM iteration count (default 100)
%   'verbose': plots log likelihood (default 'false')
%   'threshold': if the decrease in log likelihood is less than the
%                threshold, stop.
%
% Sample Usage:  
%   [alpha,pi,A,z,LL] = bk_mixMarkovEM(y,D,K,'maxIter', 1000, 'verbose',true,'threshold', 1e-5)   
%   Runs the EM algorithm for the sequences y, whose input space dimension 
%   is D, and the number of mixture components are K. The algorithm is run 
%   1000 times unless the decrease in log likelihood is less than 1e-5. 
%   The loglikelihood is also plotted after every iteration.
%
% Output:   
%    - alpha: prior distribution of the mixture components p(z)
%    - p1: cond. dist.of initial states p(x_n(1)|z_n)
%    - A: cond. dist. of state transitions p(x_n(t)|x_n(t-1),z_n)
%    - z: posterior distribution of mixture componenets p(z|x,pi,A)
%    - LL: the bound value during the computations
%
%
%   Date: 09.08.2013
%
%   Baris Kurt. 
%   Bogazici University, Dept. of Computer Engineering
%   bariskurt@gmail.com
%-----------------------------------------------------------

% Parse arguments:
p=inputParser;
p.addParamValue('maxIter',100,@isnumeric);
p.addParamValue('threshold',0,@isnumeric);
p.addParamValue('verbose',false,@islogical);
p.parse(varargin{:});
maxIter=p.Results.maxIter;
verbose=p.Results.verbose;
threshold = p.Results.threshold;


N = size(x,1);      % num sequences
S = zeros(D,D,N);   % transition counts
x1 = zeros(D,N);    % initial states (1-of-K representation)

% Collect sufficient statistics
for n=1:N,
    x1(x{n}(1),n) = 1;
    S(:,:,n) = accumarray([x{n}(2:end)' x{n}(1:end-1)'],1,[D D]);
end
S = reshape(S,[D*D N]);

% random initialize model parameters
alpha=normalize(rand(K,1));
p1=normalize(rand(D,K));
A=normalize(rand(D,D,K));

% the posterior of latent variables:
z = zeros(K,N);

%log likelihood
LL = zeros(1,maxIter);

for iter=1:maxIter,
    
    log_p1 = logsafe(p1);
    log_A = reshape(logsafe(A),[D*D K]);
    log_alpha = logsafe(alpha);
    
    % E-step:
    llhood = (log_p1' * x1) + (log_A' * S) + repmat(log_alpha,[1 N]);
    z = normalize_exp(llhood);
    
    % Likelihood:
    LL(iter) = sum(log_sum_exp(llhood));
    % Alternative way
    % LL(iter) = sum(sum(llhood .* z)) - sum(sum(z .* logsafe(z)));
    
    % M-step:
    p1 = normalize(x1*z');
    A = normalize(reshape(S*z',[D D K]));
    alpha = normalize(sum(z,2));
    
    % plot after iteration ? 
    if verbose,
        plot(LL(1:iter));
        title('Log Likelihood');
        xlabel('Iterations');
        drawnow;
    end
    
    % exit if a threshold given ? 
    if threshold > 0 && iter >1 && (LL(iter) - LL(iter-1) < threshold),
        LL = LL(1:iter);
        break;
    end
    
end

end %bk_mixMarkovEM

%
%------------  UTILITIES ----------------
%

function X = normalize_exp(X)
%   
% Numerically stable exp + normalize.
% Applies to the 1st dimension of tensor X
%
  dm = ones(1, length(size(X)));
  dm(1) = size(X, 1); 
  X = X - repmat(max(X), dm);  
  X = normalize(exp(X));
end

function A=normalize(A)
%   
% Normalizes each column (1st dim) of a given tensor A
%
  Z = sum(A);
  Z = Z + (Z==0);
  dm = ones(1, length(size(A)));
  dm(1) = size(A, 1); 
  A = A./repmat(Z, dm);
end

function log_x = logsafe(x, cutpoint)
%
% Takes the logarithm of x, avoiding the values to go to -Inf for numerical
% stability.
%
    if nargin<2,
        cutpoint = -500;
    end
    log_x = log(x);
    log_x(isinf(log_x)) = cutpoint;
end

function L=log_sum_exp(X)
%
% log(sum(exp(X))) operation with avoiding underflow
%
    dm = ones(1, length(size(X)));
    dm(1) = size(X,1);
    mx = max(X);
    mx2 = repmat(mx, dm);
    L = mx + log(sum(exp(X-mx2)));
end