function [M, band_set, iter] =MHBSCUR(Y,opts)

% Matrix Hyperspectral Band Selection based on matrix CUR decomposition

% --------------------------------------------
% Input:
%                  Y       -    HSI band matrix
%           
%           opts           -   Structure value in Matlab. The fields are
%           opts.tol       -   termination tolerance
%           opts.max_iter  -   maximum number of iterations
%           opts.beta      -   stepsize for dual variable updating in ADMM
%           opts.Ls        -   spatial graph Laplacian
%           opts.Lc        -   spectral graph Laplacian
%           opts.lambda    -   lambda for sparse component (tune)
%           opts.rs        -   number of rows for CUR (proportional to
%           rln(mn))=20(ln(4625500))=307
%           opts.cs        -   number of columns for CUR (see above) round
%           (20*ln(220))= 108
%
%           opts.gamma1    -   parameter for spectral GL (tune)
%           opts.gamma2    -   parameter for spatial GL (tune)
%           opts.tau       -   step size in CUR gradient descent (<1)
%           (tune)
%           opts.k         -   number of desired bands (usually known)
%           opts.DEBUG     -   0 or 1
%
% Output:
%               M          -   desired band set M=Y(:,k)
%


%% Read Parameters
if ~exist('opts', 'var')
    opts = [];
end    
if isfield(opts, 'tol');         tol = opts.tol;              end
if isfield(opts, 'max_iter');    max_iter = opts.max_iter;    end
if isfield(opts, 'beta');        beta = opts.beta;            end
if isfield(opts, 'Ls');          Ls = opts.Ls;                end
if isfield(opts, 'Lc');          Lc = opts.Lc;                end
if isfield(opts, 'rs');          rs = opts.rs;                end
if isfield(opts, 'cs');          cs = opts.cs;                end
if isfield(opts, 'gamma1');      gamma1 = opts.gamma1;        end
if isfield(opts, 'gamma2');      gamma2 = opts.gamma2;        end
if isfield(opts, 'tau');         tau = opts.tau;              end
if isfield(opts, 'k');           k = opts.k;                  end
if isfield(opts, 'lambda');      lambda = opts.lambda;        end






%% Initialize
dim = size(Y);
[d1,d2] = size(Y);
B = zeros(dim);
S = B;
Z = B;
Zhat = B;


%% Random sampling for CUR
rng(1);
I = randperm(d1,rs); 
J = randperm(d2,cs);
C = Y(:,J); 
R = Y(I,:);
U = 0.5*(C(I,:)+R(:,J)); 
[W,sig,V] = svds(U,k);
B = (C*V)*pinv(sig)*(W'*R);


%% ADMM
for iter = 1 : max_iter
    Bk = B;
    Sk = S;
    Zk = Z;
    
    %% update B (low rank component)
    tmp = B-Y+S+Z+Zhat;
    
    GI = tmp(I,:)+(gamma2/2)*(C(I,:)*V)*pinv(sig)*((W'*R)*Lc)+(gamma1/2)*(Ls(I,:)*(C*V))*pinv(sig)*(W'*R);
    GJ = tmp(:,J)+(gamma2/2)*(C*V)*pinv(sig)*((W'*R)*Lc(:,J))+(gamma1/2)*(Ls*(C*V))*pinv(sig)*(W'*R(:,J));

    C = C-tau*GJ;
    R = R-tau*GI;
    U = 0.5*(C(I,:)+R(:,J)); 
    [W,sig,V] = svds(U,k);
    B = (C*V)*pinv(sig)*(W'*R);
    
    %% update S
    S = prox_l1(Y-B-Z+Zhat,lambda/beta);
    
    %% update Z
    Z= prox_l1(Y-B-S+Zhat,1/beta);
    
    %% Check for convergence
    dZhat = Y-B-S-Z;
    chgB = max(abs(Bk(:)-B(:)));
    chgS = max(abs(Sk(:)-S(:)));
    chgZ = max(abs(Zk(:)-Z(:)));
    chg = max([ chgB chgS  chgZ max(abs(dZhat(:))) ]);
    
    if chg < tol
        break;
    end 

    %% update Zhat
    Zhat = Zhat+Y-B-S-Z;
end

%% compute k-means clustering
[~, ~, ~, D] = kmeans(B.',k,'maxiter',100,'replicates',50,'emptyaction','singleton');
[~,I] = min(D); % find indices for bands with centroids
M = Y(:,I); % select k bands from original data
band_set = I;
