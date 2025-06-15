%vfi.m
% Use a value-function-iteration procedure to obtain the policy function in a tradable-endowment, small open economy.

clear all

%The endowment process is
%y_t - 1 = rho (y_{t-1} - 1) + sigg epsilon_t
%where E(epsilon_t) and E(epsilon_t)^2=sigg^2  and 
rho = 0.4;
sigg = 0.05;

%Discretize this process with a two-state Markov process  to match the standard deviation and correlation of y_t:
p = (1+rho)/2;
pai = [p 1-p;1-p p]; 
ss = sigg/sqrt(1-rho^2);
ygrid = [1+ss;1-ss];
ny = numel(ygrid);

nd = 200;% # of grid points for debt, d_t

r = 0.04;  %interest rate  

betta = 0.954; %discount factor

sigg = 2; %intertemporal elasticity of consumption

G = 0.2; %0.22 %constant component of domestic absorption (e.g., wasteful government spending)

%Natural borrowing limit
nbl = (min(ygrid)-G)/r;
%debt grid
dupper = min(nbl, 19);
dlower = 15;
dgrid = linspace(dlower,dupper,nd);
dgrid = dgrid(:);

n = ny*nd;

d = repmat(dgrid',ny,1);
d = d(:);

yix = repmat((1:ny)',nd,1);
y = ygrid(yix);

ctry = bsxfun(@(x,y) x+y,y-(1+r)*d-G,dgrid');% consumption of tradables

starve = find(max(ctry,[],2)<0);
if ~isempty(starve)
warning('Natural debt limit violated')
end

utry = (ctry.^(1-sigg) - 1) / (1-sigg);

utry(ctry<=0) = -inf;

clear ctry

v = zeros(ny,nd);

dpix = zeros(n,1);

dist = 1;

while dist> 1e-10

Evptry = pai *  v;
Evptry  = repmat(Evptry,nd,1);

[vnew, dpixnew] = max(utry+betta*Evptry,[],2);

dist = max(abs(vnew-v(:)))  + max(abs(dpixnew-dpix));

v(:) = vnew;

dpix = dpixnew;
dpix(starve) = 1;

end %while dist>1e-10

dp = dgrid(dpix);

pais = zeros(n);
for k=1:n
jp = dpix(k);
pais(k,ny*(jp-1)+1:ny*jp) = pai(yix(k),:);
end

c = y+dp - (1+r)*d - G;
tb = y-c-G;
ca = -(dp-d);

upais = uncond_distrib(pais);

aux = zeros(ny,nd);
aux(:) = upais;
paid = sum(aux);
paid = paid(:);
clear aux

%save vfi.mat nd ny   r betta sigg  pai ygrid dgrid y   d  v dpix dp c tb nd pais G upais paid

function P = uncond_distrib(TPM,tol);
% P = uncond_distrib(TPM) computes the unconditions distribution P associated with the transition probability matrix TPM.
%P = uncond_distrib(TPM,tol) accepts a tolerance for the precision of P (default 1e-8). 
%
%(c) Martin Uribe, May 31, 2013.

if nargin<2
tol = 1e-8;
end

K = size(TPM,1);
dist_tpm = 1;
P = ones(1,K)/K;
while dist_tpm>tol
Pnew = P*TPM;
dist_tpm = max(abs(P(:)-Pnew(:)));
P = Pnew;
end
P = P(:);
end