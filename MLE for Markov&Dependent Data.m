%% density function mfile

function res = densityfun(theta, x, y)
 
res = 1/sqrt(2*pi)*exp(-(y-(theta(1)+theta(2)*x)).^2/2);
 
end

%% define the scaled log likelihood function
function res = likelihood(func_handle, theta, x, y)
 
res = -sum(log(func_handle(theta, x, y)))/length(x);
 
end


%% Define MLE function 
% The MLE takes in parameters of datasets x and y, initial values for theta, self-defined significance level for hypothesis testing and the hypothesis; returns the estimated theta, value of log likelihood function, confidence interval and p-value.

function [theta_hat, llik, lower_CI, upper_CI, p_value] = MLE(x, y, theta0, sig_level, hypothesis)
 
%K is length(theta_hat)
 
[theta_hat,llik,~,~,~,hessian] = fminunc(@(theta) likelihood(@densityfun, theta, x, y),theta0);
 
K = length(theta_hat); 
n = length(x);
variance_mat = inv(hessian);
 
sigma_hat = zeros([1,K]);
 
for i=1:K
    sigma_hat(i) = sqrt(variance_mat(i,i))/sqrt(n);
end
 
z_statistic = (theta_hat-hypothesis)./sigma_hat;
p_value = 2*normcdf(-abs(z_statistic),0,1);
 
%calculate confidence interval
threshold = abs(icdf('Normal',sig_level/2,0,1));
lower_CI = hypothesis - threshold * sigma_hat;
upper_CI = hypothesis + threshold * sigma_hat;
 
 
end

%% The main program

% simulate data
epsilon  = normrnd(0,1,[1,1000]);
 
X = zeros([1,1000]);
X(1) = 2;
 
for i = 2:1000
    X(i) = X(i-1)*0.6+1+epsilon(i-1);
end
 
%x is X(i)
x = X(1:999);
%y is X(i-1)
y = X(2:1000);
    
theta0 = [1,1];
sig_level = 0.05;
hypothesis = [1, 0.6];
 
[theta_hat, llik, lower_CI, upper_CI, p_value] = MLE(x, y, theta0, sig_level, hypothesis);
