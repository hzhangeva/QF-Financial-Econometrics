%% 1. define the density function

% theta is a vector contains 3 elements, parameter a, b, and c for Vasicek Model

function res = densityfun(theta,h,x,xs)
 
res = (2251799813685248.*exp(-(x - xs).^2./(2.*theta(3).^2.*h)).*((theta(1).*(theta(2) - xs).*(x - xs).*(theta(1).^4.*h.^4 - 5.*theta(1).^3.*h.^3 + 20.*theta(1).^2.*h.^2 - 60.*theta(1).*h + 120))./(120.*theta(3).^2) - (((theta(1).*h.*(theta(1).*theta(2).^2 - 2.*theta(1).*theta(2).*xs - theta(3).^2 + theta(1).*xs.^2))./(2.*theta(3).^2) - (theta(1).^2.*h.^2.*(3.*theta(1).*theta(2).^2 - 6.*theta(1).*theta(2).*xs - 2.*theta(3).^2 + 3.*theta(1).*xs.^2))./(6.*theta(3).^2) + (theta(1).^3.*h.^3.*(7.*theta(1).*theta(2).^2 - 14.*theta(1).*theta(2).*xs - 4.*theta(3).^2 + 7.*theta(1).*xs.^2))./(24.*theta(3).^2) - (theta(1).^4.*h.^4.*(15.*theta(1).*theta(2).^2 - 30.*theta(1).*theta(2).*xs - 8.*theta(3).^2 + 15.*theta(1).*xs.^2))./(120.*theta(3).^2)).*(h.*theta(3).^2 - x.^2 + 2.*x.*xs - xs.^2))./(theta(3).^2.*h) + (theta(1).^2.*(theta(2) - xs).*(x - xs).*(3.*h.*theta(3).^2 - x.^2 + 2.*x.*xs - xs.^2).*(- 50.*theta(1).^3.*theta(2).^2.*h.^2 + 100.*theta(1).^3.*theta(2).*h.^2.*xs + theta(1).^3.*theta(3).^2.*h.^3 - 50.*theta(1).^3.*h.^2.*xs.^2 + 60.*theta(1).^2.*theta(2).^2.*h - 120.*theta(1).^2.*theta(2).*h.*xs + 100.*theta(1).^2.*theta(3).^2.*h.^2 + 60.*theta(1).^2.*h.*xs.^2 - 40.*theta(1).*theta(2).^2 + 80.*theta(1).*theta(2).*xs - 140.*theta(1).*theta(3).^2.*h - 40.*theta(1).*xs.^2 + 120.*theta(3).^2))./(240.*theta(3).^6) + 1))./(5644425081792261.*theta(3).*h.^(1./2));
end


%% 2. define the scaled log likelihood function
function res = likelihood(func_handle, theta, h, x, xs)
 
res = -sum(log(func_handle(theta, h, x ,xs)))/(length(x));
 
end


function [theta_hat, llik, sigma_hat, lower_CI, upper_CI, p_value] = MLE(data, theta0, sig_level, hypothesis, h)
 
%K is length(theta_hat)
xs = data(1:end-h);
x = data(h+1:end);
 
[theta_hat,llik,~,~,~,hessian] = fminunc(@(theta) likelihood(@densityfun, theta, h, x, xs),theta0);
 
K = length(theta_hat); 
%disp(K)
%pay attention here
n = length(x);
variance_mat = inv(hessian);
 
sigma_hat = zeros(1,K);
 
for i=1:K
    sigma_hat(i) = sqrt(variance_mat(i,i))/sqrt(n);
end
%disp(sigma_hat)
z_statistic = (theta_hat-hypothesis)./sigma_hat;
p_value = 2*normcdf(-abs(z_statistic),0,1);
 
%calculate confidence interval
threshold = abs(icdf('Normal',sig_level/2,0,1));
lower_CI = theta_hat - threshold * sigma_hat;
upper_CI = theta_hat + threshold * sigma_hat;
 
 
end


%% 3. MLE estimation
data = csvread('FRB_H15.csv');
data = data(7000:length(data)-12000);
data = data/100;
plot(data)
% Suppose 5 days dependent
h=5;
theta0 = [0.03,0.02,0.03];% b  = 0.02/0.07
sig_level = 0.05;
hypothesis = [0.005,0.094,0.0033];
[theta_hat, llik, sigma_hat, lower_CI, upper_CI, p_value] = MLE(data, theta0, sig_level, hypothesis, h);