%% density function mfile
function res = densityfun(theta, sigma, data)
res = 1/sqrt(2*pi)/sigma*exp(-(data-theta).^2/2/sigma^2);
end
%% main program
%give initial value
theta0=0;
sigma = 0.1;
mu = 1;
%generate data from N(1,0.1)
rr = 666;
rng(rr);
data = normrnd(mu,sigma,[1,100]);
%define likelihood function
likelihood = @(theta) -sum(log(densityfun(theta, sigma, data)));
%optimize likelihood function
[val,fval,~,~,grad,hessian] = fminunc(likelihood,theta0);
fprintf('The estimated MLE estimator is %d. \n', val);
valuetobe = sum(data)/length(data);
fprintf("The estimated MLE estimator should be %d. \n", valuetobe);
fprintf('The likelihood function value is %d. \n', -fval);
%Estimate estimator's standard deviation
sigma_hat = sqrt((1/(hessian/length(data)))/length(data));
fprintf("Standard deviation of MLE estimator is %d. \n", sigma_hat);
%calculate p-value
z_statistic = (val-mu)/sigma_hat;
p_value = 2*normcdf(-abs(z_statistic),0,1);
fprintf("Two sided p-value of MLE estimator is %d. \n", p_value);
%calculate confidence interval
alpha = 0.05;
threshold = abs(icdf('Normal',alpha/2,0,1));
lower = mu - threshold * sigma_hat;
upper = mu + threshold * sigma_hat;
fprintf("Confidence interval is [%d, %d]. \n", lower,upper);