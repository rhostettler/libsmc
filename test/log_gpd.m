clear variables;
sigma = rand(1);
kappa = rand(1);
mu = 0;
py = @(y) exp(y)/sigma.*(1+kappa*(exp(y)-mu)/sigma).^(-1/kappa-1);
y = (0.1:0.01:5).';

figure(1); clf();
plot(y, py(y));