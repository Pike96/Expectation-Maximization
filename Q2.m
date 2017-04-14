clear; close all;
mu1 = -1; sigma1 = 1;
mu2 = 1; sigma2 = 1;
r1 = mvnrnd(mu1,sigma1,100);
r2 = mvnrnd(mu2,sigma2,100);
X = [r1;r2]; % Cascade data points
X = X(randperm(size(X,1)),:); % Shuffle them

figure(1)
plot(X(:,1), 'o');
title('Data Points without Labels')
