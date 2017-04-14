mu = [1; 2; 3];
sigma = [3 -1 1; -1 5 3; 1 3 4];
r = mvnrnd(mu,sigma,10000);
plot(r(:,1),r(:,2),'+')
