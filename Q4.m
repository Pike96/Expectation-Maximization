clear; close all;

%% a.
% Read
n = 272;
data = textread('faithful.dat.txt', '%f', n*3, 'headerlines', 26);
duration = data(2:3:n*3);
waiting = data(3:3:n*3);
X = [duration waiting];

% Scatter plot
figure(1);
scatter(duration,waiting)
title('Scatter plot of duration and waiting time');
xlabel('Duaration time'); ylabel('Waiting time');

% K-means
[idx,C] = kmeans(X,2);
figure(2)
plot(X(idx==1,1),X(idx==1,2),'g.','MarkerSize',12)
hold on
plot(X(idx==2,1),X(idx==2,2),'r.','MarkerSize',12)
plot(C(:,1),C(:,2),'kx', 'MarkerSize',15,'LineWidth',3)
legend('Cluster 1','Cluster 2','Centroids','Location','NW')
title('K-means Cluster Assignments')
xlabel('Duaration time'); ylabel('Waiting time');
hold off

%% b.
% GMM-EM
% Initialization
k = randperm(n);
mu = X(k(1:2), :);

sigma1 = cov(X);
sigma2 = cov(X);

p = [0.5 0.5];

ud1 = bsxfun(@minus, X, mu(1,:));
ud2 = bsxfun(@minus, X, mu(2,:));

phi(:,1)=exp(-1/2*sum((ud1*inv(sigma1).*ud1),2))/sqrt((2*pi)^2*det(sigma1));
phi(:,2)=exp(-1/2*sum((ud2*inv(sigma2).*ud2),2))/sqrt((2*pi)^2*det(sigma2));
    
L_before = sum(log(p(1).*phi(:,1)+p(2).*phi(:,2)))/n;

for i = 1:2000
    %E-step
    phi_w = bsxfun(@times, phi, p);

    gamma = bsxfun(@rdivide, phi_w, sum(phi_w, 2));
    
    mu_temp = mu;
    
    % M-step
    mu(1,:) = gamma(:, 1)' * X ./ sum(gamma(:, 1), 1);
    mu(2,:) = gamma(:, 2)' * X ./ sum(gamma(:, 2), 1);

    XS1 = bsxfun(@minus, X, mu(1, :));
    XS2 = bsxfun(@minus, X, mu(2, :));

    for j=1:n
        sigma1 = sigma1 + (gamma(j, 1) .* (XS1(j, :)' * XS1(j, :)));
        sigma2 = sigma2 + (gamma(j, 2) .* (XS2(j, :)' * XS2(j, :)));
    end

    sigma1 = sigma1 ./ sum(gamma(:, 1));
    sigma2 = sigma2 ./ sum(gamma(:, 2));
     
    p = [mean(gamma(:,1)) mean(gamma(:,2))];
    
    ud1 = bsxfun(@minus, X, mu(1,:));
    ud2 = bsxfun(@minus, X, mu(2,:));

    phi(:,1)=exp(-1/2*sum((ud1*inv(sigma1).*ud1),2))/sqrt((2*pi)^2*det(sigma1));
    phi(:,2)=exp(-1/2*sum((ud2*inv(sigma2).*ud2),2))/sqrt((2*pi)^2*det(sigma2));
    
    L_after = sum(log(p(1).*phi(:,1)+p(2).*phi(:,2)))/n;
    
    % Check convergence
    if L_after == L_before
        break;
    else
        L_before = L_after;
    end
            
end
sigma = cat(3,sigma1,sigma2);
% build GMM
obj = gmdistribution(mu,sigma,p);


% 2D projection
figure(3);
ezcontourf(@(x,y) pdf(obj,[x y]),[1.5 5.5], [40 100]);
hold on
scatter(duration,waiting,'w')
title('Scatter plot and GMM');
xlabel('Duaration time'); ylabel('Waiting time');
