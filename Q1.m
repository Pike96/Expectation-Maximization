clear; close all;
mu = [1 2 3];
sigma = [3 -1 1; -1 5 3; 1 3 4];
z = [normrnd(0,1) normrnd(0,1) normrnd(0,1)];
A = chol(sigma,'lower');
x = (A*z' + mu')';
