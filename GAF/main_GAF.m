%Gramian Summation Angular Field and Gramian Difference Angular Field 
clc;
clear;
clear all;
load('a10_s1_t1_inertial.mat');
ax = d_iner(:,6);%1-6
% GADF_ax=GADF(ax');
% imshow(mat2gray(GADF_ax));

GASF_ax=GASF(ax');
imshow(mat2gray(GASF_ax));
% ay = iner(:,2);
% az = iner(:,3);

% gx = iner(:,4);
% gy = iner(:,5);
% gz = iner(:,6);