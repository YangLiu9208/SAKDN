clc;
clear;
clear all;
csv=importdata('carrying.csv');
csvdata=csv.data;
ax = csvdata(:,2); %1-3
GADF_ax=GADF(ax');
imshow(mat2gray(GADF_ax));

% GASF_ax=GASF(ax');
% imshow(mat2gray(GASF_ax));
% ay = iner(:,2);
% az = iner(:,3);

% gx = iner(:,4);
% gy = iner(:,5);
% gz = iner(:,6);