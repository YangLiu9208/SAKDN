%Calculate the GAF inages in UTD-MHAD dataset
clc;
clear;
clear all;
GASF_or_GADF='GASF';
Inertial_path='E:\Berkeley MHAD\Accelerometer\Shimmer06\';
Inertial_GAF_path='E:\Berkeley MHAD\Accelerometer\Shimmer06_GAF\';
Inertial_data_dir = dir(Inertial_path);
foldername = {Inertial_data_dir(:).name};
Inertial_data_dir = setdiff(foldername,{'.','..'});
if ~exist([Inertial_GAF_path],'dir')
  mkdir([Inertial_GAF_path]);
end
for i=1:length(Inertial_data_dir)
    
    path=cell2mat(Inertial_data_dir(i));
    Inertial_data=load([Inertial_path,path]);
    path=path(1:end-4);
    %if ~exist([Inertial_GAF_path,path],'dir')
    %    mkdir([Inertial_GAF_path,path]);
    %end
    if GASF_or_GADF=='GASF'
        ax = Inertial_data(:,1);
        GASF_ax=GASF(ax');
        ay = Inertial_data(:,2);
        GASF_ay=GASF(ay');
        az = Inertial_data(:,3);
        GASF_az=GASF(az');
        GASF_a=zeros(size(GASF_ax,1),size(GASF_ax,2),3);
        GASF_a(:,:,1)=mat2gray(GASF_ax);
        GASF_a(:,:,2)=mat2gray(GASF_ay);
        GASF_a(:,:,3)=mat2gray(GASF_az);
        imwrite(GASF_a,[Inertial_GAF_path,path,'_GASF_a.jpg']);

    else if GASF_or_GADF=='GADF'
        ax = Inertial_data.d_iner(:,1);
        GADF_ax=GADF(ax');
        ay = Inertial_data.d_iner(:,2);
        GADF_ay=GADF(ay');
        az = Inertial_data.d_iner(:,3);
        GADF_az=GADF(az');
        GADF_a=zeros(size(GADF_ax,1),size(GADF_ax,2),3);
        GADF_a(:,:,1)=mat2gray(GADF_ax);
        GADF_a(:,:,2)=mat2gray(GADF_ay);
        GADF_a(:,:,3)=mat2gray(GADF_az);
        imwrite(GADF_a,[Inertial_GAF_path,path,'_GADF_a.jpg']);

        gx = Inertial_data.d_iner(:,4);
        GADF_gx=GADF(gx');
        gy = Inertial_data.d_iner(:,5);
        GADF_gy=GADF(gy');
        gz = Inertial_data.d_iner(:,6);
        GADF_gz=GADF(gz');
        GADF_g=zeros(size(GADF_gx,1),size(GADF_gx,2),3);
        GADF_g(:,:,1)=mat2gray(GADF_gx);
        GADF_g(:,:,2)=mat2gray(GADF_gy);
        GADF_g(:,:,3)=mat2gray(GADF_gz);
        imwrite(GADF_g,[Inertial_GAF_path,path,'_GADF_g.jpg']);
        end
    end
            
end
