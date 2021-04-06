%Calculate the GAF inages in UTD-MHAD dataset
clc;
clear;
clear all;
GASF_or_GADF='GASF';
Inertial_path='D:\Multi-modal Action Recognition\MMAct\orientation_clip\';
%Inertial_GAF_path='D:\Postdoctoral Research\Multi-modal Action Recognition\Datasets\MMAct\trimmed-selected\acc_phone_clip_GAF\';
Inertial_data_dir = dir(Inertial_path);
foldername = {Inertial_data_dir(:).name};
Inertial_data_subject_dir = setdiff(foldername,{'.','..'});
flag=1;
for i=1:length(Inertial_data_subject_dir)
    scene_path=[Inertial_path,cell2mat(Inertial_data_subject_dir(i))];
    scene_data_dir = dir(scene_path);
    foldername = {scene_data_dir.name};
    Inertial_data_scene_dir = setdiff(foldername,{'.','..'});
    for j=1:length(Inertial_data_scene_dir)
        session_path=[Inertial_path,cell2mat(Inertial_data_subject_dir(i)),'\',cell2mat(Inertial_data_scene_dir(j))];
        session_data_dir = dir(session_path);
        foldername = {session_data_dir.name};
        Inertial_data_session_dir = setdiff(foldername,{'.','..'});
        for k=1:length(Inertial_data_session_dir)
            data_path=[Inertial_path,cell2mat(Inertial_data_subject_dir(i)),'\',cell2mat(Inertial_data_scene_dir(j)),'\',Inertial_data_session_dir(k)];
            path_dir=dir([cell2mat(data_path) '/*.csv' ]);
            foldername = {path_dir.name};
            data_dir = setdiff(foldername,{'.','..'});
            for p=1:length(data_dir)
                fprintf('%d',flag);
                fprintf('\n');
                flag=flag+1;
                Inertial=importdata([cell2mat(data_path),'\',cell2mat(data_dir(p))]);
                if ~isempty(Inertial)
                    Inertial_data=Inertial.data;
                    path=cell2mat(data_dir(p));
                    path=path(1:end-4);
%                     if ~exist([cell2mat(data_path),'\',path,'_GAF'],'dir')
%                         mkdir([cell2mat(data_path),'\',path,'_GAF']);
%                     end
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
                        imwrite(GASF_a,[cell2mat(data_path),'\',data_path{2},'_',data_path{4},'_',data_path{6},'_', path,'_GASF_orientation.jpg']);%_acc_phone,_acc_watch,gyro,orientation

    %                     gx = Inertial_data.d_iner(:,4);
    %                     GASF_gx=GASF(gx');
    %                     gy = Inertial_data.d_iner(:,5);
    %                     GASF_gy=GASF(gy');
    %                     gz = Inertial_data.d_iner(:,6);
    %                     GASF_gz=GASF(gz');
    %                     GASF_g=zeros(size(GASF_gx,1),size(GASF_gx,2),3);
    %                     GASF_g(:,:,1)=mat2gray(GASF_gx);
    %                     GASF_g(:,:,2)=mat2gray(GASF_gy);
    %                     GASF_g(:,:,3)=mat2gray(GASF_gz);
    %                     imwrite(GASF_g,[Inertial_GAF_path,path,'\GASF_g.jpg']);
                    else if GASF_or_GADF=='GADF'
                        ax = Inertial_data(:,1);
                        GADF_ax=GADF(ax');
                        ay = Inertial_data(:,2);
                        GADF_ay=GADF(ay');
                        az = Inertial_data(:,3);
                        GADF_az=GADF(az');
                        GADF_a=zeros(size(GADF_ax,1),size(GADF_ax,2),3);
                        GADF_a(:,:,1)=mat2gray(GADF_ax);
                        GADF_a(:,:,2)=mat2gray(GADF_ay);
                        GADF_a(:,:,3)=mat2gray(GADF_az);
                        imwrite(GADF_a,[cell2mat(data_path),'\',data_path{2},'_',data_path{4},'_',data_path{6},'_',path,'_GADF_acc_phone.jpg']);%_acc_phone,_acc_watch,gyro,orientation

    %                     gx = Inertial_data.d_iner(:,4);
    %                     GADF_gx=GADF(gx');
    %                     gy = Inertial_data.d_iner(:,5);
    %                     GADF_gy=GADF(gy');
    %                     gz = Inertial_data.d_iner(:,6);
    %                     GADF_gz=GADF(gz');
    %                     GADF_g=zeros(size(GADF_gx,1),size(GADF_gx,2),3);
    %                     GADF_g(:,:,1)=mat2gray(GADF_gx);
    %                     GADF_g(:,:,2)=mat2gray(GADF_gy);
    %                     GADF_g(:,:,3)=mat2gray(GADF_gz);
    %                     imwrite(GADF_g,[Inertial_GAF_path,path,'\GADF_g.jpg']);
                        end
                    end
                end
            end
        end     
    end
end
