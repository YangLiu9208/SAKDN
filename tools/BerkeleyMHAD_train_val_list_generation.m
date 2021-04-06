%Calculate the GAF inages in UTD-MHAD dataset
clc;
clear;
clear all;
%----------提取iDT特征----------%
vid_data_path='D:\Berkeley-MHAD\Camera\';
fid=fopen('BerkeleyMHAD_train_list.txt','w');%写入文件路径
fid2=fopen('BerkeleyMHAD_val_list.txt','w');%写入文件路径
vid_data_dir = dir(vid_data_path);%找到所有后缀为.avi的视频
foldername = {vid_data_dir(:).name};
vid_data_dir = setdiff(foldername,{'.','..'});
flag=1;
for i=1:length(vid_data_dir)
    scene_path=[vid_data_path,cell2mat(vid_data_dir(i))];
    scene_data_dir = dir(scene_path);
    foldername = {scene_data_dir.name};
    Inertial_data_scene_dir = setdiff(foldername,{'.','..'});
    for j=1:length(Inertial_data_scene_dir)
        session_path=[vid_data_path,cell2mat(vid_data_dir(i)),'\',cell2mat(Inertial_data_scene_dir(j))];
        session_data_dir = dir(session_path);
        foldername = {session_data_dir.name};
        Inertial_data_session_dir = setdiff(foldername,{'.','..'});
        for k=1:length(Inertial_data_session_dir)
            data_path=[vid_data_path,cell2mat(vid_data_dir(i)),'\',cell2mat(Inertial_data_scene_dir(j)),'\',cell2mat(Inertial_data_session_dir(k))];
            path_dir=dir(data_path);
            foldername = {path_dir.name};
            data_dir = setdiff(foldername,{'.','..'});
            for p=1:length(data_dir)
                avi_path=[vid_data_path,cell2mat(vid_data_dir(i)),'\',cell2mat(Inertial_data_scene_dir(j)),'\',cell2mat(Inertial_data_session_dir(k)),'\',cell2mat(data_dir(p))];
                vidDir = dir([avi_path]);
                each_foldername = {vidDir(:).name};
                each_vid_data_dir = setdiff(each_foldername,{'.','..'});
                for m=1:length(each_vid_data_dir)
                    frame_path=[vid_data_path,cell2mat(vid_data_dir(i)),'\',cell2mat(Inertial_data_scene_dir(j)),'\',cell2mat(Inertial_data_session_dir(k)),'\',cell2mat(data_dir(p)),'\', cell2mat(each_vid_data_dir(m))];
                    frame_data_dir = dir([frame_path,'\*.pgm']);
                    foldername = {frame_data_dir.name};
                    each_frame_data_dir_dir = setdiff(foldername,{'.','..'});
                    if ~isempty(strfind(cell2mat(Inertial_data_session_dir(k)),'S01')) ||~isempty(strfind(cell2mat(Inertial_data_session_dir(k)),'S02')) || ~isempty(strfind(cell2mat(Inertial_data_session_dir(k)),'S03')) || ~isempty(strfind(cell2mat(Inertial_data_session_dir(k)),'S04')) ...
                            ||~isempty(strfind(cell2mat(Inertial_data_session_dir(k)),'S05')) || ~isempty(strfind(cell2mat(Inertial_data_session_dir(k)),'S06')) || ~isempty(strfind(cell2mat(Inertial_data_session_dir(k)),'S07'))
                    
                        if contains(cell2mat(data_dir(p)),'A01')
                            action_class=0;
                        end
                        if contains(cell2mat(data_dir(p)),'A02')
                            action_class=1;
                        end
                        if contains(cell2mat(data_dir(p)),'A03')
                            action_class=2;
                        end
                        if contains(cell2mat(data_dir(p)),'A04')
                            action_class=3;
                        end
                        if contains(cell2mat(data_dir(p)),'A05')
                            action_class=4;  
                        end
                        if contains(cell2mat(data_dir(p)),'A06')
                            action_class=5;
                        end
                        if contains(cell2mat(data_dir(p)),'A07')
                            action_class=6;
                        end
                        if contains(cell2mat(data_dir(p)),'A08')
                            action_class=7;
                        end
                        if contains(cell2mat(data_dir(p)),'A09')
                            action_class=8;
                        end
                        if contains(cell2mat(data_dir(p)),'A10')
                            action_class=9;  
                        end
                        if contains(cell2mat(data_dir(p)),'A11')
                            action_class=10;
                        end

                        fprintf(fid,'%s ',frame_path);  
                        fprintf(fid,'%d ',length(each_frame_data_dir_dir));  
                        fprintf(fid,'%d\n',action_class);  
                    
                    
                    
                    end
                    
                    if ~isempty(strfind(cell2mat(Inertial_data_session_dir(k)),'S08')) ||~isempty(strfind(cell2mat(Inertial_data_session_dir(k)),'S09')) || ~isempty(strfind(cell2mat(Inertial_data_session_dir(k)),'S10')) || ~isempty(strfind(cell2mat(Inertial_data_session_dir(k)),'S11')) ...
                            ||~isempty(strfind(cell2mat(Inertial_data_session_dir(k)),'S12')) 
                    
                        if contains(cell2mat(data_dir(p)),'A01')
                            action_class=0;
                        end
                        if contains(cell2mat(data_dir(p)),'A02')
                            action_class=1;
                        end
                        if contains(cell2mat(data_dir(p)),'A03')
                            action_class=2;
                        end
                        if contains(cell2mat(data_dir(p)),'A04')
                            action_class=3;
                        end
                        if contains(cell2mat(data_dir(p)),'A05')
                            action_class=4;  
                        end
                        if contains(cell2mat(data_dir(p)),'A06')
                            action_class=5;
                        end
                        if contains(cell2mat(data_dir(p)),'A07')
                            action_class=6;
                        end
                        if contains(cell2mat(data_dir(p)),'A08')
                            action_class=7;
                        end
                        if contains(cell2mat(data_dir(p)),'A09')
                            action_class=8;
                        end
                        if contains(cell2mat(data_dir(p)),'A10')
                            action_class=9;  
                        end
                        if contains(cell2mat(data_dir(p)),'A11')
                            action_class=10;
                        end

                        fprintf(fid2,'%s ',frame_path);  
                        fprintf(fid2,'%d ',length(each_frame_data_dir_dir));  
                        fprintf(fid2,'%d\n',action_class);  
                    
                    
                    
                    end
                end
                end
            end
      end
end

fclose(fid);
fclose(fid2);
