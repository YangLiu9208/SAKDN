%Calculate the GAF inages in UTD-MHAD dataset
clc;
clear;
clear all;
%----------提取iDT特征----------%
vid_data_path='E:\MMAct_resize_frames\';
fid=fopen('MMAct_cross_view_train_list.txt','w');%写入文件路径
fid2=fopen('MMAct_cross_view_val_list.txt','w');%写入文件路径
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
                    frame_data_dir = dir([frame_path,'\*.jpg']);
                    foldername = {frame_data_dir.name};
                    each_frame_data_dir_dir = setdiff(foldername,{'.','..'});
                    if ~isempty(strfind(cell2mat(Inertial_data_scene_dir(j)),'cam1')) ||~isempty(strfind(cell2mat(Inertial_data_scene_dir(j)),'cam2')) || ~isempty(strfind(cell2mat(Inertial_data_scene_dir(j)),'cam3'))
                    
                        if contains(cell2mat(each_vid_data_dir(m)),'carrying')&&~contains(cell2mat(each_vid_data_dir(m)),'carrying_heavy')&&~contains(cell2mat(each_vid_data_dir(m)),'carrying_light')
                            action_class=0;
                        end
                        if contains(cell2mat(each_vid_data_dir(m)),'carrying_heavy')
                            action_class=1;
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'carrying_light'))
                            action_class=2;
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'checking_time'))
                            action_class=3;
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'closing'))
                            action_class=4;  
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'crouching'))
                            action_class=5;
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'drinking'))
                            action_class=6;
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'entering'))
                            action_class=7;
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'exiting'))
                            action_class=8;
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'fall'))
                            action_class=9;  
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'jumping'))
                            action_class=10;
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'kicking'))
                            action_class=11;
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'loitering'))
                            action_class=12;
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'looking_around'))
                            action_class=13;  
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'transferring_object'))
                            action_class=14;  
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'opening'))
                            action_class=15;
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'picking_up'))
                            action_class=16;
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'pocket_in'))
                            action_class=17;
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'pocket_out'))
                            action_class=18;
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'pointing'))
                            action_class=19;
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'pulling'))
                            action_class=20;  
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'pushing'))
                            action_class=21;
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'running'))
                            action_class=22;
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'setting_down'))
                            action_class=23;
                        end
                        if contains(cell2mat(each_vid_data_dir(m)),'sitting')&&~contains(cell2mat(each_vid_data_dir(m)),'sitting_down')
                            action_class=24;
                         end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'sitting_down'))
                            action_class=25;
                        end
                        
                        if contains(cell2mat(each_vid_data_dir(m)),'standing')&&~contains(cell2mat(each_vid_data_dir(m)),'standing_up')
                            action_class=26;  
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'standing_up'))
                            action_class=27;  
                        end
                        
                        if contains(cell2mat(each_vid_data_dir(m)),'talking')&&~contains(cell2mat(each_vid_data_dir(m)),'talking_on_phone')
                            action_class=28;
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'talking_on_phone'))
                            action_class=29;  
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'throwing'))
                            action_class=30;
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'transferring_object'))
                            action_class=31;
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'using_pc'))
                            action_class=32;
                        end
                        
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'using_phone'))
                            action_class=33;
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'walking'))
                            action_class=34;  
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'waving_hand'))
                            action_class=35;
                        end

                        fprintf(fid,'%s ',frame_path);  
                        fprintf(fid,'%d ',length(each_frame_data_dir_dir));  
                        fprintf(fid,'%d\n',action_class);  
                    
                    
                    
                    end
                    
                    if ~isempty(strfind(cell2mat(Inertial_data_scene_dir(j)),'cam4'))
                    
                       if contains(cell2mat(each_vid_data_dir(m)),'carrying')&&~contains(cell2mat(each_vid_data_dir(m)),'carrying_heavy')&&~contains(cell2mat(each_vid_data_dir(m)),'carrying_light')
                            action_class=0;
                        end
                        if contains(cell2mat(each_vid_data_dir(m)),'carrying_heavy')
                            action_class=1;
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'carrying_light'))
                            action_class=2;
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'checking_time'))
                            action_class=3;
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'closing'))
                            action_class=4;  
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'crouching'))
                            action_class=5;
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'drinking'))
                            action_class=6;
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'entering'))
                            action_class=7;
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'exiting'))
                            action_class=8;
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'fall'))
                            action_class=9;  
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'jumping'))
                            action_class=10;
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'kicking'))
                            action_class=11;
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'loitering'))
                            action_class=12;
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'looking_around'))
                            action_class=13;  
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'transferring_object'))
                            action_class=14;  
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'opening'))
                            action_class=15;
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'picking_up'))
                            action_class=16;
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'pocket_in'))
                            action_class=17;
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'pocket_out'))
                            action_class=18;
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'pointing'))
                            action_class=19;
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'pulling'))
                            action_class=20;  
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'pushing'))
                            action_class=21;
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'running'))
                            action_class=22;
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'setting_down'))
                            action_class=23;
                        end
                        if contains(cell2mat(each_vid_data_dir(m)),'sitting')&&~contains(cell2mat(each_vid_data_dir(m)),'sitting_down')
                            action_class=24;
                         end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'sitting_down'))
                            action_class=25;
                        end
                        
                        if contains(cell2mat(each_vid_data_dir(m)),'standing')&&~contains(cell2mat(each_vid_data_dir(m)),'standing_up')
                            action_class=26;  
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'standing_up'))
                            action_class=27;  
                        end
                        
                        if contains(cell2mat(each_vid_data_dir(m)),'talking')&&~contains(cell2mat(each_vid_data_dir(m)),'talking_on_phone')
                            action_class=28;
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'talking_on_phone'))
                            action_class=29;  
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'throwing'))
                            action_class=30;
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'transferring_object'))
                            action_class=31;
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'using_pc'))
                            action_class=32;
                        end
                        
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'using_phone'))
                            action_class=33;
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'walking'))
                            action_class=34;  
                        end
                        if ~isempty(strfind(cell2mat(each_vid_data_dir(m)),'waving_hand'))
                            action_class=35;
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
