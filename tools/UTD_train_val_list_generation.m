%Calculate the GAF inages in UTD-MHAD dataset
clc;
clear;
clear all;
%----------提取iDT特征----------%
vid_data_path='E:\UTD-MHAD\Frames2\';
fid=fopen('UTD_rgb_train_list.txt','w');%写入文件路径
fid2=fopen('UTD_rgb_val_list.txt','w');%写入文件路径
vid_data_dir = dir(vid_data_path);%找到所有后缀为.avi的视频
foldername = natsort({vid_data_dir(:).name});
vid_data_dir = natsort(setdiff(foldername,{'.','..'}));
flag=1;
for i=1:length(vid_data_dir)
    scene_path=[vid_data_path,cell2mat(vid_data_dir(i))];
    scene_data_dir = dir([scene_path,'\*.jpg']);
    foldername = {scene_data_dir.name};
    Inertial_data_scene_dir = setdiff(foldername,{'.','..'});
    if ~isempty(strfind(cell2mat(vid_data_dir(i)),'s1')) ||~isempty(strfind(cell2mat(vid_data_dir(i)),'s3')) || ~isempty(strfind(cell2mat(vid_data_dir(i)),'s5')) || ~isempty(strfind(cell2mat(vid_data_dir(i)),'s7'))
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a1'))
            action_class=0;
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a2'))
            action_class=1;
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a3'))
            action_class=2;  
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a4'))
            action_class=3;
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a5'))
            action_class=4;
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a6'))
            action_class=5;
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a7'))
            action_class=6;  
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a8'))
            action_class=7;
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a9'))
            action_class=8;
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a10'))
            action_class=9;
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a11'))
            action_class=10;  
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a12'))
            action_class=11;
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a13'))
            action_class=12;
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a14'))
            action_class=13;
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a15'))
            action_class=14;  
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a16'))
            action_class=15;
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a17'))
            action_class=16;
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a18'))
            action_class=17;
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a19'))
            action_class=18;  
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a20'))
            action_class=19;
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a21'))
            action_class=20;  
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a22'))
            action_class=21;
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a23'))
            action_class=22;
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a24'))
            action_class=23;
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a25'))
            action_class=24;  
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a26'))
            action_class=25;
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a27'))
            action_class=26;
        end
        fprintf(fid,'%s ',scene_path);  
        fprintf(fid,'%d ',length(Inertial_data_scene_dir));  
        fprintf(fid,'%d\n',action_class);  
    end
    
    if ~isempty(strfind(cell2mat(vid_data_dir(i)),'s2')) ||~isempty(strfind(cell2mat(vid_data_dir(i)),'s4')) || ~isempty(strfind(cell2mat(vid_data_dir(i)),'s6')) || ~isempty(strfind(cell2mat(vid_data_dir(i)),'s8'))
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a1'))
            action_class=0;
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a2'))
            action_class=1;
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a3'))
            action_class=2;  
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a4'))
            action_class=3;
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a5'))
            action_class=4;
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a6'))
            action_class=5;
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a7'))
            action_class=6;  
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a8'))
            action_class=7;
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a9'))
            action_class=8;
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a10'))
            action_class=9;
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a11'))
            action_class=10;  
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a12'))
            action_class=11;
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a13'))
            action_class=12;
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a14'))
            action_class=13;
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a15'))
            action_class=14;  
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a16'))
            action_class=15;
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a17'))
            action_class=16;
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a18'))
            action_class=17;
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a19'))
            action_class=18;  
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a20'))
            action_class=19;
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a21'))
            action_class=20;  
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a22'))
            action_class=21;
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a23'))
            action_class=22;
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a24'))
            action_class=23;
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a25'))
            action_class=24;  
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a26'))
            action_class=25;
        end
        if ~isempty(strfind(cell2mat(vid_data_dir(i)),'a27'))
            action_class=26;
        end
        fprintf(fid2,'%s ',scene_path);  
        fprintf(fid2,'%d ',length(Inertial_data_scene_dir));  
        fprintf(fid2,'%d\n',action_class);  
    end
           
        
end
 fclose(fid);
 fclose(fid2);

