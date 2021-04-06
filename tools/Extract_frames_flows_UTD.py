__author__ = 'yjxiong'

import os
import glob
import sys
import cv2
from pipes import quote
from multiprocessing import Pool, current_process

import argparse

out_path = r"D:/Multi-modal Action Recognition/UTD-MHAD/Frames/"

def dump_frames(vid_path):
    video = cv2.VideoCapture(vid_path)
    vid_name = vid_path.split('/')[-1].split('.')[0]
    out_full_path = os.path.join(out_path, vid_name)

    fcount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass
    file_list = []
    for i in range(fcount):
        ret, frame = video.read()
        assert ret
        cv2.imwrite('{}/{:06d}.jpg'.format(out_full_path, i+1), frame)
        access_path = '{}/{:06d}.jpg'.format(vid_name, i+1)
        file_list.append(access_path)
    print('{} done'.format(vid_name))
    sys.stdout.flush()
    return file_list


def run_optical_flow(vid_item, dev_id=0):
    vid_path = vid_item[0]
    vid_id = vid_item[1]
    vid_name = vid_path.split('/')[-1].split('.')[0]
    out_full_path = os.path.join(out_path, vid_name)
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass

    current = current_process()
    dev_id = (int(current._identity[0]) - 1) % NUM_GPU
    image_path = '{}/img'.format(out_full_path)
    flow_x_path = '{}/flow_x'.format(out_full_path)
    flow_y_path = '{}/flow_y'.format(out_full_path)

    cmd = os.path.join(df_path + 'build/extract_gpu')+' -f {} -x {} -y {} -i {} -b 20 -t 1 -d {} -s 1 -o {} -w {} -h {}'.format(
        quote(vid_path), quote(flow_x_path), quote(flow_y_path), quote(image_path), dev_id, out_format, new_size[0], new_size[1])

    os.system(cmd)
    print('{} {} done'.format(vid_id, vid_name))
    sys.stdout.flush()
    return True


def run_warp_optical_flow(vid_item, dev_id=0):
    vid_path = vid_item[0]
    vid_id = vid_item[1]
    vid_name = vid_path.split('/')[-1].split('.')[0]
    out_full_path = os.path.join(out_path, vid_name)
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass

    current = current_process()
    dev_id = (int(current._identity[0]) - 1) % NUM_GPU
    flow_x_path = '{}/flow_x'.format(out_full_path)
    flow_y_path = '{}/flow_y'.format(out_full_path)

    cmd = os.path.join(df_path + 'build/extract_warp_gpu')+' -f {} -x {} -y {} -b 20 -t 1 -d {} -s 1 -o {}'.format(
        vid_path, flow_x_path, flow_y_path, dev_id, out_format)

    os.system(cmd)
    print('warp on {} {} done'.format(vid_id, vid_name))
    sys.stdout.flush()
    return True

def nonintersection(lst1, lst2):
    lst3 = [value for value in lst1 if ((value.split("/")[-1]).split(".")[0]) not in lst2]
    return lst3

def main():
    parser = argparse.ArgumentParser(description="extract optical flows")
    #parser.add_argument("src_dir",type=str,default='D:/Multi-modal Action Recognition/UTD-MHAD/RGB/')
    #parser.add_argument("out_dir",type=str,default='D:/Multi-modal Action Recognition/UTD-MHAD/Frames/')
    parser.add_argument("--num_worker", type=int, default=8)
    parser.add_argument("--Frames_or_Flows", type=str, default='Frames', choices=['Frames', 'Flows'])
    parser.add_argument("--flow_type", type=str, default='tvl1', choices=['tvl1', 'warp_tvl1'])
    parser.add_argument("--df_path", type=str, default='./lib/dense_flow/', help='path to the dense_flow toolbox')
    parser.add_argument("--out_format", type=str, default='dir', choices=['dir','zip'],
                        help='path to the dense_flow toolbox')
    parser.add_argument("--ext", type=str, default='avi', choices=['avi','mp4'], help='video file extensions')
    parser.add_argument("--new_width", type=int, default=0, help='resize image width')
    parser.add_argument("--new_height", type=int, default=0, help='resize image height')
    parser.add_argument("--num_gpu", type=int, default=8, help='number of GPU')
    parser.add_argument("--resume", type=str, default='no', choices=['yes','no'], help='resume optical flow extraction instead of overwriting')

    args = parser.parse_args()

    src_path = r"D:/Multi-modal Action Recognition/UTD-MHAD/RGB/"
    out_path = r"D:/Multi-modal Action Recognition/UTD-MHAD/Frames/"

    num_worker = args.num_worker
    flow_type = args.flow_type
    df_path = args.df_path
    out_format = args.out_format
    ext = args.ext
    new_size = (args.new_width, args.new_height)
    NUM_GPU = args.num_gpu
    resume = args.resume
    Frames_or_Flows=args.Frames_or_Flows
    if not os.path.isdir(out_path):
        print("creating folder: "+out_path)
        os.makedirs(out_path)
    print("reading videos from folder: ", src_path)
    print("selected extension of videos:", ext)
    vid_list = glob.glob(src_path+'*.'+ext)
    print("total number of videos found: ", len(vid_list))
    if(resume == 'yes'):
        com_vid_list = os.listdir(out_path)
        vid_list = nonintersection(vid_list, com_vid_list)
        print("resuming from video: ", vid_list[0]) 
    #pool = Pool(num_worker)
    if Frames_or_Flows=='Frames':
        for i in range(len(vid_list)):
            vid_list[i]=eval(repr( vid_list[i]).replace(r"\\", r"/"))
            dump_frames(vid_list[i])
    #if flow_type == 'tvl1':
    #    pool.map(run_optical_flow, zip(vid_list, range(len(vid_list))))
    #elif flow_type == 'warp_tvl1':
    #    pool.map(run_warp_optical_flow, zip(vid_list, range(len(vid_list))))

if __name__ == '__main__':
    main()