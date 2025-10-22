import os
import pandas as pd
import json
import subprocess
import glob


csv_path = '/home/pty_ssd/EgoEye/qa_pairs/temporal_egtea.csv'
long_video_dir = '/home/pty_ssd/EGTEA'  
clip_dir = 'clips_egtea'        
output_json = 'temporal_egtea.json'


os.makedirs(clip_dir, exist_ok=True)


df = pd.read_csv(csv_path)


with open('/home/pty_ssd/ego4d_exo_gaze/ego4d_exo_gaze_annotations/annotations/keystep_train.json', 'r') as f:
    keystep_data = json.load(f)
videoid2takename = {ann['take_uid']: ann['take_name'] for ann in keystep_data['annotations'].values()}

def find_egoexo_video(take_name):
    search_dir = f'/home/pty_ssd/ego4d_exo_gaze/takes/{take_name}/frame_aligned_videos/'
    files = glob.glob(os.path.join(search_dir, '*214-1.mp4'))
    if files:
        return files[0]
    else:
        return None

qa_dict = {}

for idx, row in df.iterrows():
    video_id = row['video_id']
    group_id = row['group_id']
    question = row['Question']
    answer_options = row['Answer Options'].split('\n')
    correct_answer = row['Correct Answer']


    frame_indices = [int(f.replace('.jpg', '')) for f in group_id.strip().split('\n')]
    frame_indices.sort()
    start_frame = frame_indices[0]
    end_frame = frame_indices[-1]


    fps = 30
    start_time = start_frame / fps
    end_time = end_frame / fps
    duration = end_time - start_time


    is_egoexo = video_id in videoid2takename
    if is_egoexo:
        take_name = videoid2takename[video_id]
        long_video_path = find_egoexo_video(take_name)
        if long_video_path is None:
            continue
    else:
        long_video_path = os.path.join(long_video_dir, f'{video_id}.mp4')

    clip_name = f'{start_frame}_{end_frame}.mp4'
    video_clip_dir = os.path.join(clip_dir, video_id)
    os.makedirs(video_clip_dir, exist_ok=True)  
    clip_path = os.path.join(video_clip_dir, clip_name)


    if not os.path.exists(clip_path):
        cmd = [
            'ffmpeg',
            '-ss', str(start_time),
            '-i', long_video_path,
            '-t', str(duration),
            '-c', 'copy',
            '-y',  
            clip_path
        ]
        print(' '.join(cmd))
        subprocess.run(cmd, check=True)


    frame_paths = [
        os.path.join(f"{idx}.jpg") for idx in frame_indices
    ]

 
    qa_item = {
        "clip_name": clip_name,
        "clip_path": os.path.basename(clip_path),
        "frames": frame_paths,
        "question": question,
        "answer_options": answer_options,
        "correct_answer": correct_answer
    }

    if video_id not in qa_dict:
        qa_dict[video_id] = []
    qa_dict[video_id].append(qa_item)

with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(qa_dict, f, ensure_ascii=False, indent=2)
