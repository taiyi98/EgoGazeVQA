import os
import json
import csv
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gaze_trajectory import plot_gaze_trajectory



def load_video(image_paths, image_dir, video_id):
    image_files = [os.path.join(image_dir, video_id, path) for path in image_paths]
    return image_files

def get_gaze_info_from_json(json_file, video_id, group_id):
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    json_name = os.path.basename(json_file)
    
    
    video_data = data.get(video_id)
    if not video_data:
        return None
    
    gaze_info_list = []
    for image_file in group_id:
        if json_name == 'egtea.json':
            frame = int(image_file.split('.')[0].split('_')[1]) # egtea
        else:
            frame = int(image_file.split('/')[-1].split('.')[0])  # ego4d, egoexo
        for narration in video_data["narrations"]:
            if narration["timestamp_frame"] == frame:
                gaze_info_list.append(narration["gaze_info"])
                break  

    return gaze_info_list


def get_gaze_info_from_csv(csv_folder, video_id, group_id):

    csv_file = os.path.join(csv_folder, f"{video_id}.csv")
    if not os.path.exists(csv_file):
        return None

  
    gaze_dict = {}
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = row['frame']
            gaze_str = row['gaze'].strip('()')
            gaze_x, gaze_y = map(float, gaze_str.split(','))
            gaze_dict[frame] = {'gaze_x': gaze_x, 'gaze_y': gaze_y}


    gaze_info_list = []
    for image_file in group_id:
        gaze_info = gaze_dict.get(image_file)
        if gaze_info:
            gaze_info_list.append(gaze_info)
        else:
            gaze_info_list.append(None)  
    return gaze_info_list

def calc_mae(gaze_es_list, gaze_gd_list):
    dists = []
    for es, gd in zip(gaze_es_list, gaze_gd_list):
        if es is None or gd is None:
            continue
        dx = es['gaze_x'] - gd['gaze_x']
        dy = es['gaze_y'] - gd['gaze_y']
        dist = np.sqrt(dx**2 + dy**2)
        dists.append(dist)
    if len(dists) == 0:
        return None
    return np.mean(dists)

def calc_mse(gaze_es_list, gaze_gd_list):
    errors = []
    for es, gd in zip(gaze_es_list, gaze_gd_list):
        if es is None or gd is None:
            continue
        dx = es['gaze_x'] - gd['gaze_x']
        dy = es['gaze_y'] - gd['gaze_y']
        errors.append(dx**2 + dy**2)
    if len(errors) == 0:
        return None
    return np.mean(errors)

all_mse_total = []
all_mae_total = []  

datasets = ['ego4d', 'egoexo', 'egtea']
categories = ['spatial', 'causal', 'temporal']
for dataset in datasets:
    all_mae = []
    all_mse = []
    for category in categories:
       
        file_path = f"./qa_pairs/{category}_{dataset}.csv"
        file_name = os.path.basename(file_path)
        new_file = os.path.splitext(file_name)[0]
        base_folder = f"./datasets/{dataset}"
        gazees_folder = f"./ablation/gazees_vllm/{dataset}"
        narration_json = f"./narrations/{dataset}.json"
        model_name = 'Qwen2.5-VL-7B-Instruct'
        output_csv = f"./results/gazees_vllm/{model_name}_{new_file}.csv"

        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_id = row['video_id']
                image_paths = row['group_id'].split("\n")
                gaze_es_list = get_gaze_info_from_csv(gazees_folder, video_id, image_paths)
                gaze_gd_list = get_gaze_info_from_json(narration_json, video_id, image_paths)
                if gaze_es_list is None or gaze_gd_list is None:
                    continue
                mse = calc_mse(gaze_es_list, gaze_gd_list)
                mae = calc_mae(gaze_es_list, gaze_gd_list)
                if mse is not None:
                    all_mse.append(mse)
                    all_mse_total.append(mse)
                if mae is not None:
                    all_mae.append(mae)
                    all_mae_total.append(mae)

    if all_mse:
        print(f" {dataset} MSE: {np.mean(all_mse):.4f}")
    if all_mae:
        print(f"{dataset} MAE: {np.mean(all_mae):.4f}")




                

