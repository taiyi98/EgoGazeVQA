import os
import json
import cv2
import glob
import pandas as pd
from math import ceil


def read_annotations(annotations_file):
    """读取标注的 JSON 文件"""
    with open(annotations_file, 'r') as f:
        annotations_data = json.load(f)
    return annotations_data


def get_video_and_gaze_paths(take_name, takes_root):
    """根据 take_name 获取对应的视频和眼动数据文件路径"""
    # 使用 glob 查找所有符合模式的视频文件
    video_folder = os.path.join(takes_root, take_name, 'frame_aligned_videos')
    video_files = glob.glob(os.path.join(video_folder, '*214-1.mp4'))  # 查找以 214-1.mp4 结尾的视频文件

    # 如果找不到符合条件的视频文件，返回 None
    if not video_files:
        print(f"警告: {video_folder} 中没有找到符合条件的视频文件（以 214-1.mp4 结尾）。")
        return None, None

    # 选择第一个符合条件的视频文件
    video_path = video_files[0]
    gaze_data_path = os.path.join(takes_root, take_name, 'eye_gaze', 'general_eye_gaze_2d.csv')
    
    return video_path, gaze_data_path


def load_gaze_data(gaze_data_path):
    """加载眼动数据并返回 DataFrame"""
    if not os.path.exists(gaze_data_path):
        print(f"警告: {gaze_data_path} 文件不存在，跳过处理该视频。")
        return None
    
    gaze_data = pd.read_csv(gaze_data_path)

    # 对眼动数据进行重复，以便与视频帧率对齐（10 fps -> 30 fps）
    x_values = gaze_data['x'].tolist()
    y_values = gaze_data['y'].tolist()

    # 每个眼动数据项重复 3 次
    x_values = [num for num in x_values for _ in range(3)]
    y_values = [num for num in y_values for _ in range(3)]

    # 将数据返回为对齐后的 DataFrame
    gaze_data_aligned = pd.DataFrame({'x': x_values, 'y': y_values})

    return gaze_data_aligned


def normalize_gaze_coordinates(gaze_x, gaze_y, frame_width, frame_height):
    """将眼动坐标归一化到 [0, 1] 范围"""
    normalized_x = round(gaze_x / frame_width, 3)
    normalized_y = round(gaze_y / frame_height, 3)
    return normalized_x, normalized_y


def process_video(take_uid, take_info, video_path, gaze_data, fps, output_folder):
    """处理视频帧并保存眼动数据"""
    cap = cv2.VideoCapture(video_path)
    narrations = []

    take_name = take_info['take_name']
    scenario = take_info['scenario']

    # 遍历每个步骤
    for segment in take_info['segments']:
        end_time = segment['end_time']
        step_description = segment['step_description']

        # 计算对应的帧号
        frame_num = ceil(end_time * fps)

        if frame_num < len(gaze_data):
            gaze_x, gaze_y = gaze_data.loc[frame_num, ['x', 'y']]

            # 获取视频帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                # 获取帧的高宽
                frame_height, frame_width = frame.shape[:2]

                # 归一化眼动坐标
                normalized_x, normalized_y = normalize_gaze_coordinates(gaze_x, gaze_y, frame_width, frame_height)

                # 保存图片
                image_path = os.path.join(output_folder, f'{frame_num}.jpg')
                cv2.imwrite(image_path, frame)
                relative_image_path = os.path.join(take_uid, f'{frame_num}.jpg')

                # 更新标注数据
                gaze_info = {
                    'gaze_x': normalized_x,
                    'gaze_y': normalized_y,
                }

                narrations.append({
                    'timestamp_sec': end_time,
                    'timestamp_frame': frame_num,
                    'description': step_description,
                    'gaze_info': gaze_info,
                    'image_path': relative_image_path
                })

    cap.release()

    # 构造 take_uid 对应的标注信息
    take_annotations = {
        "take_name": take_name,
        "scenario": scenario,
        "narrations": narrations
    }

    return take_annotations


def save_annotations_to_json(all_annotations, output_folder):
    """将所有标注数据保存为一个 JSON 文件"""
    output_json_path = os.path.join(output_folder, 'procedure_understanding.json')
    with open(output_json_path, 'w') as json_out:
        json.dump(all_annotations, json_out, indent=4)
    print(f"标注文件已保存到 {output_json_path}")


def main():
    # 配置路径
    annotations_file = '/home/pty_ssd/ego4d_exo_gaze/ego4d_exo_gaze_annotations/annotations/keystep_train.json'
    takes_root = '/home/pty_ssd/ego4d_exo_gaze/takes'
    output_root = '/home/pty_ssd/ego4d_exo_gaze/output_0214_keystep'

    # 创建输出文件夹
    if not os.path.exists(output_root):
        os.mkdir(output_root)

    # 读取标注数据
    annotations_data = read_annotations(annotations_file)

    all_annotations = {}

    # 遍历所有 UID
    for take_uid, take_info in annotations_data['annotations'].items():
        take_name = take_info['take_name']
        video_path, gaze_data_path = get_video_and_gaze_paths(take_name, takes_root)
        if video_path is None or not os.path.exists(gaze_data_path):
            print(f"警告: {gaze_data_path} 或视频文件不存在，跳过处理该视频。")
            continue

        gaze_data = load_gaze_data(gaze_data_path)
        if gaze_data is None:
            continue  # 如果 gaze_data 加载失败，则跳过该视频

        # 打开视频并获取帧率
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # 创建每个 UID 的文件夹
        uid_folder = os.path.join(output_root, take_uid)
        if not os.path.exists(uid_folder):
            os.mkdir(uid_folder)

        # 处理视频帧并获取标注数据
        annotations_per_step = process_video(take_uid, take_info, video_path, gaze_data, fps, uid_folder)

        save_annotations_to_json(annotations_per_step, uid_folder)

        # 将标注数据添加到总体标注列表
        all_annotations[take_uid] = annotations_per_step


    # 保存标注数据到一个 JSON 文件
    save_annotations_to_json(all_annotations, output_root)

    print("所有视频处理完成，标注 JSON 文件已保存！")


if __name__ == "__main__":
    main()
