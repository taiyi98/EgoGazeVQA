import os
import json
import cv2
import glob
import pandas as pd
from math import ceil


def read_annotations(annotations_file):
    """Read annotation data from JSON file."""
    with open(annotations_file, 'r') as f:
        annotations_data = json.load(f)
    return annotations_data


def get_video_and_gaze_paths(take_name, takes_root):
    """
    Get video and gaze data file paths based on take_name.
    
    Args:
        take_name: Name of the video take
        takes_root: Root directory containing all takes
    
    Returns:
        video_path: Path to the video file
        gaze_data_path: Path to the gaze data CSV file
    """
    video_folder = os.path.join(takes_root, take_name, 'frame_aligned_videos')
    video_files = glob.glob(os.path.join(video_folder, '*214-1.mp4'))

    if not video_files:
        print(f"Warning: No video file ending with 214-1.mp4 found in {video_folder}")
        return None, None

    video_path = video_files[0]
    gaze_data_path = os.path.join(takes_root, take_name, 'eye_gaze', 'general_eye_gaze_2d.csv')
    
    return video_path, gaze_data_path


def load_gaze_data(gaze_data_path):
    """
    Load gaze data and align frame rate from 10fps to 30fps.
    
    Args:
        gaze_data_path: Path to the gaze data CSV file
    
    Returns:
        DataFrame with aligned gaze data (x, y coordinates)
    """
    if not os.path.exists(gaze_data_path):
        print(f"Warning: {gaze_data_path} does not exist, skipping.")
        return None
    
    gaze_data = pd.read_csv(gaze_data_path)

    x_values = gaze_data['x'].tolist()
    y_values = gaze_data['y'].tolist()

    x_values = [num for num in x_values for _ in range(3)]
    y_values = [num for num in y_values for _ in range(3)]

    gaze_data_aligned = pd.DataFrame({'x': x_values, 'y': y_values})

    return gaze_data_aligned


def normalize_gaze_coordinates(gaze_x, gaze_y, frame_width, frame_height):
    """
    Normalize gaze coordinates to [0, 1] range.
    
    Args:
        gaze_x: X coordinate of gaze point
        gaze_y: Y coordinate of gaze point
        frame_width: Width of the video frame
        frame_height: Height of the video frame
    
    Returns:
        normalized_x, normalized_y: Normalized coordinates
    """
    normalized_x = round(gaze_x / frame_width, 3)
    normalized_y = round(gaze_y / frame_height, 3)
    return normalized_x, normalized_y


def process_video(take_uid, take_info, video_path, gaze_data, fps, output_folder):
    """
    Process video frames and save gaze information.
    
    Args:
        take_uid: Unique identifier for the take
        take_info: Dictionary containing take metadata
        video_path: Path to the video file
        gaze_data: DataFrame containing gaze coordinates
        fps: Frame rate of the video
        output_folder: Directory to save extracted frames
    
    Returns:
        Dictionary containing narrations with gaze information
    """
    cap = cv2.VideoCapture(video_path)
    narrations = []

    take_name = take_info['take_name']
    scenario = take_info['scenario']

    for segment in take_info['segments']:
        end_time = segment['end_time']
        step_description = segment['step_description']

        frame_num = ceil(end_time * fps)

        if frame_num < len(gaze_data):
            gaze_x, gaze_y = gaze_data.loc[frame_num, ['x', 'y']]

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                frame_height, frame_width = frame.shape[:2]

                normalized_x, normalized_y = normalize_gaze_coordinates(
                    gaze_x, gaze_y, frame_width, frame_height
                )

                image_path = os.path.join(output_folder, f'{frame_num}.jpg')
                cv2.imwrite(image_path, frame)
                relative_image_path = os.path.join(take_uid, f'{frame_num}.jpg')

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

    take_annotations = {
        "take_name": take_name,
        "scenario": scenario,
        "narrations": narrations
    }

    return take_annotations


def save_annotations_to_json(all_annotations, output_folder):
    """
    Save all annotation data to a single JSON file.
    
    Args:
        all_annotations: Dictionary containing all annotations
        output_folder: Directory to save the JSON file
    """
    output_json_path = os.path.join(output_folder, 'procedure_understanding.json')
    with open(output_json_path, 'w') as json_out:
        json.dump(all_annotations, json_out, indent=4)
    print(f"Annotation file saved to {output_json_path}")


def main():
    annotations_file = '/home/pty_ssd/ego4d_exo_gaze/ego4d_exo_gaze_annotations/annotations/keystep_train.json'
    takes_root = '/home/pty_ssd/ego4d_exo_gaze/takes'
    output_root = '/home/pty_ssd/ego4d_exo_gaze/output_0214_keystep'

    if not os.path.exists(output_root):
        os.mkdir(output_root)

    annotations_data = read_annotations(annotations_file)
    all_annotations = {}

    for take_uid, take_info in annotations_data['annotations'].items():
        take_name = take_info['take_name']
        video_path, gaze_data_path = get_video_and_gaze_paths(take_name, takes_root)
        
        if video_path is None or not os.path.exists(gaze_data_path):
            print(f"Warning: Video or gaze data not found for {take_name}, skipping.")
            continue

        gaze_data = load_gaze_data(gaze_data_path)
        if gaze_data is None:
            continue

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        uid_folder = os.path.join(output_root, take_uid)
        if not os.path.exists(uid_folder):
            os.mkdir(uid_folder)

        annotations_per_step = process_video(take_uid, take_info, video_path, gaze_data, fps, uid_folder)
        save_annotations_to_json(annotations_per_step, uid_folder)

        all_annotations[take_uid] = annotations_per_step

    save_annotations_to_json(all_annotations, output_root)
    print("All videos processed successfully!")


if __name__ == "__main__":
    main()
