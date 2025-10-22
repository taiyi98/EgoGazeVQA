import os
import json
import csv
import base64
import argparse
import cv2
from openai import OpenAI


def group_frames_and_generate_text(image_folder, json_file, target_group_index, group_size=9):
    """
    Groups images from a folder and retrieves narration and gaze information from JSON.
    
    Args:
        image_folder: Path to the folder containing image files
        json_file: Path to the JSON file containing narration data
        target_group_index: Index of the target group to process
        group_size: Number of images per group (default: 9)
    
    Returns:
        text_pairs: List of formatted text pairs with frame captions and gaze info
        group: List of image filenames in the target group
    """
    try:
        with open(json_file, "r") as f:
            narrations = json.load(f)

        video_id = os.path.basename(image_folder)
        print(f"Video ID: {video_id}\nCurrent group: {target_group_index}")
        
        if video_id not in narrations:
            raise ValueError(f"Video ID {video_id} not found in the JSON file.")
        
        narration_data = narrations[video_id]["narration_pass_1"]["narrations"]

        frame_to_narration = {
            item["timestamp_frame"]: {
                "narration_text": item["narration_text"],
                "gaze_info": item.get("gaze_info", {})
            }
            for item in narration_data
        }

        image_files = sorted(
            [f for f in os.listdir(image_folder) if f.endswith('.jpg')],
            key=lambda x: int(os.path.splitext(x)[0])
        )

        text_pairs = []
        for i in range(0, len(image_files), group_size):
            group_index = i // group_size
            if group_index == target_group_index:
                group = image_files[i:i + group_size]
                for j, image_file in enumerate(group):
                    frame_number = int(os.path.splitext(image_file)[0])
                    if frame_number in frame_to_narration:
                        narration_text = frame_to_narration[frame_number]["narration_text"]
                        gaze_info = frame_to_narration[frame_number]["gaze_info"]
                        gaze_text = f"Gaze:({gaze_info.get('gaze_x', 0)},{gaze_info.get('gaze_y', 0)})"
                        text_pair = f"Frame {j + 1}: {narration_text}; {gaze_text}"
                        text_pairs.append(text_pair)
                    else:
                        print(f"No narration data for frame {frame_number}, skipping.")
        
        return text_pairs, group
    
    except Exception as e:
        print(f"Error: {e}")
        return [], []


def encode_images_from_folder(folder_path, group_id):
    """
    Encode image files to Base64 format.
    
    Args:
        folder_path: Path to the folder containing images
        group_id: List of image filenames to encode
    
    Returns:
        List of Base64 encoded image data URIs
    """
    image_data_list = []
    
    for image_file in group_id:
        image_path = os.path.join(folder_path, image_file)
        with open(image_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode("utf-8")
            image_data_list.append(f"data:image/jpeg;base64,{base64_image}")
    
    return image_data_list


def append_to_json(target_group_index, group_id, caption, completion_content, output_path):
    """
    Append generated QA data to JSON file.
    
    Args:
        target_group_index: Index of the current target group
        group_id: List of image filenames in the group
        caption: Frame captions and gaze information
        completion_content: Generated QA content from the model
        output_path: Path to save the JSON file
    """
    new_entry = {
        "current_group": target_group_index,
        "group_id": group_id,
        "caption": caption,
        "completion_content": completion_content
    }

    if os.path.exists(output_path):
        with open(output_path, "r") as json_file:
            existing_data = json.load(json_file)
    else:
        existing_data = []

    existing_data.append(new_entry)

    with open(output_path, "w") as json_file:
        json.dump(existing_data, json_file, indent=4)

    print(f"New entry appended to {output_path}")


def save_completion_to_csv(video_id, group_id, completion_content, output_csv):
    """
    Parse completion content and save to CSV file.
    
    Args:
        video_id: ID of the video
        group_id: List of image filenames
        completion_content: Generated QA content to parse
        output_csv: Path to the output CSV file
    """
    lines = completion_content.split("\n")
    question = ""
    answer_options = []
    correct_answer = ""
    is_answer_section = False

    question_keywords = ["what", "where", "when", "why", "which", "how"]

    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("### Question"):
            for j in range(i + 1, len(lines)):
                next_line = lines[j].strip()
                if next_line.startswith("### Answer Options:"):
                    break
                if any(keyword in next_line.lower() for keyword in question_keywords):
                    question = next_line.strip()
        elif line.startswith("### Answer Options:"):
            is_answer_section = True
        elif line.startswith("### Correct Answer:"):
            is_answer_section = False
            correct_answer = lines[i + 1][0].strip()
        elif is_answer_section and any(line.startswith(f"{opt}:") for opt in ["A", "B", "C", "D", "E"]):
            answer_options.append(line)
    
    file_exists = os.path.exists(output_csv)
    data = [{
        "video_id": video_id,
        "group_id": "\n".join(group_id),
        "Question": question,
        "Answer Options": "\n".join(answer_options),
        "Correct Answer": correct_answer
    }]

    with open(output_csv, "a", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["video_id", "group_id", "Question", "Answer Options", "Correct Answer"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()
   
        writer.writerows(data)
    
    print(f"Data appended to {output_csv}")


def visualize_gaze_for_group(json_file, video_id, group_ids, output_dir, radius=20, color=(0, 0, 255), thickness=2):
    """
    Visualize gaze points on video frames and save the results.
    
    Args:
        json_file: Path to the JSON file containing video data
        video_id: ID of the video to process
        group_ids: List of image filenames to visualize
        output_dir: Directory to save visualized images
        radius: Radius of the gaze circle
        color: Color of the gaze circle (default: red)
        thickness: Thickness of the circle
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with open(json_file, "r") as f:
        data = json.load(f)
    
    if video_id not in data:
        print(f"Video ID {video_id} not found in the JSON file.")
        return
    
    video_data = data[video_id]
    if "narration_pass_1" not in video_data or "narrations" not in video_data["narration_pass_1"]:
        print(f"No narration data found for Video ID {video_id}.")
        return
    
    narrations = video_data["narration_pass_1"]["narrations"]
    
    for narration in narrations:
        image_path = narration.get("image_path")
        if not image_path or os.path.basename(image_path) not in group_ids:
            continue
        
        gaze_info = narration.get("gaze_info")
        if not gaze_info:
            continue
        
        gaze_x = int(gaze_info.get("gaze_x", -1))
        gaze_y = int(gaze_info.get("gaze_y", -1))
        confidence = gaze_info.get("confidence", 0)
        
        if gaze_x < 0 or gaze_y < 0 or confidence < 1.0:
            continue
        
        full_image_path = os.path.join("/home/pty_ssd/Output1211", image_path)
        if not os.path.exists(full_image_path):
            print(f"Image {full_image_path} not found, skipping.")
            continue
        
        image = cv2.imread(full_image_path)
        if image is None:
            print(f"Failed to load image {full_image_path}, skipping.")
            continue
        
        cv2.circle(image, (gaze_x, gaze_y), radius, color, thickness)
        
        output_image_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_image_path, image)
        print(f"Saved visualized image: {output_image_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate spatial intent QA pairs for egocentric videos")
    parser.add_argument('--video_id', type=str, required=True, help="Video ID to process")
    parser.add_argument('--target_index', type=int, required=True, help="Target group index")
    args = parser.parse_args()

    video_id = args.video_id
    target_group_index = args.target_index

    image_folder = f"/home/pty_ssd/output0207/{video_id}"
    json_file = "/home/pty_ssd/narration_with_gaze_600s.json"
    output_json = f"/home/pty_ssd/Qwen2-VL/QA_benchmark0207/spatial/json/{video_id}.json"
    output_csv = f"/home/pty_ssd/Qwen2-VL/QA_benchmark0207/spatial/csv/{video_id}.csv"

    text_pairs, group_id = group_frames_and_generate_text(image_folder, json_file, target_group_index)
    print(f"Group ID: {group_id}")
    image_data_list = encode_images_from_folder(image_folder, group_id)

    joint_text = "\n".join(text_pairs)

    try:
        client = OpenAI(
            api_key='YOUR_API_KEY',
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        completion = client.chat.completions.create(
            model="qwen-vl-max-latest",
            messages=[
                {
                    "role": "system",
                    "content": [{
                        "type": "text",
                        "text": (
                            "You are an expert in understanding gaze information and spatial localization in ego-centric video data. "
                            "Your task is to generate spatial localization and gaze-aware video QA benchmarks. "
                            "These benchmarks are designed to evaluate nuanced understanding of gaze-based spatial relations in video scenes. "
                            "You will be provided with the following input data:\n"
                            "1. RGB keyframe: A visual snapshot of the scene captured from the first-person perspective.\n"
                            "2. Keyframe caption: A textual description of the keyframe content, including objects and their spatial arrangement.\n"
                            "3. Ego-centric gaze information: Includes gaze fixation points.\n\n"
                            "Requirements:\n"
                            "1. Create only one Question: Formulate spatial localization questions based on the given input with a strong emphasis on the ego-centric gaze. "
                            "Questions should incorporate both gaze dynamics and spatial relationships in the scene, such as:\n"
                            "- The object's position relative to the gaze direction (e.g., 'Where is the object I am looking at relative to my position?').\n"
                            "- Combining gaze focus and object-to-object spatial relations (e.g., 'What is the position of the object I looked at relative to another object?').\n"
                            "2. Generate Answer Options: Provide five plausible answer options for each question. Ensure that:\n"
                            "- Only one option is correct.\n"
                            "- The other options are plausible but incorrect, requiring nuanced understanding of gaze fixation, object relationships, and spatial layout to differentiate.\n"
                            "3. Focus on Detailed Spatial and Gaze-Based Relations: Unlike traditional benchmarks with simple spatial answers (e.g., 'on the table'), "
                            "your answers should include detailed gaze-based spatial relationships (e.g., 'On the table, to the right of the object I looked at for the longest time'). "
                            "This tests the model's ability to interpret both gaze data and spatial relations.\n\n"
                            "Example 1:\n"
                            "### Question:\n"
                            "What is the relative relationship between the knife and my current fixation?\n"
                            "### Answer Options:\n"
                            "A: The knife is on the countertop, to the left of my current fixation.\n"
                            "B: The knife is near the edge of the countertop, behind my current fixation.\n"
                            "C: The knife is near the edge of the countertop, to the right of my current fixation.\n"
                            "D: The knife is on the countertop, in front of my current fixation.\n"
                            "E: The knife is near the edge of the countertop, to the left of the cutting board and my current fixation.\n"
                            "### Correct Answer:\n"
                            "C: The knife is near the edge of the countertop, to the right of my current fixation.\n\n"
                            "Example 2:\n"
                            "### Question:\n"
                            "What object did you focus on the most, and where is it located relative to other objects?\n"
                            "### Answer Options:\n"
                            "A: The coffee mug, on the countertop, right side of the book.\n"
                            "B: The plate, on the table, left side of the spoon.\n"
                            "C: The notebook, in the drawer, right side of the pen.\n"
                            "D: The bowl, on the counter, left side of the cup.\n"
                            "E: The coffee mug, on the table, left side of the book.\n"
                            "### Correct Answer:\n"
                            "E: The coffee mug, on the table, left side of the book."
                        )
                    }]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": image_data_list},
                        {"type": "text", "text": f"Given inputs: {joint_text}\nGenerate a spatial localization, gaze-aware video QA benchmark and give the correct answer"}
                    ]
                }
            ]
        )
        
        print(completion.choices[0].message.content)
        completion_text = completion.choices[0].message.content
        append_to_json(target_group_index, group_id, joint_text, completion_text, output_json)
        save_completion_to_csv(video_id, group_id, completion_text, output_csv)

    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    main()
