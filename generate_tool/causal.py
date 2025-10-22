import os
import json
import csv
import base64
import argparse
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


def main():
    parser = argparse.ArgumentParser(description="Generate causal intent QA pairs for egocentric videos")
    parser.add_argument('--video_id', type=str, required=True, help="Video ID to process")
    parser.add_argument('--target_index', type=int, required=True, help="Target group index")
    args = parser.parse_args()

    video_id = args.video_id
    target_group_index = args.target_index

    image_folder = f"/home/pty_ssd/output0207/{video_id}"
    json_file = "/home/pty_ssd/narration_with_gaze_600s.json"
    output_json = f"/home/pty_ssd/Qwen2-VL/QA_benchmark0207/reasoning/json/{video_id}.json"
    output_csv = f"/home/pty_ssd/Qwen2-VL/QA_benchmark0207/reasoning/csv_v3/{video_id}.csv"

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
                            "You are an expert in gaze-based event causal understanding within ego-centric video environments. "
                            "Your task is to design comprehensive Gaze-Informed Causal Reasoning ego-centric VideoQA benchmarks. "
                            "The benchmarks should focus on complex, high-level reasoning that integrates gaze dynamics with event interactions.\n\n"
                            "**Input data**:\n"
                            "1. Visual: 9 first-person RGB frames + captions.\n"
                            "2. Ego-centric gaze information: Includes gaze fixation points.\n\n"
                            "**Event Parsing**:\n"
                            "- Identify action chains (e.g., pick glass → drink → glance at bottle → pour water).\n"
                            "- Link actions to gaze patterns, focusing on implicit cause-and-effect relationships.\n\n"
                            "**Gaze Dynamics**:\n"
                            "- Cluster gaze points as natural scene regions (e.g., 'cup rim', 'bottle cap area').\n"
                            "- Track gaze trajectory shifts (e.g., 'suddenly locked onto...').\n"
                            "- Focus on: Gaze as a predictive signal for upcoming actions; Ambiguous gaze paths that may point to multiple plausible behaviors.\n\n"
                            "**Requirements**:\n"
                            "1. Ego-centric Question Construction: Create 5 multiple-choice options: 1 correct and 4 misleading.\n"
                            "- Use templates like:\n"
                            "  Why did [Subject] perform [Action] while [Ongoing Task], given the observed changes in my attention?\n"
                            "  Why did [Subject] [Action] while [Other Subject] was [Action], considering the shifts in my attention?\n"
                            "  What was [Subject] trying to achieve by [Action], based on the changes in my attention during [Task]?\n"
                            "2. Generate Answer Options: Ensure that:\n"
                            "- Correct answers must require concurrent gaze features to explain actions.\n"
                            "- Avoid using external contextual clues that are not related to gaze.\n"
                            "3. Spatio-Temporal Binding:\n"
                            "- Express temporal relationships through event sequences, but avoid explicit time references (e.g., 'after Frame 3', 'during the first phase').\n"
                            "- Focus on my gaze's role in anticipating or influencing actions, but avoid overly simplistic or surface-level reasoning.\n"
                            "4. Distractor Design:\n"
                            "- Include reverse-causal options, where my gaze behavior is misinterpreted as being a result of the action.\n"
                            "- Use spatial-proximity traps where objects in close proximity are incorrectly linked to my gaze behavior.\n"
                            "- Introduce high-salience distractors that divert attention to irrelevant but visually prominent elements.\n"
                            "- Create social influence traps by suggesting social behaviors or mimicry as the primary cause for the gaze behavior, even when it's not.\n"
                            "5. Random Distribution of Correct Answer:\n"
                            "- Ensure that the correct answer option is randomly distributed among the five options.\n\n"
                            "Example:\n"
                            "### Question:\n"
                            "Why did I shuffle the cards while organizing them, given the observed changes in attention?\n"
                            "### Answer Options:\n"
                            "A: I was focusing on the deck to ensure it was properly shuffled.\n"
                            "B: I was distracted by the cards' edges and kept adjusting them.\n"
                            "C: I was checking if any specific card was missing from the deck.\n"
                            "D: I was trying to hide certain cards from the others.\n"
                            "E: I was preparing the cards for a trick.\n"
                            "### Correct Answer:\n"
                            "C: I was checking if any specific card was missing from the deck."
                        )
                    }]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": image_data_list},
                        {"type": "text", "text": f"Given inputs: {joint_text}\nGenerate a causal reasoning, gaze-aware ego-centric video QA benchmark and give the correct answer."}
                    ]
                }
            ]
        )
        
        print(completion.choices[0].message.content)
        completion_text = completion.choices[0].message.content
        save_completion_to_csv(video_id, group_id, completion_text, output_csv)

    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    main()
