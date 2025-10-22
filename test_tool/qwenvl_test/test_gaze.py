import os
import torch
import csv
import json
import gc
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from torch.cuda.amp import autocast


def load_model(model_name):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        f"/home/pty/Qwen2.5-VL/pretrained/{model_name}", torch_dtype="auto", device_map="auto"
    )
    min_pixels = 256*28*28
    max_pixels = 448*28*28
    processor = AutoProcessor.from_pretrained(f"/home/pty/Qwen2.5-VL/pretrained/{model_name}", min_pixels=min_pixels, max_pixels=max_pixels)
    return model, processor


def load_video(image_paths, image_dir, video_id):

    image_files = [os.path.join(image_dir, video_id, path) for path in image_paths]
    return image_files

def get_gaze_info_from_json(json_file, video_id, group_id):

    with open(json_file, 'r') as f:
        data = json.load(f)

    video_data = data.get(video_id)
    if not video_data:
        return None
    
    gaze_info_list = []
    for image_file in group_id:
        frame = int(image_file.split('.')[0].split('_')[1])
        for narration in video_data["narrations"]:
            if narration["timestamp_frame"] == frame:
                gaze_info_list.append(narration["gaze_info"])
                break  

    return gaze_info_list


datasets = ['egtea']
categories = ['temporal', 'causal']
for dataset in datasets:
    for category in categories:
        print(f"----Processing {category}_{dataset}----\n")
        csv_file = f"/home/pty_ssd/EgoEye/qa_pairs/{category}_{dataset}.csv"
        image_dir = f"/home/pty_ssd/EgoEye/datasets/{dataset}"
        model_name = 'Qwen2.5-VL-7B-4D_EXO'
        file_name = os.path.basename(csv_file)
        new_file = os.path.splitext(file_name)[0]
        output_csv = f"/home/pty_ssd/EgoEye/results/lora_sft/{model_name}-{new_file}-gaze.csv"
        narration_json = f"/home/pty_ssd/EgoEye/narrations/{dataset}.json"
            
        model, processor = load_model(model_name)
        results = []

        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_id = row['video_id']
                question = row['Question']
                answer_options = row['Answer Options']
                correct_answer = row['Correct Answer']

                image_paths = row['group_id'].split("\n")
                image_files = load_video(image_paths, image_dir, video_id)

                gaze_info_list = get_gaze_info_from_json(narration_json, video_id, image_paths)
                if not gaze_info_list:
 
                    continue

                gaze_info_text = "Gaze information for the relevant frames:\n"
                for i, gaze_info in enumerate(gaze_info_list):
                    gaze_x = gaze_info.get("gaze_x", "N/A")
                    gaze_y = gaze_info.get("gaze_y", "N/A")
                    gaze_info_text += f"Frame {i+1}: Gaze({gaze_x}, {gaze_y})\n"
                
                joint_text = gaze_info_text
                
                input_question = gaze_info_text+(f"I provide you with a video and the normalized gaze coordinates for each corresponding frame."
                                                "You need to follow these steps to answer the questions:\n"
                                                "1.Observe the position of the annotated gaze points in each frame. The coordinate is from left to right for the x-axis and from top to bottom for the y-axis."
                                                "2.Analyze the video while considering the gaze point information and then answer the questions."
                                                f"3.Question:{question}\nOptions:\n{answer_options}\n"
                                                "Choose the most appropriate option. Return the letter of the correct option.")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": image_files,  
                            },
                            {"type": "text", "text": input_question},
                        ],
                    }
                ]


                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )

                inputs = inputs.to("cuda")

    
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=128)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    output_text = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                    qwenvl_model_answer = output_text[0].strip()

                del inputs, image_inputs, video_inputs
                torch.cuda.empty_cache()
                gc.collect()

                print(f'{input_question}\nModel Answer: {qwenvl_model_answer} \nCorrect Answer: {correct_answer}')
    
                results.append({
                    'video_id': video_id,
                    'Question': question,
                    'Answer Options': answer_options,
                    'Model_Answer': qwenvl_model_answer,
                    'Reference_Answer': correct_answer
                })
   

            
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['video_id', 'Question', 'Answer Options', 'Model_Answer', 'Reference_Answer'])
            writer.writeheader()
            writer.writerows(results)

        print(f"Results saved to {output_csv}")
