import os
import numpy as np
import torch
import cv2
import base64
from torchvision import transforms
import csv
import random
from PIL import Image
import gc
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gaze_trajectory import plot_gaze_trajectory


def load_model(model_name):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        f"Qwen2.5-VL/pretrained/{model_name}", torch_dtype="auto", device_map="auto"
    )
    min_pixels = 256*28*28
    max_pixels = 448*28*28
    processor = AutoProcessor.from_pretrained(f"Qwen2.5-VL/pretrained/{model_name}", min_pixels=min_pixels, max_pixels=max_pixels)
    return model, processor


def load_video(image_paths, image_dir, video_id):

    image_files = [os.path.join(image_dir, video_id, path) for path in image_paths]
    return image_files


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

datasets = ['egtea']
categories = ['temporal', 'causal']
for dataset in datasets:
    for category in categories:
        print(f"----Processing {category}_{dataset}.csv----\n")
 
        file_path = f"./qa_pairs/{category}_{dataset}.csv"
        file_name = os.path.basename(file_path)
        new_file = os.path.splitext(file_name)[0]
        base_folder = f"./datasets/{dataset}"
        gazees_folder = f"./ablation/gazees_vllm/{dataset}"
        model_name = 'Qwen2.5-VL-7B-Instruct'
        output_csv = f"./results/gazees_vllm/{model_name}_{new_file}.csv"
        
    

        model, processor = load_model(model_name)

        results = []

        
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            all_rows = [row for row in reader]
            sample_rows = random.sample(all_rows, 100)  
            for row in sample_rows:
                video_id = row['video_id']
                question = row['Question']
                answer_options = row['Answer Options']
                correct_answer = row['Correct Answer']
                
               
                image_paths = row['group_id'].split("\n")
                image_files = load_video(image_paths, base_folder, video_id)
                gaze_info_list = get_gaze_info_from_csv(gazees_folder, video_id, image_paths)
                salience_image_base64 = plot_gaze_trajectory(image_files[0], gaze_info_list)


                input_question = ("I provide you with a Picture{Frame 0} and a video{Frame 1-9}. Choose the correct option based on the first-person perspective scene question.\n"
                                "You must follow these steps to answer the question:\n"
                                "1. {Frame 0} is the saliency grayscale map of the gaze trajectory from the first-person perspective, with gaze sequence from low brightness to high brightness.\n"
                                "2. Remember the location and time sequence of gaze areas in {Frame 0}.\n"
                                "3. Observe the video according to the gaze sequence in step 2.\n"
                                f"4. Question:\n{question}\nOptions:\n{answer_options}\n"
                                "Choose the most appropriate option. Only return the letter of the correct option.")
                            
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": f"data:image;base64,{salience_image_base64}",
                            },
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
                image_inputs, video_inputs= process_vision_info(messages)
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

