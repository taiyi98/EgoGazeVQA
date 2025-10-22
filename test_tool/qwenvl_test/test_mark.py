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




datasets = ['egtea']
categories = ['spatial', 'causal', 'temporal']
for dataset in datasets:
    for category in categories:
        print(f"----Processing {category}_{dataset}----\n")
        csv_file = f"/home/pty_ssd/EgoEye/qa_pairs/{category}_{dataset}.csv"
        image_dir = f"/home/pty_ssd/EgoEye/datasets/visual_mark/{dataset}"
        model_name = 'Qwen2.5-VL-7B-4D_EXO'
        file_name = os.path.basename(csv_file)
        new_file = os.path.splitext(file_name)[0]
        output_csv = f"/home/pty_ssd/EgoEye/results/lora_sft/{model_name}-{new_file}-mark.csv"
    
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
                
                input_question =("I provide you with a video that contains gaze information. For each frame, the gaze point will be marked on the image with a red heart-shaped circle.\n" 
                                "Choose the correct option based on the first-person perspective scene question. You must follow these steps to answer the question:\n"
                                "1.Focus on the objects marked by each red heart-shaped circle.\n"
                                "2.Observe the video chronological order according to the gaze sequence in step 1.\n"
                                f"3.Question:\n{question}\nOptions:\n{answer_options}\n"
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
                break


        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['video_id', 'Question', 'Answer Options', 'Model_Answer', 'Reference_Answer'])
            writer.writeheader()
            writer.writerows(results)

        print(f"Results saved to {output_csv}")
