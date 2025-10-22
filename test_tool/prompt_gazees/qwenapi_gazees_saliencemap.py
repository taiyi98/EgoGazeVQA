import os
import json
import csv
import base64
import pandas as pd
from openai import OpenAI
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gaze_trajectory import plot_gaze_trajectory


def encode_images_from_folder(base_folder, video_id, group_id):

    image_data_list = []
    folder_path = os.path.join(base_folder, video_id)
    
    for image_file in group_id:
        image_path = os.path.join(folder_path, image_file.strip())
        if os.path.exists(image_path):
            with open(image_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode("utf-8")
                image_data_list.append(f"data:image/jpeg;base64,{base64_image}")
        else:
            print(f"No File: {image_path}")
    
    return image_data_list, image_path


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


def main():
    datasets = ['egtea']
    categories = ['spatial', 'temporal', 'causal']
    for dataset in datasets:
        for category in categories:
            print(f"----Processing {category}_{dataset}.csv----\n")
        
            file_path = f"/home/pty_ssd/EgoEye/qa_pairs/{category}_{dataset}.csv"
            file_name = os.path.basename(file_path)
            new_file = os.path.splitext(file_name)[0]
            base_folder = f"/home/pty_ssd/EgoEye/datasets/{dataset}"
            gazees_folder = f"/home/pty_ssd/EgoEye/ablation/gazees_vllm/{dataset}"  
            api_key = ""  #  API Key
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"  
            
            df = pd.read_csv(file_path)
            client = OpenAI(api_key=api_key, base_url=base_url)
            
            results = []

            for index, row in df.iterrows():
                video_id = row["video_id"]
                group_id = row["group_id"].split("\n")  
                question = row["Question"]
                answer_options = row["Answer Options"]
                reference = row["Correct Answer"]

                image_data_list, image_path = encode_images_from_folder(base_folder, video_id, group_id)

        
                if not image_data_list:
                    print(f"jump {video_id}")
                    continue

  
                gaze_info_list = get_gaze_info_from_csv(gazees_folder, video_id, group_id)

                salience_image_base64 = plot_gaze_trajectory(image_path, gaze_info_list)
             

                input_question = ("I provide you with a Picture{Frame 0} and a video{Frame 1-9}. Choose the correct option based on the first-person perspective scene question.\n"
                                "You must follow these steps to answer the question:\n"
                                "1. {Frame 0} is the saliency grayscale map of the gaze trajectory from the first-person perspective, with gaze sequence from low brightness to high brightness.\n"
                                "2. Remember the location and time sequence of gaze areas in {Frame 0}.\n"
                                "3. Observe the video according to the gaze sequence in step 2.\n"
                                f"4. Question:\n{question}\nOptions:\n{answer_options}\n"
                                "Choose the most appropriate option. Only return the letter of the correct option.")


                
                try:
                    completion = client.chat.completions.create(
                        model="qwen-vl-max-latest",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url":{"url":f"data:image/png;base64,{salience_image_base64}"}},
                                    {"type": "video", "video": image_data_list},
                                    {"type": "text", "text": input_question}
                                ]
                            }
                        ]
                    )
                    model_answer = completion.choices[0].message.content

                except Exception as e:
                    model_answer = f"API fail: {e}"
                
                # print(f'{input_question}\nModel Answer: {model_answer} \nCorrect Answer: {reference}')

                results.append({
                    "video_id": video_id,
                    "question": question,
                    "answer_options": answer_options,
                    "model_answer": model_answer,
                    "reference_answer": reference
                })

            results_df = pd.DataFrame(results)
            results_df.to_csv(f"/home/pty_ssd/EgoEye/results/prompt_gazees/{new_file}.csv", index=False, encoding="utf-8")




if __name__ == "__main__":
    main()
