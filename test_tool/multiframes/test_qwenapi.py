import os
import json
import base64
import cv2
from PIL import Image
import io
from openai import OpenAI
import pandas as pd
import time

def extract_frames_from_video(video_path, num_frames):

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    images = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
    
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            images.append(img)
    cap.release()
    return images

def encode_images_to_base64(images):

    image_data_list = []
    for img in images:
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image_data_list.append(f"data:image/jpeg;base64,{base64_image}")
    return image_data_list

def main():
    datasets = ['ego4d','egoexo','egtea']
    categories = ['spatial','temporal','causal']
    for dataset in datasets:
        for category in categories:
            start_time = time.time() 
            print(f"----Processing {category}_{dataset}.json----\n")
          
            json_path = f"/home/pty_ssd/EgoEye/qa_pairs/{category}_{dataset}.json"
            file_name = os.path.basename(json_path)
            new_file = os.path.splitext(file_name)[0]
            base_folder = f"/home/pty_ssd/EgoEye/datasets/clips_video/{dataset}"
            api_key = ""  
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1" 
            
            with open(json_path, "r", encoding="utf-8") as f:
                qa_data = json.load(f)
            client = OpenAI(api_key=api_key, base_url=base_url)
            
            results = []

            for video_id, qa_list in qa_data.items():
                for qa in qa_list:
                    clip_name = qa["clip_name"]
                    video_path = os.path.join(base_folder, video_id, clip_name)
                    if not os.path.exists(video_path):                     
                        continue

                   
                    images = extract_frames_from_video(video_path, num_frames=9)
                    if not images:
                        continue
                    image_data_list = encode_images_to_base64(images)

                    question = qa["question"]
                    answer_options = "\n".join(qa["answer_options"])
                    reference = qa["correct_answer"]

                    input_question = ("I provide you with a video. Choose the correct option based on the first-person perspective scene question.\n"
                                    f"Question:\n{question}\nOptions:\n{answer_options}\n"
                                    "Choose the most appropriate option. Only return the letter of the correct option.")
                    
                    try:
                        completion = client.chat.completions.create(
                            model="qwen-vl-max-latest",
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "video", "video": image_data_list},
                                        {"type": "text", "text": input_question}
                                    ]
                                }
                            ]
                        )
                        model_answer = completion.choices[0].message.content

                    except Exception as e:
                        model_answer = f"API 调用失败: {e}"
                    
                    print(f'{input_question}\nModel Answer: {model_answer} \nCorrect Answer: {reference}')

                    results.append({
                        "video_id": video_id,
                        "clip_name": clip_name,
                        "question": question,
                        "answer_options": answer_options,
                        "model_answer": model_answer,
                        "reference_answer": reference
                    })
             


            results_df = pd.DataFrame(results)
            results_df.to_csv(f"results/multiframes/frame9/{new_file}.csv", index=False, encoding="utf-8")
            print(f"save results/multiframes/frame9/{new_file}.csv。")

            end_time = time.time()
            elapsed_min = (end_time - start_time) / 60

if __name__ == "__main__":
    main()

