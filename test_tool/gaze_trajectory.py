import cv2
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO

def plot_gaze_trajectory(image_path, gaze_info_list, output_image_path="gaze_trajectory.png", sigma=20, radius=60, weight_factor=20, alpha=0.5):


    first_image = cv2.imread(image_path)
    height, width, _ = first_image.shape  

    
    salience_map = np.zeros((height, width), dtype=np.float32)

    
    step_values = np.linspace(10, 100, len(gaze_info_list))  

    for i, gaze_info in enumerate(gaze_info_list):
        gaze_x = int(gaze_info['gaze_x'] * width)  
        gaze_y = int(gaze_info['gaze_y'] * height)


        for dx in range(-radius, radius + 1, 10):  
            for dy in range(-radius, radius + 1, 10):  
               
                dist = np.sqrt(dx ** 2 + dy ** 2)
                if dist <= radius:
                    
                    target_x = gaze_x + dx
                    target_y = gaze_y + dy
                    
               
                    target_x = max(0, min(target_x, salience_map.shape[1] - 1))
                    target_y = max(0, min(target_y, salience_map.shape[0] - 1))
                    
               
                    salience_map[target_y, target_x] += weight_factor * np.exp(-dist ** 2 / (2 * (sigma ** 2)))
                
              
                gaze_x = max(0, min(gaze_x, salience_map.shape[1] - 1))
                gaze_y = max(0, min(gaze_y, salience_map.shape[0] - 1))

   
        salience_map[gaze_y, gaze_x] += weight_factor * step_values[i]  
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1

    salience_map = cv2.GaussianBlur(salience_map, (ksize, ksize), 0)


    salience_map_normalized = cv2.normalize(salience_map, None, 0, 255, cv2.NORM_MINMAX)


    image_stream = BytesIO()
    _, encoded_img = cv2.imencode('.png', salience_map_normalized.astype(np.uint8)) 
    image_stream.write(encoded_img)
    image_stream.seek(0)
    image_base64 = base64.b64encode(image_stream.read()).decode('utf-8')

    return image_base64

