import pandas as pd
import os

def calculate_accuracy(file_path):

    df = pd.read_csv(file_path)
    
    # correct_count = (df["Model_Answer"] == df["Reference_Answer"]).sum()
    # correct_count = (df["InternVL_Model_Answer"] == df["Reference_Answer"]).sum()
    correct_count = (df["model_answer"] == df["reference_answer"]).sum()
    total_count = len(df)
    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    
    return round(accuracy, 2)

if __name__ == "__main__":
    folder_path = "/home/pty_ssd/EgoEye/results/multiframes/frame4"
    for file_name in os.listdir(folder_path):
        if file_name.endswith("_merged.csv"):
            file_path = os.path.join(folder_path, file_name)
            accuracy = calculate_accuracy(file_path)
            print(f"{file_name} Acc: {accuracy:.2f}%")
