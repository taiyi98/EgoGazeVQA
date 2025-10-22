#!/bin/bash

BASE_DIR="EgoGazeVQA"

for sub_dir in "$BASE_DIR"/*; do
    if [ -d "$sub_dir" ]; then
        video_id=$(basename "$sub_dir")
        
        echo "Processing video ID: $video_id"
        
        for idx in {0..3..2}; do
            echo "Running: python spatial.py --video_id $video_id --target_index $idx"
            python spatial.py --video_id "$video_id" --target_index "$idx"
        done
        
        echo "Finished processing $video_id"
        echo "----------------------------------------"
    fi
done
