#!/bin/bash

# Set the source and destination directories
src_dir="/Users/kaushik/Documents/OceanWater/gorilla-berkeley-tool-call-eval/berkeley-function-call-leaderboard/data/multi_turn_func_doc"
dest_dir="/Users/kaushik/Documents/OceanWater/gorilla-berkeley-tool-call-eval/berkeley-function-call-leaderboard/data/all_functions"

# Create the destination directory if it doesn't exist
mkdir -p "$dest_dir"

# Loop through each JSON file in the source directory
for file in "$src_dir"/*.json; do
    # Get the base filename
    base_filename=$(basename "$file")
    
    # Prepend 'multi_turn_func_doc_' and move to the destination directory
    mv "$file" "$dest_dir/multi_turn_func_doc_$base_filename"
done

echo "Files moved and renamed successfully."