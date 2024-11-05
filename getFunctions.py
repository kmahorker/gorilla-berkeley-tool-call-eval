import json
import os

folder_path = "/Users/kaushik/Documents/OceanWater/gorilla-berkeley-tool-call-eval/berkeley-function-call-leaderboard/data"

def process_bfcl_file(file_path, ext):
    with open(file_path, 'r') as f:
        print("FILE PATH: ", file_path)
        function_set = []
        current_file_name = os.path.basename(file_path)
        destination_path = os.path.join(f"{folder_path}/{ext}", f'functions_{current_file_name}')
        for line in f:
            # Process each line of the BFCL file
            # Extract the unique set of functions from the file
            # Add the function to the set of functions
            # Write the set of functions to a file
            line_dict = json.loads(line)
            print(line_dict)
            print()
            for function in line_dict["function"]:
                function_set.append(function)
                
        # remove duplicates by checking f.name
        function_set = [f for n, f in enumerate(function_set) if f["name"] not in [g["name"] for g in function_set[:n]]]
        
        print("FUNCTION SET: ", function_set)
                
        with open(destination_path, 'w') as f:
            for function in function_set:
                f_json = json.dumps(function)
                f.write(f_json + "\n")
            
# Get all files starting with BFCL
bfcl_files = [f for f in os.listdir(folder_path) if f.startswith('BFCL') and f.find('multi_turn') == -1]
multi_turn_files = [f for f in os.listdir(f"{folder_path}/multi_turn_func_doc")]

# Process each BFCL file

def process_multi_turn_files(ext="multi_turn_func_doc"):
    for bfcl_file in multi_turn_files:
        file_path = os.path.join(f"{folder_path}/{ext}", bfcl_file)
        process_bfcl_file(file_path, ext)
    
process_multi_turn_files()