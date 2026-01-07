import os
import random
import shutil

# --- Configuration ---
source_dir = 'data/test_256'
dest_dir = 'data/raw'
num_images_to_move = 300
# Define supported image extensions
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

def select_and_move_images():
    # 1. Create destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"Created destination folder: {dest_dir}")

    # 2. Get list of all image files in source directory
    all_files = [f for f in os.listdir(source_dir) 
                 if f.lower().endswith(valid_extensions)]
    
    # Check if there are enough images
    if len(all_files) < num_images_to_move:
        print(f"Warning: Only found {len(all_files)} images. Moving all of them.")
        selected_files = all_files
    else:
        # 3. Randomly sample the files
        selected_files = random.sample(all_files, num_images_to_move)

    # 4. Move the files
    print(f"Moving {len(selected_files)} images...")
    for filename in selected_files:
        source_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(dest_dir, filename)
        
        # Use shutil.copy() if you want to keep originals, 
        # or shutil.move() to relocate them entirely.
        shutil.copy(source_path, dest_path) 

    print("Task complete.")

if __name__ == "__main__":
    select_and_move_images()