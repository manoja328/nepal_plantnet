# import kagglehub
# path = kagglehub.dataset_download("nirmalsankalana/plantdoc-dataset")
# path
# # /workspace3/e33778/.cache/kagglehub/datasets/nirmalsankalana/plantdoc-dataset/versions/7
# path = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset"); path
# # /workspace3/e33778/.cache/kagglehub/datasets/abdallahalidev/plantvillage-dataset/versions/3'

# ls /workspace3/e33778/.cache/kagglehub/datasets/abdallahalidev/plantvillage-dataset/versions/3/plantvillage\ dataset
# ls /workspace3/e33778/.cache/kagglehub/datasets/abdallahalidev/plantvillage-dataset/versions/3/plantvillage\ dataset/color
# ls /workspace3/e33778/.cache/kagglehub/datasets/nirmalsankalana/plantdoc-dataset/versions/7/test/
# ls /workspace3/e33778/.cache/kagglehub/datasets/nirmalsankalana/plantdoc-dataset/versions/7/train/


import os
import shutil
from tqdm import tqdm

class_mapping = {
    "Apple_Scab_Leaf": "Apple___Apple_scab",
    "Apple_leaf": "Apple___healthy",
    "Apple_rust_leaf": "Apple___Cedar_apple_rust",
    "Blueberry_leaf": "Blueberry___healthy",
    "Cherry_leaf": "Cherry_(including_sour)___healthy",
    "Corn_Gray_leaf_spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_leaf_blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_rust_leaf": "Corn_(maize)___Common_rust_",
    "Peach_leaf": "Peach___healthy",
    "Potato_leaf_early_blight": "Potato___Early_blight",
    "Potato_leaf_late_blight": "Potato___Late_blight",
    "Raspberry_leaf": "Raspberry___healthy",
    "Soyabean_leaf": "Soybean___healthy",
    "Squash_Powdery_mildew_leaf": "Squash___Powdery_mildew",
    "Strawberry_leaf": "Strawberry___healthy",
    "Tomato_Early_blight_leaf": "Tomato___Early_blight",
    "Tomato_Septoria_leaf_spot": "Tomato___Septoria_leaf_spot",
    "Tomato_leaf": "Tomato___healthy",
    "Tomato_leaf_bacterial_spot": "Tomato___Bacterial_spot",
    "Tomato_leaf_late_blight": "Tomato___Late_blight",
    "Tomato_leaf_mosaic_virus": "Tomato___Tomato_mosaic_virus",
    "Tomato_leaf_yellow_virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_mold_leaf": "Tomato___Leaf_Mold",
    "grape_leaf": "Grape___healthy",
    "grape_leaf_black_rot": "Grape___Black_rot"
}


pvillage = "/workspace3/e33778/.cache/kagglehub/datasets/abdallahalidev/plantvillage-dataset/versions/3/plantvillage\ dataset/color"
pdoc_train = "/workspace3/e33778/.cache/kagglehub/datasets/nirmalsankalana/plantdoc-dataset/versions/7/train"
pdoc_test = "/workspace3/e33778/.cache/kagglehub/datasets/nirmalsankalana/plantdoc-dataset/versions/7/test"

## copy pviallge images to dataset/train
os.makedirs("dataset/train", exist_ok=True)  # Ensure train directory exists
for k, v in class_mapping.items():
    source_dir = os.path.join(pvillage, k)
    dest_dir = f"dataset/train/{v}"
    os.makedirs(dest_dir, exist_ok=True)  # Ensure destination directory exists
    file_list = os.listdir(source_dir)
    for filename in tqdm(file_list, desc=f"Copying train images for {v}"):
        source_file = os.path.join(source_dir, filename)
        destination_file = os.path.join(dest_dir, filename)
        shutil.copy2(source_file, destination_file)


for k, v in class_mapping.items():
    source_dir = os.path.join(pdoc_train, k)
    dest_dir = f"dataset/train/{v}"
    file_list = os.listdir(source_dir)
    for filename in tqdm(file_list, desc=f"Copying train images for {v}"):
        source_file = os.path.join(source_dir, filename)
        destination_file = os.path.join(dest_dir, filename)
        shutil.copy2(source_file, destination_file)


for k, v in class_mapping.items():
    source_dir = os.path.join(pdoc_test, k)
    dest_dir = f"dataset/test/{v}"
    os.makedirs(dest_dir, exist_ok=True)  # Ensure destination directory exists
    file_list = os.listdir(source_dir)
    for filename in tqdm(file_list, desc=f"Copying test images for {v}"):
        source_file = os.path.join(source_dir, filename)
        destination_file = os.path.join(dest_dir, filename)
        shutil.copy2(source_file, destination_file)