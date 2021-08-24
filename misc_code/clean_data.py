import os
from tqdm import tqdm
import torch
import torchvision


path_to_original_images = "/home/nthom/Documents/datasets/vgg_face_dataset/images/"
path_to_good_images = "/home/nthom/Documents/datasets/vgg_face_dataset/good_images/"
path_to_problem_images = "/home/nthom/Documents/datasets/vgg_face_dataset/problem_images/"
image_names = sorted(os.listdir(path_to_original_images))

num_images = len(image_names)

good_count = 0
bad_count = 0
for file in tqdm(image_names):
    print(file)
    try:
        img_path = path_to_original_images + file
        image = torchvision.io.read_image(img_path)
        os.rename(path_to_original_images + file, path_to_good_images + file)
        good_count+=1
    except:
        bad_count+=1
        os.rename(path_to_original_images+file, path_to_problem_images+file)

print(good_count, bad_count)