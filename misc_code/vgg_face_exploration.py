import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# path_to_images = "/home/nthom/Documents/datasets/vgg_face_dataset/images/"
path_to_images = "/home/nthom/Documents/datasets/vgg_face_dataset/images_cropped_224x224/"
path_to_labels = "/home/nthom/Documents/datasets/vgg_face_dataset/labels/"
path_to_identities = "/home/nthom/Documents/datasets/vgg_face_dataset/files/"
path_to_csv = "/home/nthom/Documents/similarity_classifiers/misc_code/vgg_face_224x224_identity_labels.csv"

cropped_images_df = pd.read_csv(path_to_csv)
names_list = cropped_images_df['file_path'].tolist()
labels_list = cropped_images_df['identity_label'].tolist()

classes = cropped_images_df.identity_label.unique()
classes_w_under_500_samples = []
classes_w_under_450_samples = []
classes_w_under_425_samples = []
classes_w_under_400_samples = []
classes_w_under_300_samples = []
classes_w_under_200_samples = []

current_class = 0
current_class_count = 0
for index, label in enumerate(labels_list):
    if label != current_class:
        if current_class_count < 500:
            classes_w_under_500_samples.append(current_class)
        if current_class_count < 450:
            classes_w_under_450_samples.append(current_class)
        if current_class_count < 425:
            classes_w_under_425_samples.append(current_class)
        if current_class_count < 400:
            classes_w_under_400_samples.append(current_class)
        if current_class_count < 300:
            classes_w_under_300_samples.append(current_class)
        if current_class_count < 200:
            classes_w_under_200_samples.append(current_class)
        current_class = label
        current_class_count = 0
    current_class_count+=1

image_filenames = sorted(os.listdir(path_to_images))
image_filenames_dict = {}
for index, image_name in enumerate(image_filenames):
    image_filenames_dict[image_name] = 0

label_filenames = sorted(os.listdir(path_to_labels))
identity_filenames = sorted(os.listdir(path_to_identities))

image_filenames_wo_extensions = []
label_filenames_wo_extensions = []

for file in tqdm(image_filenames):
    image_filenames_wo_extensions.append(file[:-4])
for file in tqdm(label_filenames):
    label_filenames_wo_extensions.append(file[:-4])

difference_list = list(set(image_filenames_wo_extensions) - set(label_filenames_wo_extensions)) + \
                  list(set(label_filenames_wo_extensions) - set(image_filenames_wo_extensions))

for item in tqdm(difference_list):
    try:
        # image_filenames.remove(item + ".jpg")
        image_filenames_dict.pop(item + ".jpg")
    except Exception as e:
        print(f"*** {e} ***")

output_list = [["file_path", "identity_label"]]
image_filenames = sorted(list(image_filenames_dict.keys()))
current_identity = image_filenames[0][:-13]
identity_counter = 0
for file in tqdm(image_filenames):
    if file[:-13] != current_identity:
        current_identity = file[:-13]
        identity_counter += 1
    output_list.append([path_to_images+file, identity_counter])

output_df = pd.DataFrame(output_list)

output_df.to_csv("./vgg_face_224x224_identity_labels.csv", header=False, index=False)

# print(len(image_filenames), len(label_filenames),
#       (len(image_filenames)-len(label_filenames)))
#
# print(f"First 10 image filenames: {image_filenames[:10]} \n\n"
#       f"First 10 label filenaes: {label_filenames[:10]}")

