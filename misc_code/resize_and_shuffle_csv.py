import pandas as pd
df = pd. read_csv("/home/nthom/Documents/similarity_classifiers/misc_code/csvs/vgg_face_224x224_identity_labels_2_classes.csv")
# df = df[:21018]
# df = df[:21017]
df = df.sample(frac=1, random_state=62)
df.to_csv("/home/nthom/Documents/similarity_classifiers/misc_code/csvs/vgg_face_224x224_identity_labels_2_classes_shuffled.csv", index=False)
