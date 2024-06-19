'''Use this script to generate a training dataset for I-JEPA, the downstream predictor network and a test dataset from Doges-77-Breeds.
'''

import glob
import os
from pathlib import Path
from sklearn.model_selection import train_test_split


dataset_root_folder='./Doges_77/'

#make sure the directories are one level above to avoid conflicts/confusion.
# ssl_train_folder = Path('../ssl_train/')
downstream_train_folder = Path('./dataset/train/')
downstream_test_folder = Path('./dataset/test/')

#create directories
# ssl_train_folder.mkdir(exist_ok=True, parents=True)
downstream_train_folder.mkdir(exist_ok=True, parents=True)
downstream_test_folder.mkdir(exist_ok=True, parents=True)

ratio_for_testing = 0.3 
ratio_for_downstream = 0.7

for x in glob.glob(dataset_root_folder+'**', recursive=True):

	if x.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
		folder_name = x.split('/')[-2].lower().replace(' ', '_')
		file_name = x.split('/')[-1]

		#create new directories in the downstream train and test folder.
		new_train_folder_path = Path(f'{downstream_train_folder}/{folder_name}/')
		new_test_folder_path = Path(f'{downstream_test_folder}/{folder_name}/')
		new_train_folder_path.mkdir(parents=True, exist_ok=True)
		new_test_folder_path.mkdir(parents=True, exist_ok=True)

		#move everything to the train downstream folder first,
		os.rename(x, f'{downstream_train_folder}/{folder_name}/{file_name}')



images = []
labels = []

for x in glob.glob(os.path.join(downstream_train_folder, '**'), recursive=True):

	if x.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):

		folder_name = x.split('/')[-2].lower().replace(' ', '_')
		images.append(x)
		labels.append(folder_name)



downstream_train_images, downstream_test_images, downstream_train_labels, downstream_test_labels = train_test_split(images, labels, test_size=ratio_for_testing, random_state=42, stratify=labels)

# ssl_train_images, downstream_train_images, _, downstream_train_labels = train_test_split(downstream_train_images, downstream_train_labels, test_size=ratio_for_downstream, stratify=downstream_train_labels)


#write to test folder.
for image_path, folder_name in zip(downstream_test_images, downstream_test_labels):

	file_name = image_path.split('/')[-1]
	
	os.rename(image_path, os.path.join(downstream_test_folder, folder_name, file_name))



# #write to ssl folder.
# for image_path in ssl_train_images:

# 	file_name = image_path.split('/')[-1]
	
# 	os.rename(image_path, os.path.join(ssl_train_folder, file_name))