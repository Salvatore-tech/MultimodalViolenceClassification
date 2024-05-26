import os
import glob
import shutil
from sklearn.model_selection import train_test_split

########################################################################################
#Change the 3 variables below for your own need.

# The path of features, relative to the script's location, original test set values
root_path = '../../../../Data/vggish-features/test'

#Make it None, if you haven't splitted the set. But remember we only make 1 split and always use them. The split made in original is made on RGB set.
#This step is required because, even if we set the random_state, train_test_split changes when we are looking for different features.
#If you already made the split, set here the path of train_filenames.list
train_list = 'train_filenames.list'
#The below variable only improtant, if train_list is None
#train_list_to_write = os.path.abspath('train_filenames.list') 

#Name of the .list file you want the give to training set.  
train_list_path = os.path.abspath('audio.list')

#Name of the .list file you want the give to test set.  
test_list_path = os.path.abspath('audio_test.list')

######################################################
#The 4 code line below should only be used if we are planning to seperate the original test videos (.mp4 files)
#If you set seperatE_videos=False, it will not do anything
seperate_videos=False
videos_path ='../../../../Data/Test/videos'
new_train_videos_path = '../../../../Data/train-videos-new'
new_test_videos_path = '../../../../Data/test-videos-new'
######################################################
#Only set it true with RGB or Flow, otherwise it throws an error.
split_annotation = False 
annotation_path = '../../../../Data/Test/annotations.txt'
annotation_train_path = 'annotations.txt'
annotation_test_path = 'annotations_test.txt'
########################################################################################

def get_train_test_filenames(path, train_list:list=None):
    # Get the absolute path to the root_path
    abs_root_path = os.path.abspath(path)

    # Find all .npy files and sort them
    features = sorted(glob.glob(os.path.join(abs_root_path, "*.npy")))

    #Features are repeating for the same file. We extract the file names.
    filenames = list(set([(os.path.basename(feat).split('.npy')[0])[:-3] for feat in features]))  

    #This means we already made the split and saved the filenames of the train set.
    if train_list is not None:
        #We read our train_filenames from train_list variable which is a path
        with open(train_list, 'r') as file:
            train_list = file.read().splitlines()

        if isinstance(train_list, list):
            train_filenames = []
            test_filenames = []
            for file in filenames:
                if any(filter_value in file for filter_value in train_list):
                    train_filenames.append(file)
                else:
                    test_filenames.append(file)

            train_filenames, test_filenames = sorted(train_filenames), sorted(test_filenames)
        else:
            raise TypeError("train_list must be a Ptyhon list. You provided different type.")
    
    #In this case, we havent made the split.
    elif train_list is None:
        #Labels for each file in same order
        labels = [ 0 if '_label_A' in os.path.basename(file) else 1 for file in filenames]

        #Splitting the files to train and test
        train_filenames, test_filenames, train_labels, test_labels = train_test_split(filenames, labels, test_size=0.2, stratify=labels, random_state=42)
        train_filenames, test_filenames = sorted(train_filenames), sorted(test_filenames)

        #Creating a file to store training video filenames for further use.
        with open(train_list_to_write, 'w+') as f:
            for file in train_filenames:
                newline = file + '\n'
                f.write(newline)

    return train_filenames, test_filenames, features

#Calling the function
train_filenames, test_filenames, features = get_train_test_filenames(root_path, train_list)

#Train List
train_normal = []

#Sorting the files by label(because they did like that), writing the .list file
with open(train_list_path, 'w+') as f:
    for feat in features:
        if any(filename in feat for filename in train_filenames):
            if '_label_A' in feat: 
                train_normal.append(feat)
            else: 
                newline = feat + '\n'
                f.write(newline)

    for feat in train_normal:
        newline = feat + '\n'
        f.write(newline)


#Test List
test_normal = []

with open(test_list_path, 'w+') as f:
    for feat in features:
        if any(filename in feat for filename in test_filenames):
            if '_label_A' in feat: 
                test_normal.append(feat)
            else: 
                newline = feat + '\n'
                f.write(newline)
                
    for feat in test_normal:
        newline = feat + '\n'
        f.write(newline)

train_annotation = []
test_annotation = []
#Split the annotation file
if split_annotation:
    with open(annotation_path, 'r') as file:
        annotation = file.read().splitlines()
    for anno in annotation:
        if any(filename in anno for filename in train_filenames):
            train_annotation.append(anno)
        elif any(filename in anno for filename in test_filenames):
            test_annotation.append(anno)
        else:
            raise ValueError(f"There is a problem with this anno it is not exist in test or train: {anno}")
        
    with open(annotation_train_path, 'w+') as f:
        for anno in train_annotation:
            newline = anno + '\n'
            f.write(newline)

    with open(annotation_test_path, 'w+') as f:
        for anno in test_annotation:
            newline = anno + '\n'
            f.write(newline)


#Copying the test videos into two seperate folder as they splitted above.
if seperate_videos:
    #shutil.rmtree(new_train_videos_path + "/")
    #shutil.rmtree(new_test_videos_path + "/")
    for filename in os.listdir(videos_path):
        if filename.endswith('.mp4'):  
            if filename[:-4] in train_filenames:
                # Copy to train
                shutil.copy(os.path.join(videos_path, filename), new_train_videos_path)
            elif filename[:-4] in test_filenames:
                # Copy to train
                shutil.copy(os.path.join(videos_path, filename), new_test_videos_path)
            else:
                raise OSError(f"File not found in both train and test filenames list you splitted: {filename}")
print('Writing Done!')