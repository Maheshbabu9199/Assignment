# Creating more training dataset using data augmentation techniques

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os, glob, cv2, keras

generator = ImageDataGenerator(
    rotation_range=0.4,
    zoom_range=0.2,
    vertical_flip=True,
    horizontal_flip=True,
    )

# Configuring the images directory
img_dir = "Images"

# images
data_path = os.path.join(img_dir,'*g')
print(data_path)

files = glob.glob(data_path)
data = []

for f1 in files:
    img = cv2.imread(f1)
    data.append(img)

for img in data:
    x= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    i = 0
    # path, dirs, files = next(os.walk("Images"))
    # file_count = len(files) #to find number of files in folder

    for batch in generator.flow (x, batch_size=1, save_to_dir ="New_Images",save_prefix="new_",save_format='jpg'):
        i+=1
        if i>10:
            break
"""
# Converting the image to array
arr = img_to_array(img)
print("arr shape is: ",arr.shape)
print("Again reshaped the shape :", arr)

arr = arr.reshape((1,)+arr.shape)
print("new shape: ",arr.shape)

i=0
for batch in generator.flow(arr, batch_size=1, save_to_dir="New_Images", save_prefix="new_image", save_format="jpg"):
    if i>20:
        break
    i+=1"""