
#for downloading the dataset in zip file format
!kaggle datasets download -d xhlulu/140k-real-and-fake-faces

#extracting the images from zip file
import zipfile
zip_ref = zipfile.ZipFile('/content/140k-real-and-fake-faces.zip','r')
zip_ref.extractall('/content')
zip_ref.close()

#importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import tensorflow as tf
from keras.preprocessing import image
import seaborn as sns
from sklearn import metrics

"""#### Data loading"""

#main path is provided in which images are being downloaded
main_path = '/content/real_vs_fake/real-vs-fake'
#joining the path to the train,valid,test directories using the join() function available in os.path module
train_dir = os.path.join(main_path, 'train')
valid_dir = os.path.join(main_path, 'valid')
test_dir = os.path.join(main_path, 'test')

#listdir() will give the subfolders that are present in the particular directory
print("Train_dir splitted_data: ", os.listdir(train_dir))
print("Valid_dir splitted_data: ", os.listdir(valid_dir))
print("Test_dir splitted_data: ", os.listdir(test_dir))

#creating the dictionaries for real_images and fake_images
real_images_df = {
    "splitted_data":[],
    "image_path":[],
    "label":[]
}
fake_images_df = {
    "splitted_data":[],
    "image_path":[],
    "label":[]
}
for splitted_data in os.listdir(main_path): #iterate on each train, valid and test folder
    for label in os.listdir(main_path + "/" + splitted_data): #iterate on fake and real labels
        if(label=='real'):
        #glob.glob() is a function which is used to search for a particular extension files that means it returns a list containing all the files related to given extension
        #iterate on images in folders
            for img in glob.glob(main_path + "/" + splitted_data + "/" + label + "/*.jpg"):
                real_images_df["splitted_data"].append(splitted_data)
                real_images_df["image_path"].append(img)
                real_images_df["label"].append(label)
        else:
            for img in glob.glob(main_path + "/" + splitted_data + "/" + label + "/*.jpg"):
                fake_images_df["splitted_data"].append(splitted_data)
                fake_images_df["image_path"].append(img)
                fake_images_df["label"].append(label)

#converting the dictionaries into a dataframes
real_images_df =pd.DataFrame(real_images_df)
fake_images_df = pd.DataFrame(fake_images_df)
real_images_df

fake_images_df

#concatenating the dataframes real_images and fake_images
images_df = pd.concat([real_images_df, fake_images_df])
images_df

#understanding what is present in the column splitted_data of real_images_df
real_df=real_images_df['splitted_data']
real_df

#finding the count of the each type of data in splitted_data
real_test_count=real_train_count=real_valid_count=0
for data in real_df:
    if(data=='test'):
        real_test_count+=1
    if(data=='train'):
        real_train_count+=1
    if(data=='valid'):
        real_valid_count+=1
print("real_test_count:",real_test_count)
print("real_train_count:",real_train_count)
print("real_valid_count:",real_valid_count)

#understanding what is present in the column splitted_data of fake_images_df
fake_df=fake_images_df['splitted_data']
fake_df

#finding the count of the each type of data in splitted_data of fake_images_df-
fake_test_count=0
fake_train_count=0
fake_valid_count=0
for data in fake_df:
    if(data=='test'):
        fake_test_count+=1
    if(data=='train'):
        fake_train_count+=1
    if(data=='valid'):
        fake_valid_count+=1
print("fake_test_count:",fake_test_count)
print("fake_train_count:",fake_train_count)
print("fake_valid_count:",fake_valid_count)

import tensorflow as tf
from tensorflow import keras

"""##### Data Preprocessing which involves data resizing,normalization and augmentation"""

#normalization which is done to convert the large pixels format(0-225) to 0-1 which will be helpful for mathematical applications
image_train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                                  rescale=1./255.,
                                  horizontal_flip=True,
                                  )

image_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)

train_ds = image_train_gen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    batch_size=64,
    class_mode='binary',
)

valid_ds = image_gen.flow_from_directory(
    valid_dir,
    target_size=(256, 256),
    batch_size=64,
    class_mode='binary'
)

test_ds = image_gen.flow_from_directory(
    test_dir,
    target_size=(256, 256),
    batch_size=64,
    class_mode='binary',
    shuffle=False)

#this function will plot the images it takes image and label as parameters
#figsize=[12,12] it will create like a container of 12x12 inches and loop starts iterating it will plot 16 images
#in this area a subplot is plotted with as 4x4 and i+1 refers to the value between 1-16 then the image is displayed using imshow function
#axis('off') says the axis for every current plot willn't be visualized
def plot_images(img, label):
    plt.figure(figsize=[12, 12])
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(img[i])
        plt.axis('off')
        if label[i] == 0:
            plt.title("Fake")
        else:
            plt.title("Real")

img,lbl = next(train_ds)
plot_images(img,lbl)

for img, label in train_ds:
    print("Values: ", img[0])
    print("Label: ", label[0])
    break

for img, label in train_ds:
    print(img.shape)
    print(label.shape)
    break

#defining the input size of the model
input_shape = (256,256,3)

#create the model's architecture and compile it
def get_model(input_shape):
    #an input layer of the architecture is created with the input_tensor of defined input_shape
    input = tf.keras.Input(shape=input_shape)
    #densenet121 is instantiated with weights as imagenet,include_top is the top classification layer which sets as False and input_tensor is our created input layer.
    densenet = tf.keras.applications.DenseNet121( weights="imagenet", include_top=False, input_tensor = input)
    #global average Pooling layer has added to the output of the densenet121 to reduce the feature map to the single value
    x = tf.keras.layers.GlobalAveragePooling2D()(densenet.output)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x) #binary classification

    model = tf.keras.Model(densenet.input, output)

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy",tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall')])

    return model

#to get the summary of the model
model_ft = get_model(input_shape)

model_ft.summary()

checkpoint_filepath = "fake_vs_real_cp.h5"

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min', #minimize the loss value
    save_best_only=True)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                     patience=5,
                                                     restore_best_weights=True,
                                                    )

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                 factor=0.2,
                                                 patience=3)

history_ft = model_ft.fit(train_ds,
                       epochs = 5,
                       validation_data = valid_ds,
                       callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr])

model_ft.save('deepfake.keras')
model_ft.save('realvsfake.h5')
plt.plot(history_ft.history['accuracy'])
plt.plot(history_ft.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper right')
plt.show()


plt.plot(history_ft.history['loss'])
plt.plot(history_ft.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper right')
plt.show()

# Evaluate the model on the test set
test_loss, test_acc,precision,recall = model_ft.evaluate(test_ds)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)
print("Precision:", precision)
print("Recall:", recall)
f1_score = 2 * (precision * recall) / (precision + recall)
print("F1-Score :",f1_score)

model = get_model(input_shape)
# Restore the weights
model.load_weights('/content/fake_vs_real_cp.h5')

from google.colab import drive
drive.mount('/content/drive')

# Evaluate the model on the test set
test_loss, test_acc,precision,recall = model.evaluate(test_ds)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)
print("Precision:", precision)
print("Recall:", recall)
f1_score = 2 * (precision * recall) / (precision + recall)
print("F1-Score :",f1_score)

#confusion matrix
predicted_labels = model.predict(test_ds)
true_labels = test_ds.classes
cm=metrics.confusion_matrix(true_labels, predicted_labels.round())
print(cm)

#testing the model by giving the new image
test_image = tf.keras.preprocessing.image.load_img('/content/Screenshot 2024-06-05 133808.png', target_size=(256, 256, 3))
plt.imshow(test_image)


test_image_arr = tf.keras.preprocessing.image.img_to_array(test_image)
test_image_arr = np.expand_dims(test_image, axis=0)
test_image_arr = test_image_arr/255.


result = model.predict(test_image_arr)

plt.title(f"This image is {100 * (1 - result[0][0]):.2f}% Fake and {100 * result[0][0]:.2f}% Real.")
