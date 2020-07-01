
# Mod 4 Project - Image Classification

* Student name: Alex Beat
* Student pace: part time
* Scheduled project review date/time: 06/24/20 @12pm pacific
* Instructor name: James Irving
* Blog post URL: NA

# Intro

* Problem: Build a model that can classify whether a given patient has pneumonia, given a chest x-ray image.
* Audience: Medical business, imaging labs. 
* Business Questions: How can a successful model help save medical professionals time, money and promote better accuracy in patient diagnosis. 

## Content
The dataset is organized into 2 folders (train, test) and contains subfolders for each image category (Pneumonia/Normal). There are 5,385 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).

Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care.

For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.

(Content info and dataset provided by Paul Mooney on Kaggle: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia?)

# Select Colab directory and unzip virtually in colab.

This first section will help those who want to or need to run Keras and Tensorflow in Colab. I was forced to run my notebooks in the cloud using Colab because my OS was too old and wouldn't run the necesary versions of Keras. Working with Colab is very fast with image loading, as long as you set it up correctly. If not, it will be slower than a notebook on your own machine. This blog post was used to help load in your entire zip file and virtually unzip your image files in one move in colab as opposed to how Colab normally would load in each image individually: https://medium.com/datadriveninvestor/speed-up-your-image-training-on-google-colab-dc95ea1491cf

### Mount google drive connection to colab. 


```python
from google.colab import drive

# choose your drive file path
drive.mount('/content/drive', force_remount=True)

# cd to the uppermost folder in gdrive to save your zip file
%cd ~
%cd ..
```

    Go to this URL in a browser: https://accounts.google.com/o/oauth2/auth?client_id=947318989803-6bn6qk8qdgf4n4g3pfee6491hc0brc4i.apps.googleusercontent.com&redirect_uri=urn%3aietf%3awg%3aoauth%3a2.0%3aoob&response_type=code&scope=email%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdocs.test%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive.photos.readonly%20https%3a%2f%2fwww.googleapis.com%2fauth%2fpeopleapi.readonly
    
    Enter your authorization code:
    ··········
    Mounted at /content/drive
    /root
    /


### Create file paths


```python
import os,glob

# check current directory to see if you're at the top
print(os.path.abspath(os.curdir))

# file path to save your zip file dataset
source_folder = r'content/drive/My Drive/Datasets/'

# variable for your filename
file = glob.glob(source_folder+'xray.zip',recursive=True)[0]
file
```

    /





    'content/drive/My Drive/Datasets/xray.zip'



### Zip virtual copy/unzip/removal.


```python
# variable for the path
zip_path = file

# copies the zip file
!cp "{zip_path}" .

# unzip file virtually in colab
!unzip -q xray.zip

# removes copied zip
!rm xray.zip
```


```python
# check directory to see virtual files are in your main folder
import os,glob
print(os.path.abspath(os.curdir))
os.listdir()

```

    /





    ['srv',
     'home',
     'etc',
     'tmp',
     'opt',
     'run',
     'lib',
     'root',
     'sys',
     'usr',
     'lib64',
     'var',
     'media',
     'proc',
     'boot',
     'sbin',
     'mnt',
     'bin',
     'dev',
     'xray',
     'content',
     '.dockerenv',
     'tools',
     'datalab',
     'swift',
     'tensorflow-1.15.2',
     'dlib-19.18.0-cp27-cp27mu-linux_x86_64.whl',
     'dlib-19.18.0-cp36-cp36m-linux_x86_64.whl',
     'lib32']



# Step 1: Load the Data / Preprocessing


```python
!pip install -U fsds_100719
from fsds_100719.imports import *

import pandas as pd
import numpy as np
np.random.seed(111)

```

    fsds_1007219  v0.7.22 loaded.  Read the docs: https://fsds.readthedocs.io/en/latest/ 



<style  type="text/css" >
</style><table id="T_49fa5596_b73b_11ea_8337_0242ac1c0002" ><caption>Loaded Packages and Handles</caption><thead>    <tr>        <th class="col_heading level0 col0" >Handle</th>        <th class="col_heading level0 col1" >Package</th>        <th class="col_heading level0 col2" >Description</th>    </tr></thead><tbody>
                <tr>
                                <td id="T_49fa5596_b73b_11ea_8337_0242ac1c0002row0_col0" class="data row0 col0" >dp</td>
                        <td id="T_49fa5596_b73b_11ea_8337_0242ac1c0002row0_col1" class="data row0 col1" >IPython.display</td>
                        <td id="T_49fa5596_b73b_11ea_8337_0242ac1c0002row0_col2" class="data row0 col2" >Display modules with helpful display and clearing commands.</td>
            </tr>
            <tr>
                                <td id="T_49fa5596_b73b_11ea_8337_0242ac1c0002row1_col0" class="data row1 col0" >fs</td>
                        <td id="T_49fa5596_b73b_11ea_8337_0242ac1c0002row1_col1" class="data row1 col1" >fsds_100719</td>
                        <td id="T_49fa5596_b73b_11ea_8337_0242ac1c0002row1_col2" class="data row1 col2" >Custom data science bootcamp student package</td>
            </tr>
            <tr>
                                <td id="T_49fa5596_b73b_11ea_8337_0242ac1c0002row2_col0" class="data row2 col0" >mpl</td>
                        <td id="T_49fa5596_b73b_11ea_8337_0242ac1c0002row2_col1" class="data row2 col1" >matplotlib</td>
                        <td id="T_49fa5596_b73b_11ea_8337_0242ac1c0002row2_col2" class="data row2 col2" >Matplotlib's base OOP module with formatting artists</td>
            </tr>
            <tr>
                                <td id="T_49fa5596_b73b_11ea_8337_0242ac1c0002row3_col0" class="data row3 col0" >plt</td>
                        <td id="T_49fa5596_b73b_11ea_8337_0242ac1c0002row3_col1" class="data row3 col1" >matplotlib.pyplot</td>
                        <td id="T_49fa5596_b73b_11ea_8337_0242ac1c0002row3_col2" class="data row3 col2" >Matplotlib's matlab-like plotting module</td>
            </tr>
            <tr>
                                <td id="T_49fa5596_b73b_11ea_8337_0242ac1c0002row4_col0" class="data row4 col0" >np</td>
                        <td id="T_49fa5596_b73b_11ea_8337_0242ac1c0002row4_col1" class="data row4 col1" >numpy</td>
                        <td id="T_49fa5596_b73b_11ea_8337_0242ac1c0002row4_col2" class="data row4 col2" >scientific computing with Python</td>
            </tr>
            <tr>
                                <td id="T_49fa5596_b73b_11ea_8337_0242ac1c0002row5_col0" class="data row5 col0" >pd</td>
                        <td id="T_49fa5596_b73b_11ea_8337_0242ac1c0002row5_col1" class="data row5 col1" >pandas</td>
                        <td id="T_49fa5596_b73b_11ea_8337_0242ac1c0002row5_col2" class="data row5 col2" >High performance data structures and tools</td>
            </tr>
            <tr>
                                <td id="T_49fa5596_b73b_11ea_8337_0242ac1c0002row6_col0" class="data row6 col0" >sns</td>
                        <td id="T_49fa5596_b73b_11ea_8337_0242ac1c0002row6_col1" class="data row6 col1" >seaborn</td>
                        <td id="T_49fa5596_b73b_11ea_8337_0242ac1c0002row6_col2" class="data row6 col2" >High-level data visualization library based on matplotlib</td>
            </tr>
    </tbody></table>



        <script type="text/javascript">
        window.PlotlyConfig = {MathJaxConfig: 'local'};
        if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
        if (typeof require !== 'undefined') {
        require.undef("plotly");
        requirejs.config({
            paths: {
                'plotly': ['https://cdn.plot.ly/plotly-latest.min']
            }
        });
        require(['plotly'], function(Plotly) {
            window._Plotly = Plotly;
        });
        }
        </script>
        


    [i] Pandas .iplot() method activated.


## Define folder structure for use with .flow.


```python
#  Set up directory paths from gdrive
train_dir_normal = 'xray/train/NORMAL'
train_dir_pneum = 'xray/train/PNEUMONIA'
test_dir_normal = 'xray/test/NORMAL'
test_dir_pneum = 'xray/test/PNEUMONIA'
all_dir_paths = [train_dir_normal, train_dir_pneum, test_dir_normal,
                 test_dir_pneum]
```


```python
# use loop to get number of samples from all folders
for dataset in all_dir_paths: 
  print(f'There are {len(os.listdir(dataset))} images in {str(dataset)} folder.') 
```

    There are 1342 images in xray/train/NORMAL folder.
    There are 3419 images in xray/train/PNEUMONIA folder.
    There are 234 images in xray/test/NORMAL folder.
    There are 390 images in xray/test/PNEUMONIA folder.
    There are 0 images in xray/val/NORMAL folder.
    There are 0 images in xray/val/PNEUMONIA folder.



```python
import glob,os
```


```python
# use glob to get filenames of all images in each folder
train_files_normal = glob.glob(train_dir_normal+'/*.jpeg')
train_files_pneum = glob.glob(train_dir_pneum+'/*.jpeg')
all_train_files = [*train_files_normal,*train_files_pneum]

test_files_normal = glob.glob(test_dir_normal+'/*.jpeg')
test_files_pneum = glob.glob(test_dir_pneum+'/*.jpeg')
all_test_files = [*test_files_normal,*test_files_pneum]
```

## Functions

These functions were adapted from this Colab notebook about CNNs. https://colab.research.google.com/drive/1fwXPY3IDHxNiv7YgOpt3p5BvUaO4VruB?usp=sharing


```python
from PIL import Image
from keras.preprocessing import image
from imageio import imread
from skimage.transform import resize
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
%matplotlib inline 
import itertools

def read_img(img_path,target_size=(64, 64)):
  """This function reads in the image using the image path filenames created by glob. 
  Then it turns the image into an arrary for use with modeling. Specify target pixel 
  size using target_size=(n,n). Default is 64,64 pixels."""
  
  img = image.load_img(img_path, target_size=target_size)
  img = image.img_to_array(img)
  
  return img

def load_train_test_images(training_filenames_normal, training_filenames_pneum,
                        test_filenames_normal, test_filenames_pneum,
                        img_size=(64,64)):
    """Reads in training, test and val filenames, uses read_img() to change to 
    change img to numpy array, then produces X and y data splits, in addition to
    creating proper binary labels for modeling.

    ylabels are encoded as 0=normal, 1=pneumonia
    Returns:  X_train, X_test, y_train, y_test"""
    
    display('[i] LOADING IMAGES')


 # create empty lists to contain the image filenames and another to contain
 # the classification ylabel for each image.
    train_img = []
    train_label = []

# reads in and classifies training normal label
    for img_path in tqdm(training_filenames_normal):
        train_img.append(read_img(img_path,target_size=img_size))
        train_label.append(0)

# reads in and classifies training penumonia label
    for img_path in tqdm(training_filenames_pneum):
        train_img.append(read_img(img_path,target_size=img_size))
        train_label.append(1)


 # create empty lists to contain the image filenames and another to contain
 # the classification ylabel for each image.
    test_img = []
    test_label = []

# reads in and classifies test normal label
    for img_path in tqdm(test_filenames_normal):
        test_img.append(read_img(img_path,target_size=img_size))
        test_label.append(0)

# reads in and classifies test penumonia label
    for img_path in tqdm(test_filenames_pneum):
        test_img.append(read_img(img_path,target_size=img_size))
        test_label.append(1)



# create your X_train and y_train variables for use in modeling
    X_train = np.array(train_img, np.float32)
    y_train = np.array(train_label)

    X_test = np.array(test_img, np.float32)
    y_test = np.array(test_label)


# Prints the length of each split for use in batching and knowledge of data. 
    print('\n[i] Length of Splits:')
    print(f"X_train={len(X_train)}, X_test={len(X_test)}")

    # return X_train, X_test, y_train, y_test
    return X_train, X_test, y_train, y_test


def train_test_datagens(X_train, X_test, y_train, y_test,
                            BATCH_SIZE = 32):
                              
    """Takes in your training and test data and creates ImageDataGenerators 
    for train,test,val data. This will normalize your image array data.
    Returns: training_set,test_set,val_set"""

    ## Create training and test data image generators. 
    # This will normalize the image pixel data.
    from keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale = 1./255,
                                       shear_range = 0.2, 
                                        zoom_range = 0.2)

    test_datagen = ImageDataGenerator(rescale = 1./255)

    training_set = train_datagen.flow(X_train,y=y_train,batch_size=BATCH_SIZE)
    test_set = test_datagen.flow(X_test,y=y_test,batch_size=BATCH_SIZE)
    
    return training_set, test_set


# final eval of model showing report, confusion matrix and acc/loss graphs
def evaluate_model(y_test, y_pred, model_history):
    """Takes in your target, target predictions, model history.
        Returns a metric report, confusion matrix and plots for 
        accuracy and loss."""
    
    ## Classification Report / Scores 
    print(metrics.classification_report(y_test,y_pred))

    # confusion matrix
    fig, ax = plt.subplots(figsize=(12,6))
    cm = metrics.confusion_matrix(y_test, y_pred, normalize='true')

    # Add title and axis labels
    plt.imshow(cm, interpolation='nearest', cmap='Blues' )
    plt.title('Confusion Matrix. 0=Norm, 1=Pneum')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Add appropriate axis scales
    class_names = set(y_test) # Get class labels to add to matrix
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    # Add labels to each cell
    thresh = cm.max() / 2. # Used for text coloring below
    # Here we iterate through the confusion matrix and append labels to our visualization 
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, round(cm[i, j], 2),
                    horizontalalignment='center',
                    color='white' if cm[i, j] > thresh else 'black')

    # classes = list(range(len(cm)))  
    # plt.xticks(classes, classes)
    # plt.yticks(classes, classes)

    plt.colorbar()
    plt.show()


    # loss and acc plots
    fig, ax = plt.subplots(figsize=(12,6),ncols=2)

    train_acc = model_history.history['acc']
    test_acc = model_history.history['val_acc']
    train_loss = model_history.history['loss']
    test_loss = model_history.history['val_loss']

    epochs = range(len(train_acc))
    ax[0].plot(epochs, train_acc, 'g', label='Training acc')
    ax[0].plot(epochs, test_acc, 'b', label='Test acc')
    ax[0].legend()

    ax[1].plot(epochs, train_loss, 'g', label='Training loss')
    ax[1].plot(epochs, test_loss, 'b', label='Test loss')
    ax[1].legend()

    ax[0].set(title='Training and Testing accuracy')
    ax[1].set(title='Training and Testing loss')
    
    plt.tight_layout()
    plt.show()

```


```python
## USING FUNCTIONS TO LOAD IN IMAGES 
X_train,X_test,y_train,y_test = load_train_test_images(train_files_normal, 
                                                           train_files_pneum, 
                                                           test_files_normal, 
                                                           test_files_pneum,
                                                          img_size=(64,64))

training_set, test_set = train_test_datagens(X_train, X_test, y_train, y_test,
                                                        BATCH_SIZE=100)

print('Training set shape ', training_set[0][0].shape)

```


    '[i] LOADING IMAGES'


    100%|██████████| 1341/1341 [00:23<00:00, 56.80it/s]
    100%|██████████| 3418/3418 [00:18<00:00, 187.43it/s]
    100%|██████████| 234/234 [00:02<00:00, 82.49it/s]
    100%|██████████| 390/390 [00:01<00:00, 224.94it/s]


    
    [i] Length of Splits:
    X_train=4759, X_test=624
    Training set shape  (100, 64, 64, 3)


# Step 2: EDA and Visualization

## View an image

Test to view an image using filename to make sure they loaded into notebook correctly. 


```python
# view a regular image file
from keras.preprocessing import image
import matplotlib.pyplot as plt
%matplotlib inline

file_normal = train_files_normal[0] # uses first normal filename from glob list
file_pneum = train_files_pneum[0] # uses first pneumonia filename from glob list

img_n = image.load_img(file_normal)
img_p = image.load_img(file_pneum)

# need subplot titles and each plot still
fig, ax = plt.subplots(figsize=(12,6),ncols=2)


ax[0].imshow(img_n)
ax[0].set(title='Normal')

ax[1].imshow(img_p)
ax[1].set(title='Pneumonia')
plt.show()

```


![png](output_27_0.png)


Test to view image as a tensor array. 


```python
feature_img = train_files_normal[1] # uses first normal filename from glob list

f_img = image.load_img(feature_img, target_size=(64,64))

# view as a tensor image
img_tensor = image.img_to_array(f_img)
img_tensor = np.expand_dims(img_tensor, axis=0)

# Follow the Original Model Preprocessing
img_tensor /= 255.

# Check tensor shape
print(img_tensor.shape)

# Preview an image
plt.imshow(img_tensor[0])
plt.show()
```

    (1, 64, 64, 3)



![png](output_29_1.png)


# Step 4: Modeling

## BUILD BASELINE CNN


```python
np.random.seed(111)
import keras
from keras import layers
from keras import models
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
import datetime


# timer for model
original_start = datetime.datetime.now()
start = datetime.datetime.now()

# Initialising the CNN
model = Sequential()

# Step 1 - Convolution
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(64 ,64,  3)))
# Step 2 - Pooling
model.add(layers.MaxPooling2D((2, 2)))

# Adding a second convolutional layer
model.add(layers.Conv2D(64, (3, 3), activation='relu',
                        input_shape=(64 ,64,  3)))
# Pooling
model.add(layers.MaxPooling2D((2, 2)))

# Adding a third convolutional layer
model.add(layers.Conv2D(128, (3, 3), activation='relu',
                        input_shape=(64 ,64,  3)))
# Pooling
model.add(layers.MaxPooling2D((2, 2)))

# Step 3 - Flattening
model.add(layers.Flatten())

# Step 4 - Full connection
model.add(layers.Dense(units = 512, activation = 'relu'))

model.add(layers.Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
model.compile(optimizer = 'sgd', 
                   loss = 'binary_crossentropy',
                   metrics = ['acc'])
print()
display(model.summary())

# Fitting the CNN to the images using fit_generator
history = model.fit_generator(training_set,
                             steps_per_epoch = 500,
                             epochs = 4,
                             validation_data = test_set,
                             validation_steps =100, verbose=1, workers=-1)
# end timer
end = datetime.datetime.now()
elapsed = end - start
print('Training took a total of {}'.format(elapsed))                              
```

    
    Model: "sequential_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_4 (Conv2D)            (None, 62, 62, 32)        896       
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, 31, 31, 32)        0         
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 29, 29, 64)        18496     
    _________________________________________________________________
    max_pooling2d_5 (MaxPooling2 (None, 14, 14, 64)        0         
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, 12, 12, 128)       73856     
    _________________________________________________________________
    max_pooling2d_6 (MaxPooling2 (None, 6, 6, 128)         0         
    _________________________________________________________________
    flatten_2 (Flatten)          (None, 4608)              0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 512)               2359808   
    _________________________________________________________________
    dense_4 (Dense)              (None, 1)                 513       
    =================================================================
    Total params: 2,453,569
    Trainable params: 2,453,569
    Non-trainable params: 0
    _________________________________________________________________



    None


    Epoch 1/4
    500/500 [==============================] - 58s 117ms/step - loss: 0.5215 - acc: 0.7509 - val_loss: 0.7036 - val_acc: 0.6467
    Epoch 2/4
    500/500 [==============================] - 51s 102ms/step - loss: 0.3070 - acc: 0.8701 - val_loss: 0.4160 - val_acc: 0.8556
    Epoch 3/4
    500/500 [==============================] - 51s 101ms/step - loss: 0.2403 - acc: 0.9009 - val_loss: 0.4806 - val_acc: 0.8467
    Epoch 4/4
    500/500 [==============================] - 51s 102ms/step - loss: 0.2078 - acc: 0.9162 - val_loss: 0.3440 - val_acc: 0.8605
    Training took a total of 0:03:31.504518



```python
# training loss and accuracy
results_train = model.evaluate(X_train, y_train)
results_train
```

    4759/4759 [==============================] - 1s 132us/step





    [13.932544560297888, 0.9495692253112793]




```python
# test loss and accuracy
results_test = model.evaluate(X_test, y_test)
results_test
```

    624/624 [==============================] - 0s 167us/step





    [59.18795563624455, 0.8589743375778198]




```python
# # Your code here; save the model for future reference 
# model.save('baseline_model.h5')

```

### Interpreting Results


```python
y_preds = model.predict_classes(X_test).flatten()

evaluate_model(y_test,y_preds,history)
```

                  precision    recall  f1-score   support
    
               0       0.93      0.68      0.78       234
               1       0.83      0.97      0.90       390
    
        accuracy                           0.86       624
       macro avg       0.88      0.82      0.84       624
    weighted avg       0.87      0.86      0.85       624
    



![png](output_37_1.png)



![png](output_37_2.png)


Model.evaluate shows my test accuracy at 86% and loss at 59%. My metrics classification report from sklearn also shows my accuracy at 86% with an excellent recall of 97% of pneumonia diagnosis. 

# Visualize Feature Map Layers

This will help show you what's happening with the images after each layer of the model network and how the patterns are developed. 


```python
from keras import models
import math 

# Extract model layer outputs
layer_outputs = [layer.output for layer in model.layers[:6]]

# Rather then a model with a single output, we are going to make a model to display the feature maps
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
```


```python
model.summary()
```

    Model: "sequential_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_4 (Conv2D)            (None, 62, 62, 32)        896       
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, 31, 31, 32)        0         
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 29, 29, 64)        18496     
    _________________________________________________________________
    max_pooling2d_5 (MaxPooling2 (None, 14, 14, 64)        0         
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, 12, 12, 128)       73856     
    _________________________________________________________________
    max_pooling2d_6 (MaxPooling2 (None, 6, 6, 128)         0         
    _________________________________________________________________
    flatten_2 (Flatten)          (None, 4608)              0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 512)               2359808   
    _________________________________________________________________
    dense_4 (Dense)              (None, 1)                 513       
    =================================================================
    Total params: 2,453,569
    Trainable params: 2,453,569
    Non-trainable params: 0
    _________________________________________________________________



```python
# Returns an array for each activation layer
activations = activation_model.predict(img_tensor)

first_layer_activation = activations[0]
print(first_layer_activation.shape)

# We slice the third channel and preview the results
plt.matshow(first_layer_activation[0, :, :, 3], cmap='viridis')
plt.show()
```

    (1, 62, 62, 32)



![png](output_43_1.png)


REPEAT FOR ALL LAYERS


```python
fig, axes = plt.subplots(2,4, figsize=(12,8))

layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

for i in range(8):
    row = i//4
    column = i%4
    ax = axes[row, column]
    cur_layer = activations[i]
    ax.matshow(cur_layer[0, :, :, 29], cmap='viridis')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_title(layer_names[i])
```


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-40-c4b608dc2808> in <module>()
          9     column = i%4
         10     ax = axes[row, column]
    ---> 11     cur_layer = activations[i]
         12     ax.matshow(cur_layer[0, :, :, 29], cmap='viridis')
         13     ax.xaxis.set_ticks_position('bottom')


    IndexError: list index out of range



![png](output_45_1.png)


# Step 5: Model Hyperparams Updates

## Second model with RMSprop


```python
np.random.seed(111)

from keras import layers
from keras import models
from keras import optimizers
import datetime


# timer for model
start = datetime.datetime.now()

# Initialising the CNN
rms_model = Sequential()

# Step 1 - Convolution
rms_model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(64 ,64,  3)))
# Step 2 - Pooling
rms_model.add(layers.MaxPooling2D((2, 2)))

# Adding a second convolutional layer
rms_model.add(layers.Conv2D(64, (3, 3), activation='relu',
                        input_shape=(64 ,64,  3)))
# Pooling
rms_model.add(layers.MaxPooling2D((2, 2)))

# Adding a third convolutional layer
rms_model.add(layers.Conv2D(128, (3, 3), activation='relu',
                        input_shape=(64 ,64,  3)))
# Pooling
rms_model.add(layers.MaxPooling2D((2, 2)))

# Step 3 - Flattening
rms_model.add(layers.Flatten())

# Step 4 - Full connection
rms_model.add(layers.Dense(units = 512, activation = 'relu'))

rms_model.add(layers.Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
rms_model.compile(optimizer = 'rmsprop', 
                   loss = 'binary_crossentropy',
                   metrics = ['acc'])
print()
display(rms_model.summary())

# Fitting the CNN to the images using fit_generator
rms_history = rms_model.fit_generator(training_set,
                             steps_per_epoch = 500,
                             epochs = 6,
                             validation_data = test_set,
                             validation_steps =100, verbose=1, workers=-1)
# end timer
end = datetime.datetime.now()
elapsed = end - start
print('Training took a total of {}'.format(elapsed))                              
```

    
    Model: "sequential_3"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_7 (Conv2D)            (None, 62, 62, 32)        896       
    _________________________________________________________________
    max_pooling2d_7 (MaxPooling2 (None, 31, 31, 32)        0         
    _________________________________________________________________
    conv2d_8 (Conv2D)            (None, 29, 29, 64)        18496     
    _________________________________________________________________
    max_pooling2d_8 (MaxPooling2 (None, 14, 14, 64)        0         
    _________________________________________________________________
    conv2d_9 (Conv2D)            (None, 12, 12, 128)       73856     
    _________________________________________________________________
    max_pooling2d_9 (MaxPooling2 (None, 6, 6, 128)         0         
    _________________________________________________________________
    flatten_3 (Flatten)          (None, 4608)              0         
    _________________________________________________________________
    dense_5 (Dense)              (None, 512)               2359808   
    _________________________________________________________________
    dense_6 (Dense)              (None, 1)                 513       
    =================================================================
    Total params: 2,453,569
    Trainable params: 2,453,569
    Non-trainable params: 0
    _________________________________________________________________



    None


    Epoch 1/6
    500/500 [==============================] - 52s 104ms/step - loss: 0.2852 - acc: 0.8812 - val_loss: 0.3833 - val_acc: 0.9035
    Epoch 2/6
    500/500 [==============================] - 52s 103ms/step - loss: 0.1239 - acc: 0.9531 - val_loss: 0.2609 - val_acc: 0.9188
    Epoch 3/6
    500/500 [==============================] - 51s 103ms/step - loss: 0.0905 - acc: 0.9668 - val_loss: 0.6276 - val_acc: 0.9005
    Epoch 4/6
    500/500 [==============================] - 51s 102ms/step - loss: 0.0698 - acc: 0.9747 - val_loss: 0.1512 - val_acc: 0.9279
    Epoch 5/6
    500/500 [==============================] - 51s 103ms/step - loss: 0.0568 - acc: 0.9804 - val_loss: 0.3148 - val_acc: 0.9150
    Epoch 6/6
    500/500 [==============================] - 51s 103ms/step - loss: 0.0442 - acc: 0.9839 - val_loss: 0.0586 - val_acc: 0.9183
    Training took a total of 0:05:08.859220



```python
# training loss and accuracy

results_train = rms_model.evaluate(X_train, y_train)
results_train
```

    4759/4759 [==============================] - 0s 104us/step





    [23.047839572937193, 0.9693213105201721]




```python
# test loss and accuracy

results_test = rms_model.evaluate(X_test, y_test)
results_test
```

    624/624 [==============================] - 0s 104us/step





    [261.0955595114292, 0.8814102411270142]



HOW TO SAVE h5 model


```python
# # Your code here; save the model for future reference 
# model.save('chest_xray_downsampled_data.h5')

```

HOW TO LOAD A PREVIOUS SAVED h5 model


```python
# from keras.models import load_model

# model = load_model('chest_xray_all_data.h5')
# # As a reminder 
# model.summary()  
```

### Interpreting Results


```python
y_preds = rms_model.predict_classes(X_test).flatten()

evaluate_model(y_test,y_preds,rms_history)

```

                  precision    recall  f1-score   support
    
               0       0.97      0.71      0.82       234
               1       0.85      0.99      0.91       390
    
        accuracy                           0.88       624
       macro avg       0.91      0.85      0.86       624
    weighted avg       0.89      0.88      0.88       624
    



![png](output_56_1.png)



![png](output_56_2.png)


Model.evaluate shows my test accuracy at 88% and loss at 261%. My metrics classification report from sklearn also shows my accuracy at 88% with an excellent recall of 99% of pneumonia diagnosis. 

## Third model with ADAM


```python
np.random.seed(111)

from keras import layers
from keras import models
from keras import optimizers
import datetime


# timer for model
start = datetime.datetime.now()

# Initialising the CNN
adam_model = Sequential()

# Step 1 - Convolution
adam_model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(64 ,64,  3)))
# Step 2 - Pooling
adam_model.add(layers.MaxPooling2D((2, 2)))

# Adding a second convolutional layer
adam_model.add(layers.Conv2D(64, (3, 3), activation='relu',
                        input_shape=(64 ,64,  3)))
# Pooling
adam_model.add(layers.MaxPooling2D((2, 2)))

# Adding a third convolutional layer
adam_model.add(layers.Conv2D(128, (3, 3), activation='relu',
                        input_shape=(64 ,64,  3)))
# Pooling
adam_model.add(layers.MaxPooling2D((2, 2)))

# Step 3 - Flattening
adam_model.add(layers.Flatten())

# Step 4 - Full connection
adam_model.add(layers.Dense(units = 512, activation = 'relu'))

adam_model.add(layers.Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
adam_model.compile(optimizer = 'adam', 
                   loss = 'binary_crossentropy',
                   metrics = ['acc'])
print()
display(adam_model.summary())

# Fitting the CNN to the images using fit_generator
adam_history = adam_model.fit_generator(training_set,
                             steps_per_epoch = 500,
                             epochs = 6,
                             validation_data = test_set,
                             validation_steps =100, verbose=1, workers=-1)
# end timer
end = datetime.datetime.now()
elapsed = end - start
print('Training took a total of {}'.format(elapsed))                              
```

    
    Model: "sequential_4"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_10 (Conv2D)           (None, 62, 62, 32)        896       
    _________________________________________________________________
    max_pooling2d_10 (MaxPooling (None, 31, 31, 32)        0         
    _________________________________________________________________
    conv2d_11 (Conv2D)           (None, 29, 29, 64)        18496     
    _________________________________________________________________
    max_pooling2d_11 (MaxPooling (None, 14, 14, 64)        0         
    _________________________________________________________________
    conv2d_12 (Conv2D)           (None, 12, 12, 128)       73856     
    _________________________________________________________________
    max_pooling2d_12 (MaxPooling (None, 6, 6, 128)         0         
    _________________________________________________________________
    flatten_4 (Flatten)          (None, 4608)              0         
    _________________________________________________________________
    dense_7 (Dense)              (None, 512)               2359808   
    _________________________________________________________________
    dense_8 (Dense)              (None, 1)                 513       
    =================================================================
    Total params: 2,453,569
    Trainable params: 2,453,569
    Non-trainable params: 0
    _________________________________________________________________



    None


    Epoch 1/6
    500/500 [==============================] - 52s 104ms/step - loss: 0.1979 - acc: 0.9182 - val_loss: 0.1766 - val_acc: 0.9327
    Epoch 2/6
    500/500 [==============================] - 52s 103ms/step - loss: 0.1075 - acc: 0.9604 - val_loss: 0.2425 - val_acc: 0.9276
    Epoch 3/6
    500/500 [==============================] - 52s 103ms/step - loss: 0.0853 - acc: 0.9676 - val_loss: 0.3909 - val_acc: 0.8835
    Epoch 4/6
    500/500 [==============================] - 52s 103ms/step - loss: 0.0683 - acc: 0.9745 - val_loss: 0.2292 - val_acc: 0.9251
    Epoch 5/6
    500/500 [==============================] - 52s 103ms/step - loss: 0.0592 - acc: 0.9786 - val_loss: 0.2745 - val_acc: 0.9198
    Epoch 6/6
    500/500 [==============================] - 51s 103ms/step - loss: 0.0455 - acc: 0.9834 - val_loss: 0.1770 - val_acc: 0.9358
    Training took a total of 0:05:10.536690



```python
# training loss and accuracy
results_train = adam_model.evaluate(X_train, y_train)
results_train
```

    4759/4759 [==============================] - 0s 100us/step





    [34.364630120890745, 0.9436856508255005]




```python
# test loss and accuracy
results_test = adam_model.evaluate(X_test, y_test)
results_test
```

    624/624 [==============================] - 0s 101us/step





    [76.02700419931313, 0.9038461446762085]



### Interpreting Results


```python
y_preds = adam_model.predict_classes(X_test).flatten()

evaluate_model(y_test,y_preds,adam_history)

```

                  precision    recall  f1-score   support
    
               0       0.89      0.85      0.87       234
               1       0.91      0.94      0.92       390
    
        accuracy                           0.90       624
       macro avg       0.90      0.89      0.90       624
    weighted avg       0.90      0.90      0.90       624
    



![png](output_63_1.png)



![png](output_63_2.png)


Model.evaluate shows my test accuracy at 90% and loss at 76%. My metrics classification report from sklearn also shows my accuracy at 90% with a good recall of 94% of pneumonia diagnosis. 

Best model so far for recall was adam with 90% accuracy. I'm going to move forward with the adam optimizer. 

- baseline sgd
test accuracy at 86% and loss at 59%. sklearn also shows my accuracy at 86% with an excellent recall of 97% of pneumonia diagnosis.  
- rmsprop
test accuracy at 88% and loss at 261%. sklearn also shows my accuracy at 88% with an excellent recall of 99% of pneumonia diagnosis.  
- adam
test accuracy at 90% and loss at 76%. sklearn also shows my accuracy at 90% with a good recall of 94% of pneumonia diagnosis. 

## Fourth model with ADAM. Add Dropout. 


```python
np.random.seed(111)

from keras import layers
from keras import models
from keras import optimizers
import datetime


# timer for model
start = datetime.datetime.now()

# Initialising the CNN
adam_drop_model = Sequential()

# Step 1 - Convolution
adam_drop_model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(64 ,64,  3)))
# Step 2 - Pooling
adam_drop_model.add(layers.MaxPooling2D((2, 2)))

# Adding a second convolutional layer
adam_drop_model.add(layers.Conv2D(64, (3, 3), activation='relu',
                        input_shape=(64 ,64,  3)))
# Pooling
adam_drop_model.add(layers.MaxPooling2D((2, 2)))

# Adding a third convolutional layer
adam_drop_model.add(layers.Conv2D(128, (3, 3), activation='relu',
                        input_shape=(64 ,64,  3)))
# Pooling
adam_drop_model.add(layers.MaxPooling2D((2, 2)))

# Step 3 - Flattening
adam_drop_model.add(layers.Flatten())

# Step 4 - Full connection
adam_drop_model.add(layers.Dense(units = 128, activation = 'relu'))

# Dropout applied to the full connection layer
adam_drop_model.add(layers.Dropout(0.3))

# Step 5 - Full connection
adam_drop_model.add(layers.Dense(units = 512, activation = 'relu'))

# output
adam_drop_model.add(layers.Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
adam_drop_model.compile(optimizer = 'adam', 
                   loss = 'binary_crossentropy',
                   metrics = ['acc'])
print()
display(adam_drop_model.summary())

# Fitting the CNN to the images using fit_generator
adam_drop_history = adam_drop_model.fit_generator(training_set,
                             steps_per_epoch = 500,
                             epochs = 6,
                             validation_data = test_set,
                             validation_steps =100, verbose=1, workers=-1)
# end timer
end = datetime.datetime.now()
elapsed = end - start
print('Training took a total of {}'.format(elapsed))                              
```

    
    Model: "sequential_6"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_16 (Conv2D)           (None, 62, 62, 32)        896       
    _________________________________________________________________
    max_pooling2d_16 (MaxPooling (None, 31, 31, 32)        0         
    _________________________________________________________________
    conv2d_17 (Conv2D)           (None, 29, 29, 64)        18496     
    _________________________________________________________________
    max_pooling2d_17 (MaxPooling (None, 14, 14, 64)        0         
    _________________________________________________________________
    conv2d_18 (Conv2D)           (None, 12, 12, 128)       73856     
    _________________________________________________________________
    max_pooling2d_18 (MaxPooling (None, 6, 6, 128)         0         
    _________________________________________________________________
    flatten_6 (Flatten)          (None, 4608)              0         
    _________________________________________________________________
    dense_12 (Dense)             (None, 128)               589952    
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 128)               0         
    _________________________________________________________________
    dense_13 (Dense)             (None, 512)               66048     
    _________________________________________________________________
    dense_14 (Dense)             (None, 1)                 513       
    =================================================================
    Total params: 749,761
    Trainable params: 749,761
    Non-trainable params: 0
    _________________________________________________________________



    None


    Epoch 1/6
    500/500 [==============================] - 52s 103ms/step - loss: 0.2168 - acc: 0.9110 - val_loss: 0.3420 - val_acc: 0.9193
    Epoch 2/6
    500/500 [==============================] - 51s 103ms/step - loss: 0.1113 - acc: 0.9594 - val_loss: 0.3227 - val_acc: 0.9237
    Epoch 3/6
    500/500 [==============================] - 51s 103ms/step - loss: 0.0905 - acc: 0.9654 - val_loss: 0.1518 - val_acc: 0.9342
    Epoch 4/6
    500/500 [==============================] - 51s 103ms/step - loss: 0.0743 - acc: 0.9722 - val_loss: 0.3058 - val_acc: 0.9043
    Epoch 5/6
    500/500 [==============================] - 52s 103ms/step - loss: 0.0631 - acc: 0.9767 - val_loss: 0.3568 - val_acc: 0.9153
    Epoch 6/6
    500/500 [==============================] - 51s 103ms/step - loss: 0.0545 - acc: 0.9800 - val_loss: 0.5074 - val_acc: 0.8813
    Training took a total of 0:05:09.359047



```python
# training loss and accuracy
results_train = adam_drop_model.evaluate(X_train, y_train)
results_train
```

    4759/4759 [==============================] - 1s 107us/step





    [15.850227282823347, 0.9598655104637146]




```python
# test loss and accuracy
results_test = adam_drop_model.evaluate(X_test, y_test)
results_test
```

    624/624 [==============================] - 0s 108us/step





    [188.30264454621536, 0.8028846383094788]




### Interpreting Results


```python
y_preds = adam_drop_model.predict_classes(X_test).flatten()

evaluate_model(y_test,y_preds,adam_drop_history)

```

                  precision    recall  f1-score   support
    
               0       0.97      0.49      0.65       234
               1       0.76      0.99      0.86       390
    
        accuracy                           0.80       624
       macro avg       0.87      0.74      0.76       624
    weighted avg       0.84      0.80      0.78       624
    



![png](output_71_1.png)



![png](output_71_2.png)


My test accuracy is 80%. Loss is 188. Recall is 99%.


```python
print('Adam model')
adam_y_preds = adam_model.predict_classes(X_test).flatten()
print(metrics.classification_report(y_test, adam_y_preds))
'\n'
print('Adam model with dropout')
dropout_y_preds = adam_drop_model.predict_classes(X_test).flatten()
print(metrics.classification_report(y_test, dropout_y_preds))
```

    Adam model
                  precision    recall  f1-score   support
    
               0       0.89      0.85      0.87       234
               1       0.91      0.94      0.92       390
    
        accuracy                           0.90       624
       macro avg       0.90      0.89      0.90       624
    weighted avg       0.90      0.90      0.90       624
    
    Adam model with dropout
                  precision    recall  f1-score   support
    
               0       0.97      0.49      0.65       234
               1       0.76      0.99      0.86       390
    
        accuracy                           0.80       624
       macro avg       0.87      0.74      0.76       624
    weighted avg       0.84      0.80      0.78       624
    


Results: The Adam model with no dropout was best by showing a recall in pneumonia patients of 94% while still having a recall of healthy patients of 85%, compared to the dropout model 49%. With precision, the adam model with no drop was more precise with 91% pneumonia, 89% healthy classes, compared to the drop model with only 76% precision for pneumonia patients. F1 score for adam no drop is at 0.9. That's the best model.   

## Conclusion Summary

Based on this final model, it seems that the process was able to classify 94% of all pneumonia patients in the test set. More tuning for the future to try and get more of the patients properly classified. 

With a high precision of 91% as well, the model is not only capturing a majority of the pneumonia case, but also being precise in diagnosis. Same goes for patients with a precision of 89%. 

CNN models are best for this type of work over dense layered networks. The model works fast, processing thousands of images in less than five minutes due to it's ability to use small image resolution size. It uses such small images while still being able to break down the images using feature maps to search for patterns and accurately obtain results. Adjusting the model to work with higher res images could possibly increase accuracy performance with only a slight time increase.

This will greatly increase processing time and cutting costs in the medical field and will allow labor hours to be redirected towards other pressing matters and care for more patients. Expenses can be allocated, diagnosis accuracy will go up, catching pneumonia earlier. Business reputation will go up. Patients will recognize the better care and work from the company. 


```python

```
