
# coding: utf-8

# In[30]:


import os
os.chdir(r"F:\Kunal\Job_Applications\DeeperSystem\train.rotfaces\train")
os.getcwd()


# In[31]:


import pandas as pd

df=pd.read_csv(r"F:\Kunal\Job_Applications\DeeperSystem\train.rotfaces\train.truth.csv")
df.head()


# In[32]:


train_fn=df['fn'].values
train_fn[:5]


# In[33]:


train_imgs=os.listdir()
train_imgs[0:5]


# We can notice that, file name sequence is not matching in train_imgs and train_fn.
# 
# Therefore we may wrongly assign label for each file in train_imgs if we take label from df as it is

# In[34]:


print(len(train_fn))
print(len(train_imgs))


# In[35]:


import numpy as np

sum(train_fn==train_imgs)


# Note that only 1084 file names are matching with their positions (index) in train_imgs and train_fn
# 

# In[36]:


dct={}

for index,row in df.iterrows():
    dct[row['fn']]=row['label']
    


# In[37]:


label=[dct[f] for f in train_imgs]
len(label)


# Now we have proper label (label) for each file name (train_imgs)

# In[38]:


import cv2
import numpy as np


# In[39]:


# to check the image
im=cv2.imread(train_imgs[101],0) 
im2=cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
cv2.imshow("image",im2)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[40]:


im2.shape


# In[41]:



import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# Reading all image as gray scale image and converting all of them to 3 channel image
# And finally storing all the image vectors in matrix(list/array)
matrix=[]
for img in tqdm(train_imgs[:7000]):
    im=cv2.cvtColor(cv2.imread(img,0),cv2.COLOR_GRAY2BGR)
    matrix.append(np.array(im))
    


# In[42]:


# converting list(matrix) into numpy array
matrix=np.array(matrix)
matrix.shape


# In[43]:


# Playing around with label 

label=label[:7000]
set(label)


# There are only 4 class labels. Lets represent them in numerical form.

# In[44]:


# representing label in numerical form
dct={'rotated_left':0,'rotated_right':1,'upright':2,'upside_down':3}
label=[dct[item] for item in label]
label[:10]


# In[45]:


# converting data type to float and normalizing the values by dividing each value by max value i.e. 255

matrix=matrix.astype('float')
matrix/=255
matrix[0]


# In[46]:


import keras


# In[47]:


from keras.utils import np_utils

# representing label as one hot encoded vector
label=np_utils.to_categorical(label)
label[:5]


# In[48]:


from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import layers
import matplotlib.pyplot as plt


# In[49]:


# checking image
im1=matrix[0]
cv2.imshow("img",im1)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ## Neural Network Architecture

# In[50]:


# Network Architecture
model = Sequential()
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(64,64,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(250, activation='relu'))
model.add(Dense(4, activation='softmax'))


# In[51]:


# Network (model) compilation
model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])


# In[53]:


# Training network (model)
hist = model.fit(matrix, label, 
           batch_size=256, epochs=10, validation_split=0.2 )


# ## Saving Model and Weights

# In[108]:



os.chdir(r"F:\Kunal\Job_Applications\DeeperSystem\train.rotfaces")

# serialising model to json
json_model=model.to_json()
with open('model.json','w') as file:
    file.write(json_model)
# serialising weights to hdf5
model.save_weights("model.h5")


# ## Loading Model and Weights

# In[109]:


# Load model and weights

from keras.models import model_from_json

json_file=open("model.json","r")
json_model=json_file.read()
json_file.close()
model=model_from_json(json_model)
model.load_weights("model.h5")

# after laoding model and weights, compile it and use it for precdiction and evaluation on test data


# ## Testing

# In[62]:


import os
from tqdm import tqdm
import cv2
import numpy as np

# getting test images
test_imgs=os.listdir(r"F:\Kunal\Job_Applications\DeeperSystem\test.rotfaces\test")
len(test_imgs)


# In[63]:


os.chdir(r"F:\Kunal\Job_Applications\DeeperSystem\test.rotfaces\test")
print(os.getcwd())

# reading all test images and converting them into matrix vector
matrix_test=[]
for img in tqdm(test_imgs):
    im=cv2.cvtColor(cv2.imread(img,0),cv2.COLOR_GRAY2BGR)
    matrix_test.append(np.array(im))
matrix_test=np.array(matrix_test)
matrix_test.shape


# In[66]:


"""# checking image
im=cv2.imread(matrix_test[0],0) # to read image as gray image
im=cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
cv2.imshow("image",im)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

im=cv2.imread(matrix_test[0],0) # reading gray scale image
cv2.imshow("image",im)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[67]:


from keras.models import load_model

# predicting the class for each image
predictions=model.predict_classes(matrix_test)
len(predictions)


# In[68]:


predictions[:10]


# In[69]:


# creating a DataFrame with image_fn and predicted_class_value

df=pd.DataFrame(columns=['fn','prediction'])
df['fn']=test_imgs
df['prediction']=list(predictions)
df.head()


# In[70]:


# lets create one more dictionary that can be used to replace prediction values by their original orientation name
dct={0:'rotated_left',1:'rotated_right',2:'upright',3:'upside_down'}
for i in range(len(df)):
    df.iloc[i,1]=dct[df.iloc[i,1]]
df.head()


# In[71]:


# generating "test.preds.csv" file 
df.to_csv(r"F:\Kunal\Job_Applications\DeeperSystem\test.rotfaces\test.preds.csv")


# ## Correcting images by rotating it

# In[72]:


#dct={'rotated_left':0,'rotated_right':1,'upright':2,'upside_down':3}


# Rotating all predicted images to UPRIGHT (0 degree rotation) based on its rotated angle

matrix_corrected=[]
for index,pred in tqdm(enumerate(predictions)):
    if pred==0: # for rotation_left
        rot=cv2.getRotationMatrix2D((32,32),270,1)
        op=cv2.warpAffine(matrix_test[index],rot,(64,64))
        matrix_corrected.append(op)
        
    elif pred==1: # for rotation_right
        rot=cv2.getRotationMatrix2D((32,32),90,1)
        op=cv2.warpAffine(matrix_test[index],rot,(64,64))
        matrix_corrected.append(op)
        
    elif pred==2: # for upright (no need to rotate image)
        matrix_corrected.append(matrix_test[index])
        
    elif pred==3: # for upside_down
        rot=cv2.getRotationMatrix2D((32,32),180,1)
        op=cv2.warpAffine(matrix_test[index],rot,(64,64))
        matrix_corrected.append(op)
        


# In[73]:


# converting list (matrix_corrected) to numpy array
matrix_corrected=np.array(matrix_corrected)
matrix_corrected.shape


# In[104]:


# saving all the corrected images into new folder called "F:\Kunal\Job_Applications\DeeperSystem\test.rotfaces\corrected"


os.chdir(r"F:\Kunal\Job_Applications\DeeperSystem\test.rotfaces\corrected")

# for corrected images, keeping same image names as test images (only changing extension to .png)
for index,img in enumerate(test_imgs):
    fn="{}.png".format(img[:-4])
    cv2.imwrite(fn,matrix_corrected[index])
    
    


# ## Approach

# 1. Very first thing I tried to match the sequence of image file names from train.truth.csv and train 
# 2. read all the images and converted them into vector (matrix) format
# 3. Verified with random images by plotting them
# 4. Converted label into numerical representation after understanding about how many unique labels are present in a dataset
# 5. Designed neural network architecture
# 6. compiled and trained network on train dataset
# 7. saved the model weights
# 8. Read all test images and similar to train images , converted them into vector (matrix) format
# 9. predicted the classes for each test image 
# 10. Rotated all those predicted images which were predicted to be as NON-UPRIGHT so that all of them can be seen UPRIGHT (i.e like non-rotated images)
# 11. Finally stored the vector (matrix) representation of those rotated images

# ## Future Scope (How to improve model performance further)

# 1. I have sampled only 7000 images from train dataset and got decent result as we can see.
# 2. So, training the same network on more images (all from train dataset) would definitely improve the accuracy.
# 3. Also increasing the number of epochs would help network learn more and therefore model acuuracy will improve.
# 4. Adding more convolutional layers may increase the model performance. But we have to be careful that model should not overfit.
# 5. We can play around with kernel size and also number of kernels used , which may help improve model performance to some extent.
# 

# ## Instructions to run code

# 1. I am providing model in json format (model.json) and also weights in hdf5 (model.h5) format
# 2. Run the code under "Loading Model and Weights" section and then you will get the model that you can use for evaluation and prediction on test data
