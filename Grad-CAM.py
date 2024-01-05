#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
from random import shuffle 
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.utils.data_utils import Sequence


# In[2]:


def label_image_one_hot_encoder(img):
  ## Helper for process_data
  label = img.split('_')[0]
  if label == '0': return 0
  elif label == '1': return 1


def process_data(image_list, DATA_FOLDER, IMG_SIZE):
  ## Helper for manual_pre_process
  ## Creates an array of images, labels, and file path
  ## Shuffles the array before returning
  data_df = []
  for img in tqdm(image_list):
    path = os.path.join(DATA_FOLDER, img)
    label = label_image_one_hot_encoder(img)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    data_df.append([np.array(img), np.array(label), path])
  shuffle(data_df)
  return data_df


def manual_pre_process(dir, IMG_SIZE):
  '''
  Creates an array of images, labels, and files from a directory of image files
  
  Args:
    dir: string, folder name
    IMG_SIZE: int, image height and width

  Returns 
    X: (n x IMG_SIZE x IMG_SIZE) numpy array of images
    y: (n,) numpy array of labels
    files: (n,) numpy array of files

  '''
  image_lst = os.listdir(dir)
  data_df = process_data(image_lst, dir, IMG_SIZE)
  X = np.array([i[0] for i in data_df]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
  y = np.array([i[1] for i in data_df])
  files = np.array([i[2] for i in data_df])
  return X, y, files
print("function completed")



# In[3]:


val_imgs = 'training_data/fold_0/all'
class_info = {0: 'hem', 1: 'hem'}

X, y, files = manual_pre_process(val_imgs, 224)


# In[4]:


import tensorflow as tf
from keras import backend as K

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


# In[5]:


model = tf.keras.models.load_model('densenet_location.h5', custom_objects={"precision_m": precision, "recall_m": recall})


# In[6]:


def decode_prediction(pred):
  # This function thresholds a probability to produce a prediction
  pred = tf.where(pred < 0.5, 0, 1)
  return pred.numpy()


# In[7]:


img = X[59]
label = y[59]
path = files[59]

pred_raw = model.predict(np.expand_dims(img, axis=0))[0][0]
pred = decode_prediction(pred_raw)
pred_label = class_info[pred]

plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title(pred_label + ' ' + str(pred_raw))


# In[8]:


from tensorflow.keras.models import Model



def GradCam(model, img_array, layer_name, eps=1e-8):
    '''
    Creates a grad-cam heatmap given a model and a layer name contained with that model
    

    Args:
      model: tf model
      img_array: (img_width x img_width) numpy array
      layer_name: str


    Returns 
      uint8 numpy array with shape (img_height, img_width)

    '''

    gradModel = Model(
			inputs=[model.inputs],
			outputs=[model.get_layer(layer_name).output,
				model.output])
    
    with tf.GradientTape() as tape:
			# cast the image tensor to a float-32 data type, pass the
			# image through the gradient model, and grab the loss
			# associated with the specific class index
      inputs = tf.cast(img_array, tf.float32)
      (convOutputs, predictions) = gradModel(inputs)
      loss = predictions[:, 0]
		# use automatic differentiation to compute the gradients
    grads = tape.gradient(loss, convOutputs)
    
    # compute the guided gradients
    castConvOutputs = tf.cast(convOutputs > 0, "float32")
    castGrads = tf.cast(grads > 0, "float32")
    guidedGrads = castConvOutputs * castGrads * grads
    
     # Debug: Print the shape and content of guidedGrads
    print("guidedGrads shape:", guidedGrads.shape)
    print("guidedGrads content:", guidedGrads.numpy())
    
		# the convolution and guided gradients have a batch dimension
		# (which we don't need) so let's grab the volume itself and
		# discard the batch
    convOutputs = convOutputs[0]
    guidedGrads = guidedGrads[0]
    # compute the average of the gradient values, and using them
		# as weights, compute the ponderation of the filters with
		# respect to the weights
    print(guidedGrads.shape)
    weights = tf.reduce_mean(guidedGrads,axis=0)
   # weights = tf.reduce_mean(guidedGrads, axis=0)
    weights = tf.expand_dims(weights, axis=0)  # Add a dimension to match the shape of convOutputs
    #cam = tf.reduce_sum(tf.multiply(guidedGrads, convOutputs), axis=-1)
    cam = tf.expand_dims(tf.reduce_sum(tf.multiply(guidedGrads, convOutputs), axis=-1), axis=-1)


    #cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
  
    # grab the spatial dimensions of the input image and resize
		# the output class activation map to match the input image
		# dimensions
    (w, h) = (img_array.shape[2], img_array.shape[1])
   
    """
     print(cam.shape)
    print(cam.numpy())
    print("cam shape:", cam.shape)
    print("cam data type:", cam.dtype)
    print("w, h:", w, h) 
    print("guidedGrads shape:", guidedGrads.shape)
    print("guidedGrads content:", guidedGrads.numpy())
    print("convOutputs shape:", convOutputs.shape)
    print("convOutputs content:", convOutputs.numpy())
    """
    heatmap = cv2.resize(cam.numpy(), (w, h))
		# normalize the heatmap such that all values lie in the range
		# [0, 1], scale the resulting values to the range [0, 255],
		# and then convert to an unsigned 8-bit integer
    numer = heatmap - np.min(heatmap)
    denom = (heatmap.max() - heatmap.min()) + eps
    heatmap = numer / denom
    # Normalize the heatmap values so that they lie in the range [0, 1]
    heatmap_normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    # Apply the rainbow colormap to the heatmap
    #heatmap_colorized = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_RAINBOW)
    # ...

     # Convert the heatmap_normalized tensor to type CV_8UC1
    heatmap_normalized = heatmap_normalized.astype(np.uint8)

    # Apply the rainbow colormap to the heatmap
    heatmap_colorized = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)

    # heatmap = (heatmap * 255).astype("uint8")
		# return the resulting heatmap to the calling function
    return heatmap


def sigmoid(x, a, b, c):
    return c / (1 + np.exp(-a * (x-b)))

def superimpose(img_bgr, cam, thresh, emphasize=False):
    
    '''
    Superimposes a grad-cam heatmap onto an image for model interpretation and visualization.
    

    Args:
      image: (img_width x img_height x 3) numpy array
      grad-cam heatmap: (img_width x img_width) numpy array
      threshold: float
      emphasize: boolean

    Returns 
      uint8 numpy array with shape (img_height, img_width, 3)

    '''
    heatmap = cv2.resize(cam, (img_bgr.shape[1], img_bgr.shape[0]))
    if emphasize:
        heatmap = sigmoid(heatmap, 50, thresh, 1)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    hif = .8
    superimposed_img = heatmap * hif + img_bgr
    superimposed_img = np.minimum(superimposed_img, 255.0).astype(np.uint8)  # scale 0 to 255  
    superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    
    return superimposed_img_rgb


# In[9]:


## Grad-CAM heatmap for the last convolutional layer in the model, Conv_1
model.summary()
#mobilenet_layers = model.get_layer('mobilenetv2_1.00_224').layers
#for layer in mobilenet_layers:
#   print(layer.name)
layer_name = 'dropout_1'  # Example layer name from MobileNetV2
grad_cam = GradCam(model, np.expand_dims(img, axis=0), layer_name)

grad_cam_superimposed = superimpose(img, grad_cam, 0.5, emphasize=True)


plt.figure(figsize=(12, 5))
ax = plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Original Image')

plt.figure(figsize=(12, 5))
ax = plt.subplot(1, 2, 1)
plt.imshow(grad_cam)
plt.axis('off')
plt.title('Original Image')



ax = plt.subplot(1, 2, 2)
plt.imshow(grad_cam_superimposed)
plt.axis('off')
plt.title('dense_5 Grad-CAM heat-map')
plt.tight_layout()




# In[172]:





# In[11]:





# In[ ]:




