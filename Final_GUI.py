#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk

def load_model(path):
    model = keras.models.load_model(path)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"])
    return model
path=r"densenet_location.h5"
model=load_model(path)
word_dict = {0:'all',1:'hem'}
from tensorflow.keras.models import Model
import tensorflow as tf
import cv2


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
#     print("guidedGrads shape:", guidedGrads.shape)
#     print("guidedGrads content:", guidedGrads.numpy())
    
		# the convolution and guided gradients have a batch dimension
		# (which we don't need) so let's grab the volume itself and
		# discard the batch
    convOutputs = convOutputs[0]
    guidedGrads = guidedGrads[0]
    # compute the average of the gradient values, and using them
		# as weights, compute the ponderation of the filters with
		# respect to the weights
    #print(guidedGrads.shape)
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

def predict(imagePath, img_label_original, img_label_grad_cam):
    test_image = image.load_img(imagePath, target_size=(224, 224))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    sample = word_dict[np.argmax(result)]

    l2 = tk.Label(my_w, text="The Predicted sample is " + sample, font=my_font2)
    l2.place(x=160, y=405)

    layer_name = 'dropout_1'
    img = test_image.squeeze()
    grad_cam = GradCam(model, np.expand_dims(img, axis=0), layer_name)

    grad_cam_superimposed = superimpose(img, grad_cam, 0.5, emphasize=True)

    # Convert the Grad-CAM NumPy array to a PhotoImage object
    grad_cam_img = Image.fromarray(grad_cam_superimposed)
    grad_cam_img_tk = ImageTk.PhotoImage(grad_cam_img)

    # Update the Grad-CAM Label's image
    img_label_grad_cam.config(image=grad_cam_img_tk)
    img_label_grad_cam.image = grad_cam_img_tk

    # Load and display the original image in the first Label
    original_img = Image.open(imagePath)
    original_img = original_img.resize((224, 224))
    original_img_tk = ImageTk.PhotoImage(original_img)
    img_label_original.config(image=original_img_tk)
    img_label_original.image = original_img_tk

def upload_file(img_label_original, img_label_grad_cam):
    f_types = [('BMP Files', '*.bmp'),('JPG Files','*.jpg')]  # type of files to select
    global filename
    filename = tk.filedialog.askopenfilename(multiple=True, filetypes=f_types)

    for f in filename:
        # Load and display the original image in the first Label
        original_img = Image.open(f)
        original_img = original_img.resize((224, 224))
        original_img_tk = ImageTk.PhotoImage(original_img)
        img_label_original.config(image=original_img_tk)
        img_label_original.image = original_img_tk

        # Predict and display Grad-CAM in the second Label
        #redict(f, img_label_original, img_label_grad_cam)

 # Keep the window open
my_w = tk.Tk()
my_w.geometry("800x600")  # Size of the window
my_w.title('ALL Classification System')
my_font1=('times', 24, 'bold')
my_font2=('times', 18)
l1 = tk.Label(my_w,text='Upload Files & display',font=my_font2)
l1.place(x=190,y=60)
l2 = tk.Label(my_w, text="ALL Classificaion System",font=my_font1)
l2.place(x=125,y=15)

image_label_original = tk.Label(my_w)
image_label_original.place(x=200, y=130)

image_label_grad_cam = tk.Label(my_w)
image_label_grad_cam.place(x=200, y=450)

b1 = tk.Button(my_w, text='Choose File',
   width=20,command = lambda:upload_file(image_label_original, image_label_grad_cam))
b1.place(x=240,y=100)
# f=r"F:\DUMP\logo.png"
# img=Image.open(f) # read the image file
# img=img.resize((224,224)) # new width & height
# img=ImageTk.PhotoImage(img)
# e1 =tk.Label(my_w)
# e1.place(x=190,y=130).
# e1.image = img
# e1['image']=img``````````````````

b2 = tk.Button(my_w, text='Predict',
       width=20,command = lambda:predict(filename[0],image_label_original, image_label_grad_cam))
b2.place(x=240,y=375)



my_w.mainloop() 


# In[ ]:





# In[ ]:




