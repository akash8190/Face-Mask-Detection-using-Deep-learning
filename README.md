# Face-Mask-Detection-using-Deep-learning

![title](https://assets.losspreventionmedia.com/uploads/2020/07/Mask-Detection-1280x720-1.jpg)

## Project explanation

In this project, we’ll discuss our two-phase COVID-19 face mask detector, detailing how our computer vision/deep learning pipeline will be implemented.We’ll use this Python script to train a face mask detector and review the results.

Given the trained COVID-19 face mask detector, we’ll proceed to implement two more additional Python scripts used to:

1)Detect COVID-19 face masks in images.
2)Detect face masks in real-time video streams.

## Two-phase COVID-19 face mask detector

![title](https://www.pyimagesearch.com/wp-content/uploads/2020/04/face_mask_detection_phases.png)


In order to train a custom face mask detector, we need to break our project into two distinct phases, each with its own respective sub-steps (as shown by Figure 1 above):

Training: Here we’ll focus on loading our face mask detection dataset from disk, training a model (using Keras/TensorFlow) on this dataset, and then serializing the face mask detector to disk
Deployment: Once the face mask detector is trained, we can then move on to loading the mask detector, performing face detection, and then classifying each face as with_mask or without_mask

##  COVID-19 face mask detection dataset

![title](https://www.pyimagesearch.com/wp-content/uploads/2020/04/face_mask_detection_dataset.jpg)

This dataset consists of 1,376 images belonging to two classes:

with_mask: 690 images
without_mask: 686 images
 
# Our goal is to train a custom deep learning model to detect whether a person is or is not wearing a mask. 

## Implementing our COVID-19 face mask detector training script with Keras and TensorFlow

Now that we’ve reviewed our face mask dataset, Now we can use Keras and TensorFlow to train a classifier to automatically detect whether a person is wearing a mask or not.

To accomplish this task, we’ll be fine-tuning the MobileNet V2 architecture, a highly efficient architecture that can be applied to embedded devices with limited computational capacity (ex., Raspberry Pi, Google Coral, NVIDIA Jetson Nano, etc.)
Deploying our face mask detector to embedded devices could reduce the cost of manufacturing such face mask detection systems, hence why we choose to use this architecture.









