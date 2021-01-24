# Birb Cam

A Raspberry Pi project designed to detect animals visiting my balcony with a webcam. Store images from their visits and present the temporal patterns of their visitation in a simple flask app. 

## Project Structure

1. Web App
1. Camera App
1. Cloud Function 
1. Notebooks

## Web App

A flask based web app used to visualize the images and mode results through time and manually evaluate and revise the labels applied to each image.

### Visualization

![visualization](imgs/readme/birbcam_visualization.png)

![visualization 2](imgs/readme/birbcam_visualization_2.png)

### Model Evaluation

![evaluate](imgs/readme/birbcam_evaluate.png)


### Label Revision

![revise](imgs/readme/birbcam_revise.png)

## Camera App

The camera-app directory contains the Python script birbcam.py which runs in perpetuity on the Raspberry Pi 4.

![architecture](imgs/readme/birbcam_architecture.png)

The main() function handles setting up the above architecture in which the camera loop runs during the day time and uses OpenCV to retrieve images from the webcam and then detect if changes have occurred in each new frame. Images with sufficient change are pushed into a multiprocessing queue where the image processor function retrieves them and classifies them using the fast.ai model. Presently this is done using Google Cloud Functions given an issue in which using a PyTorch model trained on x86_64 build in an aarch64 build of the library produces drastically different results. Following classification, the images are saved locally

## Cloud Function

This Python cloud function is used to facilitate model inference while the bugs making inference on the aarch64 builds of PyTorch. Images are passed into the function by converting them into base64 encoded strings and passing them to the REST API endpoint. The endpoint returns a JSON object containing the model inference results.

## Notebooks

The notebooks contain a variety of experiments and manual workflows for things like preparing new training data.


## Associated Blog Posts

* [Birb Cam - January 23, 2021](http://everettsprojects.com/2021/01/23/birbcam.html)