
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/dust.jpg "dust road"
[image2]: ./examples/center.jpg "center"
[image3]: ./examples/left_1.jpg "Recovery Image"
[image4]: ./examples/left_2.jpg "Recovery Image"
[image5]: ./examples/left_3.jpg "Recovery Image"
[image6]: ./examples/center.jpg "Normal Image"
[image7]: ./examples/center_flip.jpg "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter and 3x3 filter sizes and depths between 32 and 128 (model.py lines 44-48) 

The model includes RELU layers to introduce nonlinearity (code line 44-48), and the data is normalized in the model using a Keras lambda layer (code line 43) and corpping layer(code line 42). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 50).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road on both straight and curve road, and recovering from dust road.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to use deep neural network that has image pre-processing layers, convolution layers, dropout, flatten and fully connected layers with number of output class 1.

My first step was to use a convolution neural network model similar to the NVIDIA modal I thought this model might be appropriate because it's a widely used model, it has 5 convolution layer to learn about the image and 4 fully connected layer to learn the measurement.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that 1 dropout layers are added 

Then I increase epoch to 5. Since my data is not big, one dropout(or no dropout layer) and a small epoch is enough. More dropout requires larger epoch, it takes time and the outcome is not improved much. About 7000 images is enough for a simple trail like the left trail. I didn't train the right trail, to make the car drive one the right, it make sence to gather more data, add more dropout layers and use larger epoch.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track at sharp turns, the connection point of bridge and road. The car also leans left on long straight road. To improve the driving behavior in these cases, I collect more recovery data at failure points.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 42-54) consisted of a convolution neural network with the following layers and layer sizes: 
1. 5 convolution layers from 24 to 64, with filter size 5*5 and 3*3
2. 4 fully connection layers, reducing output to 1
3. 1 dropout layers that keeps 50% data.

I tried to visualize the model using keras's plot model and grahper, but cannot find the right version and class to import. Will do more research in the futuer

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to pull back when it start driving off the road. These images show what a recovery looks like starting from left/right side of the road, pull it back to center and drive straight for a while :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

To train the car not to drive on dust road, i also recorded recovering from dust road. Images look like:
![alt text][image1]


After the collection process, I had 7225 number of data points. I then preprocessed this data by:
1. clip upper half of the image since it's not useful to the driver and will add more noise to data
2. filtered out all image that measurement is 0.0.


I finally randomly shuffled the data set and put 20% of the data into a validation set. Training loss and validation loss is between 0.03 and 0.04

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by training loss and validation is close. I used an adam optimizer so that manually training the learning rate wasn't necessary.
