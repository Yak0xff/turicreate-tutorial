# Activity Classifier


Activity classification is the task of identifying a pre-defined set of physical actions using motion-sensory inputs. Such sensors include accelerometers, gyroscopes, thermostats, and more found in most handheld devices today.

Possible applications include counting swimming laps using a watch's accelerometer data, turning on Bluetooth controlled lights when recognizing a certain gesture using gyroscope data from a handheld phone, or creating shortcuts to your favorite phone applications using hand gestures.

The activity classifier in Turi Create creates a deep learning model capable of detecting  temporal features in sensor data, lending itself well to the task of activity classification.


#### Introductory Tutorial

In this tutorial we create a model to classify physical activities done by users of a handheld phone, using both accelerometer and gyroscope data. We will use data from the [HAPT experiment](http://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions) which contains recording sessions of multiple users, each performing certain physical activities. The performed activities are walking, climbing up stairs, climbing down stairs, sitting, standing, and laying.

First you need to download the data from [here](http://archive.ics.uci.edu/ml/machine-learning-databases/00341/HAPT%20Data%20Set.zip) in zip format. The code below assumes the data was unzipped into a directory named `HAPT Data Set`. This folder contains 3 types of files - a file containing the performed activities for each experiment, files containing the collected accelerometer samples, and files containing the collected gyroscope samples.
