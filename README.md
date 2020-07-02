# Drone_Parking_Evaluation
Automated driving - GUI for the visual evaluation of drone data

This project is designed to evaluate automated parking functions using aerial images taken by drones, this detection software was designed to run in the Cloud, so that multiple devices can make use of it. For this purpose, an iOS App was developed to be used as GUI. 

When developing driver assistance systems and automated driving functions, drones can be used to efficiently evaluate vehicle tests. This is done by processing image and video recordings of certain test scenarios, for example in automated parking attempts. Reference data can be extracted from the image and video recordings, which contain the geometric relationships such as distances 
and orientations between the objects involved in the traffic situation - in particular the vehicles - and the infrastructure such as lane markings. 

This document is divided in the following sections
1. Requirements
2. Command line options
3. Cloud Architecture 
4. Image Processing algorithm
## Requirements
* OpenCV

## Command line options
* **Path** (*of the image without the .jpg extension*)  Evaluate the image in the provided path. Linux example: ./Drone_Parking_Evaluation/openCV/Drone_Parking_Evaluation data/Situation_3_Auswahl/Situation_3_fern 
![run Situation_3_fern](/img/example.jpg)
* **-tune nah/fern/mittel** open the fine tuning console for the test images. Linux example: ./Drone_Parking_Evaluation/openCV/Drone_Parking_Evaluation -tune nah
![tune car detector](/img/TuneCanny.jpg)
![tune line detector](/img/TuneHough.jpg)


## Claud Architecture 
The iOS allows the user to select pictures directly from their phone or to import them from a given URL. A preview box shows the picture to make sure the right one is selected. The user must add metadata in CSV format. It can be manually typed or imported from an URL as well. Manual changes of the metadata are always possible. Once the user provided an image and the metadata, the request can be sent to AWS for further processing.

A sequence diagram detailing the interactions between User <-> App <-> AWS is provided.

The AWS infrastructure follows a serverless pattern, making use of the API Gateway, a S3 Bucket for storing the images and metadata, a noSQL DynamoDB database for storing the requests and lambda functions for handling the interactions between the components, as well as processing the image.


## Image Processing algorithm
The Image Processing algorithm was develop on C++ to improve the computation time, 
the code is divided mainly in two functions focused on  the cars and line detection. In the following paragraph is presented the overview of the followed approach.

First in order to improve the system speed the true RGB color image was converted into a grayscale image then a Gaussian filter is used to remove the noise of the image. The borders of the objects where found used a Canny. After this point the line and car detection systems follow different methodologies.

For the car detection a series of dilatation, filling corners, erosions, and closings where applied to remove the small noisy particles, close the shapes of the cars and subtracts them from the image, afterwards, a vertical kernel was used to disconnect the cars from the street line, finally morphological operators where used to filter the by area.

The line detection followed a parallel approach that uses several hough detectors and and morphological operators that focused in the know shape of the street lines, here the biggest challenge was to detect straight lines that where twisted due to camera distortion.

Finally the points that determinate the rectangles around the cars and the start and end point of the lines, where translated 2D-3D using a unit sphere approach, ant the know position and elevation of the camera. Due to the camera was not calibrated and the calibration parameters were not  provided this method can be improved.

