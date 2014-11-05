/*
* FaceSpaceTracker.h
*
*  Created on: Jul 31, 2014
*      Author: Marcel Padilla
*/



#ifndef FACESPACETRACKER_H_
#define FACESPACETRACKER_H_

// I have to include them here
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//namespace FreeCamFaceTrack {

class FaceSpaceTracker {



private: // all variables private, they are all specified in the cfg file or used internally


	//=================================================================
	/*
	About the .cfg File:
	To change the configuration of the FaceSpaceTracker open FaceSpaceTrackerDefault.cfg or create or your own file myConfigFile.cfg and using any text editor (e.g. notepad++, gedit).

	It is also important to note that the tracker object will always be loaded without calibration and assumes the camera to be the origin. By calibrating, the origin shifts to the position of the user. Since there is no standard distance to the desktop screen it is recommended to calibrate at least once before using the tracker to never calibrate. For example your application could start with a message saying "get into your desired position and press any button to continue."

	The Lines inside the .cfg file are all written in the following format:
	NameOfVariable TypeOfInput= Value
	NOTE: do not change the order of the lines. It is important for identification. Also keep the blank spaces in between the words. Only change the values on the most right.

	FIRST TIME SET UP:
	Since the screen size variables are unused by the tracker but necessary for visualization applications you do not need to set them as long as you only need raw tracking data.
	However, the cameraVerticalFOV is important even though most cameras have FOV of about 40°-45°. Using the default wont give bad results most of the time but it will be better to figure out your camera's models FOV.
	All the other variables from the .cfg file should only affect the performance. If your tracking is slow or lagging , you can read the meaning of each variable and adjust its numbers. Keep in mind that the light conditions of the users face is important. Generally more light gives faster results.

	Multiple Camera issue:
	Until now facespace Tracker will always grab the same default camera. Change the integer value of cameraDeviceNumber to pick a different camera. 

	*/

	//=================================================================
	// editable .cfg variables

	// The following variables appear in the same order inside the .cfg file

	
	int cameraDeviceNumber;
	/*Opencv opens capture devices with integers beginning from 0. By using the value 0 opencv will grab the default camera such as the only camera connected. If you have multiple cameras attached and want to work with a specific one at start up, try increasing the integer until you are successful. e.g. 0, 1, 2, 3 (if you have 4 cameras) */

	double cameraVerticalFOV;
	/*This is the vertical angle of the viewing volume of your camera. The camera should hopefully have a constant angle that is specified in the tech specs of the camera model.*/

	double realLifeScreenHeight;
	double realLifeScreenWidth;
	/* The physical sizes of the monitor being used given by default unit in millimeters. Note that these two quantities are not used inside the tracker, but are essential for */

	double realLifeFaceHeight;
	/*This estimated the height of the face and is vital for determining the position. However, a rough estimate is enough. To see what exact size suits you, call setShowCamFrame(true) and watch the green rectangle. You can then hold a ruler at your face an measure the height of the green rectangle. Small mistakes have little effects.*/

	double detectionScaleIncreaseRate;
	/*This has to be a value greater than 1. It determines how much to increase the search area after every attempt within the face detection algorithm in opencv. The closer the value gets to 1 the more precise will the face height while reducing the performance.*/
	
	bool allowFrameResizing;
	/*Decides whether the face detection should run on the original dimensions of the picture taken by the camera or on the later defined target resolutions. Use this to downscale the camera images to increase performance while loosing precision. Low resolutions are also prone to not be able to detect a face to far away from the camera*/
	
	int cameraTargetXresolution;
	int cameraTargetYresolution;
	/*These are the target resolutions for the down scaling of the camera images if allowFrameResizing==1*/

	double imageMarginRelation; 
	/*Given the height of the face found in the previous tracking, this number determines how much additional area to search for the face in the next tracking. e.g. if the value was 0.25 and the face was 100 pixels big in the last detection, then in the next tracking the face detection will run on a cropped area up to 25 pixels further than the previous face location in all directions. Reducing this increases performance at the cost of loosing fast tracking.*/
	
	int previousVectorDataCols;
	int previousPixelDataCols;
	/*These values determine how many previous values should be included when smoothening the next data value. This next value is then calculated by the average of the new track data with the previous one. "Vector" refers to double position data while "Pixel" refers to int x,y position and height of the face inside the camera frame.*/

	double errorToleranceXY;
	double errorToleranceDepth;
	/*These values determine the maximum treshhold to classify the result as unwanted jittering (if so, the tracking data will be replaced by the previous data). It refers to the pixel data mentioned above.*/

	double eyeXrelation;
	double eyeYrelation;
	/*These values are used to determine where the center of the eyes is inside the face rectangle. e.g. eyeXrelation=0.5 means middle line between left/right, and eyeYrelation=0.4 means sligtly above the above/below middle line*/

	int skipAmount;
	/*This decides how many times to smoothen the track data before detecting a face again. Bigger values boost the performance greatly while increasing a delay. Experimenting with this and the Smoothening Iterations can create satisfying results on slow machines.  */

	std::string faceTrackingFileName;
	/*This determines which method to use to detect the frontal face with the camera.  Other .xml files might be faster or more reliable.  This should specify the absolute or relative path to the file. In this example, e.g. "haarcascade_frontalface_alt.xml" if that file is located in the working directory. */

	bool allowProfileTracking;
	/*Determines if the tracker should try to detect the side of the face when the users face is lost after previously tracking it. Due to performance issues it willnot attempt to search for profiles on the entire camera image.*/

	std::string profileTrackingFileName;
	/*This determines which method to use to detect the side of the face with the camera.  Other .xml files might be faster or more reliable.  This should specify the absolute or relative path to the file. In this example, e.g. "haarcascade_frontalface_alt.xml" if that file is located in the working directory. */




	//=================================================================
	//variables that are not configurable in the .cfg

	cv::CascadeClassifier face_cascade;	
	cv::CascadeClassifier profile_cascade;
	// opencv specific object needed to performe face detection
	
	cv::VideoCapture captureDevice;
	// this object is used to use the camera

	cv::Mat captureFrame;
	cv::Mat grayscaleFrame;
	// These are the two image holders in form of opencv specific matricies.

	std::vector<cv::Rect> faces;
	//create a vector array to store the face found by x, y, and height

	std::string configFile;
	// location of the configuration file
	
	double humanPosition[3];
	// this array will hold the final tracking data

	int cameraRealXresolution;
	int cameraRealYresolution;
	//These are the actuall resolutions as the camera sees them without downscaling
	
	char trackStatus;
	// a character used to save from where the last tracking was successfull if at all. eg. F=frontSide, N=Nothing, R=RightSide, L=LeftSide 
	char localTrackStatus; 
	// like the trackStatus but needed to write on one without loosing the other. Useful for alternating to start with left or right profile tracking.
	
	std::vector< std::vector<double> > previousVectorData;
	std::vector< std::vector<double> > previousPixelData;
	// The vectors needed to carry the smoothening iteration data. Collum 0 holds the most up to date data and is being pushed backwards as new data comes in.

	int lastTrackPixelData[3];
	// An array that hold the last x, y position and height of the very last face rectangle.
	
	bool calibrateX;
	bool calibrateY;
	bool calibrateZ;
	// If these bool values are true, the next face detection will set the x/y/z value to be the 0 by subtracting the current absolute coordinate. After that the bool are set to false again.

	double calibrationValueX;
	double calibrationValueY;
	double calibrationValueZ;
	// Carry the values of the calibration mentioned above

	int calibrationPixelX;
	int calibrationPixelY;
	// needed to feed the tracker with false position data when the face is completly lost. These are update when calibrated and cause the tracker to think that we moved back to the origin

	int numberOfFrames; 
	//needed to count the number of smoothenings we did to then trigger a facedetection after skipAmount many frames.

	bool showCamFrame;
	// bool to open another window and to display what the camera tracks

	bool smoothonly;
	// needed to run the trackUpdate without facedetection.
	
	// unit conversion
	// note: inside the program runs ins mm, but all user in & output will be converted
	//double lengthConversionScale; // mm to ?, set to 1 for mm.



	// methods of FaceSpaceTracker
public:

	/*To use this documentation, we assume that you have covered the getting started guide.  Use Ctrl+F to search this site. If the tracker object is named "myTracker" then the constructor is one of the following:*/

	//=================================================================
	// the constructors
	FaceSpaceTracker();
	/*Creates a tracker object with the default camera using the settings specified in the FaceSpaceTrackerDefault.cfg that lies in the working directory.*/
	FaceSpaceTracker(std::string);
	/*Creates a tracker object with the default camera using the settings specified in by the file accessed in the Path. Absolute or relative.*/


	//any of the functions bellow will be called by: myTracker.functionName(input if any);

	//=================================================================
	// the main thing for the user

	void updateTrackData();
	/*Use this function once every time you go though the main loop, preferably right before you use the results. Be sure to read the configuration details before to get the best results.*/

	//essential get methods
	double getXposition();
	double getYposition();
	double getZposition();
	//	Returns the X,Y,Z values of the position of the head in millimeters.
	//NOTE: From the perspective of the user facing the camera:
	//+X is right
	//+Y is up
	//-Z is forward
	//If calibrateCenter() was not called once, the origin is at the camera, otherwise the origin is at the position of the face at the moment calibrateCenter() was called.

	//=================================================================
	// calibration functions:

	void calibrateCenter();
	/*This tells the tracker that the next time that new trackdata is collected, it will be used to define the current head position as the center, meaning that  getXposition == getYposition== getZposition== Zero.*/

	void calibrateXonly();
	void calibrateYonly();
	void calibrateZonly();
	/*As in calibrateCenter(), this runs the calibration only on a specific axis.*/

	void setXcalibration(double);
	void setYcalibration(double);
	void setZcalibration(double);
	/*Manually overwrite the calibration value used before.*/

	void resetCalibration();	
	/*Removes the calibration option made by calibrateCenter().*/

	//=================================================================
	// config file options

	void setCurrentConfigFile(std::string);
	/*Switches the current config file from which to load from or save to. This function does not reboot the tracker and is useful to edit the config files.*/

	void saveConfigFile();
	/*Reads the config file that is specified by the constructor or by setCurrentConfigFile(...). This will reboot the tracker.*/

	void readConfigFile();
	/*Saves the current settings to a .cfg file specified by setCurrentConfigFile(...). If the configuration file is not present, it will create a new one.*/

	void loadConfigFile(std::string);
	/*This is like the special constructor where you set the path to the .cfg file. It reboots the tracker with the specified .cfg settings.*/

	//=================================================================
	// other camera functions

	void reportStatus();
	/*Prints out all essential parameter and variable values into the console.*/

	void cameraInit();
	/*This reboots the tracker using the same .cfg file as before if no new one had been set using .setCurrentConfigFile(std::string name).*/

	void cameraOff();
	/*This disconnects the camera from the tracking object. Use it if you want to use the camera without destroying the tracker object.*/

	void updateTrackDataSmoothOnly();	
	/*Runs the .updateTrackData() but replaces actual new data with previous track data. This is exactly what the updateTracker() does when  skipping track frames as specified in skipamount in the .cfg file.*/

	char getTrackStatus();
	/*returns a series of characters specifiyng the result of the last traking. N=nothing found. F= frontal Face. R = right looking profile. L= left looking profile.*/

	//=================================================================
	// other get methods

	double getAbsoluteXposition();
	double getAbsoluteYposition();
	double getAbsoluteZposition();
	/*This is just like the getXYZposition functions but without the calibration data. It assumes that the camera is the origin. Note that this Z value will always return you the perpendicular distance to the camera.*/

	cv::Mat getCaputureFrame();
	/*Returns a matrix object as defined in opencv. It carries the last image taken by the camera when the tracker was updated.*/

	cv::Mat getCroppedCaptureFrame();
	/*Returns a matrix object as defined in opencv. It carries a cropped out version of the image taken by the camera when the tracker was updated. The cropping area is around the face of the user where the camera expects to find the face in the next picture.*/

	int getCameraRealXresolution();
	int getCameraRealYresolution();
	/*Returns the original camera resolution dimensions as they where before being resized*/

	bool getShowCamFrame();
	void setShowCamFrame(bool);
	/*If the variable showCamFrame is true, an active window will open to show what the camera sees and where it is activly tracking*/

	bool nextFrameWillBeTracked();
	bool lastFrameWasTracked();
	/*This function returns the bools to know in advance or after that the last/next call for facedetection takes a few milliseconds longer than the smoothening calls.*/

	void setTrackStatus(char);
	/*
	Sets the char value that says if and how the face was found in the last tracking. Possible values are:
	'N'= nothing found
	'F'= frontal face found
	'R'= Right looking face found
	'L'= Left looking face found
	This one is handled inside the program, but it can be usefull to force the programm to look again everywhere.
	*/


	//=================================================================
	// .cfg file variable methods

	// the following functions are just the get and set methods of the variables inside the configuration files.
	// To view the effects of these methods please refer to the configuration explanation

	// get methods for .cfg file variables
	int getCameraDeviceNumber();
	double getCameraVerticalFOV();
	double getScreenHeight();
	double getScreenWidth();
	double getFaceHeight();
	double getDetectionScaleIncreaseRate();
	bool getAllowFrameResizing();
	int getCameraTargetXresolution();
	int getCameraTargetYresolution();
	double getImageMarginRelation();
	int getPreviousVectorDataCols();
	int getPreviousPixelDataCols();
	double getErrorToleranceXY();
	double getErrorToleranceDepth();
	double getEyeXrealtion();
	double getEyeYrealtion();
	int getSkipAmount();
	std::string getFaceTrackingFileName();
	bool getAllowProfileTracking();
	std::string getProfileTrackingFileName();

	// set methods for .cfg file variables
	void setCameraDeviceNumber(int);
	void setCamVerticalFOV(double);
	void setScreenHeigth(double);
	void setScreenWidth(double);
	void setFaceHeigth(double);
	void setDetectionScaleIncreaseRate(double);
	void setAllowFrameRezising(bool);
	void setCameraTargetXresolution(int);
	void setCameraTargetYresolution(int);
	void setImageTrackMarginRelation(double);
	void setTrackDataSmoothingSteps(int);
	void setTrackPixelDataSmoothingSteps(int);
	void setErrorToleranceXY(double);
	void setErrorToleranceDepth(double);
	void setEyeXrealtion(double);
	void setEyeYrealtion(double);
	void setSkipAmount(int);
	void setFaceTrackingFileName(std::string);
	void setAllowProfileTracking(bool);
	void setProfileTrackingFileName(std::string);



	//=================================================================
	//=================================================================
	//=================================================================
	// other methods used by the tracker

	// The following methods handle the tracking and the smoothing. They are private because they all act on private variables and depend on each other.
	// Feel free to edit them, but for the basic usage there will be no need to look at these.

private:

	void loadCaptureDevice();
	/*This method loads the camera with the cameraDeviceNumber and if the camera was busy or not found it will search for other device numbers. If none are available an error message will be send*/

	void setUpSmoothing();
	/*Called to resize the initially empty arrays for the smoothening into the demanded sizes.*/

	void resetSmoothers();
	/*Replaces all the values inside the smoothening arrays to 0 if they describe a position and to 1 if they describe the face height*/

	void displayCameraInput();
	/*This method will only be called when showCamFrame was set true and displays what the Tracker sees when searching for the image. The different squares show the cropped area, the face found in opencv and the position of the eye*/

	void leftProfileTransformation();
	void leftProfileInverseTransformation();
	/*These function are call before and after searching the captureframe for the left looking face. It mirrors the caputrefram and transforms the tracking positions because the haarcascade_profileface.xml and searches right looking faces*/

	void trackPixelDataSmoother(int , int , int);
	/*Smoothens the face position and size integer values by calculating the mean of the new data and the previous data. It also ignores changes if the change is below the errorToleranceXY/Depth treshhold to avoid jittering*/

	void trackVectorSmoother(double*);
	/* smoothens the x y z double coordinates by calculating the mean of the new data and the previous data.*/

	void trackFilter(double , double , double );
	/*It reads the head position and size and runs through the smoothers and interpretes and calibrates the result. This is where the new positions are being assigned.*/

	void croppedFaceTrack( char side);
	void fullFaceTrack(char side);
	/*searches the cropped part of the capture frame where the face is expected or the entire captureFrame for the face or the profile. The value of char determines what it searches in that case
	'F' = search for frontal face
	'R' =search for right looking face
	'L' =search for left looking face
	To se if any face was found check with if(faces.size() > 0) if there is any need.
	*/

	void profileTrackCycle(char side);
	/*This method only simplyfies the tracking. If the last tracking turned out to be left sided, then it will start left sided tracking and then try right sided. btw: frontal face tracking is always tried first since it is the most essential*/

	void updateCameraFrame();
	/*This method updates the private variable image matrix and resizes it to the target resolutions if resizing is allowed.*/

	void faceTrack();
	/*Attempts to find the face efficiently by calling croppedFaceTrack() and fullFaceTrack() depending on the Trackstatuses of the attempted trackings. Note: due to heavy performance, the fullFaceTrack is only being called with the 'F' argument.*/

	void trackUpdate();
	/*This is the first function being called to trigger most of the private functions. updateTrackData directly calls this function which decides to smoothen only by using past information or to go for face tracking. After that it calls the trackfilter. If nothing was found it will feed the filter with information as if the face was moved back to the center of the screen.*/


	//public:
	//	FaceSpaceTracker();
	//	virtual ~FaceSpaceTracker();
};

//} /* namespace FaceSpaceTracker */

#endif /* FACESPACETRACKER_H_ */
