/*
* FaceSpaceTracker.cpp
*
*  Created on: Jul 31, 2014
*      Author: Marcel Padilla
		Last upload: 14.11 (reads year, month)
*/


#include <iostream>
#include <stdio.h>
#include <fstream>

#include <string>

#include <ctime> // for time measurement
#include <math.h> // use constant_PI from here and sin

double constant_PI=3.14159265359;

// FaceSpaceTracker class, includes opencv
#include "FaceSpaceTracker.h"

using namespace std;
using namespace cv; // opencv



//namespace FaceSpaceTracker {
//
//FaceSpaceTracker::FaceSpaceTracker() {
//	// TODO Auto-generated constructor stub
//
//}
//
//FaceSpaceTracker::~FaceSpaceTracker() {
//	// TODO Auto-generated destructor stub
//}
//
//} /* namespace FaceSpaceTracker */

//=================================================================================================
//=================================================================================================
//=================================================================================================
/*
* This is a facetracking class by the use of a webcam that is set directly above the screen.
* Used to return the estimated Position of the eyes in 3D space
*
* The origin of the default coordinate system is at the camera lens itself. When you call for a full calibration the origin becomes the position of the face at the moment of calibration
* +X = Right
* +Y = Up
* -Z = Depth
* Everything is measured in millimeters
* Note that when launching the tracker, it will take a few runs to fill the smoothenings stack with data.
*
* The constructor reads the settings from the FaceSpaceTracker.cfg file
* To update the Tracking Position, do Objectname.updateTrackData();"
* and then get Objectname.getXhumanPosition() etc.
*
*
* */
//=================================================================================================
//=================================================================================================
//=================================================================================================


//=================================================================================================
//=================================================================================================
//=================================================================================================
// public methods
//=================================================================================================
//=================================================================================================
//=================================================================================================
// for a more detailed explaination what these function do, check the FaceSpaceTracker website


FaceSpaceTracker::FaceSpaceTracker(){
	// constructor with default config file
	configFile = "FaceSpaceTrackerDefault.cfg";
	cameraInit();
}

FaceSpaceTracker::FaceSpaceTracker(std::string configLocation){
	// constructor with specified config file
	configFile = configLocation;
	cameraInit();
}

void FaceSpaceTracker::cameraInit(){
	// initializes all camera Settings and loads the camera

	//check if camera device is open, if yes, close it
	if(captureDevice.isOpened()) cameraOff();

	cout << " FaceSpaceTracker : read config file = "<< configFile << endl;
	readConfigFile();

	// init what is not set by the the configfile
	trackStatus='N'; // N = nothing
	calibrateX=false;
	calibrateY=false;
	calibrateZ=false;
	smoothonly=false;
	showCamFrame = false;
	calibrationValueX=0;
	calibrationValueY=0;
	calibrationValueZ=0;
	calibrationPixelX=0;
	calibrationPixelY=0;
	numberOfFrames=0;
	//lastTimeOfUpdate=0; // unfinished, to update trackdata after specified time intervals


	// load the different cascades
	cout << " FaceSpaceTracker : load face cascade = " << faceTrackingFileName << endl;
	face_cascade.load(faceTrackingFileName);

	if( allowProfileTracking) {
		cout << " FaceSpaceTracker : load profile cascade = " << profileTrackingFileName << endl;
		profile_cascade.load(profileTrackingFileName);
	}

	cout << " FaceSpaceTracker : open capture device" << endl;
	loadCaptureDevice();

	// make sure the frames are not empty
	captureDevice >> captureFrame;
	captureDevice >> captureFrame;  // for misterious reasons, i need to call this twice to correctly read the cameraReal_resolution
	grayscaleFrame = captureFrame;

	// read resolutions. the opencv matrix holding the video capture automatically carries the correct sizes.
	cameraRealXresolution = captureFrame.cols;
	cameraRealYresolution = captureFrame.rows;
	
}

void FaceSpaceTracker::cameraOff(){
	// disconnects the captureDevice from the Object
	captureDevice.release();
}


void FaceSpaceTracker::updateTrackData(){
	// intended to call once per frame, updates the Position data
	if(  numberOfFrames % (skipAmount+1) == 0){ // trackStatus =='N' ||
		// only update after every "skipAmount" of frames and if the face was previously found
		trackUpdate();
	} else {
		//when the actuall facetracking is taking a brake, the previous data is still smoothend
		updateTrackDataSmoothOnly();
	}
	// remember that this call happened
	numberOfFrames++;
	//	numberOfFrames=clock() - lastTime;
}

void FaceSpaceTracker::updateTrackDataSmoothOnly(){
	// use in between the face detection to smoothen the frames while reducing cpu usage
	smoothonly=true;
	trackUpdate();
	smoothonly=false;
}

void FaceSpaceTracker::calibrateCenter(){
	// set true so in the next update the calibration will happen
	calibrateX = true;
	calibrateY = true;
	calibrateZ = true;
}

void FaceSpaceTracker::resetCalibration(){
	// remove x,y z calibration
	calibrationValueX=0;
	calibrationValueY=0;
	calibrationValueZ=0;
}

void FaceSpaceTracker::loadConfigFile(std::string newfile){
	// switch the camera Settings
	configFile = newfile;
	// initialize with new camera
	cameraInit(); 
}

// get the calibrated position
double FaceSpaceTracker::getXposition(){
	return humanPosition[0];
}

double FaceSpaceTracker::getYposition(){
	return humanPosition[1];
}

double FaceSpaceTracker::getZposition(){
	return humanPosition[2];
}

// get the Position as if without calibration of xyz
double FaceSpaceTracker::getAbsoluteXposition(){
	return previousVectorData[0][0];
}

double FaceSpaceTracker::getAbsoluteYposition(){
	return previousVectorData[1][0];
}

double FaceSpaceTracker::getAbsoluteZposition(){
	return previousVectorData[2][0];
}

int FaceSpaceTracker::getCameraDeviceNumber(){
	return cameraDeviceNumber;
}

char FaceSpaceTracker::getTrackStatus(){
	// see if a face was found, frontal, L or R in the last tracking attempt
	return trackStatus;
}

cv::Mat FaceSpaceTracker::getCaputureFrame(){
	// return what the camera sees
	return captureFrame;
}

cv::Mat FaceSpaceTracker::getCroppedCaptureFrame(){
	// return what the camera tracks

	// update imageMargin
	int imageMargin = lastTrackPixelData[2]*imageMarginRelation; // floor that value
	int ancorX = lastTrackPixelData[0] - imageMargin;
	int ancorY = lastTrackPixelData[1] - imageMargin;
	int ancorHeight = lastTrackPixelData[2] + 2*imageMargin;
	int ancorWidth = lastTrackPixelData[2] + 2*imageMargin;

	// make sure we do not cross the boarders
	if( ancorX < 0 ) ancorX=0;
	if( ancorY < 0 ) ancorY=0;
	if( ancorX + ancorWidth >= captureFrame.cols ) ancorWidth = captureFrame.cols - ancorX;
	if( ancorY + ancorHeight >= captureFrame.rows ) ancorHeight = captureFrame.rows - ancorY;

	// return the cut out of the image
	return captureFrame( Rect(ancorX , ancorY , ancorWidth , ancorHeight ) );
}

// get some of the parameters
double FaceSpaceTracker::getScreenHeight(){
	return realLifeScreenHeight;
}

double FaceSpaceTracker::getScreenWidth(){
	return realLifeScreenWidth;
}

double FaceSpaceTracker::getFaceHeight(){
	return realLifeFaceHeight;
}

double FaceSpaceTracker::getDetectionScaleIncreaseRate(){
	return detectionScaleIncreaseRate;
}

double FaceSpaceTracker::getCameraVerticalFOV(){
	return cameraVerticalFOV;
}

bool FaceSpaceTracker::getAllowFrameResizing(){
	return allowFrameResizing;
}

int FaceSpaceTracker::getCameraRealXresolution(){
	return cameraRealXresolution;
}

int FaceSpaceTracker::getCameraRealYresolution(){
	return cameraRealYresolution;
}

int FaceSpaceTracker::getCameraTargetXresolution(){
	return cameraTargetXresolution;
}

int FaceSpaceTracker::getCameraTargetYresolution(){
	return cameraTargetYresolution;
}

double FaceSpaceTracker::getImageMarginRelation(){
	return imageMarginRelation;
}

int FaceSpaceTracker::getPreviousPixelDataCols(){
	return previousPixelDataCols;
}

int FaceSpaceTracker::getPreviousVectorDataCols(){
	return previousVectorDataCols;
}

double FaceSpaceTracker::getErrorToleranceXY(){
	return errorToleranceXY;
}

double FaceSpaceTracker::getErrorToleranceDepth(){
	return errorToleranceDepth;
}

double FaceSpaceTracker::getEyeXrealtion(){
	return eyeXrelation;
}

double FaceSpaceTracker::getEyeYrealtion(){
	return eyeYrelation;
}

int FaceSpaceTracker::getSkipAmount(){
	return skipAmount;
}

std::string FaceSpaceTracker::getFaceTrackingFileName(){
	return faceTrackingFileName;
}

bool FaceSpaceTracker::getAllowProfileTracking(){
	return allowProfileTracking;
}

std::string FaceSpaceTracker::getProfileTrackingFileName(){
	return profileTrackingFileName;
}

bool FaceSpaceTracker::getShowCamFrame(){
	return showCamFrame;
}

bool FaceSpaceTracker::nextFrameWillBeTracked(){
	// returns true if the next updateTrackData() call will take longer since it actually tracks the face instead of smoothening
	// this can be important since that tracking will take a few more miliseconds
	return ( trackStatus!='N' || numberOfFrames % (skipAmount+1) == 0); 
}
bool FaceSpaceTracker::lastFrameWasTracked(){
	// returns true if the last updateTrackData() call took longer since it actually tracked the face instead of smoothening
	// this can be important since that tracking will take a few more miliseconds
	return ( trackStatus!='N' || (numberOfFrames - 1) % (skipAmount+1) == 0); 
}



//=================================================================
// =========== SET METHODS ==================================
//=================================================================

void FaceSpaceTracker::setCameraDeviceNumber(int id){
	cameraDeviceNumber = id;
}

void FaceSpaceTracker::setCamVerticalFOV(double fov){
	cameraVerticalFOV = fov;
}

void FaceSpaceTracker::setScreenHeigth(double heigth){
	realLifeScreenHeight = heigth;
}

void FaceSpaceTracker::setScreenWidth(double width){
	realLifeScreenWidth = width;
}

void FaceSpaceTracker::setFaceHeigth(double faceheigth){
	realLifeFaceHeight = faceheigth;
}

void FaceSpaceTracker::setDetectionScaleIncreaseRate(double rate){
	detectionScaleIncreaseRate=rate;
}

void FaceSpaceTracker::setAllowFrameRezising(bool allow){
	allowFrameResizing=allow;
}

void FaceSpaceTracker::setCameraTargetXresolution(int pixels){
	//reset trackStatus to avoid targeting an empty area
	trackStatus ='N';
	resetSmoothers();
	cameraTargetXresolution = pixels;
}

void FaceSpaceTracker::setCameraTargetYresolution(int pixels){
	//reset trackStatus to avoid targeting an empty area
	trackStatus ='N';
	resetSmoothers();
	cameraTargetYresolution = pixels;
}

void FaceSpaceTracker::setImageTrackMarginRelation(double margin){
	imageMarginRelation = margin;
}

void FaceSpaceTracker::setTrackDataSmoothingSteps(int cols){
	previousVectorDataCols = cols;
	setUpSmoothing();
}

void FaceSpaceTracker::setTrackPixelDataSmoothingSteps(int cols){
	previousPixelDataCols = cols;
	setUpSmoothing();
}

void FaceSpaceTracker::setErrorToleranceXY(double pixels){
	errorToleranceXY = pixels;
}

void FaceSpaceTracker::setErrorToleranceDepth(double pixels){
	errorToleranceDepth = pixels;
}

void FaceSpaceTracker::setEyeXrealtion(double rel){
	eyeXrelation=rel;
}

void FaceSpaceTracker::setEyeYrealtion(double rel){
	eyeYrelation=rel;
}

void FaceSpaceTracker::setSkipAmount(int numberOfEmptyLoops){
	skipAmount = numberOfEmptyLoops;
}

void FaceSpaceTracker::setXcalibration(double value){
	calibrationValueX = value;
}

void FaceSpaceTracker::setYcalibration(double value){
	calibrationValueY = value;
}

void FaceSpaceTracker::setZcalibration(double value){
	calibrationValueZ = value;
}

void FaceSpaceTracker::setShowCamFrame(bool show ){
	showCamFrame = show;
}

void FaceSpaceTracker::setCurrentConfigFile(std::string name){
	configFile = name;
}

void FaceSpaceTracker::setTrackStatus(char status){
	trackStatus = status;
}

void FaceSpaceTracker::setFaceTrackingFileName(std::string path){
	faceTrackingFileName = path;
}

void FaceSpaceTracker::setAllowProfileTracking(bool allow){
	allowProfileTracking = allow;
}

void FaceSpaceTracker::setProfileTrackingFileName(std::string path){
	profileTrackingFileName = path;
}

void FaceSpaceTracker::readConfigFile(){
	// all the following parameters/Global variables are stored in a separate config file to be manipulated by the application
	// this loads the parameters set in the current configfile to the current tracker

	// open file for input
	ifstream inputFile;
	inputFile.open(configFile); 

	string line;
	inputFile >> line >>  line >>  cameraDeviceNumber;
	inputFile >> line >>  line >>  cameraVerticalFOV;
	inputFile >> line >>  line >>  realLifeScreenHeight;
	inputFile >> line >>  line >>  realLifeScreenWidth;
	inputFile >> line >>  line >>  realLifeFaceHeight;
	inputFile >> line >>  line >>  detectionScaleIncreaseRate;
	inputFile >> line >>  line >>  allowFrameResizing;
	inputFile >> line >>  line >>  cameraTargetXresolution;
	inputFile >> line >>  line >>  cameraTargetYresolution;
	inputFile >> line >>  line >>  imageMarginRelation;
	inputFile >> line >>  line >>  previousVectorDataCols;
	inputFile >> line >>  line >>  previousPixelDataCols;
	inputFile >> line >>  line >>  errorToleranceXY;
	inputFile >> line >>  line >>  errorToleranceDepth;
	inputFile >> line >>  line >>  eyeXrelation;
	inputFile >> line >>  line >>  eyeYrelation;
	inputFile >> line >>  line >>  skipAmount;

	inputFile >> line >> line >> line;
	faceTrackingFileName = line;

	inputFile >> line >> line >> allowProfileTracking;

	inputFile >> line >> line >> line;
	profileTrackingFileName = line;

	//close file
	inputFile.close();

	// necessary for dynamic Matrices
	setUpSmoothing(); 
	resetSmoothers();

}

void FaceSpaceTracker::saveConfigFile(){
	// open file for output
	ofstream outputfile;
	outputfile.open(configFile);

	// write everything

	outputfile << "cameraDeviceNumber int= ";
	outputfile << cameraDeviceNumber << endl;

	outputfile << "cameraVerticalFOV Degrees= ";
	outputfile << cameraVerticalFOV << endl;

	outputfile << "realLifeScreenHeight Millimeters= ";
	outputfile << realLifeScreenHeight << endl;

	outputfile << "realLifeScreenWidth Millimeters= ";
	outputfile << realLifeScreenWidth << endl;

	outputfile << "realLifeFaceHeight Millimeters= ";
	outputfile << realLifeFaceHeight<< endl;

	outputfile << "detectionScaleIncreaseRate double= ";
	outputfile << detectionScaleIncreaseRate << endl;
	
	outputfile << "allowFrameResizing bool= ";
	outputfile << allowFrameResizing << endl;

	outputfile << "cameraTargetXresolution int= ";
	outputfile << cameraTargetXresolution << endl;

	outputfile << "cameraTargetYresolution int= ";
	outputfile << cameraTargetYresolution << endl;

	outputfile << "imageMarginRelation double= ";
	outputfile << imageMarginRelation << endl;

	outputfile << "previousVectorDataColsSmootheningIterations Int= ";
	outputfile << previousVectorDataCols << endl;

	outputfile << "previousPixelDataColsSmootheningIterations Int= ";
	outputfile << previousPixelDataCols << endl;

	outputfile << "errorToleranceXY Pixels= ";
	outputfile << errorToleranceXY<< endl;

	outputfile << "errorToleranceDepth Pixels= ";
	outputfile << errorToleranceDepth<< endl;

	outputfile << "eyeXrelation Ratio= ";
	outputfile << eyeXrelation << endl;

	outputfile << "eyeYrelation Ratio= ";
	outputfile << eyeYrelation << endl;

	outputfile << "skipAmount int= ";
	outputfile << skipAmount << endl;
	
	outputfile << "faceTrackingFile Location= ";
	outputfile << faceTrackingFileName << endl;

	outputfile << "allowProfileTracking bool= ";
	outputfile << allowProfileTracking << endl;

	outputfile << "profileTrackingFile Location= ";
	outputfile << profileTrackingFileName << endl;

	//close it
	outputfile.close();
}

void FaceSpaceTracker::reportStatus(){
	cout << " ===== STATUS REPORT OF FaceSpaceTracker ======" << endl;

	cout << "x position is = " << getXposition() << endl;
	cout << "y position is = " << getYposition() << endl;
	cout << "z position is = " << getZposition() << endl;
	cout << "absolute x position is = " << getAbsoluteXposition() << endl;
	cout << "absolute y position is = " << getAbsoluteYposition() << endl;
	cout << "absolute z position is = " << getAbsoluteZposition() << endl;
	cout << "current config file is = " << configFile << endl;
	cout << "cameraDeviceNumber is = " << cameraDeviceNumber << endl;
	cout << "current face tracking xml is = " << faceTrackingFileName << endl;
	cout << "Camera X resolution = "  << cameraRealXresolution << endl;
	cout << "Camera Y resolution = "  << cameraRealYresolution << endl;
	cout << "Camera Target X resolution = "  << cameraTargetXresolution << endl;
	cout << "Camera Target Y resolution = "  << cameraTargetYresolution << endl;
	cout << "Camera vertical field of view is = "  << cameraVerticalFOV << endl;
	cout << "Face tracking file name is = " << faceTrackingFileName << endl;
	cout << "AllowProfileTracking is = " << allowProfileTracking << endl;
	cout << "Profile tracking file name is = " << profileTrackingFileName << endl;
	cout << "Screen height is = " << realLifeScreenHeight << endl;
	cout << "Screen width is = " << realLifeScreenWidth << endl;
	cout << "Face height is = " << realLifeFaceHeight << endl;
	cout << "calibrationValueX is = " << calibrationValueX << endl;
	cout << "calibrationValueY is = " << calibrationValueY << endl;
	cout << "calibrationValueZ is = " << calibrationValueZ << endl;
	cout << "calibrateX bool is = " << calibrateX << endl;
	cout << "calibrateY bool is = " << calibrateY << endl;
	cout << "calibrateZ bool is = " << calibrateZ << endl;
	cout << "imageMarginRelation is = " << imageMarginRelation << endl;
	cout << "previousVectorDataCols is = " << previousVectorDataCols << endl;
	cout << "previousPixelDataCols is = " << previousPixelDataCols << endl;
	cout << "errorToleranceXY is = " << errorToleranceXY << endl;
	cout << "errorToleranceDepth is = " << errorToleranceDepth << endl;
	cout << "eyeXrelation is = " << eyeXrelation << endl;
	cout << "eyeYrelation is = " << eyeYrelation << endl;
	cout << "skipAmount is = " << skipAmount << endl;
	cout << "showCamFrame bool is = " << showCamFrame << endl;
	cout << "trackStatus char is = " << trackStatus << endl;
	cout << "current numberOfFrames since start is = " << numberOfFrames << endl;
	//cout << " is = " <<  << endl;

	cout << " ======== END OF STATUS REPORT =========" << endl;
}



//=================================================================================================
//=================================================================================================
//=================================================================================================
/////////////////////// private METHODS /////////////////////////////
//=================================================================================================
//=================================================================================================
//=================================================================================================

void FaceSpaceTracker::loadCaptureDevice(){
	// this will be called during camerainit
	
	int currentCameraID=cameraDeviceNumber;
	bool firstCheck=true;
		
	while(!captureDevice.open(currentCameraID)) { // open a camera and simultaniously see if it exits
		// if it does not exist, try finding an alternative camera
		if( firstCheck && cameraDeviceNumber !=0 ) { // if the .cfg file captureDeviceNumber was anything else than the default value and unsuccessful, start at the beginning to finds any camera.
			currentCameraID=0; // restart searching for devices from the beginning.
			firstCheck=false;
		}else{ currentCameraID++;  } // if not, check the next port 
		cout << currentCameraID << endl;
		 // if still not, give error. Do not expect more than 100 cameras.
		if(currentCameraID > 100) {
			cout << " OpenCV could not find a camera. exiting now "<< endl;
			exit(-1);
		}
	}


}

void FaceSpaceTracker::setUpSmoothing(){
	// needed because of the dynamic sizes of the smoothening storage
	//	previousVectorData.conservativeResize(3,previousVectorDataCols);
	previousVectorData.resize(3);

	for( int i=0 ; i < 3 ; i++) previousVectorData[i].resize(previousVectorDataCols);
	//	previousVectorData.resize( previousVectorDataCols , vector<double>( 3 , 0.0 ) );

	//	previousPixelData.conservativeResize(3,previousPixelDataCols);
	previousPixelData.resize(3);
	for( int i=0 ; i < 3 ; i++) previousPixelData[i].resize(previousPixelDataCols);
	//	previousPixelData.resize( previousPixelDataCols , vector<double>( 3 , 0 ) );
}

void FaceSpaceTracker::resetSmoothers(){
	// removes all previously known data
	for( int i=0 ; i < previousVectorDataCols ; i++){
		previousVectorData[0][i]=0;
		previousVectorData[1][i]=0;
		previousVectorData[2][i]=0;
	}
	for( int i=0 ; i < previousPixelDataCols ; i++){
		previousPixelData[0][i]=0;
		previousPixelData[1][i]=0;
		previousPixelData[2][i]=1; // faceheigth
	}
	lastTrackPixelData[0] = 0;
	lastTrackPixelData[1] = 0;
	lastTrackPixelData[2] = 1;
	trackStatus = 'N'; //faceFound=false; // because we reseted the last pixel data
}

void FaceSpaceTracker::leftProfileTransformation(){
	// this function only serves to mirror the capture frame for left yawn head rotations
	cv::Mat flipper;
	cv::flip(captureFrame, flipper, 1);
	captureFrame = flipper;
	
	// also transform the ancors for the head tracking, so the last track pixel data. [2] is face height/width. cols=xresolution
	lastTrackPixelData[0] = captureFrame.cols - lastTrackPixelData[0] - lastTrackPixelData[2];
	
}

void FaceSpaceTracker::leftProfileInverseTransformation(){
	// this function cancels the mirroring made by leftProfileTransformation(). it also corrects the faces coordinates
	cv::Mat flipper;
	cv::flip(captureFrame, flipper, 1);
	captureFrame = flipper;
	// fix coordinates only if anything was found 
	if(faces.size() > 0) {
		// dont forget to shift by the face height since the offset is wrongly mirrored from top left to top right corner.
		faces[0].x = captureFrame.cols - faces[0].x - faces[0].height;
	}

	// also transform the ancors for the head tracking, so the last track pixel data, [2] is face height
	lastTrackPixelData[0] = captureFrame.cols - lastTrackPixelData[0] - lastTrackPixelData[2];


}

void FaceSpaceTracker::trackPixelDataSmoother(int faceX, int faceY, int faceHeight){

	// Method: weighted averaging with previous values, and ignore rawData with minor changes:
	if(fabs(faceX-previousPixelData[0][0]) < errorToleranceXY) faceX = previousPixelData[0][0];
	if(fabs(faceY-previousPixelData[1][0]) < errorToleranceXY) faceY = previousPixelData[1][0];
	if(fabs(faceHeight-previousPixelData[2][0]) < errorToleranceDepth) faceHeight = previousPixelData[2][0];

	double pixelData[] = {(double)faceX, (double)faceY , (double)faceHeight};

	// take average
	for(int i=0; i<3 ; i++) pixelData[i] = pixelData[i]*(double)1/(previousPixelDataCols+1);
	for( int j=0 ; j < previousPixelDataCols ; j++){
		for(int i=0; i<3 ; i++){
			pixelData[i] = pixelData[i] + previousPixelData[i][j]*((double)1/( previousPixelDataCols + 1 ));
		}
	}

	//push all vectors around like a queue
	for( int j = previousPixelDataCols-1 ; j > 0 ; j--){ // backwards loop
		for(int i=0; i<3 ; i++){
			previousPixelData[i][j] = previousPixelData[i][j-1];
		}
	}

	for(int i=0; i<3 ; i++){ // copy for return
		previousPixelData[i][0]=pixelData[i];
	}

}


void FaceSpaceTracker::trackVectorSmoother(double* newData){
	// smoothen out the offsetvector

	// Method: weighted averaging with previous values:
	for(int i=0; i<3 ; i++) newData[i] = newData[i]*(double)1/(previousVectorDataCols+1);

	for( int j=0 ; j < previousVectorDataCols ; j++){
		for(int i=0; i<3 ; i++){
			newData[i] = newData[i] + previousVectorData[i][j]*(double)1/(previousVectorDataCols+1);
		}
	}

	//push all vectors around like a queue
	for( int j = previousVectorDataCols-1 ; j > 0 ; j--){ // backwards loop
		for(int i=0; i<3 ; i++){
			previousVectorData[i][j] = previousVectorData[i][j-1];
		}
	}
	for(int i=0; i<3 ; i++){ // copy most resent data for return
		previousVectorData[i][0]=newData[i];
	}
}


void FaceSpaceTracker::trackFilter(double faceX, double faceY, double faceHeight){
	// here we interprete the trackdata to the cameraoffset. 0,0 pixel at top left

	// FILTER IT FIRST
	trackPixelDataSmoother( faceX, faceY, faceHeight);
	faceX = previousPixelData[0][0];
	faceY = previousPixelData[1][0];
	faceHeight = previousPixelData[2][0];
	//		cout << faceX << " = FaceX , " << faceY << " = FaceY , " << faceHeight << " = FaceHeight ,  AFTER SMOOTHY " << endl;

	double faceWidth = faceHeight; // keep this just in case


	// first, create pixel to mm scale factor by comparing to realLifeFaceHeigth,
	double imageScaleFactor = realLifeFaceHeight/faceHeight;

	// estimate the center between the eyes in pixels, the center is the origin;
	double eyeX = faceX + faceWidth*eyeXrelation - captureFrame.cols/2.0; // eyeX=.5 for center of eye, .7 for right eye
	double eyeY = faceY + faceHeight*eyeYrelation - captureFrame.rows/2.0;

	// estimate the distance from camera to center of camera plane (only using y hight, we guess the camera does not distort
	// Camera plane := your face plane normal to the webcam's direction
	double distanceToCamPlane = (captureFrame.rows*imageScaleFactor/2)/(tan(cameraVerticalFOV*(constant_PI/180)/2)); //


	// now estimate the position of your eyes (to the camera)
	//// -Z is depth, +X is right, +Y is up
	double newData[] = { // WARNING: we are in the camera Frame coords
		-eyeX*imageScaleFactor, // minus X because camera image is mirrored @ y-axis
		-eyeY*imageScaleFactor, // I have to flip y to have +y up. maybe because the eye coords origin is top left of webcam frame
		distanceToCamPlane //old: negative sign because i needed to flip it, maybe because of webcamVerFOV
	};

	// smooth out data by extrapolation and update cameraOffset0
	trackVectorSmoother(newData);
	for(int i=0;i<3;i++) humanPosition[i] = previousVectorData[i][0];

	if(calibrateX) {
		calibrateX = false;
		calibrationValueX = humanPosition[0];
		calibrationPixelX = lastTrackPixelData[0]; // important to have centering behaviour when loosing the face
	}
	if(calibrateY) {
		calibrateY = false;
		calibrationValueY = humanPosition[1];
		calibrationPixelY = lastTrackPixelData[1];
	}
	if(calibrateZ) {
		calibrateZ = false;
		calibrationValueZ = humanPosition[2];
	}

	// apply calibration
	humanPosition[0] -= calibrationValueX;
	humanPosition[1] -= calibrationValueY;
	humanPosition[2] -= calibrationValueZ;
	

}

void FaceSpaceTracker::croppedFaceTrack( char side){
	// here we will find the face on the cropped area of the camera image
	// the char side specifies what to track. eg F = front, R = right profile

	//determine what classifier to use
	cv::CascadeClassifier cascadeName;
	if( side == 'F' ) cascadeName = face_cascade;
	else if( allowProfileTracking ) cascadeName = profile_cascade; // if allowProfileTracking was false, then side would forcefully have to be =N at this point and we would not enter this method

	// make the transform if necessary
	if( allowProfileTracking && side == 'L' ) leftProfileTransformation();

	// update imageMargin
	int imageMargin = lastTrackPixelData[2]*imageMarginRelation;
	int ancorX = lastTrackPixelData[0] - imageMargin;
	int ancorY = lastTrackPixelData[1] - imageMargin;
	int ancorHeight = lastTrackPixelData[2] + 2*imageMargin;
	int ancorWidth = lastTrackPixelData[2] + 2*imageMargin;

	// make sure we do not cross the boarders
	if( ancorX < 0 ) ancorX=0;
	if( ancorY < 0 ) ancorY=0;
	if( ancorX + ancorWidth >= captureFrame.cols ) ancorWidth = captureFrame.cols - ancorX;
	if( ancorY + ancorHeight >= captureFrame.rows ) ancorHeight = captureFrame.rows - ancorY;

	//convert captured image to gray scale and equalize
	cvtColor(captureFrame( Rect(ancorX , ancorY , ancorWidth , ancorHeight ) ), grayscaleFrame, CV_BGR2GRAY);
	equalizeHist(grayscaleFrame, grayscaleFrame);

	// estimate the size of the face based on the previous height
	double min_face_size = lastTrackPixelData[2]*0.8;
	double max_face_size = lastTrackPixelData[2]*1.2;

	// apply the actual face detection
	cascadeName.detectMultiScale(grayscaleFrame, faces, detectionScaleIncreaseRate , 0, CV_HAAR_FIND_BIGGEST_OBJECT| CV_HAAR_SCALE_IMAGE, Size(min_face_size, min_face_size),Size(max_face_size, max_face_size));

	if(faces.size() > 0) { // now recalibrate the data back to the big image frame
		faces[0].x += ancorX;
		faces[0].y += ancorY;
	}

	// undo the tranformation if necesary
	if( allowProfileTracking && side == 'L' ) leftProfileInverseTransformation();

}

void FaceSpaceTracker::profileTrackCycle(char side){
	// in this method, depending on the value of side track that side first and then the other


	// right side case
	if(side == 'R' ) {
		croppedFaceTrack('R');
		// check the result
		if(faces.size() > 0) localTrackStatus='R';
		else {
			// track left
			croppedFaceTrack('L');
			// check the result
			if(faces.size() > 0) localTrackStatus='L';
			else {
				// mark failure if even that failed
				localTrackStatus ='N'; 
			}
		}
	} else if( side == 'L') {
		// track left
		croppedFaceTrack('L');
		// check the result
		if(faces.size() > 0) localTrackStatus='L';
		else {
			// right
			croppedFaceTrack('R');
			// check the result
			if(faces.size() > 0) localTrackStatus='R';
			else {
				// mark failure if even that failed
				localTrackStatus ='N'; 
			}
		}
	}
}

void FaceSpaceTracker::updateCameraFrame(){
	//capture a new image frame, this is the image in the showwebcam, the grayscale is used for face detection
	captureDevice >> captureFrame;

	// if allowed, resize window size
	if( allowFrameResizing ) {
		// resize the captureFrame to a different (smaller) size
		Size size(cameraTargetXresolution, cameraTargetYresolution);
		resize( captureFrame, captureFrame , size); //resize image
	}

}

void FaceSpaceTracker::fullFaceTrack(char side){

	//determine what classifier to use
	cv::CascadeClassifier cascadeName;
	if( side == 'F' ) cascadeName = face_cascade;
	else if( allowProfileTracking ) cascadeName = profile_cascade;

	// make the transform if necessary
	if( allowProfileTracking && side == 'L' ) leftProfileTransformation();

	//make gray scale and equalize
	cvtColor(captureFrame, grayscaleFrame, CV_BGR2GRAY);
	equalizeHist(grayscaleFrame, grayscaleFrame);

	// apply the actual face detection
	cascadeName.detectMultiScale(grayscaleFrame, faces, detectionScaleIncreaseRate, 2, CV_HAAR_FIND_BIGGEST_OBJECT| CV_HAAR_SCALE_IMAGE, Size(30,30));

	// undo the tranformation if necesary
	if( allowProfileTracking && side == 'L' ) leftProfileInverseTransformation();

}

void FaceSpaceTracker::faceTrack(){
	// this function is called once smoothing was ruled out and the face has to be detected

	// read with the camera
	updateCameraFrame();

	// save a variable to not forget the previous face direction. overwrite the global one at the end of this method.
	localTrackStatus = trackStatus; 

	// check if cropped search makes sense
	if( localTrackStatus != 'N' ){

		// always check for frontal face first since the side of the face is only there to avoide disconnections. The user should not look from the side for long.
		croppedFaceTrack('F');
		// mark the result
		if(faces.size() > 0) localTrackStatus='F';
		else localTrackStatus='N'; // try to not overwrite this yet to see to quickly skip ahed. // else trackStatus='N';

		// check for profile tracking
		if( allowProfileTracking && localTrackStatus != 'F'){ // only do this if no frontal face was found right before

			if( trackStatus == 'R' ) {			// if the last tracking was a RIGHT sided tracking, start right sided tracking
				profileTrackCycle('R');
			}	else if ( trackStatus == 'L' ) {		// ELSE IF the last tracking was a LEFT sided tracking, start left sided tracking
				profileTrackCycle('L');
			} else {		// if none of the above was true, check right and left just as before
				profileTrackCycle('R');
			}

		} // end allow Profile tracking

	} // end cropped searches

	if( localTrackStatus=='N' ){ // if nothing was found, search the entire image

		// check everywhere for the frontal face
		fullFaceTrack('F');
		// mark the result
		if(faces.size() > 0) localTrackStatus='F';

	} // end full searchers

	// update the global track status
	trackStatus = localTrackStatus;

	if(showCamFrame){
		// show what the camera is seeing
		displayCameraInput();
	}
}

void FaceSpaceTracker::trackUpdate(){
	// this is the mother function that handles the tracking and smoothing

	if(!smoothonly){ // if true, track using opencv

		faceTrack();

	} else {  // if only smoothing, pass previous info for smoothening, future idea: predict movement
		faces[0].x=lastTrackPixelData[0];
		faces[0].y=lastTrackPixelData[1];
		faces[0].height=lastTrackPixelData[2];
	}// end smoothonly

	// if nothing was found, feed the smoothers with old, known positions, e.g. the origin.
	if(faces.size() < 1) { 
		// create a cv::rect and push it onto the faces. the center of the frame together with the last face height
		// sadly, since we calibrate at the very end, we must reverse calculate the x, y, pixel values of the origin
		cv::Rect substitude( calibrationPixelX, calibrationPixelY, lastTrackPixelData[2], lastTrackPixelData[2]); //captureFrame.rows/2, captureFrame.cols/2
		faces.push_back(substitude);
	}

	//if tracking found, read and interpret it
	if(faces.size() > 0) {

		// update last actual position
		lastTrackPixelData[0] = faces[0].x;
		lastTrackPixelData[1] = faces[0].y;
		lastTrackPixelData[2] = faces[0].height;

		// smoothen and transform to position data
		trackFilter(faces[0].x, faces[0].y, faces[0].height);

	}

}

void FaceSpaceTracker::displayCameraInput(){

	if(faces.size() > 0)	{
		// create several rectangles to marke different objects. 1: opencv Face, 2: cropped image, 3: eye position
		Point pt1(lastTrackPixelData[0] + faces[0].width, lastTrackPixelData[1] + faces[0].height);
		Point pt2(lastTrackPixelData[0], lastTrackPixelData[1]);
		int rec2size = 8;
		Point pt1eye(lastTrackPixelData[0] + lastTrackPixelData[2]*eyeXrelation -rec2size/2, lastTrackPixelData[1] + lastTrackPixelData[2]*eyeYrelation - rec2size/2);
		Point pt2eye(lastTrackPixelData[0] + lastTrackPixelData[2]*eyeXrelation +rec2size/2, lastTrackPixelData[1] + lastTrackPixelData[2]*eyeYrelation + rec2size/2);
		int imageMargin = lastTrackPixelData[2]*imageMarginRelation;
		Point pt1track(lastTrackPixelData[0] + lastTrackPixelData[2] + imageMargin, lastTrackPixelData[1] + lastTrackPixelData[2] + imageMargin);
		Point pt2track(lastTrackPixelData[0] - imageMargin, lastTrackPixelData[1] - imageMargin);

		rectangle(captureFrame, pt1, pt2, cvScalar(0, 255, 0, 0), 1, 8, 0);
		rectangle(captureFrame, pt1eye, pt2eye, cvScalar(0,0,0, 0), 3, 8, 0);
		rectangle(captureFrame, pt1track, pt2track, cvScalar(0,0,255, 0), 3, 8, 0);
	} else {
		// if nothign was found, draw a rectangle arround the entire image
		Point pt1(0, 0);
		Point pt2(captureFrame.cols, captureFrame.rows);
		rectangle(captureFrame, pt1, pt2, cvScalar(0,0,255, 0), 6, 8, 0);
	}

	//show the output
	imshow("outputCapture", captureFrame);
	//pause for 1 ms, without it it wont show the image
	waitKey(1);
}




