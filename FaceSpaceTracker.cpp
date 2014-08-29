/*
 * FaceSpaceTracker.cpp
 *
 *  Created on: Jul 31, 2014
 *      Author: internshipdude
 */


#include <iostream>
#include <stdio.h>
#include <fstream>

#include <string>

#include <ctime> // for time measurement
#include <math.h> // use M_PI from here and sin


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
 * The origin of its coordinate system is the center of the screen, unless you calibrate, then it is perpendicuar on the screen to where you calibrated it
 * +X = Right
 * +Y = Up
 * -Z = Depth
 * Everything is measured in millimeters
 *
 * The constructor reads the settings from the FaceSpaceTracker.cfg file
 * To update the Tracking Position, do Objectname.updateTrackData();"
 * and then get Objectname.getXhumanPosition() etc.
 *
 * eyeXrelation & eyeYrelation have default values .5,.4 to give the camera the center between your eyes,
 * but it might be benificial for the 3d effect to push the x value towards the dominant eye of the user,
 * eg. right dominant eye x=.7;
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



FaceSpaceTracker::FaceSpaceTracker(){
	// constructor with default config file
	configFile = "FaceSpaceTrackerDefault.cfg";
	cameraInit();
}

FaceSpaceTracker::FaceSpaceTracker(const char* configLocation){
	// constructor with specified config file
	configFile = configLocation;
	cameraInit();
}

void FaceSpaceTracker::cameraInit(){
	// initializes all camera Settings
	cout << " FaceSpaceTracker : read config file = "<< configFile << endl;
	readConfigFile();

	// define what is not set by the the configfile
	faceFound=false;
	profileFound=false;
	calibrate=false;
	smoothonly=false;
	showCamFrame = false;
	calibrationX=0;
	calibrationY=0;
	numberOfFrames=0;
	lastTimeOfUpdate=0; // unfinished, to update trackdata after specified time intervals

	cout << " FaceSpaceTracker : load cascade = " << faceTrackingFile << endl;

	face_cascade.load(faceTrackingFile);
	//	profile_cascade.load(profileTrackingFile);

	cout << " FaceSpaceTracker : open capture device" << endl;
	int cameraNumber=0;
	if(!captureDevice.open(cameraNumber)) { // open a camera and simultaniously see if it exits
		cameraNumber++; // if not, check the next port
		if(cameraNumber > 100) { // if still not, give error
			cout << " OpenCV could not find a camera, exiting now "<< endl;
			exit(-1);
		}
	}
	// make sure the frames are not empty
	captureDevice >> captureFrame;
	grayscaleFrame = captureFrame;

	// read resolutions
	webcamYresolution = captureDevice.get(CV_CAP_PROP_FRAME_HEIGHT);
	webcamXresolution = captureDevice.get(CV_CAP_PROP_FRAME_WIDTH);

	//	 run the facetracking a few times to calibrate it. important for fixed depth.
	for(int i=0;i < 2*skipAmount*max(previousPixelDataCols,previousVectorDataCols) ; i++){
		trackUpdate();
	}
	calibrateCenter();
}

void FaceSpaceTracker::cameraOff(){
	// disconnects the captureDevice from the Object
	captureDevice.release();
}


void FaceSpaceTracker::updateTrackData(){
	// intended to call once per frame, updates the Position data
	if(!faceFound || numberOfFrames % (skipAmount+1) == 0){
		//	if(!faceFound || (clock()/CLOCKS_PER_SEC*1000 - lastTimeOfUpdate) > (skipAmount+1)){
		// only update after every "skipAmount" of frames and if the face was previously found
		trackUpdate();
		//numberOfFrames=1;
		//		lastTimeOfUpdate=clock()/CLOCKS_PER_SEC*1000;

	} else {
		//when the actuall facetracking is taking a break, the previous data is still smoothend
		updateTrackDataSmoothOnly();
	}
	// remember that a frame has been made
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
	calibrate = true;
}

void FaceSpaceTracker::resetCalibration(){
	// remove x,y z calibration
	calibrationX=0;
	calibrationY=0;
	fixedDepth=false;
}

void FaceSpaceTracker::loadConfigFile(const char* newfile){
	// switch the camera Settings
	configFile = newfile;

	cameraOff(); // free the old camera

	cameraInit(); // initialize with new camera
}

// get the calibrated position
double FaceSpaceTracker::getXHumanPosition(){
	return humanPosition[0];
}

double FaceSpaceTracker::getYHumanPosition(){
	return humanPosition[1];
}

double FaceSpaceTracker::getZHumanPosition(){
	return humanPosition[2];
}

// get the Position as if without calibration of xyz
double FaceSpaceTracker::getAbsoluteXHumanPosition(){
	return previousVectorData[0][0];
}

double FaceSpaceTracker::getAbsoluteYHumanPosition(){
	return previousVectorData[1][0];
}

double FaceSpaceTracker::getAbsoluteZHumanPosition(){
	return previousVectorData[2][0];
}

bool FaceSpaceTracker::getFaceFound(){
	return faceFound;
}

cv::Mat FaceSpaceTracker::getCaputureFrame(){
	// return what the camera sees
	return captureFrame;
}

cv::Mat FaceSpaceTracker::getCroppedCaptureFrame(){
	// return what the camera Trackes

	// update imageMargin
	int imageMargin = lastTrackPixelData[2]*imageMarginRelation;
	int ancorX = lastTrackPixelData[0] - imageMargin;
	int ancorY = lastTrackPixelData[1] - imageMargin;
	int ancorHeight = lastTrackPixelData[2] + 2*imageMargin;
	int ancorWidth = lastTrackPixelData[2] + 2*imageMargin;

	// make sure we do not cross the boarders
	if( ancorX < 0 ) ancorX=0;
	if( ancorY < 0 ) ancorY=0;
	if( ancorX + ancorWidth >= webcamXresolution ) ancorWidth = webcamXresolution - ancorX;
	if( ancorY + ancorHeight >= webcamYresolution ) ancorHeight = webcamYresolution - ancorY;

	return captureFrame( Rect(ancorX , ancorY , ancorWidth , ancorHeight ) );
}

double FaceSpaceTracker::getScreenHeight(){
	return realLifeScreenHeight;
}

double FaceSpaceTracker::getScreenWidth(){
	return realLifeScreenWidth;
}

double FaceSpaceTracker::getFaceHeight(){
	return realLifeFaceHeight;
}

bool FaceSpaceTracker::getFixedDepth(){
	return fixedDepth;
}

bool FaceSpaceTracker::getShowCamFram(){
	return showCamFrame;
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

void FaceSpaceTracker::setCamXResolution(int res){
	webcamXresolution = res;
}

void FaceSpaceTracker::setCamYResolution(int res){
	webcamYresolution = res;
}

void FaceSpaceTracker::setCamVerticalFOV(double fov){
	webcamVerticalFOV = fov;
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

void FaceSpaceTracker::setErrorToleranceXY(double err){
	errorToleranceXY = err;
}

void FaceSpaceTracker::setErrorToleranceDepth(double err){
	errorToleranceDepth = err;
}

void FaceSpaceTracker::setSkipAmount(int nr){
	skipAmount = nr;
}

void FaceSpaceTracker::setFixedDepth(bool set){
	fixedDepth=set;
}

void FaceSpaceTracker::setShowCamFrame(bool show ){
	showCamFrame = show;
}

void FaceSpaceTracker::setCurrentConfigFile(const char* name){
	configFile = name;
}

void FaceSpaceTracker::findFaceAgain(){
	faceFound = false;
}

/* LINE BY LINE STRUCTURE OF CFG:
	realLifeScreenHeight in mm
	realLifeScreenWidth // maybe not used here, but useful on client side
	realLifeFaceHeight
	webcamVerFOV degrees
	previousDataCols int
	previousPixelDataCols
	errorToleranceXY double
	errorToleranceDepth double
	eyeXrelation double
	eyeYrelation double
	skipAmount int
	faceTrackingFile the char* needed to load the .xml

 */

void FaceSpaceTracker::readConfigFile(){
	// all parameters/Global variables are stored in a separate config file to be manipulated by the application

	// open file for input
	ifstream inputFile;
	inputFile.open(configFile);

	string line;
	inputFile >> line >>  line >>  realLifeScreenHeight;
	inputFile >> line >>  line >>  realLifeScreenWidth;
	inputFile >> line >>  line >>  realLifeFaceHeight;
	inputFile >> line >>  line >>  webcamVerticalFOV;
	inputFile >> line >>  line >>  imageMarginRelation;
	inputFile >> line >>  line >>  previousVectorDataCols;
	inputFile >> line >>  line >>  previousPixelDataCols;
	inputFile >> line >>  line >>  errorToleranceXY;
	inputFile >> line >>  line >>  errorToleranceDepth;
	inputFile >> line >>  line >>  eyeXrelation;
	inputFile >> line >>  line >>  eyeYrelation;
	inputFile >> line >>  line >>  skipAmount;
	inputFile >> line >>  line >>  fixedDepth;

	inputFile >> line >> line >> line;
	faceTrackingFile = line.c_str();
	faceTrackingFileName = line;

	setUpSmoothing(); // necessary for dynamic Matrices
	resetSmoothers();

	//close file
	inputFile.close();
}

void FaceSpaceTracker::saveConfigFile(){ // TODO currently not working because of misterious eigen messages sneaking in.
	// open file for output
	ofstream outputfile;
	outputfile.open(configFile);

	// write everything

	outputfile << "realLifeScreenHeight Millimeters= ";
	outputfile << realLifeScreenHeight << endl;

	outputfile << "realLifeScreenWidth Millimeters= ";
	outputfile << realLifeScreenWidth << endl;

	outputfile << "realLifeFaceHeight Millimeters= ";
	outputfile << realLifeFaceHeight<< endl;

	outputfile << "webcamVerticalFOV Degrees= ";
	outputfile << webcamVerticalFOV << endl;

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

	outputfile << "fixedDepth bool= ";
	outputfile << fixedDepth << endl;

	outputfile << "faceTrackingFile Location= ";
	outputfile << faceTrackingFileName << endl;



	//close it
	outputfile.close();
}

void FaceSpaceTracker::reportStatus(){
	cout << " It should be working, hopefully " << endl;
}



//=================================================================================================
//=================================================================================================
//=================================================================================================
/////////////////////// private METHODS /////////////////////////////
//=================================================================================================
//=================================================================================================
//=================================================================================================


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
	faceFound=false; // because we reseted the last pixel data
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
	double eyeX = faceX + faceWidth*eyeXrelation - (double)webcamXresolution/2; // eyeX=.5 for center of eye, .7 for right eye
	double eyeY = faceY + faceHeight*eyeYrelation - (double)webcamYresolution/2;

	// estimate the distance from camera to center of camera plane (only using y hight, we guess the camera does not distort
	// Camera plane := your face plane normal to the webcam's direction
	double distanceToCamPlane = (webcamYresolution*imageScaleFactor/2)/(tan(webcamVerticalFOV*(M_PI/180)/2)); //


	// now estimate the position of your eyes (to the webcam)
	//// -Z is depth, +X is right, +Y is up
	double newData[] = { // WARNING: we are in the webcame Frame coords
			-eyeX*imageScaleFactor, // minus X because webcam image is mirrored @ y-axis
			-eyeY*imageScaleFactor, // I have to flip y to have +y up. maybe because the eye coords origin is top left of webcam frame
			distanceToCamPlane //old: negative sign because i needed to flip it, maybe because of webcamVerFOV
	};

	// NEW TRY, since camera parralel to screen, just shift the result down, cam is 1cm over screen
	//data[1] = data[1] + (realLifeScreenHeight/2+10);


	// smooth out data by extrapolation and update cameraOffset
	//	for(int i=0;i<3;i++) data[i] = trackVectorSmoother(data)[i];
	trackVectorSmoother(newData);

	for(int i=0;i<3;i++) humanPosition[i] = previousVectorData[i][0];

	if(calibrate){ // calibrate the data for the x,y coords, maybe later also the z s.t. the initial FOV is as demanded.
		calibrate = false;
		calibrationX = humanPosition[0];
		calibrationY = humanPosition[1];
		calibrationZ = humanPosition[2];
	}
	// apply calibration
	humanPosition[0] -= calibrationX;
	humanPosition[1] -= calibrationY;

	// fix the depth if necessary
	if(fixedDepth) humanPosition[2]=calibrationZ;

}

void FaceSpaceTracker::croppedFaceTrack(){

	// update imageMargin
	int imageMargin = lastTrackPixelData[2]*imageMarginRelation;
	int ancorX = lastTrackPixelData[0] - imageMargin;
	int ancorY = lastTrackPixelData[1] - imageMargin;
	int ancorHeight = lastTrackPixelData[2] + 2*imageMargin;
	int ancorWidth = lastTrackPixelData[2] + 2*imageMargin;

	// make sure we do not cross the boarders
	if( ancorX < 0 ) ancorX=0;
	if( ancorY < 0 ) ancorY=0;
	if( ancorX + ancorWidth >= webcamXresolution ) ancorWidth = webcamXresolution - ancorX;
	if( ancorY + ancorHeight >= webcamYresolution ) ancorHeight = webcamYresolution - ancorY;

	//convert captured image to gray scale and equalize
	cvtColor(captureFrame( Rect(ancorX , ancorY , ancorWidth , ancorHeight ) ), grayscaleFrame, CV_BGR2GRAY);
	equalizeHist(grayscaleFrame, grayscaleFrame);

	if(showCamFrame){ // show what i am tracking
		imshow("CROP", grayscaleFrame);
		waitKey(1);
	}

	// estimate the size of the face
	double min_face_size = lastTrackPixelData[2]*0.8;
	double max_face_size = lastTrackPixelData[2]*1.2;

	//		clock_t begindetect = clock();
	//		if(!profileFound){ // distinguish between tracking the front or the side of the face
	// mistery note: seting 1.1 to 1.5 make it much faster, but looses faceHeight accuracy. use this if only the x,y data is needed
	face_cascade.detectMultiScale(grayscaleFrame, faces, 1.1, 0, CV_HAAR_FIND_BIGGEST_OBJECT| CV_HAAR_SCALE_IMAGE, Size(min_face_size, min_face_size),Size(max_face_size, max_face_size));

	//			if(faces.size()==0){
	//				profile_cascade.detectMultiScale(grayscaleFrame, faces, 1.1, 0, CV_HAAR_FIND_BIGGEST_OBJECT| CV_HAAR_SCALE_IMAGE, Size(min_face_size, min_face_size),Size(max_face_size, max_face_size));
	//				if(faces.size()==1) profileFound=true;
	//			}
	//		} else {
	//			cout << "profile found" << endl;
	//			profile_cascade.detectMultiScale(grayscaleFrame, faces, 1.1, 0, CV_HAAR_FIND_BIGGEST_OBJECT| CV_HAAR_SCALE_IMAGE, Size(min_face_size, min_face_size),Size(max_face_size, max_face_size));
	//
	//			if(faces.size()==0){
	//				profileFound=false;
	//				face_cascade.detectMultiScale(grayscaleFrame, faces, 1.1, 0, CV_HAAR_FIND_BIGGEST_OBJECT| CV_HAAR_SCALE_IMAGE, Size(min_face_size, min_face_size),Size(max_face_size, max_face_size));
	//			}
	//		}

	if(faces.size()==1) { // now recalibrate the data back to the big image frame
		faces[0].x += ancorX;
		faces[0].y += ancorY;
	} else { // no face has been found, track all again
		faceFound=false;
	}
}

void FaceSpaceTracker::fullFaceTrack(){

	//convert captured image to gray scale and equalize
	cvtColor(captureFrame, grayscaleFrame, CV_BGR2GRAY);
	equalizeHist(grayscaleFrame, grayscaleFrame);
	//find faces and store them in the vector array, BIGGEST OBJECT makes sure we only have one!
	face_cascade.detectMultiScale(grayscaleFrame, faces, 1.1, 2, CV_HAAR_FIND_BIGGEST_OBJECT| CV_HAAR_SCALE_IMAGE, Size(30,30));

}


void FaceSpaceTracker::faceTrack(){

	//capture a new image frame, this is the image in the showwebcam, the grayscale is used for face detection
	captureDevice>>captureFrame;

	if(faceFound){ // if true, work on a cropped version for performance boost

		croppedFaceTrack();

	} // endif faceFound

	if(!faceFound){ // find the face again using the entire image

		fullFaceTrack();
	}

	//		clock_t end = clock();
	//	cout << ceil(double(end - begin)*1000/ CLOCKS_PER_SEC) << " msec = time per facedetection"<<endl;

	if(showCamFrame){
		//draw a rectangle for all found faces in the vector array on the original image
		if(faces.size()==1)	{
			Point pt1(faces[0].x + faces[0].width, faces[0].y + faces[0].height);
			Point pt2(faces[0].x, faces[0].y);
			int rec2size = 8;
			Point pt1eye(faces[0].x + faces[0].width*eyeXrelation -rec2size/2, faces[0].y + faces[0].height*eyeYrelation - rec2size/2);
			Point pt2eye(faces[0].x + faces[0].width*eyeXrelation +rec2size/2, faces[0].y + faces[0].height*eyeYrelation + rec2size/2);
			int imageMargin = lastTrackPixelData[2]*imageMarginRelation;
			Point pt1track(faces[0].x + faces[0].width + imageMargin, faces[0].y + faces[0].height + imageMargin);
			Point pt2track(faces[0].x - imageMargin, faces[0].y - imageMargin);

			rectangle(captureFrame, pt1, pt2, cvScalar(0, 255, 0, 0), 1, 8, 0);
			rectangle(captureFrame, pt1eye, pt2eye, cvScalar(0,0,0, 0), 3, 8, 0);
			rectangle(captureFrame, pt1track, pt2track, cvScalar(0,0,255, 0), 3, 8, 0);
		}
		//show the output
		imshow("outputCapture", captureFrame);
		//pause for 33ms, i replaced it with 1 ms, without it it wont show the image
		waitKey(1);
	}
}

void FaceSpaceTracker::trackUpdate(){



	if(!smoothonly){ // if true, track using opencv

		faceTrack();

	} else {  // if only smoothing, pass previous info for smoothening, future idea: predict movement
		faces[0].x=lastTrackPixelData[0];
		faces[0].y=lastTrackPixelData[1];
		faces[0].height=lastTrackPixelData[2];
	}// end smoothonly



	//if tracking found, read and interpret it
	if(faces.size()==1) {
		faceFound=true;
		// update last actual position
		lastTrackPixelData[0] = faces[0].x;
		lastTrackPixelData[1] = faces[0].y;
		lastTrackPixelData[2] = faces[0].height;

		trackFilter(faces[0].x, faces[0].y, faces[0].height);

	}else{
		// if no face detected, remember to try the entire image again!
		faceFound = false;
	}

}




