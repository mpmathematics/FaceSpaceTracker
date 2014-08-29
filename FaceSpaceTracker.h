/*
 * FaceSpaceTracker.h
 *
 *  Created on: Jul 31, 2014
 *      Author: internshipdude
 */



#ifndef FACESPACETRACKER_H_
#define FACESPACETRACKER_H_

// I have to include them here
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//namespace FreeCamFaceTrack {

class FaceSpaceTracker {



private: // all variables private, they are all specified in the cfg file or

	// opencv headtracking init
	cv::CascadeClassifier face_cascade;
	//	cv::CascadeClassifier profile_cascade;
	cv::VideoCapture captureDevice;
	cv::Mat captureFrame;
	cv::Mat grayscaleFrame;
	std::vector<cv::Rect> faces; //create a vector array to store the face found

	double humanPosition[3]; // this is the tracker data being streamed,

	// sizes in mm
	double realLifeScreenHeight;
	double realLifeScreenWidth; // unused here, will be needed for perspective distortion
	double realLifeFaceHeight; // regarding the rectangle in the videostream

	// web cam tech specs
	int webcamXresolution;
	int webcamYresolution;
	double webcamVerticalFOV; // wikipedia kinect

	// track details
	bool faceFound; // Used for reset "tracking" without recalibrating, can be read by client
	bool profileFound; // used to Track the side of a face
	double imageMarginRelation; //image margin for tracking, the ratio of the facesize to be considered in the next frame

	// SMOOTHENING Iterations is the nr. of cols in this verctor: NEWEST VECTOR TO index 0
	int previousVectorDataCols;
	int previousPixelDataCols;
	std::vector< std::vector<double> > previousVectorData;
	std::vector< std::vector<double> > previousPixelData;
	double lastTrackPixelData[3];
	double errorToleranceXY;
	double errorToleranceDepth;

	// relative eye position to the facewidth/height
	double eyeXrelation;
	double eyeYrelation;

	// calibration
	bool calibrate;
	double calibrationX;
	double calibrationY;
	double calibrationZ;
	bool fixedDepth; // a depth that can be set or calibrated

	// performance
	int skipAmount; // number of frames to skip tracking and only smooth data. heavy performance gain
	int numberOfFrames; // updated in updateTrackData
	double lastTimeOfUpdate;


	// booleans
	bool showCamFrame;
	bool smoothonly;

	// location of files:
	const char* configFile;
	const char* faceTrackingFile;
	std::string faceTrackingFileName; // needed because loading the cascade destroys the const char*

	//	const char* profileTrackingFile; // side of face

	//
	//	// extra
	//	int framecounter=0;
	//	double lengthConversionScale = 0.0393701; // mm to inch
	//	bool convertCoordinates = false;


	// method declarations
public:

	// the consstructor
	FaceSpaceTracker();
	FaceSpaceTracker(const char*);

	// the main thing for the user
	void updateTrackData();
	void updateTrackDataSmoothOnly();

	void cameraInit();
	void cameraOff();
	void reportStatus();
	void calibrateCenter();
	void resetCalibration();
	void saveConfigFile();
	void readConfigFile();
	void loadConfigFile(const char*);

	//essential get methods
	double getXHumanPosition();
	double getYHumanPosition();
	double getZHumanPosition();

	// other get methods
	double getAbsoluteXHumanPosition();
	double getAbsoluteYHumanPosition();
	double getAbsoluteZHumanPosition();
	bool getFaceFound();
	cv::Mat getCaputureFrame();
	cv::Mat getCroppedCaptureFrame();
	double getScreenHeight();
	double getScreenWidth();
	double getFaceHeight();
	bool getFixedDepth();
	bool getShowCamFram();

	//set methods
	void setScreenHeigth(double);
	void setScreenWidth(double);
	void setFaceHeigth(double);
	void setCamXResolution(int);
	void setCamYResolution(int);
	void setCamVerticalFOV(double);
	void setImageTrackMarginRelation(double);
	void setTrackDataSmoothingSteps(int);
	void setTrackPixelDataSmoothingSteps(int);
	void setErrorToleranceXY(double);
	void setErrorToleranceDepth(double);
	void setSkipAmount(int);
	void setFixedDepth(bool);
	void setShowCamFrame(bool);
	void setCurrentConfigFile(const char*);
	void findFaceAgain();



private:

	void setUpSmoothing();
	void resetSmoothers();
	void trackPixelDataSmoother(int , int , int);
	void trackVectorSmoother(double*);
	void trackFilter(double , double , double );
	void croppedFaceTrack();
	void fullFaceTrack();
	void faceTrack();
	void trackUpdate();

	//public:
	//	FaceSpaceTracker();
	//	virtual ~FaceSpaceTracker();
};

//} /* namespace FaceSpaceTracker */

#endif /* FACESPACETRACKER_H_ */
