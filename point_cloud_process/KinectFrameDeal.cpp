#include "KinectFrameDeal.h"
#include "Kinect.h"
#include <iostream>
#include <fstream>
#include <glut.h>
#include <vector>
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <time.h>
#include <stdio.h>
#include <Kinect.Face.h>
#include <windows.h>
#include <mmsystem.h>
#pragma comment(lib, "winmm.lib")

using namespace std;
using namespace cv;

IMultiSourceFrameReader		*pMultiReader;
ICoordinateMapper			*pCoorMapper;
UINT16						*depthValue = new UINT16[512 * 424];
DepthSpacePoint				*pDepthSpacePoint = new DepthSpacePoint[1920 * 1080];
ColorSpacePoint				*pColorSpacePoint = new ColorSpacePoint[1920 * 1080];
RGBQUAD						*colorShow = new RGBQUAD[1920 * 1080];
CameraSpacePoint			*pCameraSpacePoints = new CameraSpacePoint[512 * 424];
ColorSpacePoint				*pColorSpacePoints = new ColorSpacePoint[512 * 424];
PointCloud					*pPointCloud = new PointCloud[512 * 424];
Joint						joints[JointType_Count];//3D joint points coordination
ColorSpacePoint				joints_position[JointType_Count];//All joint points coordination
String joints_name[] = { "SpineBase","SpineMid","Neck","Head","ShoulderLeft","ElbowLeft","WristLeft","HandLeft","ShoulderRight","ElbowRight","WristRight","HandRight","Hipleft","KneeLeft","AnkleLeft","FootLeft","HipRight","KneeRight","AnkleRight","FootRight","SpineShoulder","HandTipLeft","ThumbLeft","HandHipRight","ThumbRight" };
static GLfloat xRot = 0.0f;
static GLfloat yRot = 0.0f;

vector<CameraSpacePoint> pointsCloud;
vector<RGBQUAD> color_all;
vector<CameraSpacePoint> pointsCloudBody;
vector<RGBQUAD> color_body;

KinectFrameDeal::KinectFrameDeal()
{

	pkinectSensor = NULL;
	pMultiReader = NULL;
	pCoorMapper = NULL;

	c_nColorWidth = 1920;
	c_nColorHeight = 1080;
	//c_pColorBuffer = new RGBQUAD[c_nColorWidth * c_nColorHeight];

}


KinectFrameDeal::~KinectFrameDeal()
{
	if (pMultiReader) {
		SafeRelease(pMultiReader);
	}
	if (pCoorMapper) {
		SafeRelease(pCoorMapper);
	}
	if (pkinectSensor) {
		pkinectSensor->Close();
		SafeRelease(pkinectSensor);
	}


}

//
//Init pkinectSensor,pMultiReader,pMapper
//
HRESULT KinectFrameDeal::initiate(int argc, char **argv)
{
	HRESULT hr = GetDefaultKinectSensor(&pkinectSensor);

	if (FAILED(hr))
	{
		cout << "Get Kinect Failed." << endl;
		return hr;
	}

	if (pkinectSensor)
	{
		hr = pkinectSensor->get_CoordinateMapper(&pCoorMapper);
		if (FAILED(hr))
		{
			cout << "Get Coordinate Mapper Failed." << endl;
			return hr;
		}

		hr = pkinectSensor->Open();
		if (FAILED(hr))
		{
			cout << "Open Kinect Failed." << endl;
			return hr;
		}


		hr = pkinectSensor->OpenMultiSourceFrameReader(FrameSourceTypes_Color | FrameSourceTypes_Depth, &pMultiReader);
		//hr = pkinectSensor->OpenMultiSourceFrameReader(FrameSourceTypes_Body | FrameSourceTypes_Color | FrameSourceTypes_Depth | FrameSourceTypes_BodyIndex, &pMultiReader);
		if (FAILED(hr))
		{
			cout << "Open MultiSourceFrameReader Failed." << endl;
			return hr;
		}
	}
	if (SUCCEEDED(hr)) {
		//Initialize OpenGL 
		//Set GLUT parameter
		imageDraw = new ImageDraw(argc, argv);
		imageDraw->initDisplayMode(GLUT_DOUBLE | GLUT_RGB);
		imageDraw->initWindowPosition(200, 100);
		imageDraw->initWindowSize(960, 540);
		imageDraw->setWindowTitle("Kinect Color");
		imageDraw->setClearColor(0, 0, 0, 0);
		imageDraw->setMatrixMode(GL_MODELVIEW);
		//ImageDraw->setOrtho2D(0, 960, 0, 540);
		glOrtho(0, 1920, 0, 1080, 0, 0);

	}

	return hr;
}

//Rendering data
void KinectFrameDeal::excuteDisplyFunc() {
	//void ChangeSize(int width, int height);
	//void SpecialKeys(int key, int x, int y);
	imageDraw->displayFunc(FrameDataDeal);

	//glutReshapeFunc(ChangeSize);
	//glutSpecialFunc(SpecialKeys);
	glutMainLoop();
}

void FrameDataDeal(void)
{

	RGBQUAD *c_pColorBuffer = new RGBQUAD[1920 * 1080];
	HRESULT hr = 0;
	int frameNum = 0;


	IMultiSourceFrame	*pMultiSourceFrame = nullptr;
	IDepthFrame			*pDepthFrame = nullptr;
	IColorFrame			*pColorFrame = nullptr;
	//IBodyFrame *pBodyFrame = NULL;
	IBodyIndexFrame		*pBodyIndexFrame = nullptr;
	IBodyFrame			*pBodyFrame = nullptr;

	cout << "while pre" << endl;
	cout << "finished!" << endl;
	while (1)
	{
		
		if (!pMultiReader) {

			return;
		}
		SafeRelease(pMultiSourceFrame);
		SafeRelease(pDepthFrame);
		SafeRelease(pColorFrame);
		SafeRelease(pBodyFrame);
		SafeRelease(pBodyIndexFrame);

		//IMultiSourceFrame	*pMultiSourceFrame = NULL;
		//IDepthFrame			*pDepthFrame = NULL;
		//IColorFrame			*pColorFrame = NULL;
		////IBodyFrame *pBodyFrame = NULL;
		//IBodyIndexFrame		*pBodyIndexFrame = NULL;
		//IBodyFrame			*pBodyFrame = NULL;

		hr = pMultiReader->AcquireLatestFrame(&pMultiSourceFrame);

		//Get Depth Frame
		if (SUCCEEDED(hr)) {
			frameNum++;
			cout << "loop number£º" << frameNum << endl;

			IDepthFrameReference *pDepthFrameReference = NULL;
			hr = pMultiSourceFrame->get_DepthFrameReference(&pDepthFrameReference);
			if (SUCCEEDED(hr)) {
				hr = pDepthFrameReference->AcquireFrame(&pDepthFrame);
			}
			SafeRelease(pDepthFrameReference);
		}

		//Get Color Frame
		if (SUCCEEDED(hr)) {
			IColorFrameReference *pColorFrameReference = NULL;
			hr = pMultiSourceFrame->get_ColorFrameReference(&pColorFrameReference);
			if (SUCCEEDED(hr)) {
				hr = pColorFrameReference->AcquireFrame(&pColorFrame);
			}
			SafeRelease(pColorFrameReference);
		}

		//Get Body_Index Frame
		/*if (SUCCEEDED(hr)) {
		IBodyIndexFrameReference *pBodyIndexFrameReference = NULL;
		hr = pMultiSourceFrame->get_BodyIndexFrameReference(&pBodyIndexFrameReference);
		if (SUCCEEDED(hr)) {
		hr = pBodyIndexFrameReference->AcquireFrame(&pBodyIndexFrame);
		}
		SafeRelease(pBodyIndexFrameReference);
		}*/

		//Get Body Frame
		/*if (SUCCEEDED(hr)) {
		IBodyFrameReference *pBodyFrameReference = NULL;
		hr = pMultiSourceFrame->get_BodyFrameReference(&pBodyFrameReference);
		if (SUCCEEDED(hr)) {
		hr = pBodyFrameReference->AcquireFrame(&pBodyFrame);
		}
		SafeRelease(pBodyFrameReference);
		}*/

		SafeRelease(pMultiSourceFrame);

		if (SUCCEEDED(hr)) {
			IFrameDescription *pColorFrameDescription = NULL;
			int nColorWidth = 0;
			int nColorHeight = 0;
			ColorImageFormat imageFormat = ColorImageFormat_None;
			UINT nColorBufferSize = 0;
			RGBQUAD *pColorBuffer = NULL;

			IFrameDescription *pDepthFrameDescription = NULL;
			int nDepthWidth = 0;
			int nDepthHeight = 0;
			USHORT nMinDepth = 0;
			USHORT nMaxDepth = 0;
			UINT nDepthBufferSize = 0;
			UINT16 *pDepthBuffer = NULL;

			UINT nBodyIndexSize = 0;
			BYTE *pBodyIndexBuffer = NULL;

			IBody *pBodys[BODY_COUNT] = { 0 };


			//Get color data
			if (SUCCEEDED(hr)) {
				hr = pColorFrame->get_FrameDescription(&pColorFrameDescription);
			}
			if (SUCCEEDED(hr)) {
				hr = pColorFrameDescription->get_Width(&nColorWidth);
			}
			if (SUCCEEDED(hr)) {
				hr = pColorFrameDescription->get_Height(&nColorHeight);
				SafeRelease(pColorFrameDescription);//release color frame description
			}
			if (SUCCEEDED(hr)) {
				hr = pColorFrame->get_RawColorImageFormat(&imageFormat);
			}
			if (SUCCEEDED(hr)) {
				if (imageFormat == ColorImageFormat_Bgra) {
					hr = pColorFrame->AccessRawUnderlyingBuffer(&nColorBufferSize, reinterpret_cast<BYTE**>(pColorBuffer));
				}
				else {
					nColorBufferSize = nColorWidth * nColorHeight * sizeof(RGBQUAD);
					pColorBuffer = c_pColorBuffer;
					hr = pColorFrame->CopyConvertedFrameDataToArray(nColorBufferSize, reinterpret_cast<BYTE*>(pColorBuffer), ColorImageFormat_Bgra);
				}
			}


			//Get depth data
			if (SUCCEEDED(hr)) {
				hr = pDepthFrame->get_FrameDescription(&pDepthFrameDescription);
			}
			if (SUCCEEDED(hr)) {
				hr = pDepthFrameDescription->get_Width(&nDepthWidth);
			}
			if (SUCCEEDED(hr)) {
				hr = pDepthFrameDescription->get_Height(&nDepthHeight);
				SafeRelease(pDepthFrameDescription);
			}
			if (SUCCEEDED(hr)) {
				hr = pDepthFrame->get_DepthMinReliableDistance(&nMinDepth);
			}
			if (SUCCEEDED(hr)) {
				hr = pDepthFrame->get_DepthMaxReliableDistance(&nMaxDepth);
			}
			if (SUCCEEDED(hr)) {
				hr = pDepthFrame->AccessUnderlyingBuffer(&nDepthBufferSize, &pDepthBuffer);
			}

			//Get Body Index Data
			/*if (SUCCEEDED(hr)) {
			hr = pBodyIndexFrame->AccessUnderlyingBuffer(&nBodyIndexSize, &pBodyIndexBuffer);
			}*/

			//Get Body Data
			/*if (SUCCEEDED(hr)) {
			hr = pBodyFrame->GetAndRefreshBodyData(BODY_COUNT, pBodys);
			}*/

			//Color camera internal parameter
			double inParC[3][3] = { { 1046.68135878074, 0, 0 },
			{ 0, 1045.43763045605, 0 },
			{ 951.379989377149, 520.812624902706, 1 } };
			//Inverse matrix of depth camera internal parameter
			double inParD_inv[3][3] = { { 0.00276484676324564, 0, 0 },
			{ 0, 0.00277213247129969, 0 },
			{ -0.714294896225351, -0.582252303092675, 1 } };
			//Relative position between color and depth camera, describe with rotation and translation
			double R[3][3] = { { 0.999999104003479, 0.00133749506823724, 5.56702949213980e-05 },
			{ -0.00133732781037416, 0.999994891406734, -0.00290322488993331 },
			{ -5.95530594968121e-05, 0.00290314783922029, 0.999995784084141 } };
			double T[3] = { 51.4260581577147,0.121582060026121,1.32480065752075 };

			ostringstream path;
			//Image name index
			int num = 2004;
			//Read segment-image and source
			for (int i = 1; i <= 1; i++) {
				path.str("");
				path << "..\\data\\img_depth_" << num << ".tif";
				Mat depth1 = imread(path.str(), IMREAD_ANYDEPTH);
				Mat depth;
				flip(depth1, depth, 1);
				pDepthBuffer = (UINT16*)depth.data;

				path.str("");
				path << "..\\data\\img_" << num << "_segmentation.png";
				Mat colorseg = imread(path.str());
				if (colorseg.data == NULL)continue;
				flip(colorseg, colorseg, 1);
				path.str("");
				path << "..\\data\\img_" << num << ".jpg";
				Mat color = imread(path.str());
				flip(color, color, 1);
				int m = 0, n = 0;
				RGBQUAD *pColorSegBuffer = new RGBQUAD[1920 * 1080];
				for (int i = 0; i < 1920 * 1080; i++) {
					pColorSegBuffer[i].rgbBlue = colorseg.data[m];
					pColorSegBuffer[i].rgbGreen = colorseg.data[++m];
					pColorSegBuffer[i].rgbRed = colorseg.data[++m];
					pColorBuffer[i].rgbBlue = color.data[n];
					pColorBuffer[i].rgbGreen = color.data[++n];
					pColorBuffer[i].rgbRed = color.data[++n];
					++n; ++m;
				}

				pointsCloud.clear();
				color_all.clear();
			
				
				Mat depth_img1(424, 512, CV_16UC1, pDepthBuffer);
				hr = pCoorMapper->MapColorFrameToDepthSpace(512 * 424, (UINT16*)pDepthBuffer, 1920 * 1080, pDepthSpacePoint);
				hr = pCoorMapper->MapDepthFrameToCameraSpace(512 * 424, (UINT16*)pDepthBuffer, 512 * 424, pCameraSpacePoints);
				if (SUCCEEDED(hr))
				{
					// Loop over output pixels
					RGBQUAD* pSrc = colorShow;
					for (int colorIndex = 0; colorIndex < 1920 * 1080; ++colorIndex)
					{
						if (colorIndex == 0)
							cout << "loop" << endl;
						// Default setting source to copy from the background pixel

						DepthSpacePoint p = pDepthSpacePoint[colorIndex];

						// Values that are negative infinity means it is an invalid color to depth mapping so we
						// Skip processing for this pixel
						if (p.X != -std::numeric_limits<float>::infinity() && p.Y != -std::numeric_limits<float>::infinity())
						{

							int depthX = static_cast<int>(p.X + 0.5f);
							int depthY = static_cast<int>(p.Y + 0.5f);

							if ((depthX >= 0 && depthX < 512) && (depthY >= 0 && depthY < 424))
							{
								RGBQUAD tmp = *(pColorSegBuffer + colorIndex);
								//Select a valid depth value based on color
								if(!(tmp.rgbBlue == 0 && tmp.rgbGreen == 0 && tmp.rgbRed == 0))
								{
									if (pCameraSpacePoints[depthX + depthY * 512].X != -std::numeric_limits<float>::infinity())
									{
										// Set source for copy to the color pixel
										*pSrc = *(pColorBuffer + colorIndex);
										color_all.push_back(*pSrc);
										pointsCloud.push_back(pCameraSpacePoints[depthX + depthY * 512]);
									}
									
								}
							}

						}
						pSrc++;
					}
				}
				
				//Write to .ply file
				cout << "output" << endl;
				ofstream ply;
				string path = "..\\code\\fitted_ model_";
				char fileName[15];
				i = 81;
				sprintf_s(fileName, "%d%s", num, "n.ply");
				string file = path + fileName;
				cout << file.c_str() << endl;
				ply.open(file.c_str());
				ply << "ply" << endl;
				ply << "format ascii 1.0" << endl;
				ply << "element vertex " << color_all.size() << endl;
				ply << "property float x" << endl;
				ply << "property float y" << endl;
				ply << "property float z" << endl;
				ply << "property uchar red" << endl;
				ply << "property uchar green" << endl;
				ply << "property uchar blue" << endl;
				ply << "end_header" << endl;
				for (int i = 0; i < color_all.size(); i++) {
					ply << pointsCloud[i].X << " " << pointsCloud[i].Y << " " << pointsCloud[i].Z << " " << (int)color_all[i].rgbRed << " " << (int)color_all[i].rgbGreen << " " << (int)color_all[i].rgbBlue << endl;
				}

				ply.close();
			}

			/////
			if (SUCCEEDED(hr) && pDepthBuffer) {
				cout << "in progress" << endl;


				//hr = pCoorMapper->MapDepthFrameToCameraSpace(nDepthWidth*nDepthHeight, (UINT16*)pDepthBuffer, nDepthWidth*nDepthHeight, pCameraSpacePoints);


			}

			//Display color image
			glPixelZoom(0.5f, -0.5f);
			glRasterPos2i(-1, 1);
			glDrawPixels(nColorWidth, nColorHeight, GL_BGRA_EXT, GL_UNSIGNED_BYTE, pColorBuffer);
			//glPopMatrix();
			glutSwapBuffers();
			////Sleep(2000);

			if (false) {



				//Save depth value
				/*ostringstream depthValuePath;
				depthValuePath << "E:\\data\\depth_value_" << frameNum << ".txt";
				ofstream writeDepth(depthValuePath.str());
				for (int i = 0; i < nDepthWidth*nDepthHeight; i++) {
				writeDepth << pDepthBuffer[i];
				if (i != nDepthWidth*nDepthHeight - 1)
				writeDepth << endl;
				}
				writeDepth.close();*/

				Mat depth_img(nDepthHeight, nDepthWidth, CV_16UC1, pDepthBuffer);
				ostringstream depth_img_path;
				depth_img_path << "E:\\data\\depth_" << frameNum << ".tif";
				cout << "save image£º" << imwrite(depth_img_path.str(), depth_img) << endl;
				//Use opencv to save image


				Mat img(nColorHeight, nColorWidth, CV_8UC4, pColorBuffer);
				Mat img_flip;
				flip(img, img_flip, 1);
				ostringstream img_path;
				img_path << "E:\\data\\img_" << frameNum << ".jpg";
				cout << "save image£º" << imwrite(img_path.str(), img) << endl;

			}

		}

		SafeRelease(pBodyFrame);
		SafeRelease(pBodyIndexFrame);
		SafeRelease(pDepthFrame);
		SafeRelease(pColorFrame);
		SafeRelease(pMultiSourceFrame);
	}

}



void ChangeSize(GLsizei w, GLsizei h)
{
	GLfloat nRange = 10.0f;

	// Prevent a divide by zero

	if (h == 0)
		h = 1;

	// Set Viewport to window dimensions

	glViewport(0, 0, 512, 424);

	// Reset projection matrix stack

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	// Establish clipping volume (left, right, bottom, top, near, far)

	if (w <= h)
		glOrtho(-nRange, nRange, -nRange*h / w, nRange*h / w, -nRange, nRange);
	else
		//glOrtho(-nRange*w / h, nRange*w / h, -nRange, nRange, -nRange, nRange);
		glOrtho(0, 512, 0, 424, -500, 0);

	// Reset Model view matrix stack

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}
void SpecialKeys(int key, int x, int y)
{
	if (key == GLUT_KEY_UP)
		xRot -= 5.0f;

	if (key == GLUT_KEY_DOWN)
		xRot += 5.0f;

	if (key == GLUT_KEY_LEFT)
		yRot -= 5.0f;

	if (key == GLUT_KEY_RIGHT)
		yRot += 5.0f;

	if (xRot > 356.0f)
		xRot = 0.0f;

	if (xRot < -1.0f)
		xRot = 355.0f;

	if (yRot > 356.0f)
		yRot = 0.0f;

	if (yRot < -1.0f)
		yRot = 355.0f;

	// Refresh the Window

	glutPostRedisplay();// This will refresh the window, so, it works the same to call RenderScene() directly 

}
