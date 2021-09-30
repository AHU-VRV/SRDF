//------------------------------------------------------------------------------
// <copyright file="FaceBasics.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

#include "stdafx.h"
#include <strsafe.h>
#include "resource.h"
#include "FaceBasics.h"
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <windows.h>
#include <mmsystem.h>
#include <iostream>
#include <fstream>


#pragma comment(lib, "winmm.lib")

using namespace cv;

String joints_name[] = { "SpineBase","SpineMid","Neck","Head","ShoulderLeft","ElbowLeft","WristLeft","HandLeft","ShoulderRight","ElbowRight","WristRight","HandRight","Hipleft","KneeLeft","AnkleLeft","FootLeft","HipRight","KneeRight","AnkleRight","FootRight","SpineShoulder","HandTipLeft","ThumbLeft","HandHipRight","ThumbRight" };

// Face property text layout offset in X axis
static const float c_FaceTextLayoutOffsetX = -0.1f;

// Face property text layout offset in Y axis
static const float c_FaceTextLayoutOffsetY = -0.125f;

// Fefine the face frame features required to be computed by this application
static const DWORD c_FaceFrameFeatures = 
    FaceFrameFeatures::FaceFrameFeatures_BoundingBoxInColorSpace
    | FaceFrameFeatures::FaceFrameFeatures_PointsInColorSpace
    | FaceFrameFeatures::FaceFrameFeatures_RotationOrientation
    | FaceFrameFeatures::FaceFrameFeatures_Happy
    | FaceFrameFeatures::FaceFrameFeatures_RightEyeClosed
    | FaceFrameFeatures::FaceFrameFeatures_LeftEyeClosed
    | FaceFrameFeatures::FaceFrameFeatures_MouthOpen
    | FaceFrameFeatures::FaceFrameFeatures_MouthMoved
    | FaceFrameFeatures::FaceFrameFeatures_LookingAway
    | FaceFrameFeatures::FaceFrameFeatures_Glasses
    | FaceFrameFeatures::FaceFrameFeatures_FaceEngagement;

/// <summary>
/// Entry point for the application
/// </summary>
/// <param name="hInstance">handle to the application instance</param>
/// <param name="hPrevInstance">always 0</param>
/// <param name="lpCmdLine">command line arguments</param>
/// <param name="nCmdShow">whether to display minimized, maximized, or normally</param>
/// <returns>status</returns>
int APIENTRY wWinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPWSTR lpCmdLine, _In_ int nCmdShow)
{
    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(lpCmdLine);

    CFaceBasics application;
    application.Run(hInstance, nCmdShow);
}

/// <summary>
/// Constructor
/// </summary>
CFaceBasics::CFaceBasics() :
    m_hWnd(NULL),
    m_nStartTime(0),
    m_nLastCounter(0),
    m_nFramesSinceUpdate(0),
    m_fFreq(0),
    m_nNextStatusTime(0),
    m_pKinectSensor(nullptr),
    m_pCoordinateMapper(nullptr),
    m_pColorFrameReader(nullptr),
    m_pD2DFactory(nullptr),
    m_pDrawDataStreams(nullptr),
    m_pColorRGBX(nullptr),
    m_pBodyFrameReader(nullptr),
	pMultiReader(nullptr)
{
    LARGE_INTEGER qpf = {0};
    if (QueryPerformanceFrequency(&qpf))
    {
        m_fFreq = double(qpf.QuadPart);
    }

    for (int i = 0; i < BODY_COUNT; i++)
    {
        m_pFaceFrameSources[i] = nullptr;
        m_pFaceFrameReaders[i] = nullptr;
    }

    // Create heap storage for color pixel data in RGBX format
    m_pColorRGBX = new RGBQUAD[cColorWidth * cColorHeight];
}


/// <summary>
/// Destructor
/// </summary>
CFaceBasics::~CFaceBasics()
{
    // Clean up Direct2D renderer
    if (m_pDrawDataStreams)
    {
        delete m_pDrawDataStreams;
        m_pDrawDataStreams = nullptr;
    }

    if (m_pColorRGBX)
    {
        delete [] m_pColorRGBX;
        m_pColorRGBX = nullptr;
    }

    // Clean up Direct2D
    SafeRelease(m_pD2DFactory);

    // Done with face sources and readers
    for (int i = 0; i < BODY_COUNT; i++)
    {
        SafeRelease(m_pFaceFrameSources[i]);
        SafeRelease(m_pFaceFrameReaders[i]);		
    }

    // Done with body frame reader
    SafeRelease(m_pBodyFrameReader);

    // Done with color frame reader
    SafeRelease(m_pColorFrameReader);

	SafeRelease(pMultiReader);

    // Done with coordinate mapper
    SafeRelease(m_pCoordinateMapper);

    // Close the Kinect Sensor
    if (m_pKinectSensor)
    {
        m_pKinectSensor->Close();
    }

    SafeRelease(m_pKinectSensor);
}

/// <summary>
/// Creates the main window and begins processing
/// </summary>
/// <param name="hInstance">handle to the application instance</param>
/// <param name="nCmdShow">whether to display minimized, maximized, or normally</param>
int CFaceBasics::Run(HINSTANCE hInstance, int nCmdShow)
{
    MSG       msg = {0};
    WNDCLASS  wc;

    // Dialog custom window class
    ZeroMemory(&wc, sizeof(wc));
    wc.style         = CS_HREDRAW | CS_VREDRAW;
    wc.cbWndExtra    = DLGWINDOWEXTRA;
    wc.hCursor       = LoadCursorW(NULL, IDC_ARROW);
    wc.hIcon         = LoadIconW(hInstance, MAKEINTRESOURCE(IDI_APP));
    wc.lpfnWndProc   = DefDlgProcW;
    wc.lpszClassName = L"FaceBasicsAppDlgWndClass";

    if (!RegisterClassW(&wc))
    {
        return 0;
    }

    // Create main application window
    HWND hWndApp = CreateDialogParamW(
        NULL,
        MAKEINTRESOURCE(IDD_APP),
        NULL,
        (DLGPROC)CFaceBasics::MessageRouter, 
        reinterpret_cast<LPARAM>(this));

    // Show window
    ShowWindow(hWndApp, nCmdShow);

    // Main message loop
	int frameNum = 0;
    while (WM_QUIT != msg.message)
    {
        Update(&frameNum);

        while (PeekMessageW(&msg, NULL, 0, 0, PM_REMOVE))
        {
            // If a dialog message will be taken care of by the dialog proc
            if (hWndApp && IsDialogMessageW(hWndApp, &msg))
            {
                continue;
            }

            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }

    return static_cast<int>(msg.wParam);
}

/// <summary>
/// Handles window messages, passes most to the class instance to handle
/// </summary>
/// <param name="hWnd">window message is for</param>
/// <param name="uMsg">message</param>
/// <param name="wParam">message data</param>
/// <param name="lParam">additional message data</param>
/// <returns>result of message processing</returns>
LRESULT CALLBACK CFaceBasics::MessageRouter(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    CFaceBasics* pThis = nullptr;

    if (WM_INITDIALOG == uMsg)
    {
        pThis = reinterpret_cast<CFaceBasics*>(lParam);
        SetWindowLongPtr(hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pThis));
    }
    else
    {
        pThis = reinterpret_cast<CFaceBasics*>(::GetWindowLongPtr(hWnd, GWLP_USERDATA));
    }

    if (pThis)
    {
        return pThis->DlgProc(hWnd, uMsg, wParam, lParam);
    }

    return 0;
}

/// <summary>
/// Handle windows messages for the class instance
/// </summary>
/// <param name="hWnd">window message is for</param>
/// <param name="uMsg">message</param>
/// <param name="wParam">message data</param>
/// <param name="lParam">additional message data</param>
/// <returns>result of message processing</returns>
LRESULT CALLBACK CFaceBasics::DlgProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    UNREFERENCED_PARAMETER(wParam);
    UNREFERENCED_PARAMETER(lParam);

    switch (message)
    {
    case WM_INITDIALOG:
        {
            // Bind application window handle
            m_hWnd = hWnd;

            // Init Direct2D
            D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &m_pD2DFactory);

            // Create and initialize a new Direct2D image renderer (take a look at ImageRenderer.h)
            // We'll use this to draw the data we receive from the Kinect to the screen
            m_pDrawDataStreams = new ImageRenderer();
            HRESULT hr = m_pDrawDataStreams->Initialize(GetDlgItem(m_hWnd, IDC_VIDEOVIEW), m_pD2DFactory, cColorWidth, cColorHeight, cColorWidth * sizeof(RGBQUAD)); 
            if (FAILED(hr))
            {
                SetStatusMessage(L"Failed to initialize the Direct2D draw device.", 10000, true);
            }

            // Get and initialize the default Kinect sensor
            InitializeDefaultSensor();
        }
        break;

        // If the titlebar X is clicked, destroy app
    case WM_CLOSE:
        DestroyWindow(hWnd);
        break;

    case WM_DESTROY:
        // Quit the main message pump
        PostQuitMessage(0);
        break;        
    }

    return FALSE;
}

/// <summary>
/// Initializes the default Kinect sensor
/// </summary>
/// <returns>S_OK on success else the failure code</returns>
HRESULT CFaceBasics::InitializeDefaultSensor()
{
    HRESULT hr;

    hr = GetDefaultKinectSensor(&m_pKinectSensor);
    if (FAILED(hr))
    {
        return hr;
    }

    if (m_pKinectSensor)
    {

        hr = m_pKinectSensor->Open();

        if (SUCCEEDED(hr))
        {
            hr = m_pKinectSensor->get_CoordinateMapper(&m_pCoordinateMapper);
        }

		if (SUCCEEDED(hr)) {
			hr = m_pKinectSensor->OpenMultiSourceFrameReader(FrameSourceTypes_Body | FrameSourceTypes_Color | FrameSourceTypes_Depth | FrameSourceTypes_BodyIndex, &pMultiReader);
		}

        if (SUCCEEDED(hr))
        {
            // Create a face frame source + reader to track each body in the fov
            for (int i = 0; i < BODY_COUNT; i++)
            {
                if (SUCCEEDED(hr))
                {
                    // Create the face frame source by specifying the required face frame features
                    hr = CreateFaceFrameSource(m_pKinectSensor, 0, c_FaceFrameFeatures, &m_pFaceFrameSources[i]);
                }
                if (SUCCEEDED(hr))
                {
                    // Open the corresponding reader
                    hr = m_pFaceFrameSources[i]->OpenReader(&m_pFaceFrameReaders[i]);
                }				
            }
        }        
    }

    if (!m_pKinectSensor || FAILED(hr))
    {
        SetStatusMessage(L"No ready Kinect found!", 10000, true);
        return E_FAIL;
    }

    return hr;
}

/// <summary>
/// Main processing function
/// </summary>
void CFaceBasics::Update(int *frameNum)
{
    if (!pMultiReader)
    {
        return;
    }

	IMultiSourceFrame	*pMultiSourceFrame = nullptr;
	IDepthFrame			*pDepthFrame = nullptr;
	IColorFrame			*pColorFrame = nullptr;
	IBodyIndexFrame		*pBodyIndexFrame = nullptr;
	IBodyFrame			*pBodyFrame = nullptr;

	HRESULT hr = pMultiReader->AcquireLatestFrame(&pMultiSourceFrame);

	if (SUCCEEDED(hr)) {
		*frameNum += 1;
		cout << "frame: " << *frameNum << endl;
	}

	//Get Depth Frame
	if (SUCCEEDED(hr)) {
		IDepthFrameReference *pDepthFrameReference = NULL;
		hr = pMultiSourceFrame->get_DepthFrameReference(&pDepthFrameReference);
		if (SUCCEEDED(hr)) {
			hr = pDepthFrameReference->AcquireFrame(&pDepthFrame);
		}
		SafeRelease(pDepthFrameReference);
	}

	//Get color frame
	if (SUCCEEDED(hr))
	{
		IColorFrameReference *pColorFrameReference = NULL;
		hr = pMultiSourceFrame->get_ColorFrameReference(&pColorFrameReference);
		if (SUCCEEDED(hr)) {
			hr = pColorFrameReference->AcquireFrame(&pColorFrame);
		}
		SafeRelease(pColorFrameReference);
	}

	//Get Body_Index Frame
	if (SUCCEEDED(hr)) {
		IBodyIndexFrameReference *pBodyIndexFrameReference = NULL;
		hr = pMultiSourceFrame->get_BodyIndexFrameReference(&pBodyIndexFrameReference);
		if (SUCCEEDED(hr)) {
			hr = pBodyIndexFrameReference->AcquireFrame(&pBodyIndexFrame);
		}
		SafeRelease(pBodyIndexFrameReference);
	}

	//Get Body Frame
	if (SUCCEEDED(hr)) {
		IBodyFrameReference *pBodyFrameReference = NULL;
		hr = pMultiSourceFrame->get_BodyFrameReference(&pBodyFrameReference);
		if (SUCCEEDED(hr)) {
			hr = pBodyFrameReference->AcquireFrame(&pBodyFrame);
		}
		SafeRelease(pBodyFrameReference);
	}
	SafeRelease(pMultiSourceFrame);

	if(SUCCEEDED(hr)){
		//Get color data
        INT64 nTime = 0;
        IFrameDescription* pFrameDescription = nullptr;
        int nWidth = 0;
        int nHeight = 0;
        ColorImageFormat imageFormat = ColorImageFormat_None;
        UINT nBufferSize = 0;
        RGBQUAD *pColorBuffer = nullptr;

        hr = pColorFrame->get_RelativeTime(&nTime);
        if (SUCCEEDED(hr))
        {
            hr = pColorFrame->get_FrameDescription(&pFrameDescription);
        }
        if (SUCCEEDED(hr))
        {
            hr = pFrameDescription->get_Width(&nWidth);
        }
        if (SUCCEEDED(hr))
        {
            hr = pFrameDescription->get_Height(&nHeight);
			SafeRelease(pFrameDescription);
        }
        if (SUCCEEDED(hr))
        {
            hr = pColorFrame->get_RawColorImageFormat(&imageFormat);
        }
        if (SUCCEEDED(hr))
        {
            if (imageFormat == ColorImageFormat_Bgra)
            {
                hr = pColorFrame->AccessRawUnderlyingBuffer(&nBufferSize, reinterpret_cast<BYTE**>(&pColorBuffer));
            }
            else if (m_pColorRGBX)
            {
				pColorBuffer = m_pColorRGBX;
                nBufferSize = cColorWidth * cColorHeight * sizeof(RGBQUAD);
                hr = pColorFrame->CopyConvertedFrameDataToArray(nBufferSize, reinterpret_cast<BYTE*>(pColorBuffer), ColorImageFormat_Bgra);
            }
            else
            {
                hr = E_FAIL;
            }
        }			

		IFrameDescription *pDepthFrameDescription = NULL;
		int nDepthWidth = 0;
		int nDepthHeight = 0;
		USHORT nMinDepth = 0;
		USHORT nMaxDepth = 0;
		UINT nDepthBufferSize = 0;
		UINT16 *pDepthBuffer = NULL;
		//Get depth data
		if (SUCCEEDED(hr)) {
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
		}

		UINT nBodyIndexSize = 0;
		BYTE *pBodyIndexBuffer = NULL;

		IBody *pBodys[BODY_COUNT] = { 0 };
		//Get body index data
		if (SUCCEEDED(hr)) {
			hr = pBodyIndexFrame->AccessUnderlyingBuffer(&nBodyIndexSize, &pBodyIndexBuffer);
		}
		//Get Body Data
		if (SUCCEEDED(hr)) {
			hr = pBodyFrame->GetAndRefreshBodyData(BODY_COUNT, pBodys);
		}

		if (SUCCEEDED(hr))
		{
			DrawStreams(nTime, pColorBuffer, nWidth, nHeight,pColorBuffer,pDepthBuffer,pBodyIndexBuffer,pBodys,*frameNum,nMinDepth,nMaxDepth);
			if (pBodys) {
				for (int i = 0; i < BODY_COUNT; i++) {
					SafeRelease(pBodys[i]);
				}
			}
		}
    }
    SafeRelease(pColorFrame);
	SafeRelease(pDepthFrame);
	SafeRelease(pBodyIndexFrame);
	SafeRelease(pBodyFrame);
}

/// <summary>
/// Renders the color and face streams
/// </summary>
/// <param name="nTime">timestamp of frame</param>
/// <param name="pBuffer">pointer to frame data</param>
/// <param name="nWidth">width (in pixels) of input image data</param>
/// <param name="nHeight">height (in pixels) of input image data</param>
void CFaceBasics::DrawStreams(INT64 nTime, RGBQUAD* pBuffer, int nWidth, int nHeight, RGBQUAD* pColorBuffer, UINT16* pDepthBuffer, BYTE* pBodyIndexBuffer, IBody** pBodys,int frameNum,USHORT nMinDepth,USHORT nMaxDepth)
{
    if (m_hWnd)
    {
        HRESULT hr;
        hr = m_pDrawDataStreams->BeginDrawing();

        if (SUCCEEDED(hr))
        {
            // Make sure we've received valid color data
            if (pBuffer && (nWidth == cColorWidth) && (nHeight == cColorHeight))
            {
                // Draw the data with Direct2D
                hr = m_pDrawDataStreams->DrawBackground(reinterpret_cast<BYTE*>(pBuffer), cColorWidth * cColorHeight * sizeof(RGBQUAD));        
            }
            else
            {
                // Recieved invalid data, stop drawing
                hr = E_INVALIDARG;
            }

            if (SUCCEEDED(hr))
            {
                // Begin processing the face frames
                ProcessFaces(frameNum, pColorBuffer, pDepthBuffer, pBodyIndexBuffer, pBodys, nMinDepth, nMaxDepth);
            }

            m_pDrawDataStreams->EndDrawing();
        }

        if (!m_nStartTime)
        {
            m_nStartTime = nTime;
        }

        double fps = 0.0;

        LARGE_INTEGER qpcNow = {0};
        if (m_fFreq)
        {
            if (QueryPerformanceCounter(&qpcNow))
            {
                if (m_nLastCounter)
                {
                    m_nFramesSinceUpdate++;
                    fps = m_fFreq * m_nFramesSinceUpdate / double(qpcNow.QuadPart - m_nLastCounter);
                }
            }
        }

        WCHAR szStatusMessage[64];
        StringCchPrintf(szStatusMessage, _countof(szStatusMessage), L" FPS = %0.2f    Time = %I64d", fps, (nTime - m_nStartTime));

        if (SetStatusMessage(szStatusMessage, 1000, false))
        {
            m_nLastCounter = qpcNow.QuadPart;
            m_nFramesSinceUpdate = 0;
        }
    }    
}

/// <summary>
/// Processes new face frames
/// </summary>
void CFaceBasics::ProcessFaces(int frameNum,RGBQUAD* pColorBuffer,UINT16* pDepthBuffer,BYTE* pBodyIndexBuffer,IBody** pBodys, USHORT nMinDepth, USHORT nMaxDepth)
{
    HRESULT hr;

    // Iterate through each face reader
    for (int iFace = 0; iFace < BODY_COUNT; ++iFace)
    {
        // Retrieve the latest face frame from this reader
        IFaceFrame* pFaceFrame = nullptr;
        hr = m_pFaceFrameReaders[iFace]->AcquireLatestFrame(&pFaceFrame);

        BOOLEAN bFaceTracked = false;
        if (SUCCEEDED(hr) && nullptr != pFaceFrame)
        {
            // Check if a valid face is tracked in this face frame
            hr = pFaceFrame->get_IsTrackingIdValid(&bFaceTracked);
        }

        if (SUCCEEDED(hr))
        {
			vector<int> pyr;
            if (bFaceTracked)
            {
				
                IFaceFrameResult* pFaceFrameResult = nullptr;
                RectI faceBox = {0};
                PointF facePoints[FacePointType::FacePointType_Count];
                Vector4 faceRotation;
                DetectionResult faceProperties[FaceProperty::FaceProperty_Count];
                D2D1_POINT_2F faceTextLayout;

                hr = pFaceFrame->get_FaceFrameResult(&pFaceFrameResult);

                // Need to verify if pFaceFrameResult contains data before trying to access it
                if (SUCCEEDED(hr) && pFaceFrameResult != nullptr)
                {
                    hr = pFaceFrameResult->get_FaceBoundingBoxInColorSpace(&faceBox);

                    if (SUCCEEDED(hr))
                    {										
                        hr = pFaceFrameResult->GetFacePointsInColorSpace(FacePointType::FacePointType_Count, facePoints);
                    }

                    if (SUCCEEDED(hr))
                    {
                        hr = pFaceFrameResult->get_FaceRotationQuaternion(&faceRotation);
                    }

                    if (SUCCEEDED(hr))
                    {
                        hr = pFaceFrameResult->GetFaceProperties(FaceProperty::FaceProperty_Count, faceProperties);
                    }

                    if (SUCCEEDED(hr))
                    {
                        hr = GetFaceTextPositionInColorSpace(pBodys[iFace], &faceTextLayout);
                    }

                    if (SUCCEEDED(hr))
                    {
                        // Draw face frame results
                        pyr = m_pDrawDataStreams->DrawFaceFrameResults(iFace, &faceBox, facePoints, &faceRotation, faceProperties, &faceTextLayout);
                    }							
                }

                SafeRelease(pFaceFrameResult);	

				
            }
            else 
            {	
                // Face tracking is not valid - attempt to fix the issue
                // A valid body is required to perform this step
                if (pBodys)
                {
                    // Check if the corresponding body is tracked 
                    // If this is true then update the face frame source to track this body
                    IBody* pBody = pBodys[iFace];
                    if (pBody != nullptr)
                    {
                        BOOLEAN bTracked = false;
                        hr = pBody->get_IsTracked(&bTracked);

                        UINT64 bodyTId;
                        if (SUCCEEDED(hr) && bTracked)
                        {
                            // Get the tracking ID of this body
                            hr = pBody->get_TrackingId(&bodyTId);
                            if (SUCCEEDED(hr))
                            {
                                // Update the face frame source with the tracking ID
                                m_pFaceFrameSources[iFace]->put_TrackingId(bodyTId);
                            }
                        }
                    }
                }
            }
			cout << "in progress" << endl;


			//Insert£ºget hand status and CameraSpacePoint
			cout << "get 3d points of joints" << endl;
			boolean bTrackedBody = false;//get the hand or not
			boolean bTracked = false;//get the body or not

			for (int i = 0; i < BODY_COUNT; i++) {
				hr = pBodys[i]->get_IsTracked(&bTracked);
				if (SUCCEEDED(hr) && bTracked) {

					HandState leftHandState = HandState_Unknown;
					pBodys[i]->get_HandLeftState(&leftHandState);
					if (0 && (leftHandState == HandState_Unknown || leftHandState == HandState_NotTracked)) {

					}
					else {
						bTrackedBody = true;
						hr = pBodys[i]->GetJoints(JointType_Count, joints);
						//
						//Convert the all node concerned
						cout << "Convert the all node concerned" << endl;
						ColorSpacePoint *p = joints_position;
						for (int jointNum = 0; jointNum < JointType_Count; jointNum++) {
							m_pCoordinateMapper->MapCameraPointToColorSpace(joints[jointNum].Position, p++);
							ColorSpacePoint tmp = *(p - 1);
						}

					}
					break;
				}
			}
			if (!bTracked) {
				cout << "no body!" << endl;
				return;
			}
			if (bTrackedBody == true) {
				cout << "Get is OK" << endl;
			}
			else {
				cout << "Get is NO" << endl;
				//continue;
			}

			//Set the initial background black
			for (UINT16 i = 0; i < 1920; i++) {
				for (UINT16 j = 0; j < 1080; j++)
				{
					colorShow[j * 1920 + i] = { 0,0,0,255 };
				}
			}

			///////////////////////////////
			//Read depth data and color data
			/*ifstream readDepth("E:\\depth_value_1.txt");
			for (int i = 0; !readDepth.eof(); i++) {
			readDepth >> depthValue[i];
			}
			Mat readColor = imread("E:\\img.jpg");*/
			//hr = pCoorMapper->MapDepthFrameToColorSpace(nDepthHeight*nDepthWidth, (UINT16*)depthValue, nDepthHeight*nDepthWidth, pColorSpacePoints);
			/////////////////////////
			pointsCloud.clear();
			color_all.clear();
			pointsCloudBody.clear();
			color_body.clear();
			hr = m_pCoordinateMapper->MapDepthFrameToColorSpace(512 * 424, (UINT16*)pDepthBuffer, 512 * 424, pColorSpacePoints);
			hr = m_pCoordinateMapper->MapDepthFrameToCameraSpace(512 * 424, (UINT16*)pDepthBuffer, 512 * 424, pCameraSpacePoints);

			if (SUCCEEDED(hr)) {
				for (int i = 0; i < 512 * 424; i++) {

					if (pColorSpacePoints[i].X != -std::numeric_limits<float>::infinity() && pColorSpacePoints[i].Y != -std::numeric_limits<float>::infinity()) {

						int colorX = static_cast<int>(pColorSpacePoints[i].X);
						int colorY = static_cast<int>(pColorSpacePoints[i].Y);

						colorX = colorX < 0 ? 0 : colorX;
						colorX = colorX > 1919 ? 1919 : colorX;
						colorY = colorY < 0 ? 0 : colorY;
						colorY = colorY > 1079 ? 1079 : colorY;

						if (pCameraSpacePoints[i].X != -std::numeric_limits<float>::infinity() && pCameraSpacePoints[i].Y != -std::numeric_limits<float>::infinity() && pCameraSpacePoints[i].Z != -std::numeric_limits<float>::infinity()) {
							pointsCloud.push_back(pCameraSpacePoints[i]);
							color_all.push_back(pColorBuffer[colorX + 1920 * colorY]);
						}

						/*if (pBodyIndexBuffer[i] != 0xff) {
						pointsCloudBody.push_back(pCameraSpacePoints[i]);
						color_body.push_back(pColorBuffer[colorX + 1920 * colorY]);

						colorShow[colorX + 1920 * colorY] = pColorBuffer[colorX + 1920 * colorY];
						}*/
					}

				}
			}

			hr = m_pCoordinateMapper->MapColorFrameToDepthSpace(512 * 424, (UINT16*)pDepthBuffer, 1920 * 1080, pDepthSpacePoint);
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

							BYTE player = pBodyIndexBuffer[depthX + (depthY * 512)];

							// If we're tracking a player for the current pixel, draw from the color camera
							if (player != 0xff)
							{

								// Set source for copy to the color pixel
								*pSrc = *(pColorBuffer + colorIndex);
								color_body.push_back(*pSrc);
								pointsCloudBody.push_back(pCameraSpacePoints[depthX + depthY * 512]);
							}
						}

					}
					pSrc++;
				}
			}

			if (true && frameNum >10) {
				int nameNum = 5002;
				//PlaySound(TEXT("E:\\bone.wav"), NULL, SND_ASYNC | SND_NODEFAULT);
				mciSendString(TEXT(" open E:\\bone.wav alias mysong"), NULL, 0, NULL);
				mciSendString(TEXT("play mysong"), NULL, 0, NULL);
				Sleep(1000);

				//Save depth value
				ostringstream depthValuePath;
				depthValuePath << "E:\\1000data\\depth_value_" << nameNum << ".txt";
				ofstream writeDepth(depthValuePath.str());
				for (int i = 0; i < 512 * 424; i++) {
					writeDepth << pDepthBuffer[i];
					if (i != 512 * 424 - 1)
						writeDepth << endl;
				}
				writeDepth.close();

				//Write to .ply file
				cout << "output" << endl;
				ofstream ply;
				string path = "e:\\1000data\\point_cloud_all_";
				char fileName[15];
				sprintf_s(fileName, "%d%s", nameNum, ".ply");
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

				//Write to .ply file
				cout << "output" << endl;
				string path1 = "e:\\1000data\\point_cloud_body_";
				char fileName1[15];
				sprintf_s(fileName1, "%d%s", nameNum, ".ply");
				string file1 = path1 + fileName1;
				cout << file1.c_str() << endl;
				ply.open(file1.c_str());
				ply << "ply" << endl;
				ply << "format ascii 1.0" << endl;
				ply << "element vertex " << color_body.size() + JointType_Count << endl;
				ply << "property float x" << endl;
				ply << "property float y" << endl;
				ply << "property float z" << endl;
				ply << "property uchar red" << endl;
				ply << "property uchar green" << endl;
				ply << "property uchar blue" << endl;
				ply << "end_header" << endl;
				int diff_z = (int)joints[1].Position.Z;
				float diff_x = joints[1].Position.X;
				for (int i = 0; i < color_body.size(); i++) {
					pointsCloudBody[i].Z -= diff_z;
					pointsCloudBody[i].X = -(pointsCloudBody[i].X - diff_x);
					pointsCloudBody[i].Z = -pointsCloudBody[i].Z;
					ply << pointsCloudBody[i].X << " " << pointsCloudBody[i].Y << " " << pointsCloudBody[i].Z << " " << (int)color_body[i].rgbRed << " " << (int)color_body[i].rgbGreen << " " << (int)color_body[i].rgbBlue << endl;
				}
				for (int i = 0; i < JointType_Count; i++) {
					joints[i].Position.Z -= diff_z;
					joints[i].Position.X = -(joints[i].Position.X - diff_x);
					joints[i].Position.Z = -joints[i].Position.Z;
					ply << joints[i].Position.X << " " << joints[i].Position.Y << " " << joints[i].Position.Z << " " << 255 << " " << 0 << " " << 0 << endl;
				}

				ply.close();

				//Save the denoised pointcloud

				//Save depth map/body depth map			
				UCHAR* pnewdepth = new UCHAR[512 * 424];
				UCHAR* pdepthbody = new UCHAR[512 * 424];
				for (int i = 0; i < 512 * 424; i++) {
					pnewdepth[i] = static_cast<UCHAR>((pDepthBuffer[i] >= nMinDepth) && (pDepthBuffer[i] <= nMaxDepth) ? (pDepthBuffer[i] % 256) : 0);
					if (pBodyIndexBuffer[i] != 0xff)
					{
						pdepthbody[i] = pnewdepth[i];
					}
					else {
						pdepthbody[i] = 255;
					}
				}
				Mat depth_img(424, 512, CV_16UC1, pDepthBuffer);
				Mat depth_img1;
				flip(depth_img, depth_img1, 1);
				ostringstream depth_img_path;
				depth_img_path << "E:\\1000data\\img_depth_" << nameNum << ".tif";
				cout << "save image£º" << imwrite(depth_img_path.str(), depth_img1) << endl;

				Mat new_depth_img(424, 512, CV_8UC1, pnewdepth);
				Mat new_depth_img1;
				flip(new_depth_img, new_depth_img1, 1);
				ostringstream new_depth_img_path;
				new_depth_img_path << "E:\\1000data\\new_img_depth_" << nameNum << ".tif";
				cout << "save image£º" << imwrite(new_depth_img_path.str(), new_depth_img1) << endl;

				Mat bodydepth_img(424, 512, CV_8UC1, pdepthbody);
				Mat bodydepth_img1;
				flip(bodydepth_img, bodydepth_img1, 1);
				ostringstream depthbody_img_path;
				depthbody_img_path << "E:\\1000data\\img_depth_body_" << nameNum << ".tif";
				cout << "save image£º" << imwrite(depthbody_img_path.str(), bodydepth_img1) << endl;

				//use to opencv save image
				Mat img_body(1080, 1920, CV_8UC4, colorShow);
				Mat img_body_flip;
				flip(img_body, img_body_flip, 1);
				ostringstream img_body_path;
				img_body_path << "E:\\1000data\\img_body_" << nameNum << ".jpg";
				cout << "save image£º" << imwrite(img_body_path.str(), img_body_flip) << endl;

				Mat img(1080, 1920, CV_8UC4, pColorBuffer);
				Mat img_flip;
				flip(img, img_flip, 1);
				ostringstream img_path;
				img_path << "E:\\1000data\\img_" << nameNum << ".jpg";
				cout << "save image£º" << imwrite(img_path.str(), img_flip) << endl;

				//Save skeleton image
				RGBQUAD* joints_img = new RGBQUAD[1920 * 1080];
				for (int i = 0; i < 1920 * 1080; i++) {
					joints_img[i].rgbBlue = 255;
					joints_img[i].rgbGreen = 255;
					joints_img[i].rgbRed = 255;
					joints_img[i].rgbReserved = 255;
				}
				Mat img_s(1080, 1920, CV_8UC4, joints_img);
				int lines[24][2] = { { 3,2 },{ 2,20 },{ 20,8 },{ 8,9 },{ 9,10 },{ 10,11 },{ 11,24 },{ 11,23 },{ 20,4 },{ 4,5 },{ 5,6 },{ 6,7 },{ 7,22 },{ 7,21 },{ 20,1 },{ 1,0 },{ 0,16 },{ 16,17 },{ 17,18 },{ 18,19 },{ 0,12 },{ 12,13 },{ 13,14 },{ 14,15 } };
				//Draw bone
				for (int i = 0; i < JointType_Count - 1; i++) {
					int thinkness = 5;
					if ((lines[i][0] == 10 && lines[i][1] == 11) || (lines[i][0] == 11 && lines[i][1] == 23) || (lines[i][0] == 11 && lines[i][1] == 24) || (lines[i][0] == 6 && lines[i][1] == 7) || (lines[i][0] == 7 && lines[i][1] == 21) || (lines[i][0] == 7 && lines[i][1] == 22)) {
						continue;
					}
					float line0X = 1920 - joints_position[lines[i][0]].X;
					float line0Y = joints_position[lines[i][0]].Y;
					float line1X = 1920 - joints_position[lines[i][1]].X;
					float line1Y = joints_position[lines[i][1]].Y;
					if (line0X >= 0 && line0Y <= 1919 && line0Y >= 0 && line0Y <= 1079 && line1X >= 0 && line1Y <= 1919 && line1Y >= 0 && line1Y <= 1079)
					{
						line(img_s, Point2f(line0X, line0Y), Point2f(line1X, line1Y), Scalar(0, 255, 0), thinkness, 16);
					}
				}
				//Draw joint point
				for (size_t i = 0; i < JointType_Count; i++)
				{
					if ((i == 11 || i == 23 || i == 24 || i == 7 || i == 21 || i == 22)) {
						continue;
					}
					float tmpX = 1920 - joints_position[i].X;
					float tmpY = joints_position[i].Y;
					if (joints[i].TrackingState == TrackingState_Inferred);
					circle(img_s, Point2f(tmpX, tmpY), 7, Scalar(0, 0, 255), -1);
				}
				ostringstream img_s_path;
				img_s_path << "E:\\1000data\\img_just_joints_" << nameNum << ".jpg";;
				cout << "save image£º" << imwrite(img_s_path.str(), img_s) << endl;

				//Save 3D joints position and head orientation data
				ostringstream joints_3d_path;
				joints_3d_path << "E:\\1000data\\Joints3DPosition_" << nameNum << ".txt";
				ofstream fJoint3DPosition(joints_3d_path.str());
				for (int i = 0; i < JointType_Count; i++) {
					float confidence = 0.0;
					if (joints[i].TrackingState == TrackingState_Tracked) {
						confidence = 0.95;
					}
					else if (joints[i].TrackingState == TrackingState_Inferred) {
						confidence = 0.8;
					}
					fJoint3DPosition << i << " " << joints_name[i] << " " << joints[i].Position.X << " " << joints[i].Position.Y << " " << joints[i].Position.Z << " " << confidence << endl;
				}


				//Save 3D joints position
				ostringstream joints_path;
				joints_path << "E:\\1000data\\Joints2DPosition_" << nameNum << ".txt";
				ofstream fJointPosition(joints_path.str());
				for (int i = 0; i < JointType_Count; i++) {
					float confidence = 0.0;
					if (joints[i].TrackingState == TrackingState_Tracked) {
						confidence = 1;
					}
					else if (joints[i].TrackingState == TrackingState_Inferred) {
						confidence = 0.9;
					}
					float tmpX = 1920 - joints_position[i].X;
					float tmpY = joints_position[i].Y;
					fJointPosition << i << " " << joints_name[i] << " " << tmpX << " " << tmpY << " " << confidence << endl;

					circle(img_body_flip, Point2f(tmpX, tmpY), 5, Scalar(0, 0, 255), -1);
					circle(img_flip, Point2f(tmpX, tmpY), 5, Scalar(0, 0, 255), -1);
				}
				fJointPosition.close();

				ostringstream img_with_joints_path, img_body_with_joints_path;
				img_with_joints_path << "E:\\1000data\\img_joints_" << nameNum << ".jpg";
				img_body_with_joints_path << "E:\\1000data\\img_body_joints_" << nameNum << ".jpg";
				cout << "save image£º" << imwrite(img_body_with_joints_path.str(), img_body_flip) << endl;
				cout << "save image£º" << imwrite(img_with_joints_path.str(), img_flip) << endl;

				/*pitch is rotated around the X axis, also called the pitch angle.
				Yaw is rotated around the Y axis, also called yaw angle.
				The roll is rotated around the Z axis, also called the roll angle */
				//fJoint3DPosition << "pitch" << " " << "yaw" << " " << "roll" << " " << pyr[0] - 10 << " " << pyr[1] << " " << pyr[2] << endl;
				if (pyr.size() != 0) {
					fJoint3DPosition << "pitch" << " " << "yaw" << " " << "roll" << " " << pyr[0] - 10 << " " << pyr[1] << " " << pyr[2] << endl;
				}
				fJoint3DPosition.close();
				return;
			}
        }			

        SafeRelease(pFaceFrame);
    }

}

/// <summary>
/// Computes the face result text position by adding an offset to the corresponding 
/// body's head joint in camera space and then by projecting it to screen space
/// </summary>
/// <param name="pBody">pointer to the body data</param>
/// <param name="pFaceTextLayout">pointer to the text layout position in screen space</param>
/// <returns>indicates success or failure</returns>
HRESULT CFaceBasics::GetFaceTextPositionInColorSpace(IBody* pBody, D2D1_POINT_2F* pFaceTextLayout)
{
    HRESULT hr = E_FAIL;

    if (pBody != nullptr)
    {
        BOOLEAN bTracked = false;
        hr = pBody->get_IsTracked(&bTracked);

        if (SUCCEEDED(hr) && bTracked)
        {
            Joint joints[JointType_Count]; 
            hr = pBody->GetJoints(_countof(joints), joints);
            if (SUCCEEDED(hr))
            {
                CameraSpacePoint headJoint = joints[JointType_Head].Position;
                CameraSpacePoint textPoint = 
                {
                    headJoint.X + c_FaceTextLayoutOffsetX,
                    headJoint.Y + c_FaceTextLayoutOffsetY,
                    headJoint.Z
                };

                ColorSpacePoint colorPoint = {0};
                hr = m_pCoordinateMapper->MapCameraPointToColorSpace(textPoint, &colorPoint);

                if (SUCCEEDED(hr))
                {
                    pFaceTextLayout->x = colorPoint.X;
                    pFaceTextLayout->y = colorPoint.Y;
                }
            }
        }
    }

    return hr;
}

/// <summary>
/// Updates body data
/// </summary>
/// <param name="ppBodies">pointer to the body data storage</param>
/// <returns>indicates success or failure</returns>
HRESULT CFaceBasics::UpdateBodyData(IBody** ppBodies)
{
    HRESULT hr = E_FAIL;

    if (m_pBodyFrameReader != nullptr)
    {
        IBodyFrame* pBodyFrame = nullptr;
        hr = m_pBodyFrameReader->AcquireLatestFrame(&pBodyFrame);
        if (SUCCEEDED(hr))
        {
            hr = pBodyFrame->GetAndRefreshBodyData(BODY_COUNT, ppBodies);
        }
        SafeRelease(pBodyFrame);    
    }

    return hr;
}

/// <summary>
/// Set the status bar message
/// </summary>
/// <param name="szMessage">message to display</param>
/// <param name="showTimeMsec">time in milliseconds to ignore future status messages</param>
/// <param name="bForce">force status update</param>
/// <returns>success or failure</returns>
bool CFaceBasics::SetStatusMessage(_In_z_ WCHAR* szMessage, ULONGLONG nShowTimeMsec, bool bForce)
{
    ULONGLONG now = GetTickCount64();

    if (m_hWnd && (bForce || (m_nNextStatusTime <= now)))
    {
        SetDlgItemText(m_hWnd, IDC_STATUS, szMessage);
        m_nNextStatusTime = now + nShowTimeMsec;

        return true;
    }

    return false;
}
Mat ConvertMat(const UINT16* pBuffer, int nWidth, int nHeight, USHORT nMinDepth, USHORT nMaxDepth)
{
	cv::Mat img(nHeight, nWidth, CV_8UC3);
	uchar* p_mat = img.data;
	const UINT16* pBufferEnd = pBuffer + (nWidth * nHeight);
	while (pBuffer < pBufferEnd)
	{
		USHORT depth = *pBuffer;
		BYTE intensity = static_cast<BYTE>((depth >= nMinDepth) && (depth <= nMaxDepth) ? (depth % 256) : 0);
		*p_mat = intensity;
		p_mat++;
		*p_mat = intensity;
		p_mat++;
		*p_mat = intensity;
		p_mat++;
		++pBuffer;
	}
	return img;
}

