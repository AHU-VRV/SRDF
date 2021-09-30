#include "main.h"
#include "Kinect.h"
#include <iostream>
#include "KinectFrameDeal.h"
using namespace std;

typedef struct {
	//int					width;				//Frame Width
	//int					height;				//Frame Height
	//ColorImageFormat	imageFormat;		//Frame Format
	UINT				nColorBufferSize;	//Color Frame Size
	RGBQUAD				*pColorFrameData;	//Color Frame RGB Data
	UINT				nDepthBufferSize;	//Depth Frame Size
	UINT16				*pDepthFrameData;	//Depth Frame Data
}FrameData;

#define SafeRelease(x) if(x!=NULL){x->Release();x=NULL;}

int frameTip = 0;
void getFrame(IMultiSourceFrameReader *pMultiReader, WAITABLE_HANDLE &multiFrameArriveEvent);
int main(int argc,char **argv) 
{
	KinectFrameDeal kinectFrameDeal;
	HRESULT hr = kinectFrameDeal.initiate(argc,argv);
	if (SUCCEEDED(hr)) {
		//Sleep(3000);
		kinectFrameDeal.excuteDisplyFunc();

	}

	//IKinectSensor *iks = nullptr;

	////initialize device
	//HRESULT hr = GetDefaultKinectSensor(&iks);
	//if (SUCCEEDED(hr)){
	//	hr = iks->Open();
	//	std::cout << hr << std::endl;
	//	if (FAILED(hr)){
	//		std::cout << "openning device failed£¡" << std::endl;
	//		return 0;
	//	}
	//}else{
	//	std::cout << "not get the device£¡" << std::endl;
	//	return 0;
	//}
	//////depth frame source
	////IColorFrameSource *pColorFrameSource = nullptr;
	////hr = iks->get_ColorFrameSource(&pColorFrameSource);
	////if (FAILED(hr)){
	////	std::cout << "error:hr = iks->get_ColorFrameSource(&iColorFrameSource);" << std::endl;
	////	return 0;
	////}
	//////depth frame reader
	////IColorFrameReader *pColorFrameReader = nullptr;
	////hr = pColorFrameSource->OpenReader(&pColorFrameReader);
	////if (FAILED(hr)){
	////	std::cout << "error:hr = iks->get_ColorFrameSource(&iColorFrameSource);" << std::endl;
	////	return 0;
	////}

	//IMultiSourceFrameReader *pMultiReader = nullptr;
	//hr = iks->OpenMultiSourceFrameReader(FrameSourceTypes_Color | FrameSourceTypes_Depth, &pMultiReader);
	//if (FAILED(hr)) {
	//	cout << "Open Multi Reader Failed." << endl;
	//	return 0;
	//}

	////Registration issue
	//WAITABLE_HANDLE multiFrameArriveEvent = 0;
	//pMultiReader->SubscribeMultiSourceFrameArrived(&multiFrameArriveEvent);
	//
	//std::cout << "xxx" << std::endl;

	//HANDLE events[] = { reinterpret_cast<HANDLE>(multiFrameArriveEvent) };
	//int xunhuan = 0;
	//while (true){
	//	xunhuan++;
	//	cout << "loop number:" << xunhuan << endl;
	//	events[0] = reinterpret_cast<HANDLE>(multiFrameArriveEvent);
	//	switch (MsgWaitForMultipleObjects(sizeof(events)/sizeof(*events),events,FALSE,INFINITE,QS_ALLINPUT))
	//	{
	//	case WAIT_OBJECT_0 + 0:
	//		getFrame(pMultiReader, multiFrameArriveEvent);
	//		break;
	//	default:
	//		break;
	//	}

	//	if (xunhuan == 500) {
	//		if (pMultiReader&&multiFrameArriveEvent) {
	//			pMultiReader->UnsubscribeMultiSourceFrameArrived(multiFrameArriveEvent);
	//			multiFrameArriveEvent = 0;
	//			SafeRelease(pMultiReader);
	//		}
	//		if (iks) {
	//			iks->Close();
	//			SafeRelease(iks);
	//		}
	//		break;
	//	}

	//}

	std::cout << "end" << std::endl;
	return 0;
}

void getFrame(IMultiSourceFrameReader *pMultiReader,WAITABLE_HANDLE &multiFrameArriveEvent) {
	if (!pMultiReader) {
		return;
	}

	IMultiSourceFrameArrivedEventArgs *pMultiFrameArrivedEventArgs = nullptr;//Frame arrival event
	IMultiSourceFrameReference *pMultiFrameReference = nullptr;//Frame reference
	IMultiSourceFrame *pMultiFrame = nullptr;//frame
	IColorFrameReference *pColorFrameReference = nullptr;
	IDepthFrameReference *pDepthFrameReference = nullptr;
	IColorFrame *pColorFrame = nullptr;
	IDepthFrame *pDepthFrame = nullptr;
	
	IFrameDescription *pColorFrameDescription = nullptr;
	IFrameDescription *pDepthFrameDescription = nullptr;
	FrameData frameData = {1920*1080,nullptr,0,nullptr};

	int width = 0;
	int height = 0;
	ColorImageFormat colorImageFormat = ColorImageFormat_Bgra;

	//Get parameter
	HRESULT hr = pMultiReader->GetMultiSourceFrameArrivedEventData(multiFrameArriveEvent, &pMultiFrameArrivedEventArgs);
	
	if (SUCCEEDED(hr)) {
		cout << ">1";
		hr = pMultiFrameArrivedEventArgs->get_FrameReference(&pMultiFrameReference);//get frame reference
		SafeRelease(pMultiFrameArrivedEventArgs);
	}
	
	if (SUCCEEDED(hr)) {
		cout << ">2";
		hr = pMultiFrameReference->AcquireFrame(&pMultiFrame);//get frame
		SafeRelease(pMultiFrameReference);
	}
	
	if (SUCCEEDED(hr)) {
		cout << ">3";
		hr = pMultiFrame->get_ColorFrameReference(&pColorFrameReference);
		
		if (SUCCEEDED(hr)) {
			cout << ">4";
			hr = pColorFrameReference->AcquireFrame(&pColorFrame);
			SafeRelease(pColorFrameReference);
		}
		
		if (SUCCEEDED(hr)) {
			cout << ">5";
			hr = pColorFrame->get_FrameDescription(&pColorFrameDescription);
		}
	}
	
	if (SUCCEEDED(hr)) {
		cout << ">6";
		hr = pMultiFrame->get_DepthFrameReference(&pDepthFrameReference);
		
		if (SUCCEEDED(hr)) {
			cout << ">7";
			hr = pDepthFrameReference->AcquireFrame(&pDepthFrame);
			SafeRelease(pDepthFrameReference);
		}
		
		if (SUCCEEDED(hr)) {
			cout << ">8";
			hr = pDepthFrame->get_FrameDescription(&pDepthFrameDescription);
		}
	}
	
	if (SUCCEEDED(hr)) {
		cout << ">9";
		hr = pColorFrameDescription->get_Width(&width);//Get frame width
	}
	
	if (SUCCEEDED(hr)) {
		cout << ">10";
		hr = pColorFrameDescription->get_Height(&height);
	}
	
	if (SUCCEEDED(hr)) {
		cout << ">11";
		hr = pColorFrame->get_RawColorImageFormat(&colorImageFormat);
	}
	//Output frame description
	frameTip++;
	
	cout << "Frame Tip:" << frameTip << "\t" << width << "*" << height << "  "<<colorImageFormat;
	if (SUCCEEDED(hr)) {
		cout << ">12";
		if (colorImageFormat == ColorImageFormat_Bgra) {
			hr = pColorFrame->AccessRawUnderlyingBuffer(&frameData.nColorBufferSize, reinterpret_cast<BYTE**>(frameData.pColorFrameData));
		}
		else {
			frameData.pColorFrameData = new RGBQUAD[height*width];
			frameData.nColorBufferSize = height*width*sizeof(RGBQUAD);
			hr = pColorFrame->CopyConvertedFrameDataToArray(frameData.nColorBufferSize, reinterpret_cast<BYTE*>(frameData.pColorFrameData), ColorImageFormat_Bgra);
			
			cout << "  "<<(int)frameData.pColorFrameData[10000].rgbRed << "_" << (int)frameData.pColorFrameData[10000].rgbGreen << "_" << (int)frameData.pColorFrameData[10000].rgbBlue << "_" << (int)frameData.pColorFrameData[10000].rgbReserved;
		}
	}


	//USHORT min = 0;
	//USHORT max = 0;
	//pDepthFrame->get_DepthMinReliableDistance(&min);
	//pDepthFrame->get_DepthMaxReliableDistance(&max);
	//cout << "Frame Tip:" << frameTip << "\t" << width << "*" << height << "  " << min << "  " << max;// << frameData.imageFormat;


	delete frameData.pColorFrameData;




	//hr = pcolorframe->copyframedatatoarray(framedatasize, framedata);
	//if (hr>0) {
	//	for (uint i = 0;i < framedatasize;i++) {
	//		std::cout << framedata[i] << std::endl;
	//	}
	//}
	SafeRelease(pColorFrameDescription);
	SafeRelease(pDepthFrameDescription);
	SafeRelease(pColorFrame);
	SafeRelease(pDepthFrame);
	SafeRelease(pMultiFrame);
	cout << "  over" << endl;
}
