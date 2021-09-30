#pragma once
#include <Kinect.h>
#include "ImageDraw.h"

#define SafeRelease(p) if(p!=NULL){p->Release();p=NULL;}

struct PointCloud {
	int x;
	int y;
	int z;
	RGBQUAD color;
};


class KinectFrameDeal
{
public:
	KinectFrameDeal();
	~KinectFrameDeal();

	HRESULT initiate(int argc,char **argv);

	void excuteDisplyFunc();

private:
	
	//glut initial parameter
	int							argc;
	char						**argv;

	IKinectSensor				*pkinectSensor;
	//IMultiSourceFrameReader		*pMultiReader;
	//ICoordinateMapper			*pCoorMapper;

	RGBQUAD						*c_pColorBuffer;

	int							c_nColorWidth;
	int							c_nColorHeight;
	//int							frameWidth;
	//int							frameHeight;

	ImageDraw					*imageDraw;

	
};

void FrameDataDeal(void);