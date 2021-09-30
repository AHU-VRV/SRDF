#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <sstream>
#include <math.h>

using namespace std;
using namespace cv;

typedef struct _CameraSpacePoint
{
	float X;
	float Y;
	float Z;
} 	D3f;
typedef struct _ImageSpacePoint
{
	float X;
	float Y;
} 	D2f;
//yjk 240

String joints_name[] = { "SpineBase","SpineMid","Neck","Head","ShoulderLeft","ElbowLeft","WristLeft","HandLeft","ShoulderRight","ElbowRight","WristRight","HandRight","Hipleft","KneeLeft","AnkleLeft","FootLeft","HipRight","KneeRight","AnkleRight","FootRight","SpineShoulder","HandTipLeft","ThumbLeft","HandHipRight","ThumbRight" };

//Progress the wrong joint point
//For the joint in 'Predicted' status combined with lcr-net results
void wrong_joint_process() {
	//Kinect left forearm; left upperarm; shoulder width; right upperarm; right forearm; left thigh; left calf; right thigh; right calf
	int standard_index[][2] = { { 4,5 },{ 3,4 },{ 3,6 },{ 6,7 },{ 7,8 },{ 9,10 },{ 10,11 },{ 13,14 },{ 14,15 } };
	//Lcr left forearm; left upperarm; shoulder width; right upperarm; right forearm; left thigh; left calf; right thigh; right calf
	int lcr_index[][2] = { { 6,8 },{ 8,10 },{ 10,11 },{ 11,9 },{ 9,7 },{ 5,3 },{ 3,1 },{ 4,2 },{ 2,0 } };

	//Kinect and lcr-net joint point pairs
	int pairs[] = { -1,-1,-1,-1,11,9,7,-1,10,8,6,-1,5,3, 1,-1,4,2,0,-1,-1,-1,-1,-1,-1 };

	//Kinect joint data
	float *read_len = new float[9];
	D3f *read_joint = new D3f[25];
	float *read_per_length = new float[9];
	int *status = new int[25];
	//Lcr joint data
	D3f *lcr_joint = new D3f[13];
	float *lcr_len = new float[9];
	float *lcr_per_total = new float[9];



	//Progress the wrong joint point
	int lcr_id = -1;
	int num = 0;
	int file_num = 1001;

	ostringstream joint_path;
	joint_path << "..\\data\\Joints3DPosition_" << file_num << ".txt";
	ifstream infile(joint_path.str(), ios::_Nocreate);

	if (infile.is_open())
	{
		lcr_id++;
		ostringstream joint_lcr_path;
		joint_lcr_path << "..\\data\\lcr_joints_" << file_num << ".txt";
		ifstream infile_lcr(joint_lcr_path.str(), ios::_Nocreate);
		if (infile_lcr.is_open())
		{

			//Read kinect joints data
			string s;
			int index = 0;
			for (int i = 0; i < 6 * 25; i++)
			{
				infile >> s;
				stringstream ss(s);
				if ((i - 2) % 6 == 0)
				{
					ss >> read_joint[i / 6].X;
				}
				else if ((i - 3) % 6 == 0)
				{
					ss >> read_joint[i / 6].Y;
				}
				else if ((i - 4) % 6 == 0) {
					ss >> read_joint[i / 6].Z;
				}
				if ((i - 5) % 6 == 0)
				{
					ss >> status[i / 6];
				}
			}
			//Read lcr-net joints data
			index = 0;
			for (int i = 0; i < 13; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					infile_lcr >> s;
					stringstream ss(s);
					switch (j)
					{
					case 0:
						ss >> lcr_joint[i].X;
					case 1:
						ss >> lcr_joint[i].Y;
					case 2:
						ss >> lcr_joint[i].Z;
					default:
						break;
					}
				}
			}
			//Let the middle of left and right hip joint points as standard
			double tmpx_kinect = (read_joint[12].X + read_joint[16].X) / 2;
			double tmpy_kinect = (read_joint[12].Y + read_joint[16].Y) / 2;
			double tmpz_kinect = (read_joint[12].Z + read_joint[16].Z) / 2;
			double tmpx_lcr = (lcr_joint[4].X + lcr_joint[5].X) / 2;
			double tmpy_lcr = (lcr_joint[4].Y + lcr_joint[5].Y) / 2;
			double tmpz_lcr = (lcr_joint[4].Z + lcr_joint[5].Z) / 2;
			for (int i = 0; i < 25; i++)
			{
				if (status[i] == 0)
				{
					read_joint[i].X = tmpx_kinect + (lcr_joint[pairs[i]].X - tmpx_lcr);
					read_joint[i].Y = tmpy_kinect + (lcr_joint[pairs[i]].Y - tmpy_lcr);
					read_joint[i].Z = tmpz_kinect + (lcr_joint[pairs[i]].Z - tmpz_lcr);
				}
			}
		}

	}
	//Save processed data
	ofstream new_kinect_file("..\\code\\fitted_ model_1001\\Joints3DPosition_1001.txt");
	for (int i = 0; i < 25; i++)
	{
		new_kinect_file << i << " " << joints_name[i] << " " << read_joint[i].X << " " << read_joint[i].Y << " " << read_joint[i].Z << " " << status[i] << endl;
	}
}

int main() {

	wrong_joint_process();
	return 0;
}