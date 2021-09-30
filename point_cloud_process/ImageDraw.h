#pragma once
class ImageDraw
{
public:
	ImageDraw();
	ImageDraw(int argc, char **argv);
	~ImageDraw();



	void initDisplayMode(unsigned int mode);	//Set display mode
	void initWindowPosition(int x, int y);	//Set the coordinate point position of the upper left corner of the window
	void initWindowSize(int width, int heigh);	//Set window size
	void setWindowTitle(char *title);	//Set window title
	void setClearColor(float red, float green, float blue, float alpha);	//Set the background color
	void setMatrixMode(unsigned int mode);	//Set projection type
	void setOrtho2D(double left, double right, double bottom, double top);	//Display window size
	void displayFunc(void(*func)(void));


	void printError();

private:

	int			nWinSizeWidth;	//
	int			nWinSizeHeight;	//

};

