#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <highgui.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

using namespace std;
using namespace cv;
#define endl '\n'

char origfile[100];
char thresfile[100];
char morphfile[100];
char cannyfile[100];
char resultfile[100];

#define SAVE_ORIG
#define SAVE_THRES
#define SAVE_MORPH
#define SAVE_CANNY
#define SAVE_RESULT
const int thres = 20;
const int op = MORPH_OPEN;
const int elem = 0;
const int size = 3;
const int iter = 2; 
// 超参数

int main()
{
	cerr << "读取视频开始" << endl;

	VideoCapture capture("C:/Users/Cindy/Desktop/技创辅的团队/RAW DATA/11.avi");  
	Mat frame;
	Mat currentframe, previousframe, difframe;

	//读取一帧处理  
	while (true)
	{
		static int cnt = 0;
		if (!capture.isOpened())
		{
			cout << "read video failure" << endl;
			return -1;
		}

		capture >> frame;
		if(!frame.data) break;
		previousframe = frame.clone(); //第一帧
		++cnt;
#ifdef SAVE_ORIG
		sprintf_s(origfile, "C:/Users/Cindy/Desktop/[Don't_Delete!]Project1/FD Related/11/ORIGINAL/Original %d.png", cnt);
 		imwrite(origfile, frame);
#endif

 		capture >> frame;
		if(!frame.data) break;
		currentframe = frame.clone();  //第二帧
		++cnt;
		if(cnt % 10 == 0) cerr << "Frame " << cnt << endl;
#ifdef SAVE_ORIG
		sprintf_s(origfile, "C:/Users/Cindy/Desktop/[Don't_Delete!]Project1/FD Related/11/ORIGINAL/Original %d.png", cnt);
 		imwrite(origfile, frame);
#endif

 		cvtColor(previousframe, previousframe, CV_BGR2GRAY);
		cvtColor(currentframe, currentframe, CV_BGR2GRAY);
		absdiff(currentframe, previousframe, difframe); //作差求绝对值

		threshold(difframe, difframe, 15, 255.0, CV_THRESH_BINARY);
#ifdef SAVE_THRES
		sprintf_s(thresfile, "C:/Users/Cindy/Desktop/[Don't_Delete!]Project1/FD Related/11/THRES/Thres %d.png", cnt/2);
		imwrite(thresfile, difframe);
#endif

		Mat element = getStructuringElement(elem, Size(size, size));
		morphologyEx(difframe, difframe, op, element, Point(-1, -1), iter); 
		// 开运算：先腐蚀，再膨胀，可清除一些小东西(亮的)，放大局部低亮度的区域
#ifdef SAVE_MORPH
		sprintf_s(morphfile, "C:/Users/Cindy/Desktop/[Don't_Delete!]Project1/FD Related/11/MORPH/Morph %d.png", cnt/2);
		imwrite(morphfile, difframe);
#endif
	}//end while (reading videocapture)
	cerr << endl 
		<< "读取视频完成" << endl;

//-----------------------------------------------------------------------------------------------------------------//

	cerr << "轮廓识别开始" << endl;
	for(int frame_idx = 1; ; frame_idx++)
	{
		sprintf_s(morphfile, "C:/Users/Cindy/Desktop/[Don't_Delete!]Project1/FD Related/11/MORPH/Morph %d.png", frame_idx);
		Mat imageSource = 
			imread(morphfile, 0); // 0使之灰度返回
		if(!imageSource.data) break;

		Mat image;

		GaussianBlur(imageSource, image, Size(3,3), 0);
		Canny(image, image, 150, 250);
#ifdef SAVE_CANNY
		sprintf_s(cannyfile, "C:/Users/Cindy/Desktop/[Don't_Delete!]Project1/FD Related/11/CANNY/Canny %d.png", frame_idx);
		imwrite(cannyfile, image);
#endif

#ifdef SAVE_RESULT
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;

		// imshow("Source Image", imageSource);
		findContours(image, contours, hierarchy, RETR_EXTERNAL, 
					CHAIN_APPROX_NONE, Point()); 

		for(int origframe_idx = 2*frame_idx - 1; origframe_idx <= 2*frame_idx; origframe_idx++) // 对前后两帧依次操作
		{
			Mat orig;
			char origfile[120];
			sprintf_s(origfile, "C:/Users/Cindy/Desktop/[Don't_Delete!]Project1/FD Related/11/ORIGINAL/Original %d.png", 
							origframe_idx);
			orig = imread(origfile);

			if(origframe_idx % 10 == 0)
				cerr << "frame " << origframe_idx << ": " << contours.size() << " contours" << endl;
			for(int contour_idx = 0; contour_idx < contours.size(); contour_idx++)
			{
				int minx = 32767, miny = 32767, maxx = 0, maxy = 0;
				for(int point_idx = 0; point_idx < contours[contour_idx].size(); point_idx++)
				{
					if(contours[contour_idx][point_idx].x < minx) minx = contours[contour_idx][point_idx].x;
					if(contours[contour_idx][point_idx].y < miny) miny = contours[contour_idx][point_idx].y;
					if(contours[contour_idx][point_idx].x > maxx) maxx = contours[contour_idx][point_idx].x;
					if(contours[contour_idx][point_idx].y > maxy) maxy = contours[contour_idx][point_idx].y;
				}

				// EXPAND
				minx = (minx - 10 < 0)? 0 : (minx - 10);
				maxx = (maxx + 10 > 1919)? 1919 : (maxx + 10);
				miny = (miny - 10 < 0)? 0 : (miny - 10);
				maxy = (maxy + 10 > 1199)? 1199 : (maxy + 10);
				
				/*void rectangle(Mat& img, Point pt1, Point pt2, 
					const Scalar& color, int thickness=1, int lineType=8,int shift=0)*/
				rectangle(orig, Point(minx, miny), Point(maxx, maxy), Scalar(0, 0, 255));
			} // endfor: contour_idx

			sprintf_s(resultfile, "C:/Users/Cindy/Desktop/[Don't_Delete!]Project1/FD Related/11/RESULT/Result %d.png", origframe_idx);
			imwrite(resultfile, orig);

		} // endfor: origframe_idx
#endif
	} // endfor: frame_idx	

	return 0;
}