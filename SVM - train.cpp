#include <stdio.h>  
#include <time.h>  
#include <opencv2/opencv.hpp>  
#include <opencv/cv.h>  
#include <iostream> 
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/ml/ml.hpp>  
#include <io.h> //查找文件相关函数: _finddata_t _findfirst, _findnext, _findclose
 
using namespace std;
using namespace cv;
using namespace cv::ml;
 
void getFiles(string path, vector<string>& files);
void getBubble(Mat& trainingImages, vector<int>& trainingLabels);
void getNoBubble(Mat& trainingImages, vector<int>& trainingLabels);

#define AUTO
 
int main()
{
    //获取训练数据
    Mat classes;
    Mat trainingData;
    Mat trainingImages;
 
    vector<int> trainingLabels;
 
    // getBubble()与getNoBubble()将获取一张图片后会将图片（特征）写入
    //  到容器中，紧接着会将标签写入另一个容器中，这样就保证了特征
    //  和标签是一一对应的关系.push_back(0)或者push_back(1)其实就是
    //  我们贴标签的过程。
    getBubble(trainingImages, trainingLabels);
    getNoBubble(trainingImages, trainingLabels);

    cout << "文件读取完毕，开始初始化模型" << endl;
    //在主函数中，将getBubble()与getNoBubble()写好的包含特征的矩阵拷贝给trainingData，将包含标签的vector容器进行类
    //型转换后拷贝到trainingLabels里，至此，数据准备工作完成，trainingData与trainingLabels就是我们要训练的数据。
    Mat(trainingImages).copyTo(trainingData);
    trainingData.convertTo(trainingData, CV_32FC1);
    Mat(trainingLabels).copyTo(classes);
    //classes.convertTo(classes, CV_32SC1); // S: signed int

    Ptr<TrainData> tData = TrainData::create(trainingData, ROW_SAMPLE, classes);

#ifndef AUTO
    // 创建分类器并设置参数
    Ptr<SVM> SVM_params = SVM::create();
    SVM_params->setType(SVM::C_SVC); // SVC：支持向量分类（Classification）
    SVM_params->setKernel(SVM::LINEAR);  //核函数
    SVM_params->setDegree(0);
    SVM_params->setGamma(1);
    SVM_params->setCoef0(0);
    SVM_params->setC(1);
    SVM_params->setNu(0);
    SVM_params->setP(0);
    SVM_params->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 0.01));
    cout << "初始化模型完毕，开始训练模型" << endl;
    // 训练分类器
    SVM_params->train(tData);
#else
    Ptr<SVM> SVM_params = SVM::create();
    cout << "初始化模型完毕，开始训练模型" << endl;
    SVM_params->trainAuto(tData);
#endif
    cout << "模型训练完毕，开始保存模型" << endl;
    //保存模型
    SVM_params->save("D:\\Documents\\Datasets\\Crazy_Waterdrops\\svmtest\\svm_auto.xml");
    cout << "模型保存完毕" << endl;
    getchar(); // 暂停一下，让我们能看到命令提示符上输出的结果
    return 0;
}
 
 
 
void getFiles(string path, vector<string>& files) // 把路径下所有文件的【路径+文件名】存在这个vector里
{
    intptr_t   hFile = 0; // pointer to handle of file
    struct _finddata_t fileinfo;
    string p;
    int i = 30;
    if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
    {
 
        do
        {
            if ((fileinfo.attrib &  _A_SUBDIR)) // unsigned attrib: 文件属性
                // _A_ARCH（存档）、_A_HIDDEN（隐藏）、_A_NORMAL（正常）、_A_RDONLY（只读）、_A_SUBDIR（文件夹）、
                // _A_SYSTEM（系统），位组合。“与”表示只看这一位的属性。
            {
                if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
                    getFiles(p.assign(path).append("\\").append(fileinfo.name), files); // 递归进入子文件夹
            }
            else
            {
                files.push_back(p.assign(path).append("\\").append(fileinfo.name));
            }
 
        } while (_findnext(hFile, &fileinfo) == 0);
 
        _findclose(hFile); 
    }
}
 


//获取正样本
//并贴标签为1
void getBubble(Mat& trainingImages, vector<int>& trainingLabels)
{
    string filePath = "D:\\Documents\\Datasets\\Crazy_Waterdrops\\svmtest\\1"; //正样本路径
    vector<string> files;
    getFiles(filePath, files);
    int number = files.size();
    for (int i = 0; i < number; i++)
    {
        Mat  SrcImage = imread(files[i].c_str());
        SrcImage = SrcImage.reshape(1, 1); // 变单通道单行
        trainingImages.push_back(SrcImage);
        trainingLabels.push_back(1);//该样本是正例，用1表示
    }
}
 
//获取负样本
//并贴标签为0
void getNoBubble(Mat& trainingImages, vector<int>& trainingLabels)
{
    string filePath = "D:\\Documents\\Datasets\\Crazy_Waterdrops\\svmtest\\0"; //负样本路径
    vector<string> files;
    getFiles(filePath, files);
    int number = files.size();
    for (int i = 0; i < number; i++)
    {
        Mat  SrcImage = imread(files[i].c_str());
        SrcImage = SrcImage.reshape(1, 1); // 变单通道单行
        trainingImages.push_back(SrcImage);
        trainingLabels.push_back(0); //该样本是负例，用0表示
    }
}

/*
【注释，勿误删】

struct _finddata_t

       {

            unsigned attrib;

            time_t time_create;

            time_t time_access;

            time_t time_write;

            _fsize_t size;

            char name[_MAX_FNAME]; // 文件名

       };

long _findfirst( char *filespec, struct _finddata_t *fileinfo )；
返回值：如果查找成功的话，将返回一个long型的唯一的查找用的句柄（就是一个唯一编号）。这个句柄将在_findnext函数中被使用。若失败，则返回-1。
参数：
filespec：标明文件的字符串，可支持通配符。比如：*.c，则表示当前文件夹下的所有后缀为C的文件。
fileinfo ：这里就是用来存放文件信息的结构体的指针。这个结构体必须在调用此函数前声明，不过不用初始化，
只要分配了内存空间就可以了。函数成功后，函数会把找到的文件的信息放入这个结构体中。

int _findnext( long handle, struct _finddata_t *fileinfo );
返回值：若成功返回0，否则返回-1。
参数：
handle：即由_findfirst函数返回回来的句柄（编号）。
fileinfo：文件信息结构体的指针。找到文件后，函数将该文件信息放入此结构体中。

int _findclose( long handle );
返回值：成功返回0，失败返回-1。
参数：
handle ：_findfirst函数返回回来的句柄。

先用_findfirst查找第一个文件，若成功则用返回的句柄调用_findnext函数查找其他的文件，
当查找完毕后，用_findclose函数结束查找。
下面我们就按照这样的思路来编写一个查找C:\WINDOWS文件夹下的所有exe可执行文件的程序。

#include <stdio.h>
#include <io.h>

const char *to_search="C:\\WINDOWS\\*.exe";        //欲查找的文件，支持通配符

int main()

{
    long handle;                                 //用于查找的句柄

    struct _finddata_t fileinfo;                 //文件信息的结构体

    handle=_findfirst(to_search, &fileinfo);     //第一次查找

    if(-1==handle)return -1;

    printf("%s\n",fileinfo.name);                         //打印出找到的文件的文件名

    while(!_findnext(handle,&fileinfo))               //循环查找其他符合的文件，知道找不到其他的为止。 !0即1继续，!(-1)即0跳出。

    {

         printf("%s\n",fileinfo.name);

   }

    _findclose(handle);                                      //别忘了关闭句柄

    system("pause");

    return 0;

}

C++: Mat Mat::reshape(int cn, int rows=0) const
cn: 表示通道数(channels), 如果设为0，则表示保持通道数不变，否则则变为设置的通道数。
rows: 表示矩阵行数。 如果设为0，则表示保持原有的行数不变，否则则变为设置的行数。
通道数不变，变行向量： Mat dst = data.reshape(0, 1);（变1行）
通道数不变，变列向量： Mat dst = data.reshape(0, data.rows*data.cols);（有多少元素是多少行）
或者转置:
Mat dst = data.reshape(0, 1);      //序列成行向量
Mat dst = data.reshape(0, 1).t();  //序列成列向量
变化之前的  rows*cols*channels = 变化之后的 rows*cols*channels 整除不了就报错
序列化时，OpenCV行优先（Matlab列优先）

————————————————
版权声明：本文为CSDN博主「北顾+」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_40238526/article/details/92686721


*/