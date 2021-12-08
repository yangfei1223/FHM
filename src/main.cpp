#include "crf.h"
#include "segmentation.h"
#include <sys/time.h>
//#include "../bmpFile.h"

//#define __DEBUG__

//调试函数
void MeanSiftSegmentation()
{

	int frameID;
	LSegmentation2D *segmentation;
	for (frameID = 0;frameID < 96;frameID++)
	{

		//baselayer
		segmentation = new LMeanShiftSegmentation2D(5, 5, 20, "/home/yangfei/Datasets/Result/KITTI/MeanShift/50x50/", ".msh");

		char fileName[256];
		//获取图像文件名
		sprintf(fileName, "/home/yangfei/Datasets/Data/KITTI/um/Images/Test/um_%06d.png", frameID);																					 //读取图像，转换到LUV颜色空间
		LLuvImage luvImage(fileName);
		printf("segmenting image %s ...\n", fileName);
		//保存文件名
		LSegmentImage segmentImage;
		sprintf(fileName, "%sum_%06d%s", segmentation->folder, frameID, segmentation->extension);
		segmentation->Segment(luvImage, segmentImage);
		if (segmentImage.GetPoints() > 0) segmentImage.LImage<int>::Save(fileName);
	}
	delete segmentation;

}

void TestMeanSiftRes()
{
	int frameID = 10;
	char filename[256];
	const char *datasetName = "um";
	const char *segScale = "70x60";
//	sprintf(filename, "Result/KITTI/%s/MeanShift/%s/Train/%s_%06d.msh", datasetName, segScale, datasetName,frameID);
    sprintf(filename, "/home/yangfei/Datasets/Result/KITTI/MeanShift/50x30//%s_%06d.msh", datasetName,frameID);
	int width, height, bands;
	int *pData;
	FILE *fp = fopen(filename, "rb");
	fread(&width, 4, 1, fp);
	fread(&height, 4, 1, fp);
	fread(&bands, 4, 1, fp);
	cout << width << " " << height << " " << bands << endl;
	pData = new int[width*height*bands];
	fread(pData, sizeof(int), width*height*bands, fp);
	fclose(fp);
	Mat Img(height, width, CV_8U);
	int *pCur = pData;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			Img.at<uchar>(y, x) = *pCur++;
		}
	}
//	imshow("seg", Img);
//	waitKey(0);
    sprintf(filename, "/home/yangfei/Datasets/Result/KITTI/MeanShift/%s_%06d_5030.png", datasetName,frameID);
    imwrite(filename,Img);
	delete[] pData;
}


#define UM 96
#define UMM 94
#define UU 100
void TestLidarRes()
{
	int frameID = 0;
	const char *datasetName = "uu";
	char imagefileName[256];
	char calibfileName[256];
	char lidarfileName[256];
	char resfileName[256];
    char filename[256];
	cLidarProcess gLidar;
	for (frameID = 0;frameID < 98;frameID++)
	{
        sprintf(filename,"%s_%06d",datasetName,frameID);
		sprintf(imagefileName, "/media/yangfei/Repository/KITTI/data_road/training/image_2/%s.png",filename);
		sprintf(calibfileName, "/media/yangfei/Repository/KITTI/data_road/training/calib/%s.txt", filename);
		sprintf(lidarfileName, "/media/yangfei/Repository/KITTI/data_road/training/velodyne/%s.bin", filename);
		sprintf(resfileName, "/home/yangfei/Workspace/HFM/scripts/LightGBMClassifier/train_pred/%s.txt", filename);
		gLidar.TestLidarFeature(imagefileName, calibfileName, lidarfileName, resfileName ,filename);

	}
}

void TestImageRes()
{
	FILE *fp = NULL;
	int i, j, k, frameID = 0;
	const char *datasetName = "um";
	char filename[256];
	int width, height, classNo;
	for (frameID = 10;frameID < 96;frameID++)
	{
		sprintf(filename, "Result/KITTI/%s/Dense/Test/%s_%06d.dns", datasetName, datasetName, frameID);
		printf("%s\n", filename);
		fp = fopen(filename, "rb");
		fread(&width, sizeof(int), 1, fp);
		fread(&height, sizeof(int), 1, fp);
		fread(&classNo, sizeof(int), 1, fp);
		//Mat res(height, width, CV_8UC3);
		sprintf(filename, "Data/KITTI/%s/Images/Test/%s_%06d.png", datasetName, datasetName, frameID);
		Mat res = imread(filename);
		for (i = 0;i < height;i++)
		{
			for (j = 0;j < width;j++)
			{
				double data[2];
				fread(data, sizeof(double), classNo, fp);
				if (data[0] > data[1])
				{
					/*Vec3b color = Vec3b(0, 255, 0);
					res.at<Vec3b>(i, j) = color;*/
					res.at<Vec3b>(i, j)[1] = min(res.at<Vec3b>(i, j)[1] + 100, 255);
				}
				/*else
				{
					Vec3b color = Vec3b(0, 0, 255);
					res.at<Vec3b>(i, j) = color;
				}*/
			}
		}
		fclose(fp);
		imshow(filename, res);
		waitKey(0);
		destroyAllWindows();
//		Rmw_Write24BitImg2BmpFile(res.data, res.cols, res.rows, "Debug/img_res.bmp");
	}
}

void TestFeatureMap()
{
	int frameID = 42;
	char filename[256];
	const char *datasetName = "uu";
	const char *featureName = "Texton";
	const char *featureExtension = ".txn";
	sprintf(filename, "Result/KITTI/%s/Feature/%s/Train/%s_%06d%s",datasetName,featureName,datasetName, frameID,featureExtension);
	Mat Img;
	int width, height, bands;
	unsigned short *pData, *pCur;
	FILE *fp = fopen(filename, "rb");
	fread(&width, 4, 1, fp);
	fread(&height, 4, 1, fp);
	fread(&bands, 4, 1, fp);
	cout << width << " " << height << " " << bands << endl;
	pData = new unsigned short[width*height*bands];
	fread(pData, sizeof(unsigned short), width*height*bands, fp);
	fclose(fp);

	if (bands == 1)
	{
		Img.create(height, width, CV_8U);
		pCur = pData;
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				Img.at<uchar>(y, x) = *pCur++;
			}
		}
	}
	if (bands == 3)
	{
		Img.create(height, width, CV_8UC3);
		pCur = pData;
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				Img.at<Vec3b>(y, x)[0] = *pCur++;
				Img.at<Vec3b>(y, x)[1] = *pCur++;
				Img.at<Vec3b>(y, x)[2] = *pCur++;
			}
		}
	}

	imshow("img", Img);
	waitKey(0);
	delete[] pData;
	if (bands == 3) 
	{
//		Rmw_Write24BitImg2BmpFile(Img.data, Img.cols, Img.rows, "Debug/4.bmp");
	}
	else
	{
//		Rmw_Write8BitImg2BmpFile(Img.data, Img.cols, Img.rows, "Debug/4.bmp");
	}
}

void TestLidarFeatureMap()
{
	int frameID = 17;
	const char *datasetName = "um";
	char imagefileName[256];
	char gtfileName[256];
	char calibfileName[256];
	char lidarfileName[256];
	char resfileName[256];
	cLidarProcess gLidar;
	//for (frameID = 17;frameID < 18;frameID++)
	{
		sprintf(imagefileName, "Data/KITTI/%s/Images/Train/%s_%06d.png", datasetName, datasetName, frameID);
		sprintf(gtfileName, "Data/KITTI/%s/GroundTruth/Train/%s_%06d.png",datasetName, datasetName, frameID);
		sprintf(calibfileName, "Data/KITTI/%s/calib/Train/%s_%06d.txt", datasetName, datasetName, frameID);
		sprintf(lidarfileName, "Data/KITTI/%s/Lidar/Train/%s_%06d.bin", datasetName, datasetName, frameID);
		gLidar.DoNext(imagefileName, gtfileName, calibfileName, lidarfileName);
		gLidar.Debug();
	}
}

#if 1
int main()
{
//	MeanSiftSegmentation();
//	TestMeanSiftRes();
	TestLidarRes();
	//TestImageRes();
	//TestFeatureMap();
//	TestLidarFeatureMap();
	return(0);
}
#endif

float t_spend=0;
#if 0
int main(int argc, char *argv[])
{

	//读取主函数参数，from和to为起始帧数
	int from = 0, to = -1;
	if (argc == 3) from = atoi(argv[1]), to = atoi(argv[2]);
	if (argc == 4) from = atoi(argv[2]), to = atoi(argv[3]);

//	LDataset *dataset = new LKITTIDataset();
//	LDataset *dataset = new LKITTIumDataset();
	//LDataset *dataset = new LKITTIummDataset();
	//LDataset *dataset = new LKITTIuuDataset();
	LDataset *dataset = new LKITTIValidationset();        //Hierarchical Fusion Condition Random Filed
//	LDataset *dataset = new LKITTIPairwiseCompare();      //Pairwise Condition Random Field
//	LDataset *dataset = new LKITTIHighOrderCompare();       //High Order Condition Random Field



	//创建CRF模型
	LCrf *crf = new LCrf(dataset);
	//用数据集构造CRF模型，包含特征，势函数等参数和网络结果的设置，添加各种对象
	dataset->SetCRFStructure(crf);
	//list data
	printf("List all images...\n");
	for (int i=0;i<dataset->allImageFiles.GetCount();i++)
	{
		printf("%s\n", dataset->allImageFiles[i]);
	}
	printf("List train images...\n");
	for (int i = 0;i < dataset->trainImageFiles.GetCount();i++)
	{
		printf("%s\n", dataset->trainImageFiles[i]);
	}
	printf("List test images...\n");
	for (int i = 0;i < dataset->testImageFiles.GetCount();i++)
	{
		printf("%s\n", dataset->testImageFiles[i]);
	}
	//无监督分割,mean-sift聚类算法生成segment图，作为高阶层
//	crf->Segment(dataset->allImageFiles, from, to);
	//训练特征，利用训练集提取不同特征，生成不同聚类中心，生成聚类中心只使用全部的训练集
//	crf->TrainFeatures(dataset->trainImageFiles);	//训练集
	//特征评估，提取图像特征，分配每个特征向量到相应的聚类中心，从而降低特征维数
//	crf->EvaluateFeatures(dataset->allImageFiles, from, to);
	//训练势函数，一阶以及高阶数据项的训练，即分类器的训练
	crf->TrainPotentials(dataset->trainImageFiles);
	//评价，即测试，用分类器输出每个类别的概率
	crf->EvaluatePotentials(dataset->testImageFiles, from, to);
	//模型推断，alpha-expansion算法求解
//    float t_spend = 0;
//    struct timeval tv1,tv2;
//    gettimeofday(&tv1,NULL);
//	crf->Solve(dataset->testImageFiles, from, to);
//    gettimeofday(&tv2,NULL);
//    t_spend=(tv2.tv_sec+tv2.tv_usec/1e+6)-(tv1.tv_sec+tv1.tv_usec/1e+6);
    printf("time spend : %f \n",t_spend);
	//结果评估统计
	crf->Confusion(dataset->testImageFiles, "Result/KITTI/val/Crf/results.txt");

	delete crf;
	delete dataset;

	return(0);

}

#endif

