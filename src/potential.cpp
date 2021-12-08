#include <string.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#endif

#define USE_LIDAR
#include "std.h"
#include "crf.h"
#include "learning.h"
#include "lidarproc.h"

//�״��ඨ��
cLidarProcess gLidar;
void LidarDoNext(bool isTrain, char *filenmae,const char *datasetname)
{
	char pathname[256];
	char imagefileName[256];
	char gtfileName[256];
	char calibfileName[256];
	char lidarfileName[256];
	sprintf(pathname, "/home/yangfei/Datasets/Data/KITTI/%s", datasetname);
	if (isTrain)
	{
		cout << "extract lidar feature ...\n";
		sprintf(imagefileName, "%s/Images/%s.png",pathname, filenmae);
		sprintf(gtfileName, "%s/GroundTruth/%s.png", pathname, filenmae);
		sprintf(calibfileName, "%s/calib/%s.txt", pathname, filenmae);
		sprintf(lidarfileName, "%s/Lidar/%s.bin", pathname, filenmae);
		gLidar.DoNext(imagefileName, gtfileName, calibfileName, lidarfileName);
	}
	else
	{
		cout << "compute lidar position ...\n";
		sprintf(imagefileName, "%s/Images/%s.png", pathname, filenmae);
		sprintf(calibfileName, "%s/calib/%s.txt", pathname, filenmae);
		sprintf(lidarfileName, "%s/Lidar/%s.bin", pathname, filenmae);
		gLidar.DoNext(imagefileName, calibfileName, lidarfileName);
	}
}

LPotential::LPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension, int setClassNo)
{
	classNo = setClassNo, dataset = setDataset, crf = setCrf, domain = setDomain;
	trainFolder = setTrainFolder, trainFile = setTrainFile, evalFolder = setEvalFolder, evalExtension = setEvalExtension;
	nodeOffset = edgeOffset = 0;
}

#ifdef MULTITHREAD

struct LEvaluateParams
{
	LPotential *potential;
	char *fileName;
};

thread_return EvaluateThread(void *par)
{
	LEvaluateParams *params = (LEvaluateParams *)par;
	params->potential->Evaluate(params->fileName);
	return(thread_defoutput);
}
#endif

void LPotential::Evaluate(LList<char *> &imageFiles, int from, int to)
{
	if (to == -1) to = imageFiles.GetCount();
#ifdef MULTITHREAD
	int i;
	int processors = GetProcessors(), running = 0;
	thread_type *threads;
	LEvaluateParams *params;
	LCrf **crfs;

	threads = new thread_type[processors];
	memset(threads, 0, processors * sizeof(thread_type));
	params = new LEvaluateParams[processors];
	crfs = new LCrf *[processors];
	InitializeCriticalSection();

	for (i = 0; i < processors; i++)
	{
		crfs[i] = new LCrf(dataset);
		dataset->SetCRFStructure(crfs[i]);
		for (int j = 0; j < crfs[i]->features.GetCount(); j++) crfs[i]->features[j]->LoadTraining();
	}

	int index = -1, ind = from;
	for (i = 0; (i < crf->potentials.GetCount()) && (index == -1); i++) if (crf->potentials[i] == this) index = i;

	if (index != -1) for (i = 0; i < processors; i++) if (ind < to)
	{
		params[i].potential = (LPotential *)crfs[i]->potentials[index], params[i].fileName = imageFiles[ind];
		params[i].potential->LoadTraining();
		threads[i] = NewThread(EvaluateThread, &params[i]);
		if (threads[i] != 0)
		{
			printf("Evaluating image %d..\n", ind);
			running++;
			ind++;
		}
	}

	while (running)
	{
		for (i = 0; i < processors; i++) if (threads[i])
		{
			if (ThreadFinished(threads[i]))
			{
				CloseThread(&threads[i]);
				running--;

				if (ind < to)
				{
					params[i].fileName = imageFiles[ind];
					threads[i] = NewThread(EvaluateThread, &params[i]);
					if (threads[i] != 0)
					{
						printf("Evaluating image %d..\n", ind);
						running++;
						ind++;
					}
				}
			}
		}
		Sleep(0);
	}
	DeleteCriticalSection();
	for (i = 0; i < processors; i++) delete crfs[i];

	delete[] crfs;
	delete[] threads;
	delete[] params;
#else
	for (int i = from; i < to; i++)
	{
		printf("Evaluating image %d %s..\n", i, imageFiles[i]);
		Evaluate(imageFiles[i]);
	}
#endif
}

int LPotential::GetNodeCount()
{
	return(0);
}

int LPotential::GetEdgeCount()
{
	return(0);
}

LUnaryPixelPotential::LUnaryPixelPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LBaseCrfLayer *setLayer, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension, int setClassNo, double setUnaryFactor) : LPotential(setDataset, setCrf, setDomain, setTrainFolder, setTrainFile, setEvalFolder, setEvalExtension, setClassNo)
{
	layer = setLayer;
	unaryFactor = setUnaryFactor;
	unaryCount = width = height = 0;
	unaryCosts = NULL;
}

LUnaryPixelPotential::~LUnaryPixelPotential()
{
	if (unaryCosts != NULL) delete[] unaryCosts;
}

void LUnaryPixelPotential::AddCosts(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes)
{
	int i, j;
	if (domain == costDomain)
	{
		if (!layer->range)		//baselayer range=0
		{
			//ѭ��ÿ��������
			for (i = 0; i < unaryCount; i++) if (!layer->active[i])		//����Ǽ���
			{
				//����ն˱ߣ����ǲ����ڵ�ǰ��ǩ�ĵ���ӵ��յ㣬��Դ���ǵ�ǰ��ǩ��cost��������ǵ�ǰ���ر�ǩ��cost
				g->add_tweights(nodes[i + layer->nodeOffset], unaryCosts[i * classNo + label], unaryCosts[i * classNo + layer->labels[i]]);
			}
		}
		else
		{
			int range = layer->range;
			for (i = 0; i < unaryCount; i++)
			{
				g->add_tweights(nodes[i * range + layer->nodeOffset], 0, (!layer->active[i]) ? unaryCosts[i * classNo + layer->labels[i]] : LMath::positiveInfinity);
				g->add_tweights(nodes[i * range + range - 1 + layer->nodeOffset], unaryCosts[i * classNo + label + range - 1], 0);
				for (j = 0; j < range - 1; j++) g->add_edge(nodes[i * range + j + layer->nodeOffset], nodes[i * range + j + 1 + layer->nodeOffset], LMath::positiveInfinity, unaryCosts[i * classNo + label + j]);
			}

		}
	}
}

double LUnaryPixelPotential::GetCost(LCrfDomain *costDomain)
{
	if (domain == costDomain)
	{
		double cost = 0;
		for (int i = 0; i < unaryCount; i++) cost += unaryCosts[i * classNo + layer->labels[i]];
		return(cost);
	}
	else return(0);
}

void LUnaryPixelPotential::SetLabels()
{
	int i, j;

	double *dominant = new double[classNo];
	unsigned char *labels = layer->labels;

	//�ҵ���С�ı�ǩ
	for (i = 0; i < unaryCount; i++)
	{
		int min = 0;
		for (j = 1; j < classNo; j++) if (unaryCosts[i * classNo + j] < unaryCosts[i * classNo + min]) min = j;
		labels[i] = min;
	}
}

//��ȡ�״�����������д�ļ�
void LLidarUnaryPixelPotential::InitTrainData(LList<char *> &trainImageFiles)
{
	int n, i, j;
	char filename[256];
	sprintf(filename, "/home/yangfei/Datasets/Result/KITTI/%s/Lidar/denseFeature.train", dataset->datasetName);
	FILE* fp = NULL;
	fp = fopen(filename, "wt");
	//�ж��Ƿ���ȷ���ļ�
	if (!fp)
	{
		printf("can not open %s !\n",filename);
		exit(-1);
	}
	for (n = 0; n < fileSum; n++)
	{
		//�״���������
		LidarDoNext(true, trainImageFiles[n],dataset->datasetName);
		//д���
		for (i = 0; i < gLidar.m_height; i++)
		{
			for (j = 0; j < gLidar.m_width; j++)
			{
				if (gLidar.m_Valid2Img.at<uchar>(i, j) > 0)     //��Ч��
				{
					//Ȩ��,����ĵ���
					fprintf(fp, "%f\t", gLidar.m_Weights.at<float>(i, j));
					//��ǩ
					fprintf(fp, "%d\t", gLidar.m_Label.at<uchar>(i, j));
					//x,y,z,ref
					fprintf(fp, "%f\t%f\t%f\t%f\t", gLidar.m_Lidar2Img.at<Vec4f>(i, j)[0], gLidar.m_Lidar2Img.at<Vec4f>(i, j)[1], gLidar.m_Lidar2Img.at<Vec4f>(i, j)[2], gLidar.m_Lidar2Img.at<Vec4f>(i, j)[3]);
					//���ʣ��߶ȷ�����߶Ȳ�
					fprintf(fp, "%f\t%f\t%f\n", gLidar.m_Curvature.at<float>(i, j), gLidar.m_HeightVar.at<float>(i, j), gLidar.m_MaxHeightDiff.at<float>(i, j));
				}
			}
		}
	}
	fclose(fp);
}

void LLidarUnaryPixelPotential::UnInitTrainData()
{

}

//��ȡ����ͼ���״�����������д�ļ�
void LLidarUnaryPixelPotential::InitEvalData(char *evalImageFile)
{
	int i, j;
	//�����״�����
	LidarDoNext(true, evalImageFile,dataset->datasetName);
	FILE *fp = NULL;
	char filename[256];
	sprintf(filename, "/home/yangfei/Datasets/Result/KITTI/%s/Lidar/denseFeatureEval/%s.test", dataset->datasetName,evalImageFile);
	fp = fopen(filename, "wt");
	if (!fp)
	{
		printf("can not open %s !\n", filename);
		exit(-1);
	}
	//д���
	for (i = 0; i < gLidar.m_height; i++)
	{
		for (j = 0; j < gLidar.m_width; j++)
		{
			if (gLidar.m_Valid2Img.at<uchar>(i, j) > 0)     //��Ч��
			{
				//x,y,z,ref
				fprintf(fp, "%f\t%f\t%f\t%f\t", gLidar.m_Lidar2Img.at<Vec4f>(i, j)[0], gLidar.m_Lidar2Img.at<Vec4f>(i, j)[1], gLidar.m_Lidar2Img.at<Vec4f>(i, j)[2], gLidar.m_Lidar2Img.at<Vec4f>(i, j)[3]);
				//���ʣ��߶ȷ�����߶Ȳ�
				fprintf(fp, "%f\t%f\t%f\n", gLidar.m_Curvature.at<float>(i, j), gLidar.m_HeightVar.at<float>(i, j), gLidar.m_MaxHeightDiff.at<float>(i, j));
			}
		}
	}
	fclose(fp);
}

void LLidarUnaryPixelPotential::UnInitEvalData()
{

}

LLidarUnaryPixelPotential::LLidarUnaryPixelPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LBaseCrfLayer *setLayer, int setClassNo, double setUnaryFactor) : LUnaryPixelPotential(setDataset, setCrf, setDomain, setLayer, NULL, NULL, NULL, NULL, setClassNo, setUnaryFactor)
{
}

LLidarUnaryPixelPotential::~LLidarUnaryPixelPotential()
{
}


//ʵ��������ȡ����д�ļ����ⲿ��ʹ��LightGBMѵ��
void LLidarUnaryPixelPotential::Train(LList<char *> &trainImageFiles)
{
	//�ļ�����
	fileSum = trainImageFiles.GetCount();
	printf("Initializing training..\n");
	//��ʼ��ѵ������
	InitTrainData(trainImageFiles);

	printf("Uninitialising training..\n");
	UnInitTrainData();


}

//ʵ����Ҳ��������ȡ������
void LLidarUnaryPixelPotential::Evaluate(char *imageFileName)
{
	char *fileName;
	FILE *f;
	int i;

	//��ʼ��
	InitEvalData(imageFileName);

	UnInitEvalData();
}


//��ʼ������ȡcost
void LLidarUnaryPixelPotential::Initialize(LLabImage &labImage, char *imageFileName)
{
	int j, k, l;
	FILE *fp;
	char LidarClassifierOutfileName[256];

	width = labImage.GetWidth();
	height = labImage.GetHeight();

	if (unaryCosts != NULL) delete[] unaryCosts;
	unaryCosts = new double[width * height * classNo];
	unaryCount = width * height;

	double *data;
	data = new double[classNo];

	double *costData = unaryCosts;
	unsigned char *labelData = layer->labels;

	LidarDoNext(false, imageFileName,dataset->datasetName);
	
	sprintf(LidarClassifierOutfileName, "/home/yangfei/Datasets/Result/KITTI/%s/Lidar/denseClassifierOut/%s.txt", dataset->datasetName,imageFileName);
	fp = fopen(LidarClassifierOutfileName, "rt");
	if (!fp)
	{
		printf("can not open %s !\n", LidarClassifierOutfileName);
		exit(-1);
	}
	for (j = 0; j < height; j++) for (k = 0; k < width; k++, costData += classNo, labelData++)
	{
		//��ͼ��cost									
		data[0] = data[1] = -log(0.5);		//Ĭ�϶���0.5
		//������״�㣬������״��ƺ���
		if (gLidar.m_Valid2Img.at<uchar>(j, k) > 0)	//��Ч��
		{
			double tmplidarProb;
			fscanf(fp, "%lf", &tmplidarProb);
			tmplidarProb = max(tmplidarProb, 1.0e-200);		//������Сֵ1e-200
			data[0] = -log(tmplidarProb);			//��������
			data[1] = -log(1 - tmplidarProb);		//��������		
		}

		for (l = 0; l < classNo; l++)
		{
			costData[l] = unaryFactor * data[l];
		}

		if (dataset->unaryWeighted) for (l = 0; l < classNo; l++) costData[l] *= dataset->unaryWeights[l];

		*labelData = 0;
		//		for(l = 1; l < classNo; l++) if(costData[l] < costData[*labelData]) *labelData = l;
	}
	fclose(fp);
	delete[] data;
}

void LLidarUnaryPixelPotential::UnInitialize()
{
	if (unaryCosts != NULL) delete[] unaryCosts;
	unaryCosts = NULL;
}


LDenseUnaryPixelPotential::LDenseUnaryPixelPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LBaseCrfLayer *setLayer, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension, int setClassNo, double setUnaryFactor, int setSubSample, int setNumberOfShapes, int setMinimumRectangleSize, int setMaximumRectangleSize, double setMaxClassRatio) : LUnaryPixelPotential(setDataset, setCrf, setDomain, setLayer, setTrainFolder, setTrainFile, setEvalFolder, setEvalExtension, setClassNo, setUnaryFactor)
{
	int i;

	maxClassRatio = setMaxClassRatio;
	buckets = NULL, numExamples = 0;
	validPointsX = validPointsY = NULL;
	pointsSum = NULL;
	targets = NULL;
	weights = NULL;

	numberOfShapes = setNumberOfShapes;

	subSample = setSubSample;
	shapes = NULL;

	minimumRectangleSize = setMinimumRectangleSize;
	maximumRectangleSize = setMaximumRectangleSize;
	integralImages = NULL;
	buckets = NULL;

	LMath::SetSeed(0);
	shapes = new LShape[numberOfShapes];
	for (i = 0; i < numberOfShapes; i++)
	{
		int x, y, width, height;

		x = (int)LMath::RandomGaussian(0, maximumRectangleSize * maximumRectangleSize / 16);
		y = (int)LMath::RandomGaussian(0, maximumRectangleSize * maximumRectangleSize / 16);
		if (x > maximumRectangleSize) x = maximumRectangleSize;
		if (y > maximumRectangleSize) y = maximumRectangleSize;
		if (x < -maximumRectangleSize) x = -maximumRectangleSize;
		if (y < -maximumRectangleSize) y = -maximumRectangleSize;

		width = minimumRectangleSize + LMath::RandomInt(maximumRectangleSize - minimumRectangleSize);
		height = minimumRectangleSize + LMath::RandomInt(maximumRectangleSize - minimumRectangleSize);
		x -= width / 2, y -= height / 2;

		shapes[i].width = width / subSample;
		shapes[i].height = height / subSample;
		shapes[i].x = ((x + ((x > 0) ? 1 : ((x < 0) ? -1 : 0)) * subSample / 2) / subSample);
		shapes[i].y = ((y + ((y > 0) ? 1 : ((y < 0) ? -1 : 0)) * subSample / 2) / subSample);
	}
}

LDenseUnaryPixelPotential::~LDenseUnaryPixelPotential()
{
	if (shapes != NULL) delete[] shapes;
}

int *LDenseUnaryPixelPotential::GetTrainBoostingValues(int index, int core)
{
	int bucket = index / numberOfShapes, i, j;
	LShape *shape = &shapes[index % numberOfShapes];

	int n = 0;
	for (i = 0; i < fileSum; i++) for (j = 0; j < pointsSum[i]; j++, n++)
	{
		featureValues[core][n] = CalculateShapeFilterResponse(i, bucket, shape, validPointsX[i][j], validPointsY[i][j]);
	}
	return(featureValues[core]);
}

int *LDenseUnaryPixelPotential::GetEvalBoostingValues(int index)
{
	int bucket = index / numberOfShapes, i, j;
	LShape *shape = &shapes[index % numberOfShapes];

	int n = 0;
	for (i = 0; i < height; i++) for (j = 0; j < width; j++, n++) featureValues[0][n] = CalculateShapeFilterResponse(0, bucket, shape, j, i);
	return(featureValues[0]);
}

int *LDenseUnaryPixelPotential::GetTrainForestValues(int index, int *pixelIndexes, int count, int core)
{
	int bucket = index / numberOfShapes, i, j;
	LShape *shape = &shapes[index % numberOfShapes];

	int n = 0, pIndex = 0;
	for (i = 0; i < fileSum; i++) for (j = 0; (j < pointsSum[i]) && (pIndex < count); j++, n++) if (pixelIndexes[pIndex] == n)
	{
		featureValues[core][pIndex] = CalculateShapeFilterResponse(i, bucket, shape, validPointsX[i][j], validPointsY[i][j]);
		pIndex++;
	}
	return(featureValues[core]);
}

int LDenseUnaryPixelPotential::GetEvalForestValue(int index, int pixelIndex)
{
	return(CalculateShapeFilterResponse(0, index / numberOfShapes, &shapes[index % numberOfShapes], pixelIndex % width, pixelIndex / width));
}

int LDenseUnaryPixelPotential::GetLength(int index)
{
	return(buckets[index / numberOfShapes]);
}

int LDenseUnaryPixelPotential::GetSize(int index)
{
	return(shapes[index % numberOfShapes].width * shapes[index % numberOfShapes].height * subSample * subSample);
}

void LDenseUnaryPixelPotential::SaveTraining()
{
	learning->SaveTraining();
}

void LDenseUnaryPixelPotential::LoadTraining()
{
	learning->LoadTraining();
}

//һԪ�ƺ��������������ص��������ѵ��
void LDenseUnaryPixelPotential::Train(LList<char *> &trainImageFiles)		//����Ϊ��ѵ�����ļ�
{
	int i;

	//�ļ�����
	fileSum = trainImageFiles.GetCount();

	printf("Initializing training..\n");

	//��ʼ��ѵ������
	InitTrainData(trainImageFiles);

	//��������
	int totalFeatures = 0;
	for (i = 0; i < features.GetCount(); i++) totalFeatures += buckets[i];

	integralBuckets = new int[features.GetCount()];
	integralBuckets[0] = 0;
	for (i = 1; i < features.GetCount(); i++) integralBuckets[i] = integralBuckets[i - 1] + buckets[i - 1];

	integralPoints = new int[fileSum];
	integralPoints[0] = pointsSum[0];
	for (i = 1; i < fileSum; i++) integralPoints[i] = integralPoints[i - 1] + pointsSum[i];

	int processors = 1;
#ifdef MULTITHREAD
	processors = GetProcessors();
#endif

	featureValues = new int *[processors];

	for (i = 0; i < processors; i++)  featureValues[i] = new int[numExamples];
	learning->Train(classWeights, numExamples, targets, totalFeatures * numberOfShapes, weights);
	SaveTraining();

	if (integralBuckets != NULL) delete[] integralBuckets;
	if (integralPoints != NULL) delete[] integralPoints;

	if (featureValues != NULL)
	{
		for (i = 0; i < processors; i++) if (featureValues[i] != NULL) delete[] featureValues[i];
		delete[] featureValues;
	}

	printf("Uninitialising training..\n");
	UnInitTrainData();

	if (classWeights != NULL) delete[] classWeights;

	if (validPointsX != NULL)
	{
		for (i = 0; i < fileSum; i++) if (validPointsX[i] != NULL) delete[] validPointsX[i];
		delete[] validPointsX;
	}
	if (validPointsY != NULL)
	{
		for (i = 0; i < fileSum; i++) if (validPointsY[i] != NULL) delete[] validPointsY[i];
		delete[] validPointsY;
	}
	if (pointsSum != NULL) delete[] pointsSum;

	if (weights != NULL)
	{
		for (i = 0; i < numExamples; i++) if (weights[i] != NULL) delete[] weights[i];
		delete[] weights;
	}
	if (targets != NULL) delete[] targets;
}

//�ƺ�������
void LDenseUnaryPixelPotential::Evaluate(char *imageFileName)
{
	char *fileName;
	FILE *f;
	int i;

	//��ʼ��
	InitEvalData(imageFileName, &width, &height);

	//��������=������*ÿ����������������
	int totalFeatures = 0;
	for (i = 0; i < features.GetCount(); i++) totalFeatures += buckets[i];

	integralBuckets = new int[features.GetCount()];
	integralBuckets[0] = 0;
	for (i = 1; i < features.GetCount(); i++) integralBuckets[i] = integralBuckets[i - 1] + buckets[i - 1];

	fileName = GetFileName(evalFolder, imageFileName, evalExtension);
	f = fopen(fileName, "wb");
	if (f == NULL) _error(fileName);

	fwrite(&width, sizeof(int), 1, f);
	fwrite(&height, sizeof(int), 1, f);
	fwrite(&classNo, sizeof(int), 1, f);

	//������
	LCostImage classifications(width, height, classNo);
	featureValues = new int *[1];

	for (i = 0; i < numberOfShapes; i++) shapes[i].x *= subSample, shapes[i].y *= subSample, shapes[i].width *= subSample, shapes[i].height *= subSample;

	featureValues[0] = new int[width * height];
	//������������
	learning->Evaluate(classifications.GetData(), width * height, totalFeatures * numberOfShapes);
	for (i = 0; i < numberOfShapes; i++) shapes[i].x /= subSample, shapes[i].y /= subSample, shapes[i].width /= subSample, shapes[i].height /= subSample;

	if (integralBuckets != NULL) delete[] integralBuckets;

	if (featureValues != NULL)
	{
		if (featureValues[0] != NULL) delete[] featureValues[0];
		delete[] featureValues;
	}

	//���������
	fwrite(classifications.GetData(), sizeof(double), width * height * classNo, f);
	fclose(f);

	delete[] fileName;
	UnInitEvalData();
}

#ifdef USE_LIDAR
//��ʼ������
void LDenseUnaryPixelPotential::Initialize(LLabImage &labImage, char *imageFileName)
{
	int j, k, l;
	char *fileName;
	FILE *f;

	int width, height;

	LidarDoNext(false, imageFileName,dataset->datasetName);

	fileName = GetFileName(evalFolder, imageFileName, evalExtension);
	f = fopen(fileName, "rb");
	if (f == NULL) _error(fileName);

	//��ȡͼ���С��Ϣ
	fread(&width, sizeof(int), 1, f);
	fread(&height, sizeof(int), 1, f);
	fread(&classNo, sizeof(int), 1, f);

	if (unaryCosts != NULL) delete[] unaryCosts;
	unaryCosts = new double[width * height * classNo];
	unaryCount = width * height;

	double *data;
	data = new double[classNo];

	double *costData = unaryCosts;
	unsigned char *labelData = layer->labels;



	char LidarClassifierOutfileName[256];
	sprintf(LidarClassifierOutfileName, "/home/yangfei/Datasets/Result/KITTI/%s/Lidar/denseClassifierOut/%s.txt",dataset->datasetName, imageFileName);
	FILE *fp = fopen(LidarClassifierOutfileName, "rt");
	for (j = 0; j < height; j++) for (k = 0; k < width; k++, costData += classNo, labelData++)
	{
		//��ͼ��cost									
		double sum = 0;
		fread(data, sizeof(double), classNo, f);
		for (l = 0; l < classNo; l++)
		{
			sum += exp(data[l]);
		}

		double cost_pos, cost_neg;
		cost_pos = cost_neg = -log(0.5);		//Ĭ�϶���0.5
		//������״�㣬������״��ƺ���
		if (gLidar.m_Valid2Img.at<uchar>(j, k) > 0)	//��Ч��
		{
			double lidarProb = 0.5;
			fscanf(fp, "%lf", &lidarProb);
			lidarProb = max(lidarProb, 1.0e-200);
			cost_pos = -log(lidarProb);		//��������
			cost_neg = -log(1 - lidarProb);		//��������
			for (l = 0; l < classNo; l++)
			{
				if (l == 0)
				{
					costData[l] = unaryFactor * (0.5*(-log(exp(data[l]) / sum))+0.5*cost_pos);
				}
				if (l == 1)
				{
					costData[l] = unaryFactor * (0.5*(-log(exp(data[l]) / sum))+0.5*cost_neg);

				}
			}
		}
		else
		{
			for (l = 0; l < classNo; l++)
			{
				if (l == 0)
				{
					costData[l] = unaryFactor * (-log(exp(data[l]) / sum));
				}
				if (l == 1)
				{
					costData[l] = unaryFactor * (-log(exp(data[l]) / sum));

				}
			}

		}

		if (dataset->unaryWeighted) for (l = 0; l < classNo; l++) costData[l] *= dataset->unaryWeights[l];

		*labelData = 0;
		//		for(l = 1; l < classNo; l++) if(costData[l] < costData[*labelData]) *labelData = l;
	}
	fclose(fp);
	fclose(f);

	delete[] fileName;
	delete[] data;
}

#else
void LDenseUnaryPixelPotential::Initialize(LLabImage &labImage, char *imageFileName)
{
	int j, k, l;
	char *fileName;
	FILE *f;

	int width, height;

	fileName = GetFileName(evalFolder, imageFileName, evalExtension);
	f = fopen(fileName, "rb");
	if (f == NULL) _error(fileName);

	fread(&width, sizeof(int), 1, f);
	fread(&height, sizeof(int), 1, f);
	fread(&classNo, sizeof(int), 1, f);

	if (unaryCosts != NULL) delete[] unaryCosts;
	unaryCosts = new double[width * height * classNo];
	unaryCount = width * height;

	double *data;
	data = new double[classNo];

	double *costData = unaryCosts;
	unsigned char *labelData = layer->labels;

	for (j = 0; j < height; j++) for (k = 0; k < width; k++, costData += classNo, labelData++)
	{
		double sum = 0;
		fread(data, sizeof(double), classNo, f);
		for (l = 0; l < classNo; l++)
		{
			sum += exp(data[l]);
		}
		for (l = 0; l < classNo; l++)
		{
			costData[l] = -unaryFactor * log(exp(data[l]) / sum);
		}
		if (dataset->unaryWeighted) for (l = 0; l < classNo; l++) costData[l] *= dataset->unaryWeights[l];

		*labelData = 0;
		//		for(l = 1; l < classNo; l++) if(costData[l] < costData[*labelData]) *labelData = l;
	}
	fclose(f);

	delete[] fileName;
	delete[] data;
}

#endif



void LDenseUnaryPixelPotential::UnInitialize()
{
	if (unaryCosts != NULL) delete[] unaryCosts;
	unaryCosts = NULL;
}

int LDenseUnaryPixelPotential::CalculateShapeFilterResponse(int index, int bucket, LShape *shape, int pointX, int pointY)
{
	LIntegralImage *integralImage = integralImages[index][bucket];

	if (integralImage == NULL) return(0);
	else
	{
		int x1 = pointX + shape->x;
		int y1 = pointY + shape->y;
		int x2 = pointX + shape->x + shape->width;
		int y2 = pointY + shape->y + shape->height;

		return(integralImage->Response(x1, y1, x2, y2));
	}
}

//��������ͼ
void LDenseUnaryPixelPotential::CalculateIntegralImages(LLabImage &labImage, LIntegralImage ***integralImage, int subSample, int *width, int *height, char *imageFileName)
{
	int i, j, k, l;

	//���
	*width = labImage.GetWidth();
	*height = labImage.GetHeight();

	//����������������*ÿ�������ľ������ĸ���
	totalFeatures = 0;
	for (i = 0; i < features.GetCount(); i++) totalFeatures += buckets[i];

	//->feature->texton->map����ά����
	//������������������ͼ��ÿ��������n���������ģ�ÿһ������������һ��texton��һ��texton����һ������ͼ
	*integralImage = new LIntegralImage *[totalFeatures];	//ÿ��ָ�����һ������ͼ

	int offset = 0;
	for (i = 0; i < features.GetCount(); i++)
	{
		LImage<unsigned short> dataImage;

		//��������������������Ѿ�ѵ�����˵ģ���ֱ������
		if (!dataset->featuresOnline)
		{
			char *fileName;
			fileName = GetFileName(features[i]->evalFolder, imageFileName, features[i]->evalExtension);
			dataImage.Load(fileName);
			delete[] fileName;
		}
		//��û��ѵ���ã���������ȡ�����������䵽��ͬ��������
		else
			((LDenseFeature *)features[i])->Discretize(labImage, dataImage, imageFileName);

		//ͨ����
		int bands = dataImage.GetBands();		//������1
		for (j = 0; j < buckets[i]; j++, offset++)		//ÿ�������ľ�����������
		{
			int count = 0;

			unsigned short *dataFrom = dataImage.GetData();		//ָ����������
			//ͳ��ÿ�����ĵ�Ԫ�ظ���
			for (k = 0; k < *height; k++)
				for (l = 0; l < *width * bands; l++, dataFrom++)
					if (*dataFrom == j)
						count++;

			if (!count)
				(*integralImage)[offset] = NULL;		//���Ϊ��
			//�����Ϊ�գ���������ͼ
			else
			{
				//���С��16�������ɰ��ֽ�����ͼ
				if (count < 16) (*integralImage)[offset] = new LIntegralImageHB();		//���ֽ�
				//����С��16��������1�ֽ�����ͼ
				else if (count < 256) (*integralImage)[offset] = new LIntegralImage1B();		//1�ֽ�
				//���С��65536��������2�ֽ�����ͼ
				else if (count < 65536) (*integralImage)[offset] = new LIntegralImage2B();		//2�ֽ�
				//����ر������4�ֽ�����ͼ
				else (*integralImage)[offset] = new LIntegralImage4B();			//4�ֽ�

				//��������
				(*integralImage)[offset]->Load(dataImage, subSample, j);
			}
		}
	}
}

void LDenseUnaryPixelPotential::InitTrainData(LList<char *> &trainImageFiles)
{
	int i, j, k;

	//�����ڴ棬�����СΪ��������int�ڴ��
	buckets = new int[features.GetCount()];
	//ָ��������ָ��
	for (i = 0; i < features.GetCount(); i++)
		buckets[i] = features[i]->GetBuckets();	//�������ĸ���

	//����ͼ������TextonBoost������ȡ
	integralImages = new LIntegralImage **[fileSum];	//��ά���飬����ͼ��ÿ��ͼ��ÿ��������ÿ����������һ����
	validPointsX = new int *[fileSum];		//��ά���飬��Ч��x����
	validPointsY = new int *[fileSum];		//��ά���飬��Ч��y����
	pointsSum = new int[fileSum];		//һά���飬ÿ��ͼ����Ч�����

	int **shortTargets;		//��ά����
	shortTargets = new int *[fileSum];		//ÿ��ͼ����Ч���Ӧ�ı�ǩ

	double **fileProbs;
	fileProbs = new double *[fileSum];		//����

	int **labelBitLists;
	labelBitLists = new int *[fileSum];		//��ǩ����

	double *classCounts, threshold;		//ÿ����ĸ�������ֵ
	int total = 0;
	classCounts = new double[classNo];
	memset(classCounts, 0, classNo * sizeof(double));

	for (i = 0; i < fileSum; i++)
	{
		char *fileName;
		//��ȡ��ǩ
		fileName = GetFileName(dataset->groundTruthFolder, trainImageFiles[i], dataset->groundTruthExtension);
		LLabelImage groundTruth(fileName, domain);
		delete[] fileName;

		int width = groundTruth.GetWidth(), height = groundTruth.GetHeight();

		//��������С
		int subWidth = (width + subSample - 1) / subSample;
		int subHeight = (height + subSample - 1) / subSample;

		for (k = 0; k < subHeight; k++) for (j = 0; j < subWidth; j++)
		{
			unsigned char gtVal = groundTruth(j * subSample, k * subSample, 0);
			//���ΪʲôҪ����0����Ϊ0����ձ�ǩ������
			if (gtVal > 0) classCounts[gtVal - 1]++, total++;
		}
	}

	threshold = total * maxClassRatio;
	double *classProbs = new double[classNo];
	for (i = 0; i < classNo; i++) if (classCounts[i] != 0) classProbs[i] = threshold / (double)classCounts[i];

	unsigned char *thisOccurence;
	thisOccurence = new unsigned char[classNo];

//#ifdef USE_LIDAR
//	FILE *fp = fopen("Result/KITTI/um/Lidar/denseFeature.train", "wt");
//#endif // USE_LIDAR
	for (i = 0; i < fileSum; i++)
	{
		dataset->GetLabelSet(thisOccurence, trainImageFiles[i]);
		unsigned int labelBits = 0;
		for (k = 0; k < classNo; k++) if (thisOccurence[k]) labelBits |= (1 << k);

		int width, height;

		char *fileName;
		fileName = GetFileName(dataset->imageFolder, trainImageFiles[i], dataset->imageExtension);
		LRgbImage rgbImage(fileName);
		delete[] fileName;

		LLabImage labImage(rgbImage);
		//�������ͼ
		CalculateIntegralImages(labImage, &integralImages[i], subSample, &width, &height, trainImageFiles[i]);

		int subWidth = (width + subSample - 1) / subSample;
		int subHeight = (height + subSample - 1) / subSample;

		LList<int> pointsX, pointsY;
		LList<int> targs, classList;
		LList<double> probs;

		fileName = GetFileName(dataset->groundTruthFolder, trainImageFiles[i], dataset->groundTruthExtension);
		LLabelImage groundTruth(fileName, domain);
		delete[] fileName;

		//���ѵ��������Ч��
		for (k = 0; k < subHeight; k++) for (j = 0; j < subWidth; j++)
		{
			unsigned char gtVal = groundTruth(j * subSample, k * subSample, 0);
			//����0�࣬��������ҪС����ֵ����ֹĳ������ѵ�����������Ӵ�����ѵ������ƽ�⣩
			if ((gtVal > 0) && ((classCounts[gtVal - 1] <= threshold) || (LMath::RandomReal() < classProbs[gtVal - 1])))
			{
				//x����
				pointsX.Add(j);
				//y����
				pointsY.Add(k);
				//��Ӧ�����
				targs.Add(gtVal - 1);		//ȥ��0����0��ʼ��
				//���ֵĸ���
				probs.Add(1.0);
				//������Ϣ�����ֵı�ǩ���������һ��ͼ��Ķ���һ����
				classList.Add(labelBits);
			}
		}

		//���ݿ���
		validPointsX[i] = new int[pointsX.GetCount()];
		memcpy(validPointsX[i], pointsX.GetArray(), pointsX.GetCount() * sizeof(int));

		validPointsY[i] = new int[pointsY.GetCount()];
		memcpy(validPointsY[i], pointsY.GetArray(), pointsY.GetCount() * sizeof(int));

		shortTargets[i] = new int[targs.GetCount()];
		memcpy(shortTargets[i], targs.GetArray(), targs.GetCount() * sizeof(int));

		fileProbs[i] = new double[probs.GetCount()];
		memcpy(fileProbs[i], probs.GetArray(), probs.GetCount() * sizeof(double));

		labelBitLists[i] = new int[classList.GetCount()];
		memcpy(labelBitLists[i], classList.GetArray(), classList.GetCount() * sizeof(int));

		pointsSum[i] = pointsX.GetCount();
		numExamples += pointsX.GetCount();

//#ifdef USE_LIDAR
//		//�״���������
//		LidarDoNext(true, trainImageFiles[i]);
//		//д���
//		for (int i = 0; i < gLidar.m_height; i++)
//		{
//			for (int j = 0; j < gLidar.m_width; j++)
//			{
//				if (gLidar.m_Valid2Img.at<uchar>(i, j) > 0)     //��Ч��
//				{
//					//Ȩ��,����ĵ���
//					fprintf(fp, "%f\t", gLidar.m_Weights.at<float>(i, j));
//					//��ǩ
//					fprintf(fp, "%d\t", gLidar.m_Label.at<uchar>(i, j));
//					//x,y,z,ref
//					fprintf(fp, "%f\t%f\t%f\t%f\t", gLidar.m_Lidar2Img.at<Vec4f>(i, j)[0], gLidar.m_Lidar2Img.at<Vec4f>(i, j)[1], gLidar.m_Lidar2Img.at<Vec4f>(i, j)[2], gLidar.m_Lidar2Img.at<Vec4f>(i, j)[3]);
//					//���ʣ��߶ȷ�����߶Ȳ�
//					fprintf(fp, "%f\t%f\t%f\n", gLidar.m_Curvature.at<float>(i, j), gLidar.m_HeightVar.at<float>(i, j), gLidar.m_MaxHeightDiff.at<float>(i, j));
//				}
//			}
//		}
//#endif	// USE_LIDAR
	}
//#ifdef USE_LIDAR
//	fclose(fp);
//#endif

	delete[] thisOccurence;

	delete[] classProbs;
	delete[] classCounts;

	//������һ����������
	int offset = 0;
	targets = new int[numExamples];
	for (i = 0; i < fileSum; i++)
	{
		memcpy(targets + offset, shortTargets[i], pointsSum[i] * sizeof(int));
		offset += pointsSum[i];
	}

	//����Ȩ��
	offset = 0;
	weights = new double *[numExamples];
	for (i = 0; i < fileSum; i++)		//ÿ��ͼ
	{
		for (j = 0; j < pointsSum[i]; j++)		//ÿ��ͼ����Ч�����
		{
			weights[offset + j] = new double[classNo];		//�����ڴ�

			for (k = 0; k < classNo; k++)
			{
				//����Ȩ��
				if ((dataset->optimizeAverage != 2) || (labelBitLists[i][j] & (1 << k))) weights[offset + j][k] = fileProbs[i][j];
				else weights[offset + j][k] = 0;
			}

		}
		//ƫ��������
		offset += pointsSum[i];
	}

	for (i = 0; i < fileSum; i++) if (labelBitLists[i] != NULL) delete[] labelBitLists[i];
	delete[] labelBitLists;

	for (i = 0; i < fileSum; i++) if (shortTargets[i] != NULL) delete[] shortTargets[i];
	delete[] shortTargets;

	for (i = 0; i < fileSum; i++) if (fileProbs[i] != NULL) delete[] fileProbs[i];
	delete[] fileProbs;

	//���Ȩ��
	classWeights = new double[classNo];

	if (dataset->optimizeAverage == 1)
	{
		memset(classWeights, 0, classNo * sizeof(double));
		for (i = 0; i < numExamples; i++) classWeights[targets[i]] += weights[i][0];
		for (i = 0; i < classNo; i++) if (classWeights[i] != 0) classWeights[i] = numExamples / (double)(classNo * classWeights[i]);
	}
	else for (i = 0; i < classNo; i++) classWeights[i] = 1.0;
}

void LDenseUnaryPixelPotential::UnInitTrainData()
{
	if (integralImages != NULL)
	{
		for (int i = 0; i < fileSum; i++)
		{
			if (integralImages[i] != NULL)
			{
				int offset = 0;
				for (int j = 0; j < features.GetCount(); j++)
				{
					for (int k = 0; k < buckets[j]; k++, offset++) if (integralImages[i][offset] != NULL) delete integralImages[i][offset];
				}
				delete[] integralImages[i];
			}
		}
		delete[] integralImages;
		integralImages = NULL;
	}
	if (buckets != NULL)
	{
		delete[] buckets;
		buckets = NULL;
	}
}

//��ʼ������
void LDenseUnaryPixelPotential::InitEvalData(char *evalImageFile, int *width, int *height)
{
	//��������
	buckets = new int[features.GetCount()];
	for (int i = 0; i < features.GetCount(); i++)
		buckets[i] = features[i]->GetBuckets();		//�������ĸ���

	//����ͼ����ά����
	integralImages = new LIntegralImage **[1];

#ifdef MULTITHREAD
	EnterCriticalSection();
#endif
	//��ȡRGBͼ��
	char *fileName;
	fileName = GetFileName(dataset->imageFolder, evalImageFile, dataset->imageExtension);
	LRgbImage rgbImage(fileName);
	delete[] fileName;
#ifdef MULTITHREAD
	LeaveCriticalSection();
#endif
	//ת����LAB�ռ�
	LLabImage labImage(rgbImage);
	//�������ͼ
	CalculateIntegralImages(labImage, &integralImages[0], 1, width, height, evalImageFile);

//#ifdef USE_LIDAR
//	//�����״�����
//	LidarDoNext(true, evalImageFile);
//	char filename[256];
//	sprintf(filename, "Result/KITTI/um/Lidar/denseFeatureEval/%s.test", evalImageFile);
//	FILE *fp = fopen(filename, "wt");
//	//д���
//	for (int i = 0; i < gLidar.m_height; i++)
//	{
//		for (int j = 0; j < gLidar.m_width; j++)
//		{
//			if (gLidar.m_Valid2Img.at<uchar>(i, j) > 0)     //��Ч��
//			{
//				//x,y,z,ref
//				fprintf(fp, "%f\t%f\t%f\t%f\t", gLidar.m_Lidar2Img.at<Vec4f>(i, j)[0], gLidar.m_Lidar2Img.at<Vec4f>(i, j)[1], gLidar.m_Lidar2Img.at<Vec4f>(i, j)[2], gLidar.m_Lidar2Img.at<Vec4f>(i, j)[3]);
//				//���ʣ��߶ȷ�����߶Ȳ�
//				fprintf(fp, "%f\t%f\t%f\n", gLidar.m_Curvature.at<float>(i, j), gLidar.m_HeightVar.at<float>(i, j), gLidar.m_MaxHeightDiff.at<float>(i, j));
//			}
//		}
//	}
//	fclose(fp);
//#endif
}

void LDenseUnaryPixelPotential::UnInitEvalData()
{
	if (integralImages != NULL)
	{
		if (integralImages[0] != NULL)
		{
			int offset = 0;
			for (int j = 0; j < features.GetCount(); j++)
			{
				for (int k = 0; k < buckets[j]; k++, offset++) if (integralImages[0][offset] != NULL) delete integralImages[0][offset];
			}
			delete[] integralImages[0];
		}
		delete[] integralImages;
		integralImages = NULL;
	}
	if (buckets != NULL)
	{
		delete[] buckets;
		buckets = NULL;
	}
}

void LDenseUnaryPixelPotential::AddFeature(LDenseFeature *feature)
{
	features.Add(feature);
}


LHeightUnaryPixelPotential::LHeightUnaryPixelPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setObjDomain, LBaseCrfLayer *setObjLayer, LCrfDomain *setLidarDomain, LBaseCrfLayer *setLidarLayer, const char *setTrainFolder, const char *setTrainFile, int setClassNo, double setUnaryFactor, int setDisparityClassNo,double setThreshold) : LUnaryPixelPotential(setDataset, setCrf, NULL, NULL, setTrainFolder, setTrainFile, NULL, NULL, setClassNo, setUnaryFactor)
{
	objDomain = setObjDomain, lidarDomain = setLidarDomain;
	objLayer = setObjLayer, lidarLayer = setLidarLayer;
	disparityClassNo = setDisparityClassNo;
	threshold = setThreshold;
}

void LHeightUnaryPixelPotential::AddCosts(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes)
{
	int i, j;
	if ((costDomain == objDomain) && (!first))
	{
		for (i = 0; i < unaryCount; i++) if (!objLayer->active[i])
		{
			g->add_tweights(nodes[i + objLayer->nodeOffset], unaryCosts[i * classNo * disparityClassNo + label * disparityClassNo + lidarLayer->labels[i]], unaryCosts[i * classNo * disparityClassNo + objLayer->labels[i] * disparityClassNo + lidarLayer->labels[i]]);
		}
	}
	if (costDomain == lidarDomain)
	{
		first = 0;

		if (!lidarLayer->range)
		{
			for (i = 0; i < unaryCount; i++) if (!lidarLayer->active[i])
			{
				g->add_tweights(nodes[i + lidarLayer->nodeOffset], unaryCosts[i * classNo * disparityClassNo + objLayer->labels[i] * disparityClassNo + label], unaryCosts[i * classNo * disparityClassNo + objLayer->labels[i] * disparityClassNo + lidarLayer->labels[i]]);
			}
		}
		else
		{
			int range = lidarLayer->range;
			for (i = 0; i < unaryCount; i++)
			{
				g->add_tweights(nodes[i * range + lidarLayer->nodeOffset], 0, (!lidarLayer->active[i]) ? unaryCosts[i * classNo * disparityClassNo + objLayer->labels[i] * disparityClassNo + lidarLayer->labels[i]] : 0);
				g->add_tweights(nodes[i * range + range - 1 + lidarLayer->nodeOffset], unaryCosts[i * classNo * disparityClassNo + objLayer->labels[i] * disparityClassNo + label + range - 1], 0);
				for (j = 0; j < range - 1; j++) g->add_edge(nodes[i * range + j + lidarLayer->nodeOffset], nodes[i * range + j + 1 + lidarLayer->nodeOffset], 0, unaryCosts[i * classNo * disparityClassNo + objLayer->labels[i] * disparityClassNo + label + j]);
			}
		}
	}
}

double LHeightUnaryPixelPotential::GetCost(LCrfDomain *costDomain)
{
	if (((costDomain == objDomain) && (!first)) || (costDomain == lidarDomain))
	{
		double cost = 0;
		for (int i = 0; i < unaryCount; i++) cost += unaryCosts[i * classNo * disparityClassNo + objLayer->labels[i] * disparityClassNo + lidarLayer->labels[i]];
		return(cost);
	}
	else return(0);
}

LHeightUnaryPixelPotential::~LHeightUnaryPixelPotential()
{
}

void LHeightUnaryPixelPotential::Train(LList<char *> &trainImageFiles)
{
	
}

void LHeightUnaryPixelPotential::SaveTraining()
{

}

void LHeightUnaryPixelPotential::LoadTraining()
{

}

void LHeightUnaryPixelPotential::Initialize(LLabImage &labImage, char *imageFileName)
{
	int i, j, k, l;

	width = labImage.GetWidth();
	height = labImage.GetHeight();
	first = 1;

	if (unaryCosts != NULL) delete[] unaryCosts;

	unaryCosts = new double[width * height * classNo * disparityClassNo];
	unaryCount = width * height;

	LidarDoNext(false, imageFileName, dataset->datasetName);
	for (k = 0; k < height; k++) for (j = 0; j < width; j++)
	{
		//���״��Ĺ���һ��
		if (gLidar.m_Valid2Img.at<uchar>(k, j) > 0)
		{

			for (i = 0; i < disparityClassNo; i++)
			{

				for (l = 0; l < classNo; l++)
				{
					double ratio;
					if (l == i)		//�����ͬ
					{
						//��һ���߶�ֱ��ͼ
						ratio = 0.8;
					}
					else ratio = threshold;
					//ratio = 1.0;
					unaryCosts[(k * width + j) * classNo * disparityClassNo + l * disparityClassNo + i] = -unaryFactor * log(ratio);
				}
			}
		}
		else
		{
			for (i = 0; i < disparityClassNo; i++)
			{

				for (l = 0; l < classNo; l++)
				{
					unaryCosts[(k * width + j) * classNo * disparityClassNo + l * disparityClassNo + i] = -unaryFactor * log(0.5);
				}
			}
		}
	}
}

void LHeightUnaryPixelPotential::UnInitialize()
{
	if (unaryCosts != NULL) delete[] unaryCosts;
	unaryCosts = NULL;
}


LPairwisePixelPotential::LPairwisePixelPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LBaseCrfLayer *setLayer, int setClassNo) : LPotential(setDataset, setCrf, setDomain, NULL, NULL, NULL, NULL, setClassNo)
{
	layer = setLayer;
}

LPottsPairwisePixelPotential::LPottsPairwisePixelPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LBaseCrfLayer *setLayer, int setClassNo) : LPairwisePixelPotential(setDataset, setCrf, setDomain, setLayer, setClassNo)
{
	pairwiseIndexes = NULL;
	pairwiseCosts = NULL;
	pairwiseCount = 0;
}

LPottsPairwisePixelPotential::~LPottsPairwisePixelPotential()
{
	if (pairwiseIndexes != NULL) delete[] pairwiseIndexes;
	if (pairwiseCosts != NULL) delete[] pairwiseCosts;
}

int LPottsPairwisePixelPotential::GetEdgeCount()
{
	return(pairwiseCount);
}

//һ��ƽ�����������Ȩ��
void LPottsPairwisePixelPotential::AddCosts(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes)
{
	int from, to, i;
	double weight;

	if (domain == costDomain) for (i = 0; i < pairwiseCount; i++)		//ѭ����Ը���
	{
		from = pairwiseIndexes[2 * i];		//���
		to = pairwiseIndexes[2 * i + 1];		//�յ�
		weight = pairwiseCosts[i];		//�ߵ�Ȩ��

		if (weight > LMath::almostZero)		//Ȩ�ش�����Сֵ
		{
			//�������cost�����û��cost
			if ((layer->active[from]) && (!layer->active[to])) g->add_tweights(nodes[to + layer->nodeOffset], 0, weight);
			else if ((!layer->active[from]) && (layer->active[to])) g->add_tweights(nodes[from + layer->nodeOffset], 0, weight);
			else if ((!layer->active[from]) && (!layer->active[to]))
			{
				//�����ǩ���
				if (layer->labels[from] == layer->labels[to]) g->add_edge(nodes[from + layer->nodeOffset], nodes[to + layer->nodeOffset], weight, weight);
				else
				{
					g->add_tweights(nodes[from + layer->nodeOffset], 0, weight);
					g->add_edge(nodes[from + layer->nodeOffset], nodes[to + layer->nodeOffset], 0, weight);
				}
			}
		}
	}
}

double LPottsPairwisePixelPotential::GetCost(LCrfDomain *costDomain)
{
	if (domain == costDomain)
	{
		double cost = 0;
		for (int i = 0; i < pairwiseCount; i++)
		{
			int from = pairwiseIndexes[2 * i];
			int to = pairwiseIndexes[2 * i + 1];
			if (layer->labels[from] != layer->labels[to]) cost += pairwiseCosts[i];
		}
		return(cost);
	}
	else return(0);
}

LEightNeighbourPottsPairwisePixelPotential::LEightNeighbourPottsPairwisePixelPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LBaseCrfLayer *setLayer, int setClassNo, double setPairwisePrior, double setPairwiseFactor, double setPairwiseBeta, double setPairwiseLWeight, double setPairwiseUWeight, double setPairwiseVWeight) : LPottsPairwisePixelPotential(setDataset, setCrf, setDomain, setLayer, setClassNo)
{
	pairwisePrior = setPairwisePrior;
	pairwiseFactor = setPairwiseFactor;
	pairwiseBeta = setPairwiseBeta;
	pairwiseLWeight = setPairwiseLWeight;
	pairwiseUWeight = setPairwiseUWeight;
	pairwiseVWeight = setPairwiseVWeight;
}

double LEightNeighbourPottsPairwisePixelPotential::PairwiseDiff(double *lab1, double *lab2, double distance)
{
	double labDiff[3];

	for (int i = 0; i < 3; i++) labDiff[i] = (lab1[i] - lab2[i]) * (lab1[i] - lab2[i]);
	//ƽ���ʽ
	return(pairwisePrior + pairwiseFactor * exp((-pairwiseLWeight * labDiff[0] - pairwiseUWeight * labDiff[1] - pairwiseVWeight * labDiff[2]) / (pairwiseBeta * distance)));
	//double lamda1, lamda2, theta_alpha, theta_beta, theta_gama;
	//double gauss_kernel, bilateral_kernel;
	////��������
	//lamda1 = 1.0;
	//lamda2 = theta_gama = 1.0;
	//theta_alpha = theta_beta = 30.0;
	//gauss_kernel = exp(-(distance*distance) / (2 * theta_alpha*theta_alpha));
	//bilateral_kernel = exp(-(distance*distance) / (2 * theta_beta*theta_beta) - (pairwiseLWeight * labDiff[0] + pairwiseUWeight * labDiff[1] + pairwiseVWeight * labDiff[2]) / (2 * theta_gama*theta_gama));
	//return pairwisePrior + pairwiseFactor*(lamda1*gauss_kernel + lamda2*bilateral_kernel);
}

//һ��ƽ����
void LEightNeighbourPottsPairwisePixelPotential::Initialize(LLabImage &labImage, char *imageFileName)
{
	int width = labImage.GetWidth();
	int height = labImage.GetHeight();

	if (pairwiseCosts != NULL) delete[] pairwiseCosts;
	if (pairwiseIndexes != NULL) delete[] pairwiseIndexes;
	pairwiseCount = 4 * height * width - 3 * (height + width) + 2;

	pairwiseCosts = new double[pairwiseCount];
	memset(pairwiseCosts, 0, pairwiseCount * sizeof(double));

	pairwiseIndexes = new int[2 * pairwiseCount];
	memset(pairwiseIndexes, 0, 2 * pairwiseCount * sizeof(int));

	double *pairwiseCostData, *labData;
	double sqrt2 = sqrt((double)2.0);
	int *pairwiseIndexData, i, j;

	pairwiseCostData = pairwiseCosts;
	pairwiseIndexData = pairwiseIndexes;
	labData = labImage.GetData();

	for (i = 0; i < height; i++) for (j = 0; j < width; j++, labData += 3)
	{
		if (j != width - 1)
		{
			*pairwiseCostData = PairwiseDiff(labData, labData + 3, 1.0);
			pairwiseIndexData[0] = i * width + j, pairwiseIndexData[1] = i * width + j + 1;
			pairwiseCostData++, pairwiseIndexData += 2;
		}
	}

	labData = labImage.GetData();
	for (i = 0; i < height; i++) for (j = 0; j < width; j++, labData += 3)
	{
		if (i != height - 1)
		{
			*pairwiseCostData = PairwiseDiff(labData, labData + 3 * width, 1.0);
			pairwiseIndexData[0] = i * width + j, pairwiseIndexData[1] = (i + 1) * width + j;
			pairwiseCostData++, pairwiseIndexData += 2;
		}
	}

	labData = labImage.GetData();
	for (i = 0; i < height; i++) for (j = 0; j < width; j++, labData += 3)
	{
		if ((j != width - 1) && (i != height - 1))
		{
			*pairwiseCostData = PairwiseDiff(labData, labData + 3 * (width + 1), sqrt2);
			pairwiseIndexData[0] = i * width + j, pairwiseIndexData[1] = (i + 1) * width + j + 1;
			pairwiseCostData++, pairwiseIndexData += 2;
		}
	}

	labData = labImage.GetData();
	for (i = 0; i < height; i++) for (j = 0; j < width; j++, labData += 3)
	{
		if ((j != 0) && (i != height - 1))
		{
			*pairwiseCostData = PairwiseDiff(labData, labData + 3 * (width - 1), sqrt2);
			pairwiseIndexData[0] = i * width + j, pairwiseIndexData[1] = (i + 1) * width + j - 1;
			pairwiseCostData++, pairwiseIndexData += 2;
		}
	}
}

void LEightNeighbourPottsPairwisePixelPotential::UnInitialize()
{
	if (pairwiseCosts != NULL) delete[] pairwiseCosts;
	if (pairwiseIndexes != NULL) delete[] pairwiseIndexes;
	pairwiseCosts = NULL;
	pairwiseIndexes = NULL;
}

LLinearTruncatedPairwisePixelPotential::LLinearTruncatedPairwisePixelPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LBaseCrfLayer *setLayer, int setClassNo, double setPairwiseFactor, double setTrauncation) : LPairwisePixelPotential(setDataset, setCrf, setDomain, setLayer, setClassNo)
{
	pairwiseIndexes = NULL;
	pairwiseCosts = NULL;
	pairwiseCount = 0;
	pairwiseFactor = setPairwiseFactor;
	truncation = setTrauncation;
}

LLinearTruncatedPairwisePixelPotential::~LLinearTruncatedPairwisePixelPotential()
{
	if (pairwiseIndexes != NULL) delete[] pairwiseIndexes;
	if (pairwiseCosts != NULL) delete[] pairwiseCosts;
}

int LLinearTruncatedPairwisePixelPotential::GetNodeCount()
{
	if (!layer->range) return(0);
	else return(pairwiseCount);
}

int LLinearTruncatedPairwisePixelPotential::GetEdgeCount()
{
	return(pairwiseCount);
}

void LLinearTruncatedPairwisePixelPotential::AddCosts(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes)
{
	int from, to, i, j;
	double weight;

	if (domain == costDomain)
	{
		if (!layer->range)
		{
			for (i = 0; i < pairwiseCount; i++)
			{
				from = pairwiseIndexes[2 * i];
				to = pairwiseIndexes[2 * i + 1];

				double lweight = 0;

				int label1 = layer->labels[from];
				int label2 = layer->labels[to];

				double a12, a21, a22;

				double diff1 = (label1 > label) ? label1 - label : label - label1;
				double diff2 = (label2 > label) ? label2 - label : label - label2;
				double diff12 = (label2 > label1) ? label2 - label1 : label1 - label2;

				if (diff1 > truncation) diff1 = truncation;
				if (diff2 > truncation) diff2 = truncation;
				if (diff12 > truncation) diff12 = truncation;

				a12 = diff1 * pairwiseCosts[i];
				a21 = diff2 * pairwiseCosts[i];
				a22 = diff12 * pairwiseCosts[i];

				weight = (a12 + a21 - a22) / 2;
				if (weight >= LMath::almostZero)
				{
					if ((layer->active[from]) && (!layer->active[to])) g->add_tweights(nodes[to + layer->nodeOffset], 0, weight);
					else if ((!layer->active[from]) && (layer->active[to])) g->add_tweights(nodes[from + layer->nodeOffset], 0, weight);
					else if ((!layer->active[from]) && (!layer->active[to]))
					{
						if (layer->labels[from] == layer->labels[to]) g->add_edge(nodes[from + layer->nodeOffset], nodes[to + layer->nodeOffset], weight, weight);
						else
						{
							g->add_tweights(nodes[from + layer->nodeOffset], 0, weight);
							g->add_edge(nodes[from + layer->nodeOffset], nodes[to + layer->nodeOffset], 0, weight);
						}
					}
				}
				if (!layer->active[from]) g->add_tweights(nodes[from + layer->nodeOffset], weight, a12);
				if (!layer->active[to]) g->add_tweights(nodes[to + layer->nodeOffset], weight, a21);
			}
		}
		else
		{
			for (i = 0; i < pairwiseCount; i++)
			{
				from = pairwiseIndexes[2 * i];
				to = pairwiseIndexes[2 * i + 1];

				int label1 = layer->labels[from];
				int label2 = layer->labels[to];
				int range = layer->range;
				for (j = 1; j < range; j++) g->add_edge(nodes[from * range + j + layer->nodeOffset], nodes[to * range + j + layer->nodeOffset], pairwiseCosts[i], pairwiseCosts[i]);

				if (((label1 >= label) && (label1 < label + range)) && ((label2 < label) || (label2 >= label + range))) g->add_edge(nodes[from * range + layer->nodeOffset], nodes[to * range + layer->nodeOffset], pairwiseCosts[i] * (truncation + range / 2.0), 0);
				else if (((label2 >= label) && (label2 < label + range)) && ((label1 < label) || (label1 >= label + range))) g->add_edge(nodes[from * range + layer->nodeOffset], nodes[to * range + layer->nodeOffset], 0, pairwiseCosts[i] * (truncation + range / 2.0));
				else if (((label1 >= label) && (label1 < label + range)) && ((label2 >= label) && (label2 < label + range)))
				{
					nodes[nodeOffset + i] = g->add_node();
					g->add_tweights(nodes[nodeOffset + i], pairwiseCosts[i] * (range + ((label2 > label1) ? label2 - label1 : label1 - label2)), 0);
					g->add_edge(nodes[to * range + layer->nodeOffset], nodes[nodeOffset + i], pairwiseCosts[i] * (truncation + range / 2.0), pairwiseCosts[i] * (truncation + range / 2.0));
					g->add_edge(nodes[from * range + layer->nodeOffset], nodes[nodeOffset + i], pairwiseCosts[i] * (truncation + range / 2.0), pairwiseCosts[i] * (truncation + range / 2.0));
				}
			}
		}
	}
}
double LLinearTruncatedPairwisePixelPotential::GetCost(LCrfDomain *costDomain)
{
	if (domain == costDomain)
	{
		double cost = 0;
		for (int i = 0; i < pairwiseCount; i++)
		{
			int from = pairwiseIndexes[2 * i];
			int to = pairwiseIndexes[2 * i + 1];

			int label1 = layer->labels[from];
			int label2 = layer->labels[to];

			double diff12 = (label2 > label1) ? label2 - label1 : label1 - label2;
			if (diff12 > truncation) diff12 = truncation;
			cost += diff12 * pairwiseCosts[i];
		}
		return(cost);
	}
	else return(0);
}


LDisparityLinearTruncatedPairwisePixelPotential::LDisparityLinearTruncatedPairwisePixelPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LBaseCrfLayer *setLayer, int setClassNo, double setPairwiseFactor, double setTruncation) : LLinearTruncatedPairwisePixelPotential(setDataset, setCrf, setDomain, setLayer, setClassNo, setPairwiseFactor, setTruncation)
{
}

void LDisparityLinearTruncatedPairwisePixelPotential::Initialize(LLabImage &labImage, char *imageFileName)
{
	int width = labImage.GetWidth();
	int height = labImage.GetHeight();
	int i, j;

	if (pairwiseCosts != NULL) delete[] pairwiseCosts;
	if (pairwiseIndexes != NULL) delete[] pairwiseIndexes;
	pairwiseCount = 4 * height * width - 3 * (height + width) + 2;

	pairwiseCosts = new double[pairwiseCount];
	for (i = 0; i < pairwiseCount; i++) pairwiseCosts[i] = pairwiseFactor;

	pairwiseIndexes = new int[2 * pairwiseCount];
	memset(pairwiseIndexes, 0, 2 * pairwiseCount * sizeof(int));

	double sqrt2 = sqrt((double)2.0);
	int *pairwiseIndexData;

	pairwiseIndexData = pairwiseIndexes;

	for (i = 0; i < height; i++) for (j = 0; j < width; j++)
	{
		if (j != width - 1)
		{
			pairwiseIndexData[0] = i * width + j, pairwiseIndexData[1] = i * width + j + 1;
			pairwiseIndexData += 2;
		}
	}

	for (i = 0; i < height; i++) for (j = 0; j < width; j++)
	{
		if (i != height - 1)
		{
			pairwiseIndexData[0] = i * width + j, pairwiseIndexData[1] = (i + 1) * width + j;
			pairwiseIndexData += 2;
		}
	}

	for (i = 0; i < height; i++) for (j = 0; j < width; j++)
	{
		if ((j != width - 1) && (i != height - 1))
		{
			pairwiseIndexData[0] = i * width + j, pairwiseIndexData[1] = (i + 1) * width + j + 1;
			pairwiseIndexData += 2;
		}
	}

	for (i = 0; i < height; i++) for (j = 0; j < width; j++)
	{
		if ((j != 0) && (i != height - 1))
		{
			pairwiseIndexData[0] = i * width + j, pairwiseIndexData[1] = (i + 1) * width + j - 1;
			pairwiseIndexData += 2;
		}
	}
}

void LDisparityLinearTruncatedPairwisePixelPotential::UnInitialize()
{
	if (pairwiseCosts != NULL) delete[] pairwiseCosts;
	if (pairwiseIndexes != NULL) delete[] pairwiseIndexes;
	pairwiseCosts = NULL;
	pairwiseIndexes = NULL;
}


LJointPairwisePixelPotential::LJointPairwisePixelPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setObjDomain, LBaseCrfLayer *setObjLayer, LCrfDomain *setLidarDomain, LBaseCrfLayer *setLidarLayer, int setClassNo, int setDisparityClassNo, double setPairwisePrior, double setPairwiseFactor, double setPairwiseBeta, double setPairwiseLWeight, double setPairwiseUWeight, double setPairwiseVWeight, double setDisparityFactor, double setTruncation, double setCrossFactor) : LPairwisePixelPotential(setDataset, setCrf, NULL, NULL, classNo)
{
	pairwiseIndexes = NULL;
	pairwiseCosts = NULL;
	pairwiseCount = 0;
	pairwisePrior = setPairwisePrior;
	pairwiseFactor = setPairwiseFactor;
	pairwiseBeta = setPairwiseBeta;
	pairwiseLWeight = setPairwiseLWeight;
	pairwiseUWeight = setPairwiseUWeight;
	pairwiseVWeight = setPairwiseVWeight;
	disparityClassNo = setDisparityClassNo;
	objDomain = setObjDomain, lidarDomain = setLidarDomain;
	objLayer = setObjLayer, lidarLayer = setLidarLayer;
	disparityClassNo = setDisparityClassNo;
	crossFactor = setCrossFactor;
	disparityFactor = setDisparityFactor, truncation = setTruncation;
}

LJointPairwisePixelPotential::~LJointPairwisePixelPotential()
{
	if (pairwiseIndexes != NULL) delete[] pairwiseIndexes;
	if (pairwiseCosts != NULL) delete[] pairwiseCosts;
}

double LJointPairwisePixelPotential::PairwiseDiff(double *lab1, double *lab2, double distance)
{
	double labDiff[3];

	for (int i = 0; i < 3; i++) labDiff[i] = (lab1[i] - lab2[i]) * (lab1[i] - lab2[i]);
	return(pairwisePrior + pairwiseFactor * exp((-pairwiseLWeight * labDiff[0] - pairwiseUWeight * labDiff[1] - pairwiseVWeight * labDiff[2]) / (pairwiseBeta * distance)));
}

int LJointPairwisePixelPotential::GetEdgeCount()
{
	return(pairwiseCount);
}

int LJointPairwisePixelPotential::GetNodeCount()
{
	if (!lidarLayer->range) return(0);
	else return(pairwiseCount);
}

void LJointPairwisePixelPotential::Initialize(LLabImage &labImage, char *imageFileName)
{
	int width = labImage.GetWidth();
	int height = labImage.GetHeight();
	first = 1;

	if (pairwiseCosts != NULL) delete[] pairwiseCosts;
	if (pairwiseIndexes != NULL) delete[] pairwiseIndexes;
	pairwiseCount = 4 * height * width - 3 * (height + width) + 2;

	pairwiseCosts = new double[pairwiseCount];
	memset(pairwiseCosts, 0, pairwiseCount * sizeof(double));

	pairwiseIndexes = new int[2 * pairwiseCount];
	memset(pairwiseIndexes, 0, 2 * pairwiseCount * sizeof(int));

	double *pairwiseCostData, *labData;
	double sqrt2 = sqrt((double)2.0);
	int *pairwiseIndexData, i, j;

	pairwiseCostData = pairwiseCosts;
	pairwiseIndexData = pairwiseIndexes;
	labData = labImage.GetData();

	for (i = 0; i < height; i++) for (j = 0; j < width; j++, labData += 3)
	{
		if (j != width - 1)
		{
			*pairwiseCostData = PairwiseDiff(labData, labData + 3, 1.0);
			pairwiseIndexData[0] = i * width + j, pairwiseIndexData[1] = i * width + j + 1;
			pairwiseCostData++, pairwiseIndexData += 2;
		}
	}

	labData = labImage.GetData();
	for (i = 0; i < height; i++) for (j = 0; j < width; j++, labData += 3)
	{
		if (i != height - 1)
		{
			*pairwiseCostData = PairwiseDiff(labData, labData + 3 * width, 1.0);
			pairwiseIndexData[0] = i * width + j, pairwiseIndexData[1] = (i + 1) * width + j;
			pairwiseCostData++, pairwiseIndexData += 2;
		}
	}

	labData = labImage.GetData();
	for (i = 0; i < height; i++) for (j = 0; j < width; j++, labData += 3)
	{
		if ((j != width - 1) && (i != height - 1))
		{
			*pairwiseCostData = PairwiseDiff(labData, labData + 3 * (width + 1), sqrt2);
			pairwiseIndexData[0] = i * width + j, pairwiseIndexData[1] = (i + 1) * width + j + 1;
			pairwiseCostData++, pairwiseIndexData += 2;
		}
	}

	labData = labImage.GetData();
	for (i = 0; i < height; i++) for (j = 0; j < width; j++, labData += 3)
	{
		if ((j != 0) && (i != height - 1))
		{
			*pairwiseCostData = PairwiseDiff(labData, labData + 3 * (width - 1), sqrt2);
			pairwiseIndexData[0] = i * width + j, pairwiseIndexData[1] = (i + 1) * width + j - 1;
			pairwiseCostData++, pairwiseIndexData += 2;
		}
	}
}

void LJointPairwisePixelPotential::UnInitialize()
{
	if (pairwiseCosts != NULL) delete[] pairwiseCosts;
	if (pairwiseIndexes != NULL) delete[] pairwiseIndexes;
	pairwiseCosts = NULL;
	pairwiseIndexes = NULL;
}

void LJointPairwisePixelPotential::AddCosts(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes)
{
	int from, to, i;
	double weight;

	if ((objDomain == costDomain) && (!first)) for (i = 0; i < pairwiseCount; i++)
	{
		from = pairwiseIndexes[2 * i];
		to = pairwiseIndexes[2 * i + 1];
		double diff = (lidarLayer->labels[from] > lidarLayer->labels[to]) ? lidarLayer->labels[from] - lidarLayer->labels[to] : lidarLayer->labels[to] - lidarLayer->labels[from];
		if (diff > truncation) diff = truncation;
		weight = pairwiseCosts[i] + crossFactor * disparityFactor * pairwiseCosts[i] * diff;

		if (weight > LMath::almostZero)
		{
			if ((objLayer->active[from]) && (!objLayer->active[to])) g->add_tweights(nodes[to + objLayer->nodeOffset], 0, weight);
			else if ((!objLayer->active[from]) && (objLayer->active[to])) g->add_tweights(nodes[from + objLayer->nodeOffset], 0, weight);
			else if ((!objLayer->active[from]) && (!objLayer->active[to]))
			{
				if (objLayer->labels[from] == objLayer->labels[to]) g->add_edge(nodes[from + objLayer->nodeOffset], nodes[to + objLayer->nodeOffset], weight, weight);
				else
				{
					g->add_tweights(nodes[from + objLayer->nodeOffset], 0, weight);
					g->add_edge(nodes[from + objLayer->nodeOffset], nodes[to + objLayer->nodeOffset], 0, weight);
				}
			}
		}
	}
	if (lidarDomain == costDomain)
	{
		if (!lidarLayer->range)
		{
			for (i = 0; i < pairwiseCount; i++)
			{
				from = pairwiseIndexes[2 * i];
				to = pairwiseIndexes[2 * i + 1];
				first = 0;

				double lweight = 0;

				int label1 = lidarLayer->labels[from];
				int label2 = lidarLayer->labels[to];
				int objLabel1 = objLayer->labels[from];
				int objLabel2 = objLayer->labels[to];

				double diff1 = (label1 > label) ? label1 - label : label - label1;
				double diff2 = (label2 > label) ? label2 - label : label - label2;
				double diff12 = (label2 > label1) ? label2 - label1 : label1 - label2;

				if (diff1 > truncation) diff1 = truncation;
				if (diff2 > truncation) diff2 = truncation;
				if (diff12 > truncation) diff12 = truncation;

				diff1 *= disparityFactor + crossFactor * disparityFactor * pairwiseCosts[i] * ((objLabel1 == objLabel2) ? 0 : 1);
				diff2 *= disparityFactor + crossFactor * disparityFactor * pairwiseCosts[i] * ((objLabel1 == objLabel2) ? 0 : 1);
				diff12 *= disparityFactor + crossFactor * disparityFactor * pairwiseCosts[i] * ((objLabel1 == objLabel2) ? 0 : 1);

				weight = (diff1 + diff2 - diff12) / 2;
				if (weight >= LMath::almostZero)
				{
					if ((lidarLayer->active[from]) && (!lidarLayer->active[to])) g->add_tweights(nodes[to + lidarLayer->nodeOffset], 0, weight);
					else if ((!lidarLayer->active[from]) && (lidarLayer->active[to])) g->add_tweights(nodes[from + lidarLayer->nodeOffset], 0, weight);
					else if ((!lidarLayer->active[from]) && (!lidarLayer->active[to]))
					{
						if (lidarLayer->labels[from] == lidarLayer->labels[to]) g->add_edge(nodes[from + lidarLayer->nodeOffset], nodes[to + lidarLayer->nodeOffset], weight, weight);
						else
						{
							g->add_tweights(nodes[from + lidarLayer->nodeOffset], 0, weight);
							g->add_edge(nodes[from + lidarLayer->nodeOffset], nodes[to + lidarLayer->nodeOffset], 0, weight);
						}
					}
				}
				if (!lidarLayer->active[from]) g->add_tweights(nodes[from + lidarLayer->nodeOffset], weight, diff1);
				if (!lidarLayer->active[to]) g->add_tweights(nodes[to + lidarLayer->nodeOffset], weight, diff2);
			}
		}
		else
		{
			for (i = 0; i < pairwiseCount; i++)
			{
				from = pairwiseIndexes[2 * i];
				to = pairwiseIndexes[2 * i + 1];

				int label1 = lidarLayer->labels[from];
				int label2 = lidarLayer->labels[to];
				int objLabel1 = objLayer->labels[from];
				int objLabel2 = objLayer->labels[to];
				int range = lidarLayer->range, j;
				double weight = disparityFactor + crossFactor * disparityFactor * pairwiseCosts[i] * ((objLabel1 == objLabel2) ? 0 : 1);

				for (j = 1; j < range; j++) g->add_edge(nodes[from * range + j + lidarLayer->nodeOffset], nodes[to * range + j + lidarLayer->nodeOffset], weight, weight);

				if (((label1 >= label) && (label1 < label + range)) && ((label2 < label) || (label2 >= label + range))) g->add_edge(nodes[from * range + lidarLayer->nodeOffset], nodes[to * range + lidarLayer->nodeOffset], weight * (truncation + range / 2.0), 0);
				else if (((label2 >= label) && (label2 < label + range)) && ((label1 < label) || (label1 >= label + range))) g->add_edge(nodes[from * range + lidarLayer->nodeOffset], nodes[to * range + lidarLayer->nodeOffset], 0, weight * (truncation + range / 2.0));
				else if (((label1 < label) || (label1 >= label + range)) && ((label2 < label) || (label2 >= label + range)))
				{
					nodes[nodeOffset + i] = g->add_node();
					g->add_tweights(nodes[nodeOffset + i], weight * (range + ((label2 > label1) ? label2 - label1 : label1 - label2)), 0);
					g->add_edge(nodes[to * range + lidarLayer->nodeOffset], nodes[nodeOffset + i], weight * (truncation + range / 2.0), weight * (truncation + range / 2.0));
					g->add_edge(nodes[from * range + lidarLayer->nodeOffset], nodes[nodeOffset + i], weight * (truncation + range / 2.0), weight * (truncation + range / 2.0));
				}
			}
		}
	}
}

double LJointPairwisePixelPotential::GetCost(LCrfDomain *costDomain)
{
	if (((objDomain == costDomain) && (!first)) || (lidarDomain == costDomain))
	{
		double cost = 0;
		for (int i = 0; i < pairwiseCount; i++)
		{
			int from = pairwiseIndexes[2 * i];
			int to = pairwiseIndexes[2 * i + 1];
			int label1 = lidarLayer->labels[from];
			int label2 = lidarLayer->labels[to];
			double diff = (lidarLayer->labels[from] > lidarLayer->labels[to]) ? lidarLayer->labels[from] - lidarLayer->labels[to] : lidarLayer->labels[to] - lidarLayer->labels[from];
			if (diff > truncation) diff = truncation;
			cost += pairwiseCosts[i] * (1 + crossFactor * disparityFactor * diff) * ((objLayer->labels[from] == objLayer->labels[to]) ? 0 : 1) + disparityFactor * diff;
		}
		return(cost);
	}
	else return(0);
}


LUnarySegmentPotential::LUnarySegmentPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension, int setClassNo, double setConsistencyPrior, double setSegmentFactor) : LPotential(setDataset, setCrf, setDomain, setTrainFolder, setTrainFile, setEvalFolder, setEvalExtension, setClassNo)
{
	consistencyPrior = setConsistencyPrior, segmentFactor = setSegmentFactor;
	unaryCosts = NULL;
}

LUnarySegmentPotential::~LUnarySegmentPotential()
{
	if (unaryCosts != NULL) delete[] unaryCosts;
}

//�߽������Ȩ��
void LUnarySegmentPotential::AddCosts(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes)
{
	int i, j, m;
	double lambdaA, lambdaB, lambdaM, gammaB;		//����
	double sumOld;

	double *costData = unaryCosts;

	//ÿ���߽ײ�ѭ��
	if (domain == costDomain) for (m = 0; m < layers.GetCount(); m++) for (i = 0; i < layers[m]->segmentCount; i++, costData += classNo + 1)	//���ϸ�����ǩ
	{
		lambdaM = costData[classNo];		//������ǩ�Ĵ���
		lambdaA = costData[label];		//��ǰ��ǩ�Ĵ���
		int parNodes = layers[m]->parent->BinaryNodes();

		if (!layers[m]->active[i])		//��ǰ�㵱ǰ������δ����
		{
			int labelBar = layers[m]->labels[i];		//��ȡ��ǩ

			if (lambdaM - lambdaA > LMath::almostZero)		//������ǩcost��
			{
				g->add_tweights(nodes[layers[m]->nodeOffset + 2 * i], 0, lambdaM - lambdaA);

				//����븸�������
				for (j = 0; j < layers[m]->segmentCounts[i]; j++)
				{
					//�������δ����
					if (!layers[m]->parent->active[layers[m]->segmentIndexes[i][j]])
					{
						g->add_edge(nodes[layers[m]->nodeOffset + 2 * i], nodes[layers[m]->parent->nodeOffset + layers[m]->segmentIndexes[i][j] * parNodes], 0, (lambdaM - lambdaA) * layers[m]->weights[i][j] / (layers[m]->weightSums[i] * layers[m]->truncation));
					}
				}
			}
			//���Ǹ�����ǩ
			if (labelBar != classNo)
			{
				sumOld = 0;
				for (j = 0; j < layers[m]->segmentCounts[i]; j++)
				{
					//��ǰ���븸���ǩ����ȵĸ���
					if (layers[m]->parent->labels[layers[m]->segmentIndexes[i][j]] != labelBar) sumOld += layers[m]->weights[i][j];
				}

				lambdaB = costData[labelBar] + sumOld * (costData[classNo] - costData[labelBar]) / (layers[m]->weightSums[i] * layers[m]->truncation);
				gammaB = costData[labelBar];

				if (lambdaM - lambdaB > LMath::almostZero) g->add_tweights(nodes[layers[m]->nodeOffset + 2 * i + 1], lambdaM - lambdaB, 0);

				if (lambdaM - gammaB > LMath::almostZero)
				{
					for (j = 0; j < layers[m]->segmentCounts[i]; j++) if (layers[m]->parent->labels[layers[m]->segmentIndexes[i][j]] == labelBar)
					{
						g->add_edge(nodes[layers[m]->nodeOffset + 2 * i + 1], nodes[layers[m]->parent->nodeOffset + layers[m]->segmentIndexes[i][j] * parNodes + parNodes - 1], (lambdaM - gammaB) * layers[m]->weights[i][j] / (layers[m]->weightSums[i] * layers[m]->truncation), 0);
					}
				}
				g->add_edge(nodes[layers[m]->nodeOffset + 2 * i], nodes[layers[m]->nodeOffset + 2 * i + 1], 0, LMath::positiveInfinity);
			}
		}
		else
		{
			if (lambdaM - lambdaA > LMath::almostZero)
			{
				double weight = (lambdaM - lambdaA) / (layers[m]->weightSums[i] * layers[m]->truncation);
				for (j = 0; j < layers[m]->segmentCounts[i]; j++)
				{
					if (!layers[m]->parent->active[layers[m]->segmentIndexes[i][j]]) g->add_tweights(nodes[layers[m]->parent->nodeOffset + layers[m]->segmentIndexes[i][j] * parNodes], 0, weight * layers[m]->weights[i][j]);
				}
			}
		}
	}
}

double LUnarySegmentPotential::GetCost(LCrfDomain *costDomain)
{
	int m, i, j;
	if (domain == costDomain)
	{
		double cost = 0;
		double *costData = unaryCosts;
		for (m = 0; m < layers.GetCount(); m++) for (i = 0; i < layers[m]->segmentCount; i++, costData += classNo + 1)
		{
			int label = layers[m]->labels[i];
			if (label != classNo)
			{
				double sum = 0;
				for (j = 0; j < layers[m]->segmentCounts[i]; j++)
				{
					if (layers[m]->parent->labels[layers[m]->segmentIndexes[i][j]] != label) sum += layers[m]->weights[i][j];
				}
				cost += costData[label] + sum * (costData[classNo] - costData[label]) / (layers[m]->weightSums[i] * layers[m]->truncation);
			}
			else cost += costData[classNo];
		}
		return(cost);
	}
	else return(0);
}

void LUnarySegmentPotential::AddLayer(LPnCrfLayer *layer)
{
	layers.Add(layer);
}

LConsistencyUnarySegmentPotential::LConsistencyUnarySegmentPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, int setClassNo, double setConsistencyPrior) : LUnarySegmentPotential(setDataset, setCrf, setDomain, NULL, NULL, NULL, NULL, setClassNo, setConsistencyPrior, 0)
{
}

//һ�����ƺ���
void LConsistencyUnarySegmentPotential::Initialize(LLabImage &labImage, char *imageFileName)
{
	int i, k, m;

	int segCount = 0;
	for (m = 0; m < layers.GetCount(); m++) segCount += layers[m]->segmentCount;

	if (unaryCosts != NULL) delete[] unaryCosts;
	unaryCosts = new double[segCount * (classNo + 1)];

	double *costData = unaryCosts;

	for (m = 0; m < layers.GetCount(); m++)
		for (i = 0; i < layers[m]->segmentCount; i++, costData += classNo + 1)
		{
			for (k = 0; k < classNo; k++) costData[k] = 0;
			costData[classNo] = consistencyPrior * layers[m]->baseSegmentCounts[i];
		}
}

void LConsistencyUnarySegmentPotential::UnInitialize()
{
	if (unaryCosts != NULL) delete[] unaryCosts;
	unaryCosts = NULL;
}

LStatsUnarySegmentPotential::LStatsUnarySegmentPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension, int setClassNo, double setConsistencyPrior, double setSegmentFactor, double setMinLabelRatio, double setAlpha, double setMaxClassRatio, int setNeighbour, LSegmentation2D *setSegmentation) : LUnarySegmentPotential(setDataset, setCrf, setDomain, setTrainFolder, setTrainFile, setEvalFolder, setEvalExtension, setClassNo, setConsistencyPrior, setSegmentFactor)
{
	minLabelRatio = setMinLabelRatio;
	alpha = setAlpha;

	numExamples = 0;
	featureValues = NULL;
	targets = NULL;
	weights = NULL;

	maxClassRatio = setMaxClassRatio;
	neighbour = setNeighbour;
}

LStatsUnarySegmentPotential::~LStatsUnarySegmentPotential()
{
}

void LStatsUnarySegmentPotential::AddFeature(LDenseFeature *feature)
{
	features.Add(feature);
}

void LStatsUnarySegmentPotential::SaveTraining()
{
	learning->SaveTraining();
}

void LStatsUnarySegmentPotential::LoadTraining()
{
	learning->LoadTraining();
}

double *LStatsUnarySegmentPotential::GetTrainBoostingValues(int index, int core)
{
	return(featureValues[index]);
}

double *LStatsUnarySegmentPotential::GetTrainSVMValues(int index, int core)
{
	return(featureValues[index]);
}

//�߽��ƺ�����������segment���������ѵ��
void LStatsUnarySegmentPotential::Train(LList<char *> &trainImageFiles)
{
	int i, j, k, l, m, discOffset = 0, classOffset = 0, r;

	numExamples = 0;

	//���ÿ�������ľ������
	buckets = new int[features.GetCount()];
	for (i = 0; i < features.GetCount(); i++) buckets[i] = features[i]->GetBuckets();		//�������

	//����ÿ�����ĸ���
	printf("Calculating probability of each class..\n");

	int totalSamples = 0;		//������
	double total = 0, threshold, *classCounts;

	int *imageSamples;		//ѵ��ͼ����

	imageSamples = new int[trainImageFiles.GetCount()];		//ͼ����
	memset(imageSamples, 0, trainImageFiles.GetCount() * sizeof(int));

	classCounts = new double[classNo];		//�����
	memset(classCounts, 0, classNo * sizeof(double));

	//����������������������ǩ������ͼ���������������
	LList<int> layerIndexList, segmentIndexList, targetList, imageIndexList, classList;
	//Ȩ��
	LList<double> weightList;		//Ȩ���б�

	unsigned char *thisOccurence;		//������
	thisOccurence = new unsigned char[classNo];

	//������ȡѵ����������
	for (m = 0; m < trainImageFiles.GetCount(); m++)	//ÿ��ͼ��
	{
		char *fileName;
		//��ʼ��ÿһ�㣬�������������Լ�����Ȩ��
		for (i = 0; i < layers.GetCount(); i++) layers[i]->Initialize(trainImageFiles[m], 0);		//������

		//��ȡ��ѵ��ͼ���ǩ���ã�ÿ��ͼ��һ������
		dataset->GetLabelSet(thisOccurence, trainImageFiles[m]);
		unsigned int labelBits = 0;		//��ǩ����Ϊ0������Ϊ0��������Ӧ����11��
		for (i = 0; i < classNo; i++) if (thisOccurence[i]) labelBits |= (1 << i);

		double *classPix;		//�������
		classPix = new double[classNo];		//�������У�ÿ���������ظ���

		if (dataset->Segmented(trainImageFiles[m]))		//����зָ�
		{
			//��ȡ��ǩ
			fileName = GetFileName(dataset->groundTruthFolder, trainImageFiles[m], dataset->groundTruthExtension);
			LLabelImage groundTruth(fileName, domain);		//��ǩͼ
			delete[] fileName;

			unsigned char *dataGT = groundTruth.GetData();		//ָ���ǩͼ��ָ��

			//ÿ�㣬ÿ���ÿ��������ѭ��
			for (i = 0; i < layers.GetCount(); i++) for (k = 0; k < layers[i]->segmentCount; k++)
			{
				memset(classPix, 0, classNo * sizeof(double));		//��ʼ��ȫ0
				//ͳ�����ز�
				for (j = 0; j < layers[i]->baseSegmentCounts[k]; j++)		//�ó����ص�ÿ������
				{
					int label = dataGT[layers[i]->baseSegmentIndexes[k][j]];		//��ȡÿ�����صı�ǩ
					if ((label > 0) && (label < classNo + 1)) classPix[label - 1]++;	//ÿ���������ı�ǩ���
				}
				for (l = 0; l < classNo; l++)
				{
					//Ѱ��������ǩ��������������ĳ����ֵ��0.5���ı�ǩ��Ϊ������ǩ
					double ratio = classPix[l] / layers[i]->baseSegmentCounts[k];
					if (ratio > 0.5)		//���ĳһ�����ı�ǩ����ռ��50%���ϣ��ʹ���������ǩ
					{
						classCounts[l] += classPix[l], total += classPix[l];		//�ܵ�ÿ����ǩ�����ظ�����ÿ�ζ�Ҫ�ۼӵ�
						layerIndexList.Add(i);		//������
						segmentIndexList.Add(k);		//�ָ�����
						classList.Add(labelBits);		//��ǩ����
						targetList.Add(l);		//��ǩ
						weightList.Add(layers[i]->baseSegmentCounts[k]);		//�Գ����ص����ظ�����Ȩ��
						imageIndexList.Add(m);		//ͼ������
						imageSamples[m]++;		//û��ͼ���������ͳ��
					}
				}
			}
		}
		//����ʼ������
		for (i = 0; i < layers.GetCount(); i++) layers[i]->UnInitialize();
		delete[] classPix;
	}
	delete[] thisOccurence;

	//����������
	threshold = total * maxClassRatio;		//�ܵ�������������һ��������
	double *classProbs = new double[classNo];
	for (i = 0; i < classNo; i++) if (classCounts[i] != 0) classProbs[i] = threshold / (double)classCounts[i];

	//������
	numExamples = targetList.GetCount();
	//��С����ֵ���������������ݼ����޳�
	for (i = numExamples - 1; i >= 0; i--) if (!((classCounts[targetList[i]] <= threshold) || (LMath::RandomReal() < classProbs[targetList[i]])))
	{
		imageSamples[imageIndexList[i]]--;
		layerIndexList.Delete(i);
		segmentIndexList.Delete(i);
		targetList.Delete(i);
		weightList.Delete(i);
		classList.Delete(i);
		numExamples--;
	}
	delete[] classProbs;

	memset(classCounts, 0, classNo * sizeof(int));
	for (i = 0; i < numExamples; i++) classCounts[targetList[i]]++;		//����ͳ�������

	totalFeatures = 0;
	for (i = 0; i < features.GetCount(); i++) totalFeatures += buckets[i];		//�������������
	if (neighbour) totalFeatures *= 2;		//˫�����������˱�����������

	int swap = learning->SwapTrainData();		//����ѵ�����ݣ�Boost����1��������Ϊ���������ԣ�ÿ��splitֻ���㵥�������������ô��
	if (swap)	//���������boost==1
	{
		featureValues = new double *[totalFeatures];		//ÿ���������е�
		for (i = 0; i < totalFeatures; i++) featureValues[i] = new double[numExamples];
	}
	else	//������
	{
		featureValues = new double *[numExamples];		//ÿ������������
		for (i = 0; i < numExamples; i++) featureValues[i] = new double[totalFeatures];
	}

	targets = new int[numExamples];
	memcpy(targets, targetList.GetArray(), numExamples * sizeof(int));		//ÿ��������ı�ǩ

	weights = new double *[numExamples];	//Ȩ��ϵ��
	for (i = 0; i < numExamples; i++)
	{
		weights[i] = new double[classNo];
		double weight = weightList[i];		//ֻ�г����ڱ�ǩ�����еı�ǩ������Ȩ��
		for (l = 0; l < classNo; l++)
		{
			//����������ĸñ�ǩ�ڵ�ǰ�����г���������Ȩ�أ������ذ��������ظ�������������Ϊ0
			if ((dataset->optimizeAverage != 2) || (classList[i] & (1 << l))) weights[i][l] = weight;
			else weights[i][l] = 0;
		}
	}

	//��ȡ���ݣ�����ÿ�����������������Ҫ����������ڴ˴�������
	printf("Reading data..\n");
#ifdef USE_LIDAR
	char filename[256];
	sprintf(filename, "/home/yangfei/Datasets/Result/KITTI/%s/Lidar/statFeature.train", dataset->datasetName);
	FILE *fp = fopen(filename, "wt");
	if (!fp)
	{
		printf("can not open %s !\n", filename);
	}
#endif
	int offset = 0;
	for (m = 0; m < trainImageFiles.GetCount(); m++)
	{
		LImage<unsigned short> *bucketImages;	//�������ͼ
		//��ͼ��
		char *fileName;
		fileName = GetFileName(dataset->imageFolder, trainImageFiles[m], dataset->imageExtension);
		LRgbImage rgbImage(fileName);		//��rgbͼ��
		delete[] fileName;

		//����ͼ
		bucketImages = new LImage<unsigned short>[features.GetCount()];		//����ͼ

		//�ٴγ�ʼ��������ָ������Լ�����Ȩ��
		for (i = 0; i < layers.GetCount(); i++) layers[i]->Initialize(trainImageFiles[m], 0);

		char **neighs = NULL;		//�ھӶ�ά����

		//�Ƿ�������
		if (neighbour)
		{
			neighs = new char *[layers.GetCount()];		//ÿ��һ��
			for (i = 0; i < layers.GetCount(); i++)		//Ӧ����д�����˰�

				for (i = 0; i < layers.GetCount(); i++)		//ÿ��
				{
					//���ָ�ͼ
					fileName = GetFileName(layers[i]->segmentation->folder, trainImageFiles[m], layers[i]->segmentation->extension);
					LSegmentImage segmentImage;
					segmentImage.Load(fileName);
					delete[] fileName;

					int width = segmentImage.GetWidth();
					int height = segmentImage.GetHeight();

					int segCount = layers[i]->segmentCount;		//�ָ���
					neighs[i] = new char[segCount * segCount];		//�ָ���*�ָ���

					//�����������˽ṹ�������һ��ʵ�Գƾ��󣬼�Ѱ�����ڵķָ�
					for (k = 0; k < height; k++) for (j = 0; j < width - 1; j++)	//����
					{
						neighs[i][segmentImage(j, k, 0) * segCount + segmentImage(j + 1, k, 0)] = 1;
						neighs[i][segmentImage(j + 1, k, 0) * segCount + segmentImage(j, k, 0)] = 1;
					}
					for (k = 0; k < height - 1; k++) for (j = 0; j < width; j++)	//����
					{
						neighs[i][segmentImage(j, k, 0) * segCount + segmentImage(j, k + 1, 0)] = 1;
						neighs[i][segmentImage(j, k + 1, 0) * segCount + segmentImage(j, k, 0)] = 1;
					}
					for (k = 0; k < segCount; k++) neighs[i][k * segCount + k] = 0;		//���������ھ�
				}
		}

		LLabImage labImage(rgbImage);
		//��������ͼ���������߻������߼�������
		for (k = 0; k < features.GetCount(); k++)
		{
			if (!dataset->featuresOnline)
			{
				fileName = GetFileName(features[k]->evalFolder, trainImageFiles[m], features[k]->evalExtension);
				bucketImages[k].Load(fileName);
				delete[] fileName;
			}
			else features[k]->Discretize(labImage, bucketImages[k], trainImageFiles[m]);
		}
		//����ֱ��ͼ����
#ifdef USE_LIDAR
		LidarDoNext(true, trainImageFiles[m],dataset->datasetName);
#endif
		for (i = 0; i < imageSamples[m]; i++)		//ÿ��ͼ���ÿ��������
		{
			//ͼ������
			int featureOffset = 0;		//����ƫ��
			for (l = 0; l < features.GetCount(); l++)		//ÿ���������ÿ��ÿ������
			{
				unsigned short *bucketData = bucketImages[l].GetData();		//ָ������ͼ��ָ��
				int *bucketCounts = new int[buckets[l]];	//�����ľ���������
				memset(bucketCounts, 0, buckets[l] * sizeof(int));
				int bucketBands = bucketImages[l].GetBands();	//������ͨ����

				int realCount = 0;

				//�߽ײ�
				LPnCrfLayer *layer = layers[layerIndexList[classOffset + i]];		//��ͼ���ڲ�

				//��������������ĳ����ص�ÿ��������ֱ��ͼ
				for (j = 0; j < layer->baseSegmentCounts[segmentIndexList[classOffset + i]]; j++)		//�ò��base����ָ���
				{
					for (r = 0; r < bucketBands; r++)
					{
						//��ȡ����ֵ
						unsigned short index = bucketData[layer->baseSegmentIndexes[segmentIndexList[classOffset + i]][j] * bucketBands + r];
						//����ֱ��ͼ
						if (index < buckets[l]) bucketCounts[index]++, realCount++;
					}
				}
				//ֱ��ͼ��һ��
				for (j = 0; j < buckets[l]; j++)
				{
					if (swap) featureValues[featureOffset + j][discOffset] = (realCount != 0) ? (bucketCounts[j] * 1000.0 / realCount) : (int)0;
					else featureValues[discOffset][featureOffset + j] = (realCount != 0) ? (bucketCounts[j] * 1000.0 / realCount) : (int)0;
				}
				if (neighbour)
				{
					//ÿ��������ѭ��������ǵ�ǰ�������ھ�
					for (k = 0; k < layer->segmentCount; k++) if (neighs[layerIndexList[classOffset + i]][k * layer->segmentCount + segmentIndexList[classOffset + i]])
					{
						//�������ص�ÿ�����أ�ͳ��ֱ��ͼ
						for (j = 0; j < layer->baseSegmentCounts[k]; j++)
						{
							for (r = 0; r < bucketBands; r++)
							{
								unsigned short index = bucketData[layer->baseSegmentIndexes[k][j] * bucketBands + r];
								if (index < buckets[l]) bucketCounts[index]++, realCount++;
							}
						}
					}
					//�������������
					for (j = 0; j < buckets[l]; j++)
					{
						if (swap) featureValues[featureOffset + (totalFeatures >> 1) + j][discOffset] = (realCount != 0) ? (bucketCounts[j] * 1000.0 / realCount) : (int)0;
						else featureValues[discOffset][featureOffset + (totalFeatures >> 1) + j] = (realCount != 0) ? (bucketCounts[j] * 1000.0 / realCount) : (int)0;
					}
				}
				if (bucketCounts != NULL) delete[] bucketCounts;
				featureOffset += buckets[l];
			}
			discOffset++;


#ifdef USE_LIDAR
			//�״�����
			double featHist[7];
			memset(featHist, 0, sizeof(featHist));
			int numLidarPoint = 0;
			LPnCrfLayer *layer = layers[layerIndexList[classOffset + i]];		//��ͼ���ڲ�
			for (j = 0; j < layer->baseSegmentCounts[segmentIndexList[classOffset + i]]; j++)		//�ò��base����ָ���
			{
				//��������
				int index = layer->baseSegmentIndexes[segmentIndexList[classOffset + i]][j];
				//��������
				int x = index % gLidar.m_width;		//��
				int y = index / gLidar.m_width;		//��
				//������״��
				if (gLidar.m_Valid2Img.at<uchar>(y, x) > 0)
				{
					//cout << "y=" << y << " x=" << x << "\n";
					featHist[0] += gLidar.m_Lidar2Img.at<Vec4f>(y, x)[0];
					featHist[1] += gLidar.m_Lidar2Img.at<Vec4f>(y, x)[1];
					featHist[2] += gLidar.m_Lidar2Img.at<Vec4f>(y, x)[2];
					featHist[3] += gLidar.m_Lidar2Img.at<Vec4f>(y, x)[3];
					featHist[4] += gLidar.m_Curvature.at<float>(y, x);
					featHist[5] += gLidar.m_MaxHeightDiff.at<float>(y, x);
					featHist[6] += gLidar.m_HeightVar.at<float>(y, x);
					numLidarPoint++;
				}
			}
			//д�״�����
			if (numLidarPoint > 0)
			{
				fprintf(fp, "%lf\t", weightList[classOffset + i]);		//дȨ��
				//cout << "weight= " << weightList[classOffset + i] << endl;
				fprintf(fp, "%d\t", !targetList[classOffset + i]);		//д��ǩ ע��0����������1�Ǹ����������˴�ȡ��
				for (j = 0;j < 6;j++)
				{
					featHist[j] /= numLidarPoint;
					fprintf(fp, "%lf\t", featHist[j]);
				}
				fprintf(fp, "%lf\n", featHist[6]);
			}
#endif
		}
		//�ͷ��ڴ�
		if (neighs != NULL)
		{
			for (i = 0; i < layers.GetCount(); i++) if (neighs[i] != NULL) delete[] neighs[i];
			delete[] neighs;
		}
		for (i = 0; i < layers.GetCount(); i++) layers[i]->UnInitialize();
		if (bucketImages != NULL) delete[] bucketImages;
		classOffset += imageSamples[m];
	}
#ifdef USE_LIDAR
	fclose(fp);
#endif
	delete[] imageSamples;
	delete[] classCounts;

	layerIndexList.Clear();
	segmentIndexList.Clear();
	targetList.Clear();
	weightList.Clear();
	imageIndexList.Clear();

	//������������Ȩ��
	classWeights = new double[classNo];

	double totalWeight = 0;

	if (dataset->optimizeAverage == 1)
	{
		memset(classWeights, 0, classNo * sizeof(double));
		for (i = 0; i < numExamples; i++) classWeights[targets[i]] += weights[i][0], totalWeight += weights[i][0];
		for (i = 0; i < classNo; i++) if (classWeights[i] != 0) classWeights[i] = totalWeight / (double)(classNo * classWeights[i]);
	}
	else for (i = 0; i < classNo; i++) classWeights[i] = 1.0;

	learning->Train(classWeights, numExamples, targets, totalFeatures, weights);
	SaveTraining();

	//�ͷ��ڴ�
	printf("Uninitialising training..\n");

	if (classWeights != NULL) delete[] classWeights;

	if (featureValues != NULL)
	{
		if (swap) for (i = 0; i < totalFeatures; i++) if (featureValues[i] != NULL) delete[] featureValues[i];
		else for (i = 0; i < numExamples; i++)  if (featureValues[i] != NULL) delete[] featureValues[i];
		delete[] featureValues;
	}

	if (weights != NULL)
	{
		for (i = 0; i < numExamples; i++) if (weights[i] != NULL) delete[] weights[i];
		delete[] weights;
	}
	if (targets != NULL) delete[] targets;
	if (buckets != NULL) delete[] buckets;
}

double *LStatsUnarySegmentPotential::GetEvalBoostingValues(int index)
{
	return(evalData[index]);
}

double *LStatsUnarySegmentPotential::GetEvalSVMValues(int index)
{
	return(evalData[index]);
}

//��������������
void LStatsUnarySegmentPotential::Evaluate(char *imageFileName)
{
	int i, j, k, l, m;
	int width, height, *dataCounts;
	double *costs;
	unsigned short **bucketData;		//����ͼ
	FILE *f;
	LImage<unsigned short> *bucketImage;		//����ͼ

#ifdef MULTITHREAD
	EnterCriticalSection();
#endif
	//��ԭͼ
	char *fileName;
	fileName = GetFileName(dataset->imageFolder, imageFileName, dataset->imageExtension);
	LRgbImage rgbImage(fileName);
	delete[] fileName;
#ifdef MULTITHREAD
	LeaveCriticalSection();
#endif
	//���������ȡ����
	LLabImage labImage(rgbImage);		//ת����lab�ռ�
	bucketImage = new LImage<unsigned short>[features.GetCount()];
	for (i = 0; i < features.GetCount(); i++)
	{
		if (!dataset->featuresOnline)
		{
			fileName = GetFileName(features[i]->evalFolder, imageFileName, features[i]->evalExtension);
			bucketImage[i].Load(fileName);
			delete[] fileName;
		}
		else features[i]->Discretize(labImage, bucketImage[i], imageFileName);
	}

	width = rgbImage.GetWidth(), height = rgbImage.GetHeight();
	buckets = new int[features.GetCount()];		//��������������ĸ���

	totalFeatures = 0;
	bucketData = new unsigned short *[features.GetCount()];
	for (i = 0; i < features.GetCount(); i++)
	{
		buckets[i] = features[i]->GetBuckets();
		bucketData[i] = bucketImage[i].GetData();
		totalFeatures += buckets[i];
	}
	//�Ƿ���������
	if (neighbour) totalFeatures *= 2;

	for (m = 0; m < layers.GetCount(); m++) layers[m]->Initialize(imageFileName);

	char **neighs = NULL;

	//��������ṹ�����ҳ�ÿ����ÿ�����ڵĳ�����
	if (neighbour)
	{
		neighs = new char *[layers.GetCount()];
		for (i = 0; i < layers.GetCount(); i++)

			for (i = 0; i < layers.GetCount(); i++)
			{
				fileName = GetFileName(layers[i]->segmentation->folder, imageFileName, layers[i]->segmentation->extension);
				LSegmentImage segmentImage;
				segmentImage.Load(fileName);
				delete[] fileName;

				int width = segmentImage.GetWidth();
				int height = segmentImage.GetHeight();

				int segCount = layers[i]->segmentCount;
				neighs[i] = new char[segCount * segCount];

				for (k = 0; k < height; k++) for (j = 0; j < width - 1; j++)
				{
					neighs[i][segmentImage(j, k, 0) * segCount + segmentImage(j + 1, k, 0)] = 1;
					neighs[i][segmentImage(j + 1, k, 0) * segCount + segmentImage(j, k, 0)] = 1;
				}
				for (k = 0; k < height - 1; k++) for (j = 0; j < width; j++)
				{
					neighs[i][segmentImage(j, k, 0) * segCount + segmentImage(j, k + 1, 0)] = 1;
					neighs[i][segmentImage(j, k + 1, 0) * segCount + segmentImage(j, k, 0)] = 1;
				}
				for (k = 0; k < segCount; k++) neighs[i][k * segCount + k] = 0;
			}
	}

	totalSegments = 0;		//�������ܵĸ���
	//����ʱÿ��������
	for (m = 0; m < layers.GetCount(); m++) totalSegments += layers[m]->segmentCount;

	costs = new double[totalSegments * classNo];		//�ܵĳ����ظ������������
	dataCounts = new int[totalFeatures];		//�����ֲ�ֱ��ͼ

	//�Ƿ��������洢ѵ�����ݣ����ھ���������
	int swap = learning->SwapEvalData();
	if (swap)
	{
		evalData = new double *[totalFeatures];
		for (i = 0; i < totalFeatures; i++)
		{
			evalData[i] = new double[totalSegments];
			memset(evalData[i], 0, totalSegments * sizeof(double));
		}
	}
	else
	{
		evalData = new double *[totalSegments];
		for (i = 0; i < totalSegments; i++)
		{
			evalData[i] = new double[totalFeatures];
			memset(evalData[i], 0, totalFeatures * sizeof(double));
		}
	}

	int segIndex = 0, r;
	//��������
#ifdef USE_LIDAR
	LidarDoNext(true, imageFileName,dataset->datasetName);
	char filename[256];
	sprintf(filename, "/home/yangfei/Datasets/Result/KITTI/%s/Lidar/statFeatureEval/%s.test",dataset->datasetName, imageFileName);
	FILE *fp = fopen(filename, "wt");
	if (!fp)
	{
		printf("can not open %s !\n", filename);
	}
#endif
	for (m = 0; m < layers.GetCount(); m++) for (i = 0; i < layers[m]->segmentCount; i++, segIndex++)
	{
		//ͼ������
		int featureOffset = 0;		//����ƫ����

		memset(dataCounts, 0, totalFeatures * sizeof(int));
		for (k = 0; k < features.GetCount(); k++)		//ÿ������
		{
			int bands = bucketImage[k].GetBands();		//��������ͨ����
			int realCount = 0;		//��ʵ����
			for (l = 0; l < layers[m]->baseSegmentCounts[i]; l++)		//�ó����ص��е�ÿ������
			{
				for (r = 0; r < bands; r++)		//ÿ��ͨ��
				{
					unsigned short index = bucketData[k][bands * layers[m]->baseSegmentIndexes[i][l] + r];
					if (index < buckets[k]) dataCounts[featureOffset + index]++, realCount++;
				}
			}
			//ֱ��ͼ��һ��
			for (l = 0; l < buckets[k]; l++)
			{
				if (swap) evalData[l + featureOffset][segIndex] = (realCount != 0) ? (dataCounts[l + featureOffset] * 1000.0 / realCount) : 0;
				else evalData[segIndex][l + featureOffset] = (realCount != 0) ? (dataCounts[l + featureOffset] * 1000.0 / realCount) : 0;
			}

			//������������
			if (neighbour)
			{
				for (l = 0; l < layers[m]->segmentCount; l++) if (neighs[m][l * layers[m]->segmentCount + i])
				{
					for (j = 0; j < layers[m]->baseSegmentCounts[l]; j++)
					{
						for (r = 0; r < bands; r++)
						{
							unsigned short index = bucketData[k][layers[m]->baseSegmentIndexes[l][j] * bands + r];
							if (index < buckets[k]) dataCounts[featureOffset + index]++, realCount++;
						}
					}
				}
				for (j = 0; j < buckets[k]; j++)
				{
					if (swap) evalData[j + featureOffset + (totalFeatures >> 1)][segIndex] = (realCount != 0) ? (dataCounts[j + featureOffset] * 1000.0 / realCount) : 0;
					else evalData[segIndex][j + featureOffset + (totalFeatures >> 1)] = (realCount != 0) ? (dataCounts[j + featureOffset] * 1000.0 / realCount) : 0;
				}
			}
			featureOffset += buckets[k];
		}

#ifdef USE_LIDAR
		//�״�����
		double featHist[7];
		memset(featHist, 0, sizeof(featHist));
		int numLidarPoint = 0;
		for (j = 0; j < layers[m]->baseSegmentCounts[i]; j++)		//�ò��base����ָ���
		{
			//��������
			int index = layers[m]->baseSegmentIndexes[i][j];
			//��������
			int x = index % gLidar.m_width;		//��
			int y = index / gLidar.m_width;		//��
												//������״��
			if (gLidar.m_Valid2Img.at<uchar>(y, x) > 0)
			{
				featHist[0] += gLidar.m_Lidar2Img.at<Vec4f>(y, x)[0];
				featHist[1] += gLidar.m_Lidar2Img.at<Vec4f>(y, x)[1];
				featHist[2] += gLidar.m_Lidar2Img.at<Vec4f>(y, x)[2];
				featHist[3] += gLidar.m_Lidar2Img.at<Vec4f>(y, x)[3];
				featHist[4] += gLidar.m_Curvature.at<float>(y, x);
				featHist[5] += gLidar.m_MaxHeightDiff.at<float>(y, x);
				featHist[6] += gLidar.m_HeightVar.at<float>(y, x);
				numLidarPoint++;
			}
		}
		//д�״�����
		if (numLidarPoint > 0)
		{
			for (j = 0;j < 6;j++)
			{
				featHist[j] /= numLidarPoint;
				fprintf(fp, "%lf\t", featHist[j]);
			}
			fprintf(fp, "%lf\n", featHist[6]);
		}
#endif
	}
#ifdef USE_LIDAR
	fclose(fp);

#endif

	fileName = GetFileName(evalFolder, imageFileName, evalExtension);
	f = fopen(fileName, "wb");
	if (f == NULL) _error(fileName);	//����Ƿ����ͼ��
	//����ʱ���г����ض�����
	learning->Evaluate(costs, totalSegments, totalFeatures);

	//������Խ��
	fwrite(costs, sizeof(double), totalSegments * classNo, f);
	fclose(f);
	delete[] fileName;

	//�ͷ��ڴ�
	if (neighs != NULL)
	{
		for (i = 0; i < layers.GetCount(); i++) if (neighs[i] != NULL) delete[] neighs[i];
		delete[] neighs;
	}

	for (m = 0; m < layers.GetCount(); m++) layers[m]->UnInitialize();

	if (dataCounts != NULL) delete[] dataCounts;
	if (evalData != NULL)
	{
		if (swap) for (i = 0; i < totalFeatures; i++) if (evalData[i] != NULL) delete[] evalData[i];
		else for (i = 0; i < totalSegments; i++) if (evalData[i] != NULL) delete[] evalData[i];
		delete[] evalData;
	}
	if (costs != NULL) delete[] costs;

	if (bucketData != NULL) delete[] bucketData;
	if (bucketImage != NULL) delete[] bucketImage;
	if (buckets != NULL) delete[] buckets;
}

#ifdef USE_LIDAR
void LStatsUnarySegmentPotential::Initialize(LLabImage &labImage, char *imageFileName)
{
	FILE *f;
	char *fileName;
	int i, k, m;

	LidarDoNext(false, imageFileName,dataset->datasetName);

	fileName = GetFileName(evalFolder, imageFileName, evalExtension);
	f = fopen(fileName, "rb");
	if (f == NULL) _error(fileName);

	//�ܵķָ������������в�
	int segCount = 0;
	for (m = 0; m < layers.GetCount(); m++) segCount += layers[m]->segmentCount;

	//������
	if (unaryCosts != NULL) delete[] unaryCosts;
	unaryCosts = new double[segCount * (classNo + 1)];		//��һ���ָ�һ���������ǩ��һ��Ϊ������ǩ

	double *costs;
	costs = new double[classNo];		//��ǩ

	double *costData = unaryCosts;		//ָ��������

	char LidarClassifierOutfileName[256];
	sprintf(LidarClassifierOutfileName, "/home/yangfei/Datasets/Result/KITTI/%s/Lidar/statClassifierOut/%s.txt", dataset->datasetName, imageFileName);
	FILE *fp = fopen(LidarClassifierOutfileName, "rt");
	//ÿ���߽ײ㣬ÿ�����еĳ����طָ�
	for (m = 0; m < layers.GetCount(); m++) for (i = 0; i < layers[m]->segmentCount; i++, costData += classNo + 1)		//������ָ��ָ����һ��
	{
		fread(costs, sizeof(double), classNo, f);		//���ļ�

		double weight = segmentFactor * layers[m]->baseSegmentCounts[i];
		double maxcost = -weight * log(alpha);		//���Ĵ��ۣ����Խض�

		double sum = 0;
		for (k = 0; k < classNo; k++) sum += exp(costs[k]);		//����֮��

		//�ж��Ƿ����״��
		int numLidarPoint = 0;
		for (int j = 0;j < layers[m]->baseSegmentCounts[i];j++)
		{
			int index = layers[m]->baseSegmentIndexes[i][j];
			//��������
			int x = index % gLidar.m_width;		//��
			int y = index / gLidar.m_width;		//��
			//ͳ���״�����
			if (gLidar.m_Valid2Img.at<uchar>(y, x) > 0)
				numLidarPoint++;
		}

		double cost_pos, cost_neg;
		cost_pos = cost_neg = -log(0.5);
		//������״��
		if (numLidarPoint > 0)
		{
			double lidarprob;
			fscanf(fp, "%lf", &lidarprob);
			cost_pos = -log(lidarprob);
			cost_neg = -log(1 - lidarprob);
			for (k = 0; k < classNo; k++)
			{
				if (k == 0)
				{
					costData[k] = weight *(0.5* (-log(exp(costs[k]) / sum) + 0.5*cost_pos));
					//costData[k] = weight *cost_pos;

				}
				if (k == 1)
				{
					costData[k] = weight * (0.5*(-log(exp(costs[k]) / sum) + 0.5*cost_neg));
					//costData[k] = weight *cost_neg;

				}
				if (dataset->unaryWeighted) costData[k] *= dataset->unaryWeights[k];
				if (costData[k] > maxcost) costData[k] = maxcost;
			}
		}
		else
		{
			for (k = 0; k < classNo; k++)
			{
				if (k == 0)
				{
					costData[k] = weight *(-log(exp(costs[k]) / sum));
					//costData[k] = weight *(-log(0.001));

				}
				if (k == 1)
				{
					costData[k] = weight * (-log(exp(costs[k]) / sum));
					//costData[k] = weight *(-log(0.999));

				}
				if (dataset->unaryWeighted) costData[k] *= dataset->unaryWeights[k];
				if (costData[k] > maxcost) costData[k] = maxcost;
			}
		}

		//������ǩ���ã������ܹ�������һ����
		costData[classNo] = maxcost + consistencyPrior * layers[m]->baseSegmentCounts[i];
	}
	if (costs != NULL) delete[] costs;

	fclose(fp);
	fclose(f);
	delete[] fileName;
}
#else
void LStatsUnarySegmentPotential::Initialize(LLabImage &labImage, char *imageFileName)
{
	FILE *f;
	char *fileName;
	int i, k, m;

	fileName = GetFileName(evalFolder, imageFileName, evalExtension);
	f = fopen(fileName, "rb");
	if (f == NULL) _error(fileName);

	//�ܵķָ������������в�
	int segCount = 0;
	for (m = 0; m < layers.GetCount(); m++) segCount += layers[m]->segmentCount;

	//������
	if (unaryCosts != NULL) delete[] unaryCosts;
	unaryCosts = new double[segCount * (classNo + 1)];		//��һ���ָ�һ���������ǩ��һ��Ϊ������ǩ

	double *costs;
	costs = new double[classNo];		//��ǩ

	double *costData = unaryCosts;		//ָ��������

	//ÿ���߽ײ㣬ÿ�����еĳ����طָ�
	for (m = 0; m < layers.GetCount(); m++) for (i = 0; i < layers[m]->segmentCount; i++, costData += classNo + 1)		//������ָ��ָ����һ��
	{
		fread(costs, sizeof(double), classNo, f);		//���ļ�

		double weight = segmentFactor * layers[m]->baseSegmentCounts[i];
		double maxcost = -weight * log(alpha);		//���Ĵ��ۣ����Խض�

		double sum = 0;
		for (k = 0; k < classNo; k++) sum += exp(costs[k]);		//����֮��

		for (k = 0; k < classNo; k++)
		{
			costData[k] = -weight * log(exp(costs[k]) / sum);
			if (dataset->unaryWeighted) costData[k] *= dataset->unaryWeights[k];
			if (costData[k] > maxcost) costData[k] = maxcost;
		}
		//������ǩ���ã������ܹ�������һ����
		costData[classNo] = maxcost + consistencyPrior * layers[m]->baseSegmentCounts[i];
	}
	if (costs != NULL) delete[] costs;

	fclose(f);
	delete[] fileName;
}
#endif	//USE_LIDAR
void LStatsUnarySegmentPotential::UnInitialize()
{
	if (unaryCosts != NULL) delete[] unaryCosts;
	unaryCosts = NULL;
}

LPairwiseSegmentPotential::LPairwiseSegmentPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LPnCrfLayer *setLayer, int setClassNo) : LPotential(setDataset, setCrf, setDomain, NULL, NULL, NULL, NULL, setClassNo)
{
	layer = setLayer;
}

LPottsPairwiseSegmentPotential::LPottsPairwiseSegmentPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LPnCrfLayer *setLayer, int setClassNo) : LPairwiseSegmentPotential(setDataset, setCrf, setDomain, setLayer, setClassNo)
{
	pairwiseIndexes = NULL;
	pairwiseCosts = NULL;
	pairwiseCount = 0;
}

LPottsPairwiseSegmentPotential::~LPottsPairwiseSegmentPotential()
{
	if (pairwiseIndexes != NULL) delete[] pairwiseIndexes;
	if (pairwiseCosts != NULL) delete[] pairwiseCosts;
}

int LPottsPairwiseSegmentPotential::GetEdgeCount()
{
	return(pairwiseCount * 3);
}

int LPottsPairwiseSegmentPotential::GetNodeCount()
{
	return(0);
}

void LPottsPairwiseSegmentPotential::AddCosts(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes)
{
	int from, to, i;
	double weight;

	if (domain == costDomain) for (i = 0; i < pairwiseCount; i++)
	{
		from = pairwiseIndexes[2 * i];
		to = pairwiseIndexes[2 * i + 1];
		weight = pairwiseCosts[i];

		if ((layer->active[from]) && (!layer->active[to]))
		{
			g->add_tweights(nodes[2 * to + layer->nodeOffset], 0, weight / 2);
			if (layer->labels[to] != classNo) g->add_tweights(nodes[2 * to + 1 + layer->nodeOffset], 0, weight / 2);
		}
		else if ((!layer->active[from]) && (layer->active[to]))
		{
			g->add_tweights(nodes[2 * from + layer->nodeOffset], 0, weight / 2);
			if (layer->labels[from] != classNo) g->add_tweights(nodes[2 * from + 1 + layer->nodeOffset], 0, weight / 2);
		}
		else if ((!layer->active[from]) && (!layer->active[to]))
		{
			g->add_edge(nodes[2 * from + layer->nodeOffset], nodes[2 * to + layer->nodeOffset], weight / 2, weight / 2);
			if (layer->labels[from] == layer->labels[to])
			{
				if (layer->labels[from] != classNo) g->add_edge(nodes[2 * from + 1 + layer->nodeOffset], nodes[2 * to + 1 + layer->nodeOffset], weight / 2, weight / 2);
			}
			else
			{
				if ((layer->labels[from] != classNo) && (layer->labels[to] != classNo))
				{
					g->add_tweights(nodes[2 * from + layer->nodeOffset + 1], 0, weight / 2);
					g->add_tweights(nodes[2 * to + layer->nodeOffset + 1], 0, weight / 2);
				}
				else if (layer->labels[to] == classNo) g->add_tweights(nodes[2 * from + layer->nodeOffset + 1], 0, weight / 2);
				else g->add_tweights(nodes[2 * to + layer->nodeOffset + 1], 0, weight / 2);
			}
		}
	}
}
double LPottsPairwiseSegmentPotential::GetCost(LCrfDomain *costDomain)
{
	if (domain == costDomain)
	{
		double cost = 0;
		for (int i = 0; i < pairwiseCount; i++)
		{
			int from = pairwiseIndexes[2 * i];
			int to = pairwiseIndexes[2 * i + 1];
			if (layer->labels[to] != layer->labels[from])
			{
				if ((layer->labels[from] == classNo) || (layer->labels[to] == classNo)) cost += pairwiseCosts[i] / 2;
				else cost += pairwiseCosts[i];
			}
		}
		return(cost);
	}
	else return(0);
}


LHistogramPottsPairwiseSegmentPotential::LHistogramPottsPairwiseSegmentPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LPnCrfLayer *setLayer, int setClassNo, double setPairwisePrior, double setPairwiseFactor, double setPairwiseBeta, int setBuckets) : LPottsPairwiseSegmentPotential(setDataset, setCrf, setDomain, setLayer, setClassNo)
{
	pairwisePrior = setPairwisePrior;
	pairwiseFactor = setPairwiseFactor;
	pairwiseBeta = setPairwiseBeta;
	buckets = setBuckets;
}

double LHistogramPottsPairwiseSegmentPotential::PairwiseDistance(double *h1, double *h2)
{
	int count = buckets * buckets * buckets;
	double diff = 0;

	for (int i = 0; i < count; i++) diff += (h1[i] - h2[i]) * (h1[i] - h2[i]);
	return(pairwisePrior + pairwiseFactor * exp(-diff / pairwiseBeta));
}

//��ʼ��
void LHistogramPottsPairwiseSegmentPotential::Initialize(LLabImage &labImage, char *imageFileName)
{
	int i, j, k, l;
	//ͼ���С
	int width = labImage.GetWidth(), height = labImage.GetHeight(), points = width * height;
	//Labת����Rgb�ռ�
	LRgbImage rgbImage(labImage);
	int count = buckets * buckets * buckets;

	if (pairwiseCosts != NULL) delete[] pairwiseCosts;
	if (pairwiseIndexes != NULL) delete[] pairwiseIndexes;

	//ȫ���ӣ�
	int *conn = new int[layer->segmentCount * layer->segmentCount];
	memset(conn, 0, layer->segmentCount * layer->segmentCount * sizeof(int));

	//�ָ�ͼ
	LSegmentImage segmentImage;
	char *fileName;

	//����ָ�ͼ
	fileName = GetFileName(layer->segmentation->folder, imageFileName, layer->segmentation->extension);
	segmentImage.Load(fileName);
	delete[] fileName;

	//ֱ��ͼ
	double *histograms = new double[layer->segmentCount * count];
	memset(histograms, 0, layer->segmentCount * count * sizeof(double));

	//ָ�����ݵ�ָ��
	int *segmentData = segmentImage.GetData();
	unsigned char *rgbData = rgbImage.GetData();

	//ͳ��ֱ��ͼ
	for (i = 0; i < points; i++, segmentData++, rgbData += 3)
	{
		int r = rgbData[0] * buckets / 256;
		int g = rgbData[1] * buckets / 256;
		int b = rgbData[2] * buckets / 256;
		histograms[(*segmentData) * count + r * buckets * buckets + g * buckets + b]++;
	}

	double *hData = histograms;
	for (i = 0; i < layer->segmentCount; i++, hData += count) if (layer->baseSegmentCounts[i])
	{
		for (j = 0; j < count; j++) hData[j] /= layer->baseSegmentCounts[i];
	}

	hData = histograms;
	for (i = 0; i < layer->segmentCount; i++, hData += buckets * buckets * buckets)
	{
		for (j = 1; j < buckets; j++) for (k = 0; k < buckets; k++) for (l = 0; l < buckets; l++) hData[j * buckets * buckets + k * buckets + l] += hData[(j - 1) * buckets * buckets + k * buckets + l];
		for (j = 0; j < buckets; j++) for (k = 1; k < buckets; k++) for (l = 0; l < buckets; l++) hData[j * buckets * buckets + k * buckets + l] += hData[j * buckets * buckets + (k - 1) * buckets + l];
		for (j = 0; j < buckets; j++) for (k = 0; k < buckets; k++) for (l = 1; l < buckets; l++) hData[j * buckets * buckets + k * buckets + l] += hData[j * buckets * buckets + k * buckets + l - 1];
	}

	for (i = 0; i < height - 1; i++) for (j = 0; j < width - 1; j++)
	{
		int index1 = segmentImage(j, i, 0), index2 = segmentImage(j + 1, i, 0), index3 = segmentImage(j, i + 1, 0);
		conn[index1 * layer->segmentCount + index2]++;
		conn[index1 * layer->segmentCount + index3]++;
		conn[index2 * layer->segmentCount + index1]++;
		conn[index3 * layer->segmentCount + index1]++;
	}

	//�ڵ������
	pairwiseCount = 0;
	for (i = 1; i < layer->segmentCount; i++) for (j = 0; j < i; j++) if (conn[j * layer->segmentCount + i]) pairwiseCount++;

	pairwiseCosts = new double[pairwiseCount];
	pairwiseIndexes = new int[2 * pairwiseCount];

	//����ƽ������
	int index = 0;
	for (i = 1; i < layer->segmentCount; i++) for (j = 0; j < i; j++)
	{
		if (conn[j * layer->segmentCount + i])
		{
			//����ֱ��ͼŷ�Ͼ��룬ת��Ϊ����
			pairwiseCosts[index] = conn[j * layer->segmentCount + i] * PairwiseDistance(histograms + i * count, histograms + j * count);
			pairwiseIndexes[2 * index] = j;
			pairwiseIndexes[2 * index + 1] = i;
			index++;
		}
	}
	if (conn != NULL) delete[] conn;
	if (histograms != NULL) delete[] histograms;
}

void LHistogramPottsPairwiseSegmentPotential::UnInitialize()
{
	if (pairwiseCosts != NULL) delete[] pairwiseCosts;
	if (pairwiseIndexes != NULL) delete[] pairwiseIndexes;
	pairwiseCosts = NULL;
	pairwiseIndexes = NULL;
}
