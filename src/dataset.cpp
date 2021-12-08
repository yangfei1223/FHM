#include <string.h>
#include <stdio.h>

#ifdef _WIN32
#include <io.h>
#else
#include <dirent.h>
#include <fnmatch.h>
#endif

#include "std.h"
#include "dataset.h"
#include "potential.h"
#include "crf.h"

LDataset::LDataset()
{
}

LDataset::~LDataset()
{
	int i;
	for (i = 0; i < allImageFiles.GetCount(); i++) delete[] allImageFiles[i];
}

void LDataset::RgbToLabel(unsigned char *rgb, unsigned char *label)
{
	label[0] = 0;
	for (int i = 0; i < 8; i++) label[0] = (label[0] << 3) | (((rgb[0] >> i) & 1) << 0) | (((rgb[1] >> i) & 1) << 1) | (((rgb[2] >> i) & 1) << 2);
}

void LDataset::LabelToRgb(unsigned char *label, unsigned char *rgb)
{
	unsigned char lab = label[0];
	rgb[0] = rgb[1] = rgb[2] = 0;
	for (int i = 0; lab > 0; i++, lab >>= 3)
	{
		rgb[0] |= (unsigned char)(((lab >> 0) & 1) << (7 - i));
		rgb[1] |= (unsigned char)(((lab >> 1) & 1) << (7 - i));
		rgb[2] |= (unsigned char)(((lab >> 2) & 1) << (7 - i));
	}
}

char *LDataset::GetFolderFileName(const char *imageFile, const char *folder, const char *extension)
{
	char *fileName;
	fileName = new char[strlen(imageFile) + strlen(folder) - strlen(imageFolder) + strlen(extension) - strlen(imageExtension) + 1];
	strcpy(fileName, folder);
	strncpy(fileName + strlen(folder), imageFile + strlen(imageFolder), strlen(imageFile) - strlen(imageFolder) - strlen(imageExtension));
	strcpy(fileName + strlen(imageFile) + strlen(folder) - strlen(imageFolder) - strlen(imageExtension), extension);
	return(fileName);
}

int LDataset::SortStr(char *str1, char *str2)
{
	return(strcmp(str1, str2));
};

void LDataset::LoadFolder(const char *folder, const char *extension, LList<char *> &list)
{
	char *fileName, *folderExt;

#ifdef _WIN32	
	_finddata_t info;
	int hnd;
	int done;

	folderExt = new char[strlen(folder) + strlen(extension) + 2];
	sprintf(folderExt, "%s*%s", folder, extension);

	hnd = (int)_findfirst(folderExt, &info);
	done = (hnd == -1);

	while (!done)
	{
		info.name[strlen(info.name) - strlen(extension)] = 0;
		fileName = new char[strlen(info.name) + 1];
		strcpy(fileName, info.name);
		list.Add(fileName);
		done = _findnext(hnd, &info);
	}
	_findclose(hnd);
#else
	struct dirent **nameList = NULL;
	int count;

	folderExt = new char[strlen(extension) + 2];
	sprintf(folderExt, "*%s", extension);

	count = scandir(folder, &nameList, NULL, alphasort);
	if (count >= 0)
	{
		for (int i = 0; i < count; i++)
		{
			if (!fnmatch(folderExt, nameList[i]->d_name, 0))
			{
				nameList[i]->d_name[strlen(nameList[i]->d_name) - strlen(extension)] = 0;
				fileName = new char[strlen(nameList[i]->d_name) + 1];
				strcpy(fileName, nameList[i]->d_name);
				list.Add(fileName);
			}
			if (nameList[i] != NULL) free(nameList[i]);
		}
		if (nameList != NULL) free(nameList);
	}
#endif
	delete[] folderExt;
}

void LDataset::Init()
{
	int index1, index2, i;

	//随机种子
	LMath::SetSeed(seed);

	LoadFolder(imageFolder, imageExtension, allImageFiles);

	//打乱顺序，filePermutations是随机交换次数
	printf("Permuting image files..\n");
	if (allImageFiles.GetCount() > 0)
		for (i = 0; i < filePermutations; i++)	//随机交换次数
		{
			index1 = LMath::RandomInt(allImageFiles.GetCount());
			index2 = LMath::RandomInt(allImageFiles.GetCount());
			allImageFiles.Swap(index1, index2);
		}

	//将数据分成两部分，训练集和测试集
	printf("Splitting image files..\n");
	for (i = 0; i < allImageFiles.GetCount(); i++)
	{
		//打乱顺序后前一半作为训练集
		if (i < proportionTrain * allImageFiles.GetCount())
			trainImageFiles.Add(allImageFiles[i]);
		//后一半作为测试集
		else if
			(i < (proportionTrain + proportionTest) * allImageFiles.GetCount()) testImageFiles.Add(allImageFiles[i]);
	}
}

void LDataset::SaveImage(LLabelImage &labelImage, LCrfDomain *domain, char *fileName)
{
	int points = labelImage.GetPoints();
	unsigned char *labelData = labelImage.GetData();
	for (int j = 0; j < points; j++, labelData++) (*labelData)++;	//全部加1，恢复有0标签的状态
	labelImage.Save(fileName, domain);
}

int LDataset::Segmented(char *imageFileName)
{
	return(1);
}

//标签配置
void LDataset::GetLabelSet(unsigned char *labelset, char *imageFileName)
{
	memset(labelset, 0, classNo * sizeof(unsigned char));

	char *fileName;
	fileName = GetFileName(groundTruthFolder, imageFileName, groundTruthExtension);
	LLabelImage labelImage(fileName, this, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::RgbToLabel);
	delete[] fileName;

	int points = labelImage.GetPoints(), i;
	unsigned char *labelData = labelImage.GetData();
	for (i = 0; i < points; i++, labelData++) if (*labelData) labelset[*labelData - 1] = 1;
}


LCamVidDataset::LCamVidDataset() : LDataset()
{
	seed = 60000;
	featuresOnline = 0;
	unaryWeighted = 0;

	classNo = 11;
	filePermutations = 10000;
	optimizeAverage = 1;

	//图像文件夹
	imageFolder = "Data/CamVid/Images/";
	//扩展名
	imageExtension = ".png";
	//标签文件夹
	groundTruthFolder = "Data/CamVid/GroundTruth/";
	//扩展名
	groundTruthExtension = ".png";
	//训练文件夹
	trainFolder = "Result/CamVid/Train/";
	//测试文件夹
	testFolder = "Result/CamVid/Crf/";

	//K-mean聚类参数，K-means应该是用于特征聚类的
	clusterPointsPerKDTreeCluster = 30;
	clusterKMeansMaxChange = 0.01;

	//位置特征参数设置
	locationBuckets = 12;
	locationFolder = "Result/CamVid/Feature/Location/";
	locationExtension = ".loc";

	//纹理特征参数设置
	textonNumberOfClusters = 50;	//filter respond聚类中心个数
	textonFilterBankRescale = 0.7;	//滤波器的参数吧
	textonKMeansSubSample = 10;		//??
	textonClusteringTrainFile = "textonclustering.dat";
	textonFolder = "Result/CamVid/Feature/Texton/";
	textonExtension = ".txn";

	//sift特征参数设置，HOG
	siftClusteringTrainFile = "siftclustering.dat";
	siftKMeansSubSample = 20;
	siftNumberOfClusters = 50;
	siftSizes[0] = 3, siftSizes[1] = 5, siftSizes[2] = 7, siftSizes[3] = 9;		//4个尺度
	siftSizeCount = 4;
	siftWindowNumber = 3;
	sift360 = 1;
	siftAngles = 8;
	siftFolder = "Result/CamVid/Feature/Sift/";
	siftExtension = ".sft";

	//color-sift特征参数设置
	colourSiftClusteringTrainFile = "coloursiftclustering.dat";
	colourSiftKMeansSubSample = 20;
	colourSiftNumberOfClusters = 50;
	colourSiftSizes[0] = 3, colourSiftSizes[1] = 5, colourSiftSizes[2] = 7, colourSiftSizes[3] = 9;		//4个尺度
	colourSiftSizeCount = 4;
	colourSiftWindowNumber = 3;
	colourSift360 = 1;
	colourSiftAngles = 8;
	colourSiftFolder = "Result/CamVid/Feature/ColourSift/";
	colourSiftExtension = ".csf";

	//LBP特征参数设置
	lbpClusteringFile = "lbpclustering.dat";
	lbpFolder = "Result/CamVid/Feature/Lbp/";
	lbpExtension = ".lbp";
	lbpSize = 11;
	lbpKMeansSubSample = 10;
	lbpNumberOfClusters = 50;

	//AdaBoost参数设置
	denseNumRoundsBoosting = 500;		//迭代次数，应该就是弱分类器个数
	denseBoostingSubSample = 5;		//降采样
	denseNumberOfThetas = 25;
	denseThetaStart = 3;
	denseThetaIncrement = 2;
	denseNumberOfRectangles = 100;		//矩形个数
	denseMinimumRectangleSize = 5;		//最小矩形大小
	denseMaximumRectangleSize = 200;		//最大矩形大小
	denseRandomizationFactor = 0.003;		//随机因子
	denseBoostTrainFile = "denseboost.dat";
	denseExtension = ".dns";
	denseFolder = "Result/CamVid/Dense/";
	denseWeight = 1.0;
	denseMaxClassRatio = 0.1;

	//mean-sift参数设置
	meanShiftXY[0] = 3.0;
	meanShiftLuv[0] = 0.3;
	meanShiftMinRegion[0] = 200;
	meanShiftFolder[0] = "Result/CamVid/MeanShift/30x03/";
	meanShiftXY[1] = 3.0;
	meanShiftLuv[1] = 0.6;
	meanShiftMinRegion[1] = 200;
	meanShiftFolder[1] = "Result/CamVid/MeanShift/30x06/";
	meanShiftXY[2] = 3.0;
	meanShiftLuv[2] = 0.9;
	meanShiftMinRegion[2] = 200;
	meanShiftFolder[2] = "Result/CamVid/MeanShift/30x09/";
	meanShiftExtension = ".msh";

	//平滑项参数设置
	pairwiseLWeight = 1.0 / 3.0;
	pairwiseUWeight = 1.0 / 3.0;
	pairwiseVWeight = 1.0 / 3.0;
	pairwisePrior = 1.5;
	pairwiseFactor = 6.0;
	pairwiseBeta = 16.0;

	//团的一些参数
	cliqueMinLabelRatio = 0.5;	//应该是主导标签比例
	cliqueThresholdRatio = 0.1;
	cliqueTruncation = 0.1;		//截断阈值，文章中的alpha吧

	//这些暂时不太清楚
	statsThetaStart = 2;
	statsThetaIncrement = 1;
	statsNumberOfThetas = 15;
	statsNumberOfBoosts = 500;
	statsRandomizationFactor = 0.1;
	statsFactor = 0.5;
	statsAlpha = 0.05;
	statsPrior = 0;
	statsMaxClassRatio = 0.5;
	statsTrainFile = "statsboost.dat";
	statsFolder = "Result/CamVid/Stats/";
	statsExtension = ".sts";

	Init();

	//创建文件夹
	int i;
	ForceDirectory(trainFolder);
	ForceDirectory(testFolder, "Train/");
	ForceDirectory(textonFolder, "Train/");
	ForceDirectory(siftFolder, "Train/");
	ForceDirectory(colourSiftFolder, "Train/");
	ForceDirectory(locationFolder, "Train/");
	ForceDirectory(lbpFolder, "Train/");
	for (i = 0; i < 3; i++) ForceDirectory(meanShiftFolder[i], "Train/");
	ForceDirectory(denseFolder, "Train/");
	ForceDirectory(statsFolder, "Train/");
	ForceDirectory(testFolder, "Val/");
	ForceDirectory(textonFolder, "Val/");
	ForceDirectory(siftFolder, "Val/");
	ForceDirectory(colourSiftFolder, "Val/");
	ForceDirectory(locationFolder, "Val/");
	ForceDirectory(lbpFolder, "Val/");
	for (i = 0; i < 3; i++) ForceDirectory(meanShiftFolder[i], "Val/");
	ForceDirectory(denseFolder, "Val/");
	ForceDirectory(statsFolder, "Val/");
	ForceDirectory(testFolder, "Test/");
	ForceDirectory(textonFolder, "Test/");
	ForceDirectory(siftFolder, "Test/");
	ForceDirectory(colourSiftFolder, "Test/");
	ForceDirectory(locationFolder, "Test/");
	ForceDirectory(lbpFolder, "Test/");
	for (i = 0; i < 3; i++) ForceDirectory(meanShiftFolder[i], "Test/");
	ForceDirectory(denseFolder, "Test/");
	ForceDirectory(statsFolder, "Test/");
}

void LCamVidDataset::RgbToLabel(unsigned char *rgb, unsigned char *label)
{
	label[0] = 0;
	for (int i = 0; i < 8; i++) label[0] = (label[0] << 3) | (((rgb[0] >> i) & 1) << 0) | (((rgb[1] >> i) & 1) << 1) | (((rgb[2] >> i) & 1) << 2);

	switch (label[0])
	{
	case 1:
		label[0] = 1; break;
	case 3:
		label[0] = 2; break;
	case 7:
		label[0] = 3; break;
	case 12:
		label[0] = 4; break;
	case 13:
		label[0] = 3; break;
	case 15:
		label[0] = 5; break;
	case 21:
		label[0] = 6; break;
	case 24:
		label[0] = 7; break;
	case 26:
		label[0] = 8; break;
	case 27:
		label[0] = 2; break;
	case 28:
		label[0] = 8; break;
	case 30:
		label[0] = 10; break;
	case 31:
		label[0] = 9; break;
	case 34:
		label[0] = 3; break;
	case 35:
		label[0] = 5; break;
	case 36:
		label[0] = 10; break;
	case 37:
		label[0] = 6; break;
	case 38:
		label[0] = 11; break;
	case 39:
		label[0] = 6; break;
	case 40:
		label[0] = 3; break;
	case 41:
		label[0] = 6; break;
	case 45:
		label[0] = 11; break;
	case 46:
		label[0] = 4; break;
	case 47:
		label[0] = 4; break;
	case 48:
		label[0] = 5; break;
	default:
		label[0] = 0; break;
	}
}

void LCamVidDataset::LabelToRgb(unsigned char *label, unsigned char *rgb)
{
	unsigned char lab = label[0];
	switch (lab)
	{
	case 1:
		lab = 1; break;
	case 2:
		lab = 3; break;
	case 3:
		lab = 7; break;
	case 4:
		lab = 12; break;
	case 5:
		lab = 15; break;
	case 6:
		lab = 21; break;
	case 7:
		lab = 24; break;
	case 8:
		lab = 28; break;
	case 9:
		lab = 31; break;
	case 10:
		lab = 36; break;
	case 11:
		lab = 38; break;
	default:
		lab = 0;
	}
	rgb[0] = rgb[1] = rgb[2] = 0;
	for (int i = 0; lab > 0; i++, lab >>= 3)
	{
		rgb[0] |= (unsigned char)(((lab >> 0) & 1) << (7 - i));
		rgb[1] |= (unsigned char)(((lab >> 1) & 1) << (7 - i));
		rgb[2] |= (unsigned char)(((lab >> 2) & 1) << (7 - i));
	}
}

//把指定文件夹下面的文件添加到相应的list中
void LCamVidDataset::AddFolder(char *folder, LList<char *> &fileList)
{
	char *fileName, *folderExt;

#ifdef _WIN32	
	_finddata_t info;
	int hnd;
	int done;

	folderExt = new char[strlen(imageFolder) + strlen(folder) + strlen(imageExtension) + 2];
	sprintf(folderExt, "%s%s*%s", imageFolder, folder, imageExtension);

	hnd = (int)_findfirst(folderExt, &info);
	done = (hnd == -1);

	while (!done)
	{
		info.name[strlen(info.name) - strlen(imageExtension)] = 0;
		fileName = new char[strlen(folder) + strlen(info.name) + 1];
		sprintf(fileName, "%s%s", folder, info.name);
		fileList.Add(fileName);
		allImageFiles.Add(fileName);	//同时添加到所有文件中
		done = _findnext(hnd, &info);
	}
	_findclose(hnd);
#else
	char *wholeFolder;
	struct dirent **nameList = NULL;
	int count;

	folderExt = new char[strlen(imageExtension) + 2];
	sprintf(folderExt, "*%s", imageExtension);

	wholeFolder = new char[strlen(imageFolder) + strlen(folder) + 1];
	sprintf(wholeFolder, "%s%s", imageFolder, folder);

	count = scandir(wholeFolder, &nameList, NULL, alphasort);
	if (count >= 0)
	{
		for (int i = 0; i < count; i++)
		{
			if (!fnmatch(folderExt, nameList[i]->d_name, 0))
			{
				nameList[i]->d_name[strlen(nameList[i]->d_name) - strlen(imageExtension)] = 0;
				fileName = new char[strlen(folder) + strlen(nameList[i]->d_name) + 1];
				sprintf(fileName, "%s%s", folder, nameList[i]->d_name);
				fileList.Add(fileName);
				allImageFiles.Add(fileName);
	}
			if (nameList[i] != NULL) free(nameList[i]);
}
		if (nameList != NULL) free(nameList);
}
	delete[] wholeFolder;
#endif
	delete[] folderExt;
}

//添加文件
void LCamVidDataset::Init()
{
	//训练集和验证集都当作训练集
	AddFolder("Train/", trainImageFiles);
	//	AddFolder("Val/", trainImageFiles);
	AddFolder("Test/", testImageFiles);
}

void LCamVidDataset::SetCRFStructure(LCrf *crf)
{
	//创建CFR主体，包含一些基本信息
	LCrfDomain *objDomain = new LCrfDomain(crf, this, classNo, testFolder, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::RgbToLabel, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::LabelToRgb);
	//添加主体信息到CRF模型
	crf->domains.Add(objDomain);
	//创建底层模型，即基于像素层CRF
	LBaseCrfLayer *baseLayer = new LBaseCrfLayer(crf, objDomain, this, 0);
	//添加底层到CRF模型
	crf->layers.Add(baseLayer);

	//高阶层，即基于segment的高阶CRF
	LPnCrfLayer *superpixelLayer[3];
	//高阶层的超像素段
	LSegmentation2D *segmentation[3];

	//分别创建高阶层模型
	int i;
	for (i = 0; i < 3; i++)
	{
		//首先进行无监督segmentation，获取不同scale的segment，此处是mean-sift分割算法
		segmentation[i] = new LMeanShiftSegmentation2D(meanShiftXY[i], meanShiftLuv[i], meanShiftMinRegion[i], meanShiftFolder[i], meanShiftExtension);
		//然后基于segment map创建高阶层
		superpixelLayer[i] = new LPnCrfLayer(crf, objDomain, this, baseLayer, segmentation[i], cliqueTruncation);
		//添加个segment map到CRF模型
		crf->segmentations.Add(segmentation[i]);
		//添加高阶层到CRF模型
		crf->layers.Add(superpixelLayer[i]);
	}

	//创建提取各个特征类
	//TextonBoost特征
	LTextonFeature *textonFeature = new LTextonFeature(this, trainFolder, textonClusteringTrainFile, textonFolder, textonExtension, textonFilterBankRescale, textonKMeansSubSample, textonNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster);
	//位置特征
	LLocationFeature *locationFeature = new LLocationFeature(this, locationFolder, locationExtension, locationBuckets);
	//Sift特征，实际上作者说明的HOG特征
	LSiftFeature *siftFeature = new LSiftFeature(this, trainFolder, siftClusteringTrainFile, siftFolder, siftExtension, siftSizeCount, siftSizes, siftWindowNumber, sift360, siftAngles, siftKMeansSubSample, siftNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster, 1);
	//基于LUV颜色空间的Sift特征
	LColourSiftFeature *coloursiftFeature = new LColourSiftFeature(this, trainFolder, colourSiftClusteringTrainFile, colourSiftFolder, colourSiftExtension, colourSiftSizeCount, colourSiftSizes, colourSiftWindowNumber, colourSift360, colourSiftAngles, colourSiftKMeansSubSample, colourSiftNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster, 1);
	//LBP特征，描述纹理的
	LLbpFeature *lbpFeature = new LLbpFeature(this, trainFolder, lbpClusteringFile, lbpFolder, lbpExtension, lbpSize, lbpKMeansSubSample, lbpNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster);

	//添加各特征到CRF模型
	crf->features.Add(textonFeature);
	crf->features.Add(locationFeature);
	crf->features.Add(lbpFeature);
	crf->features.Add(siftFeature);
	crf->features.Add(coloursiftFeature);

	//基于Dense Feature，即像素特征的底层一元势函数
	LDenseUnaryPixelPotential *pixelPotential = new LDenseUnaryPixelPotential(this, crf, objDomain, baseLayer, trainFolder, denseBoostTrainFile, denseFolder, denseExtension, classNo, denseWeight, denseBoostingSubSample, denseNumberOfRectangles, denseMinimumRectangleSize, denseMaximumRectangleSize, denseMaxClassRatio);
	//Dense Feature所用到的4种特征
	pixelPotential->AddFeature(textonFeature);
	pixelPotential->AddFeature(siftFeature);
	pixelPotential->AddFeature(coloursiftFeature);
	pixelPotential->AddFeature(lbpFeature);
	//添加数据项到CRF模型
	crf->potentials.Add(pixelPotential);
	//用于像素级分类的Boosting分类器
	LBoosting<int> *pixelBoosting = new LBoosting<int>(trainFolder, denseBoostTrainFile, classNo, denseNumRoundsBoosting, denseThetaStart, denseThetaIncrement, denseNumberOfThetas, denseRandomizationFactor, pixelPotential, (int *(LPotential::*)(int, int))&LDenseUnaryPixelPotential::GetTrainBoostingValues, (int *(LPotential::*)(int))&LDenseUnaryPixelPotential::GetEvalBoostingValues);
	//添加分类器到CRF模型
	crf->learnings.Add(pixelBoosting);
	//学习算法设置为Boosting
	pixelPotential->learning = pixelBoosting;		//基于像素的Boosting

	//创建基于8领域的Potts模型的二元势函数
	LEightNeighbourPottsPairwisePixelPotential *pairwisePotentianl = new LEightNeighbourPottsPairwisePixelPotential(this, crf, objDomain, baseLayer, classNo, pairwisePrior, pairwiseFactor, pairwiseBeta, pairwiseLWeight, pairwiseUWeight, pairwiseVWeight);
	//添加平滑项到CRF模型
	crf->potentials.Add(pairwisePotentianl);

	//基于segment的一元势函数，使用统计的方法提取的特征分布
	LStatsUnarySegmentPotential *statsPotential = new LStatsUnarySegmentPotential(this, crf, objDomain, trainFolder, statsTrainFile, statsFolder, statsExtension, classNo, statsPrior, statsFactor, cliqueMinLabelRatio, statsAlpha, statsMaxClassRatio);
	//segment所用到的特征，segment的特征是基于dense feature的归一化直方图，描述特征分布情况
	statsPotential->AddFeature(textonFeature);
	statsPotential->AddFeature(siftFeature);
	statsPotential->AddFeature(coloursiftFeature);
	statsPotential->AddFeature(locationFeature);
	statsPotential->AddFeature(lbpFeature);
	//将每个高阶层添加到高阶势函数
	for (i = 0; i < 3; i++) statsPotential->AddLayer(superpixelLayer[i]);
	//用于segment分类的Boosting分类器
	LBoosting<double> *segmentBoosting = new LBoosting<double>(trainFolder, statsTrainFile, classNo, statsNumberOfBoosts, statsThetaStart, statsThetaIncrement, statsNumberOfThetas, statsRandomizationFactor, statsPotential, (double *(LPotential::*)(int, int))&LStatsUnarySegmentPotential::GetTrainBoostingValues, (double *(LPotential::*)(int))&LStatsUnarySegmentPotential::GetEvalBoostingValues);
	//添加分类器到CRF模型
	crf->learnings.Add(segmentBoosting);
	//高阶势函数的学习设置为Boosting
	statsPotential->learning = segmentBoosting;		//基于块的Boosting
	//添加高阶势函数到CRF模型
	crf->potentials.Add(statsPotential);
}



/////////////////////////////////////////////////////////////////////////////////////////////
//
//				KITTI	Dataet
//
/////////////////////////////////////////////////////////////////////////////////////////////

//KITTI数据集，分层模型
LKITTIDataset::LKITTIDataset()
{
	seed = 10000;
	classNo = 2;	//KITTI数据集就道路和非路两类
	filePermutations = 10000;
	optimizeAverage = 1;
	featuresOnline = 0;

	//类别权重，初始化为全1
	unaryWeighted = 0;
	unaryWeights = new double[classNo];
	for (int i = 0; i < classNo; i++) unaryWeights[i] = 1.0;

	//训练集和测试集比例1:1
	proportionTrain = 0.5;
	proportionTest = 0.5;

	imageFolder = "/home/yangfei/Datasets/Data/KITTI/um/Images/";
	imageExtension = ".png";
	groundTruthFolder = "/home/yangfei/Datasets/Data/KITTI/um/GroundTruth/";
	groundTruthExtension = ".png";
	lidarFolder = "/home/yangfei/Datasets/Data/KITTI/um/Lidar/";
	lidarExtension = ".bin";
	calibFolder = "/home/yangfei/Datasets/Data/KITTI/um/calib/";
	calibExtension = ".txt";
	trainFolder = "/home/yangfei/Datasets/Result/KITTI/um/Train/";
	testFolder = "/home/yangfei/Datasets/Result/KITTI/um/Crf/";

	clusterPointsPerKDTreeCluster = 30;
	clusterKMeansMaxChange = 0.01;

	textonNumberOfClusters = 50;		//50个聚类中心
	textonFilterBankRescale = 0.7;
	textonKMeansSubSample = 20;
	textonClusteringTrainFile = "textonclustering.dat";
	textonFolder = "/home/yangfei/Datasets/Result/KITTI/um/Feature/Texton/";
	textonExtension = ".txn";

	siftClusteringTrainFile = "siftclustering.dat";
	siftKMeansSubSample = 40;
	siftNumberOfClusters = 50;
	siftSizes[0] = 5, siftSizes[1] = 7, siftSizes[2] = 9;
	siftSizeCount = 3;
	siftWindowNumber = 3;
	sift360 = 1;	//360度
	siftAngles = 8;		//HOG有8个bin代表360度的8个方向
	siftFolder = "/home/yangfei/Datasets/Result/KITTI/um/Feature/Sift/";
	siftExtension = ".sft";

	//加了颜色信息的HOG
	colourSiftClusteringTrainFile = "coloursiftclustering.dat";
	colourSiftKMeansSubSample = 40;
	colourSiftNumberOfClusters = 50;
	colourSiftSizes[0] = 5, colourSiftSizes[1] = 7, colourSiftSizes[2] = 9;
	colourSiftSizeCount = 3;
	colourSiftWindowNumber = 3;
	colourSift360 = 1;
	colourSiftAngles = 8;
	colourSiftFolder = "/home/yangfei/Datasets/Result/KITTI/um/Feature/ColourSift/";
	colourSiftExtension = ".csf";

	locationBuckets = 20;	//位置数，归一化
	locationFolder = "/home/yangfei/Datasets/Result/KITTI/um/Feature/Location/";
	locationExtension = ".loc";

	lbpClusteringFile = "lbpclustering.dat";
	lbpFolder = "/home/yangfei/Datasets/Result/KITTI/um/Feature/Lbp/";
	lbpExtension = ".lbp";
	lbpSize = 11;
	lbpKMeansSubSample = 20;
	lbpNumberOfClusters = 50;

	//第一层
	meanShiftXY[0] = 5.0;
	meanShiftLuv[0] = 3.0;
	meanShiftMinRegion[0] = 20;
	meanShiftFolder[0] = "/home/yangfei/Datasets/Result/KITTI/um/MeanShift/50x30/";
	//第一层
	meanShiftXY[1] = 5.0;
	meanShiftLuv[1] = 5.0;
	meanShiftMinRegion[1] = 20;
	meanShiftFolder[1] = "/home/yangfei/Datasets/Result/KITTI/um/MeanShift/50x50/";
	//第二层
	meanShiftXY[2] = 7.0;
	meanShiftLuv[2] = 5.0;
	meanShiftMinRegion[2] = 20;
	meanShiftFolder[2] = "/home/yangfei/Datasets/Result/KITTI/um/MeanShift/70x50/";
	//第二层
	meanShiftXY[3] = 7.0;
	meanShiftLuv[3] = 6.0;
	meanShiftMinRegion[3] = 20;
	meanShiftFolder[3] = "/home/yangfei/Datasets/Result/KITTI/um/MeanShift/70x60/";

	meanShiftExtension = ".msh";


	//Boosting参数
	denseNumRoundsBoosting = 5000;
	denseBoostingSubSample = 10;
	denseNumberOfThetas = 20;
	denseThetaStart = 3;
	denseThetaIncrement = 2;
	denseNumberOfRectangles = 100;
	denseMinimumRectangleSize = 5;
	denseMaximumRectangleSize = 200;
	denseRandomizationFactor = 0.003;
	denseBoostTrainFile = "denseboost.dat";
	denseExtension = ".dns";
	denseFolder = "/home/yangfei/Datasets/Result/KITTI/um/Dense/";
	
	denseWeight = 1.0;		//1.0->2.0
	//最大类别概率
	denseMaxClassRatio = 0.2;

	//平滑项三个通道的权重
	pairwiseLWeight = 1.0 / 3.0;
	pairwiseUWeight = 1.0 / 3.0;
	pairwiseVWeight = 1.0 / 3.0;
	pairwisePrior = 1.5;		//1.5->0
	pairwiseFactor = 6.0;		//6.0->20
	pairwiseBeta = 16.0;

	//主导标签比例
	cliqueMinLabelRatio = 0.5;
	//阈值
	cliqueThresholdRatio = 0.1;
	//截断参数
	cliqueTruncation = 0.1;		//0.1->0.5

	//Boosting参数
	statsThetaStart = 2;
	statsThetaIncrement = 1;
	statsNumberOfThetas = 15;
	statsNumberOfBoosts = 5000;
	statsRandomizationFactor = 0.1;
	statsFactor = 0.6;		//0.6->1.0
	statsAlpha = 0.05;
	statsPrior = 0.0;
	statsMaxClassRatio = 0.5;		//0.5->0.7
	statsTrainFile = "statsboost.dat";
	statsFolder = "/home/yangfei/Datasets/Result/KITTI/um/Stats/";
	statsExtension = ".sts";

	//一致性先验项
	consistencyPrior = 0.05;
	//高阶平滑项参数
	pairwiseSegmentBuckets = 8;
	pairwiseSegmentPrior = 0.0;		//0.0—>1.5
	pairwiseSegmentFactor = 2.0;
	pairwiseSegmentBeta = 40.0;

	//初始化，打乱顺序，按比例生成训练集和测试集
	Init();

	//创建目录
	int i;
	ForceDirectory(trainFolder);

	ForceDirectory(testFolder,"Train/");
	ForceDirectory(textonFolder, "Train/");
	ForceDirectory(siftFolder, "Train/");
	ForceDirectory(colourSiftFolder, "Train/");
	ForceDirectory(locationFolder, "Train/");
	ForceDirectory(lbpFolder, "Train/");
	for (i = 0; i < 4; i++)	ForceDirectory(meanShiftFolder[i], "Train/");
	ForceDirectory(denseFolder, "Train/");
	ForceDirectory(statsFolder, "Train/");

	ForceDirectory(testFolder, "Test/");
	ForceDirectory(textonFolder, "Test/");
	ForceDirectory(siftFolder, "Test/");
	ForceDirectory(colourSiftFolder, "Test/");
	ForceDirectory(locationFolder, "Test/");
	ForceDirectory(lbpFolder, "Test/");
	for (i = 0; i < 4; i++)	ForceDirectory(meanShiftFolder[i], "Test/");
	ForceDirectory(denseFolder, "Test/");
	ForceDirectory(statsFolder, "Test/");

	ForceDirectory("/home/yangfei/Datasets/Result/KITTI/um/Lidar/denseClassifierOut/Train/");
	ForceDirectory("/home/yangfei/Datasets/Result/KITTI/um/Lidar/statClassifierOut/Train/");
	ForceDirectory("/home/yangfei/Datasets/Result/KITTI/um/Lidar/denseFeatureEval/Train/");
	ForceDirectory("/home/yangfei/Datasets/Result/KITTI/um/Lidar/statFeatureEval/Train/");

	ForceDirectory("/home/yangfei/Datasets/Result/KITTI/um/Lidar/denseClassifierOut/Test/");
	ForceDirectory("/home/yangfei/Datasets/Result/KITTI/um/Lidar/statClassifierOut/Test/");
	ForceDirectory("/home/yangfei/Datasets/Result/KITTI/um/Lidar/denseFeatureEval/Test/");
	ForceDirectory("/home/yangfei/Datasets/Result/KITTI/um/Lidar/statFeatureEval/Test/");

}
void LKITTIDataset::AddFolder(char *folder, LList<char *> &fileList)
{
	char *fileName, *folderExt;

#ifdef _WIN32	
	_finddata_t info;
	int hnd;
	int done;

	folderExt = new char[strlen(imageFolder) + strlen(folder) + strlen(imageExtension) + 2];
	sprintf(folderExt, "%s%s*%s", imageFolder, folder, imageExtension);

	hnd = (int)_findfirst(folderExt, &info);
	done = (hnd == -1);

	while (!done)
	{
		info.name[strlen(info.name) - strlen(imageExtension)] = 0;
		fileName = new char[strlen(folder) + strlen(info.name) + 1];
		sprintf(fileName, "%s%s", folder, info.name);
		fileList.Add(fileName);
		allImageFiles.Add(fileName);	//同时添加到所有文件中
		done = _findnext(hnd, &info);
	}
	_findclose(hnd);
#else
	char *wholeFolder;
	struct dirent **nameList = NULL;
	int count;

	folderExt = new char[strlen(imageExtension) + 2];
	sprintf(folderExt, "*%s", imageExtension);

	wholeFolder = new char[strlen(imageFolder) + strlen(folder) + 1];
	sprintf(wholeFolder, "%s%s", imageFolder, folder);

	count = scandir(wholeFolder, &nameList, NULL, alphasort);
	if (count >= 0)
	{
		for (int i = 0; i < count; i++)
		{
			if (!fnmatch(folderExt, nameList[i]->d_name, 0))
			{
				nameList[i]->d_name[strlen(nameList[i]->d_name) - strlen(imageExtension)] = 0;
				fileName = new char[strlen(folder) + strlen(nameList[i]->d_name) + 1];
				sprintf(fileName, "%s%s", folder, nameList[i]->d_name);
				fileList.Add(fileName);
				allImageFiles.Add(fileName);
			}
			if (nameList[i] != NULL) free(nameList[i]);
		}
		if (nameList != NULL) free(nameList);
	}
	delete[] wholeFolder;
#endif
	delete[] folderExt;
}
void LKITTIDataset::Init()
{
	//训练集和验证集都当作训练集
	AddFolder("Train/", trainImageFiles);
	AddFolder("Test/", testImageFiles);
}
void LKITTIDataset::RgbToLabel(unsigned char *rgb, unsigned char *label)
{
	if (rgb[2]==255)		//r通道代表有效点
	{
		if (rgb[0]==255)	//b通道代表道路样本
		{
			label[0] = 1;		//道路
		}
		else	
			label[0] = 2;		//非路
	}
	else	
		label[0] = 0;		//无效
	
}
void LKITTIDataset::LabelToRgb(unsigned char *label, unsigned char *rgb)
{
	rgb[0] = rgb[1] = rgb[2] = 0;
	if (label[0]==2)
	{
		rgb[2] = 255;		//非路红色
	}
	else
	{
		if (label[0]==1)
		{
			rgb[2] = 255, rgb[0] = 255;		//道路粉色
		}
	}	
}
void LKITTIDataset::SetCRFStructure(LCrf *crf)
{
	//创建CRF主体对象
	LCrfDomain *objDomain = new LCrfDomain(crf, this, classNo, testFolder, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::RgbToLabel, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::LabelToRgb);
	//添加主体到模型
	crf->domains.Add(objDomain);

	//创建底层
	LBaseCrfLayer *baseLayer = new LBaseCrfLayer(crf, objDomain, this, 0);
	//添加底层到模型
	crf->layers.Add(baseLayer);

	//创建高层和分割
	LPnCrfLayer *superpixelLayer[3];
	LSegmentation2D *segmentation[3];


	//baselayer
	segmentation[0] = new LMeanShiftSegmentation2D(meanShiftXY[1], meanShiftLuv[1], meanShiftMinRegion[1], meanShiftFolder[1], meanShiftExtension);
	//建立在底层上的分割，设置父层为base层以及超像素层
	superpixelLayer[0] = new LPnCrfLayer(crf, objDomain, this, baseLayer, segmentation[0], cliqueTruncation);
	crf->segmentations.Add(segmentation[0]);		//添加分割到crf模型
	crf->layers.Add(superpixelLayer[0]);		//添加高阶层到crf模型

	int i;
	//meansSift分割
	//for (i = 1; i < 2; i++)		//2~3layer
	//{
	//	//不同尺度的聚类参数
	//	segmentation[i] = new LMeanShiftSegmentation2D(meanShiftXY[i], meanShiftLuv[i], meanShiftMinRegion[i], meanShiftFolder[i], meanShiftExtension);
	//	//建立在第一层超像素层上的层
	//	superpixelLayer[i] = new LPnCrfLayer(crf, objDomain, this, superpixelLayer[0], segmentation[i], cliqueTruncation);
	//	crf->segmentations.Add(segmentation[i]);
	//	crf->layers.Add(superpixelLayer[i]);
	//}

	//添加特征对象
	LTextonFeature *textonFeature = new LTextonFeature(this, trainFolder, textonClusteringTrainFile, textonFolder, textonExtension, textonFilterBankRescale, textonKMeansSubSample, textonNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster);
	LLocationFeature *locationFeature = new LLocationFeature(this, locationFolder, locationExtension, locationBuckets);
	LSiftFeature *siftFeature = new LSiftFeature(this, trainFolder, siftClusteringTrainFile, siftFolder, siftExtension, siftSizeCount, siftSizes, siftWindowNumber, sift360, siftAngles, siftKMeansSubSample, siftNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster, 1);
	LColourSiftFeature *coloursiftFeature = new LColourSiftFeature(this, trainFolder, colourSiftClusteringTrainFile, colourSiftFolder, colourSiftExtension, colourSiftSizeCount, colourSiftSizes, colourSiftWindowNumber, colourSift360, colourSiftAngles, colourSiftKMeansSubSample, colourSiftNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster, 1);
	LLbpFeature *lbpFeature = new LLbpFeature(this, trainFolder, lbpClusteringFile, lbpFolder, lbpExtension, lbpSize, lbpKMeansSubSample, lbpNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster);

	crf->features.Add(textonFeature);
	crf->features.Add(locationFeature);
	crf->features.Add(lbpFeature);
	crf->features.Add(siftFeature);
	crf->features.Add(coloursiftFeature);

	//一阶数据项
	LDenseUnaryPixelPotential *pixelPotential = new LDenseUnaryPixelPotential(this, crf, objDomain, baseLayer, trainFolder, denseBoostTrainFile, denseFolder, denseExtension, classNo, denseWeight, denseBoostingSubSample, denseNumberOfRectangles, denseMinimumRectangleSize, denseMaximumRectangleSize, denseMaxClassRatio);
	pixelPotential->AddFeature(textonFeature);
	pixelPotential->AddFeature(siftFeature);
	pixelPotential->AddFeature(coloursiftFeature);
	pixelPotential->AddFeature(lbpFeature);
	crf->potentials.Add(pixelPotential);

	LBoosting<int> *pixelBoosting = new LBoosting<int>(trainFolder, denseBoostTrainFile, classNo, denseNumRoundsBoosting, denseThetaStart, denseThetaIncrement, denseNumberOfThetas, denseRandomizationFactor, pixelPotential, (int *(LPotential::*)(int, int))&LDenseUnaryPixelPotential::GetTrainBoostingValues, (int *(LPotential::*)(int))&LDenseUnaryPixelPotential::GetEvalBoostingValues);
	crf->learnings.Add(pixelBoosting);
	pixelPotential->learning = pixelBoosting;

	//一阶平滑项
	crf->potentials.Add(new LEightNeighbourPottsPairwisePixelPotential(this, crf, objDomain, baseLayer, classNo, pairwisePrior, pairwiseFactor, pairwiseBeta, pairwiseLWeight, pairwiseUWeight, pairwiseVWeight));

	
	//高阶数据项
	LStatsUnarySegmentPotential *statsPotential = new LStatsUnarySegmentPotential(this, crf, objDomain, trainFolder, statsTrainFile, statsFolder, statsExtension, classNo, statsPrior, statsFactor, cliqueMinLabelRatio, statsAlpha, statsMaxClassRatio);
	statsPotential->AddFeature(textonFeature);
	statsPotential->AddFeature(siftFeature);
	statsPotential->AddFeature(coloursiftFeature);
	statsPotential->AddFeature(locationFeature);
	statsPotential->AddFeature(lbpFeature);
	
	//添加超像素层
	for (i = 0; i < 1; i++) statsPotential->AddLayer(superpixelLayer[i]);

	crf->potentials.Add(statsPotential);

	LBoosting<double> *segmentBoosting = new LBoosting<double>(trainFolder, statsTrainFile, classNo, statsNumberOfBoosts, statsThetaStart, statsThetaIncrement, statsNumberOfThetas, statsRandomizationFactor, statsPotential, (double *(LPotential::*)(int, int))&LStatsUnarySegmentPotential::GetTrainBoostingValues, (double *(LPotential::*)(int))&LStatsUnarySegmentPotential::GetEvalBoostingValues);
	crf->learnings.Add(segmentBoosting);
	statsPotential->learning = segmentBoosting;
	
	//高阶平滑项
	crf->potentials.Add(new LHistogramPottsPairwiseSegmentPotential(this, crf, objDomain, superpixelLayer[0], classNo, pairwiseSegmentPrior, pairwiseSegmentFactor, pairwiseSegmentBeta, pairwiseSegmentBuckets));

	//一致性先验项参数
	consistencyPrior = 100000;
	//一致性高阶数据项，鼓励段内一致性的
	LConsistencyUnarySegmentPotential *consistencyPotential = new LConsistencyUnarySegmentPotential(this, crf, objDomain, classNo, consistencyPrior);
	//基于第一个超像素层
	consistencyPotential->AddLayer(superpixelLayer[0]);
	crf->potentials.Add(consistencyPotential);

	//这个层暂时不清楚
	LPreferenceCrfLayer *preferenceLayer = new LPreferenceCrfLayer(crf, objDomain, this, baseLayer);
	crf->layers.Add(preferenceLayer);

}



//以下为融合模型

//KITTI数据集um
LKITTIumDataset::LKITTIumDataset()
{
	seed = 10000;
	classNo = 2;	//KITTI数据集就道路和非路两类
	filePermutations = 10000;
	optimizeAverage = 1;
	featuresOnline = 0;

	//类别权重，初始化为全1
	unaryWeighted = 0;
	unaryWeights = new double[classNo];
	for (int i = 0; i < classNo; i++) unaryWeights[i] = 1.0;

	//训练集和测试集比例1:1
	proportionTrain = 0.5;
	proportionTest = 0.5;

	datasetName = "um";			//数据集名
	imageFolder = "/home/yangfei/Datasets/Data/KITTI/um/Images/";
	imageExtension = ".png";
	groundTruthFolder = "/home/yangfei/Datasets/Data/KITTI/um/GroundTruth/";
	groundTruthExtension = ".png";
	lidarFolder = "/home/yangfei/Datasets/Data/KITTI/um/Lidar/";
	lidarExtension = ".bin";
	calibFolder = "/home/yangfei/Datasets/Data/KITTI/um/calib/";
	calibExtension = ".txt";
	trainFolder = "/home/yangfei/Datasets/Result/KITTI/um/Train/";
	testFolder = "/home/yangfei/Datasets/Result/KITTI/um/Crf/";

	clusterPointsPerKDTreeCluster = 30;
	clusterKMeansMaxChange = 0.01;

	textonNumberOfClusters = 50;		//50个聚类中心
	textonFilterBankRescale = 0.7;
	textonKMeansSubSample = 20;
	textonClusteringTrainFile = "textonclustering.dat";
	textonFolder = "/home/yangfei/Datasets/Result/KITTI/um/Feature/Texton/";
	textonExtension = ".txn";

	siftClusteringTrainFile = "siftclustering.dat";
	siftKMeansSubSample = 40;
	siftNumberOfClusters = 50;
	siftSizes[0] = 5, siftSizes[1] = 7, siftSizes[2] = 9;
	siftSizeCount = 3;
	siftWindowNumber = 3;
	sift360 = 1;	//360度
	siftAngles = 8;		//HOG有8个bin代表360度的8个方向
	siftFolder = "/home/yangfei/Datasets/Result/KITTI/um/Feature/Sift/";
	siftExtension = ".sft";

	//加了颜色信息的HOG
	colourSiftClusteringTrainFile = "coloursiftclustering.dat";
	colourSiftKMeansSubSample = 40;
	colourSiftNumberOfClusters = 50;
	colourSiftSizes[0] = 5, colourSiftSizes[1] = 7, colourSiftSizes[2] = 9;
	colourSiftSizeCount = 3;
	colourSiftWindowNumber = 3;
	colourSift360 = 1;
	colourSiftAngles = 8;
	colourSiftFolder = "/home/yangfei/Datasets/Result/KITTI/um/Feature/ColourSift/";
	colourSiftExtension = ".csf";

	locationBuckets = 20;	//位置数，归一化
	locationFolder = "/home/yangfei/Datasets/Result/KITTI/um/Feature/Location/";
	locationExtension = ".loc";

	lbpClusteringFile = "lbpclustering.dat";
	lbpFolder = "/home/yangfei/Datasets/Result/KITTI/um/Feature/Lbp/";
	lbpExtension = ".lbp";
	lbpSize = 11;
	lbpKMeansSubSample = 20;
	lbpNumberOfClusters = 50;

	//第一层
	meanShiftXY[0] = 5.0;
	meanShiftLuv[0] = 3.0;
	meanShiftMinRegion[0] = 20;
	meanShiftFolder[0] = "/home/yangfei/Datasets/Result/KITTI/um/MeanShift/50x30/";
	//第一层
	meanShiftXY[1] = 5.0;
	meanShiftLuv[1] = 5.0;
	meanShiftMinRegion[1] = 20;
	meanShiftFolder[1] = "/home/yangfei/Datasets/Result/KITTI/um/MeanShift/50x50/";
	//第二层
	meanShiftXY[2] = 7.0;
	meanShiftLuv[2] = 5.0;
	meanShiftMinRegion[2] = 20;
	meanShiftFolder[2] = "/home/yangfei/Datasets/Result/KITTI/um/MeanShift/70x50/";
	//第二层
	meanShiftXY[3] = 7.0;
	meanShiftLuv[3] = 6.0;
	meanShiftMinRegion[3] = 20;
	meanShiftFolder[3] = "/home/yangfei/Datasets/Result/KITTI/um/MeanShift/70x60/";

	meanShiftExtension = ".msh";


	//Boosting参数
	denseNumRoundsBoosting = 5000;
	denseBoostingSubSample = 10;
	denseNumberOfThetas = 20;
	denseThetaStart = 3;
	denseThetaIncrement = 2;
	denseNumberOfRectangles = 100;
	denseMinimumRectangleSize = 5;
	denseMaximumRectangleSize = 200;
	denseRandomizationFactor = 0.003;
	denseBoostTrainFile = "denseboost.dat";
	denseExtension = ".dns";
	denseFolder = "/home/yangfei/Datasets/Result/KITTI/um/Dense/";

	denseWeight = 1.0;		//1.0->2.0
							//最大类别概率
	denseMaxClassRatio = 0.2;

	//平滑项三个通道的权重
	pairwiseLWeight = 1.0 / 3.0;
	pairwiseUWeight = 1.0 / 3.0;
	pairwiseVWeight = 1.0 / 3.0;
	pairwisePrior = 1.5;		//1.5->0
	pairwiseFactor = 6.0;		//6.0->20
	pairwiseBeta = 16.0;

	//主导标签比例
	cliqueMinLabelRatio = 0.5;
	//阈值
	cliqueThresholdRatio = 0.1;
	//截断参数
	cliqueTruncation = 0.1;		//0.1->0.5

								//Boosting参数
	statsThetaStart = 2;
	statsThetaIncrement = 1;
	statsNumberOfThetas = 15;
	statsNumberOfBoosts = 5000;
	statsRandomizationFactor = 0.1;
	statsFactor = 0.6;		//0.6->1.0
	statsAlpha = 0.05;
	statsPrior = 0.0;
	statsMaxClassRatio = 0.5;		//0.5->0.7
	statsTrainFile = "statsboost.dat";
	statsFolder = "/home/yangfei/Datasets/Result/KITTI/um/Stats/";
	statsExtension = ".sts";

	//一致性先验项
	consistencyPrior = 0.05;
	//高阶平滑项参数
	pairwiseSegmentBuckets = 8;
	pairwiseSegmentPrior = 0.0;		//0.0—>1.5
	pairwiseSegmentFactor = 2.0;
	pairwiseSegmentBeta = 40.0;

	//3D雷达
	//测试结果保存路径
	lidarTestFolder = "/home/yangfei/Datasets/Result/KITTI/um/LidarCrf/";
	//深度GT图
	lidarGroundTruthFolder = "/home/yangfei/Datasets/Data/KITTI/um/GroundTruth/";		//雷达的标签，跟图像一样
	lidarGroundTruthExtension = ".png";
	lidarUnaryFactor = 5.0;//0.7;  //0.1->0.5

	//视差参数

	lidarPairwiseFactor = 1.0;//0.0005;		//0.00005->0.0005
	//平滑截断概率
	lidarPairwiseTruncation = 10.0;
	//视差类别,很大
	lidarClassNo = 2;


	//模型参数初始化
	crossUnaryWeight = 0.5;
	crossPairwiseWeight = 0.2;//-1e-4;
	crossThreshold = 1e-6;		
	//这个是要训练的
	crossTrainFile = "height.dat";

	//初始化，打乱顺序，按比例生成训练集和测试集
	Init();

	//创建目录
	int i;
	ForceDirectory(trainFolder);

	ForceDirectory(testFolder, "Train/");
	ForceDirectory(lidarTestFolder, "Train/");
	ForceDirectory(textonFolder, "Train/");
	ForceDirectory(siftFolder, "Train/");
	ForceDirectory(colourSiftFolder, "Train/");
	ForceDirectory(locationFolder, "Train/");
	ForceDirectory(lbpFolder, "Train/");
	for (i = 0; i < 4; i++)	ForceDirectory(meanShiftFolder[i], "Train/");
	ForceDirectory(denseFolder, "Train/");
	ForceDirectory(statsFolder, "Train/");

	ForceDirectory(testFolder, "Test/");
	ForceDirectory(lidarTestFolder, "Test/");
	ForceDirectory(textonFolder, "Test/");
	ForceDirectory(siftFolder, "Test/");
	ForceDirectory(colourSiftFolder, "Test/");
	ForceDirectory(locationFolder, "Test/");
	ForceDirectory(lbpFolder, "Test/");
	for (i = 0; i < 4; i++)	ForceDirectory(meanShiftFolder[i], "Test/");
	ForceDirectory(denseFolder, "Test/");
	ForceDirectory(statsFolder, "Test/");

	ForceDirectory("/home/yangfei/Datasets/Result/KITTI/um/Lidar/denseClassifierOut/Train/");
	ForceDirectory("/home/yangfei/Datasets/Result/KITTI/um/Lidar/statClassifierOut/Train/");
	ForceDirectory("/home/yangfei/Datasets/Result/KITTI/um/Lidar/denseFeatureEval/Train/");
	ForceDirectory("/home/yangfei/Datasets/Result/KITTI/um/Lidar/statFeatureEval/Train/");

	ForceDirectory("/home/yangfei/Datasets/Result/KITTI/um/Lidar/denseClassifierOut/Test/");
	ForceDirectory("/home/yangfei/Datasets/Result/KITTI/um/Lidar/statClassifierOut/Test/");
	ForceDirectory("/home/yangfei/Datasets/Result/KITTI/um/Lidar/denseFeatureEval/Test/");
	ForceDirectory("/home/yangfei/Datasets/Result/KITTI/um/Lidar/statFeatureEval/Test/");

}
void LKITTIumDataset::AddFolder(char *folder, LList<char *> &fileList)
{
	char *fileName, *folderExt;

#ifdef _WIN32	
	_finddata_t info;
	int hnd;
	int done;

	folderExt = new char[strlen(imageFolder) + strlen(folder) + strlen(imageExtension) + 2];
	sprintf(folderExt, "%s%s*%s", imageFolder, folder, imageExtension);

	hnd = (int)_findfirst(folderExt, &info);
	done = (hnd == -1);

	while (!done)
	{
		info.name[strlen(info.name) - strlen(imageExtension)] = 0;
		fileName = new char[strlen(folder) + strlen(info.name) + 1];
		sprintf(fileName, "%s%s", folder, info.name);
		fileList.Add(fileName);
		allImageFiles.Add(fileName);	//同时添加到所有文件中
		done = _findnext(hnd, &info);
	}
	_findclose(hnd);
#else
	char *wholeFolder;
	struct dirent **nameList = NULL;
	int count;

	folderExt = new char[strlen(imageExtension) + 2];
	sprintf(folderExt, "*%s", imageExtension);

	wholeFolder = new char[strlen(imageFolder) + strlen(folder) + 1];
	sprintf(wholeFolder, "%s%s", imageFolder, folder);

	count = scandir(wholeFolder, &nameList, NULL, alphasort);
	if (count >= 0)
	{
		for (int i = 0; i < count; i++)
		{
			if (!fnmatch(folderExt, nameList[i]->d_name, 0))
			{
				nameList[i]->d_name[strlen(nameList[i]->d_name) - strlen(imageExtension)] = 0;
				fileName = new char[strlen(folder) + strlen(nameList[i]->d_name) + 1];
				sprintf(fileName, "%s%s", folder, nameList[i]->d_name);
				fileList.Add(fileName);
				allImageFiles.Add(fileName);
			}
			if (nameList[i] != NULL) free(nameList[i]);
		}
		if (nameList != NULL) free(nameList);
	}
	delete[] wholeFolder;
#endif
	delete[] folderExt;
}
void LKITTIumDataset::Init()
{
	//训练集和验证集都当作训练集
	AddFolder("Train/", trainImageFiles);
	AddFolder("Test/", testImageFiles);
}
void LKITTIumDataset::RgbToLabel(unsigned char *rgb, unsigned char *label)
{
	if (rgb[2] == 255)		//r通道代表有效点
	{
		if (rgb[0] == 255)	//b通道代表道路样本
		{
			label[0] = 1;		//道路
		}
		else
			label[0] = 2;		//非路
	}
	else
		label[0] = 0;		//无效

}
void LKITTIumDataset::LabelToRgb(unsigned char *label, unsigned char *rgb)
{
	rgb[0] = rgb[1] = rgb[2] = 0;
	if (label[0] == 2)
	{
		rgb[2] = 255;		//非路红色
	}
	else
	{
		if (label[0] == 1)
		{
			rgb[2] = 255, rgb[0] = 255;		//道路粉色
		}
	}
}
void LKITTIumDataset::SetCRFStructure(LCrf *crf)
{
	//创建CRF主体对象
	LCrfDomain *objDomain = new LCrfDomain(crf, this, classNo, testFolder, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::RgbToLabel, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::LabelToRgb);
	//添加主体到模型
	crf->domains.Add(objDomain);

	//创建底层
	LBaseCrfLayer *baseLayer = new LBaseCrfLayer(crf, objDomain, this, 0);
	//添加底层到模型
	crf->layers.Add(baseLayer);

	//创建高层和分割
	LPnCrfLayer *superpixelLayer[3];
	LSegmentation2D *segmentation[3];


	//baselayer
	segmentation[0] = new LMeanShiftSegmentation2D(meanShiftXY[1], meanShiftLuv[1], meanShiftMinRegion[1], meanShiftFolder[1], meanShiftExtension);
	//建立在底层上的分割，设置父层为base层以及超像素层
	superpixelLayer[0] = new LPnCrfLayer(crf, objDomain, this, baseLayer, segmentation[0], cliqueTruncation);
	crf->segmentations.Add(segmentation[0]);		//添加分割到crf模型
	crf->layers.Add(superpixelLayer[0]);		//添加高阶层到crf模型


	//添加特征对象
	LTextonFeature *textonFeature = new LTextonFeature(this, trainFolder, textonClusteringTrainFile, textonFolder, textonExtension, textonFilterBankRescale, textonKMeansSubSample, textonNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster);
	LLocationFeature *locationFeature = new LLocationFeature(this, locationFolder, locationExtension, locationBuckets);
	LSiftFeature *siftFeature = new LSiftFeature(this, trainFolder, siftClusteringTrainFile, siftFolder, siftExtension, siftSizeCount, siftSizes, siftWindowNumber, sift360, siftAngles, siftKMeansSubSample, siftNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster, 1);
	LColourSiftFeature *coloursiftFeature = new LColourSiftFeature(this, trainFolder, colourSiftClusteringTrainFile, colourSiftFolder, colourSiftExtension, colourSiftSizeCount, colourSiftSizes, colourSiftWindowNumber, colourSift360, colourSiftAngles, colourSiftKMeansSubSample, colourSiftNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster, 1);
	LLbpFeature *lbpFeature = new LLbpFeature(this, trainFolder, lbpClusteringFile, lbpFolder, lbpExtension, lbpSize, lbpKMeansSubSample, lbpNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster);

	crf->features.Add(textonFeature);
	crf->features.Add(locationFeature);
	crf->features.Add(lbpFeature);
	crf->features.Add(siftFeature);
	crf->features.Add(coloursiftFeature);

	//一阶数据项
	LDenseUnaryPixelPotential *pixelPotential = new LDenseUnaryPixelPotential(this, crf, objDomain, baseLayer, trainFolder, denseBoostTrainFile, denseFolder, denseExtension, classNo, denseWeight, denseBoostingSubSample, denseNumberOfRectangles, denseMinimumRectangleSize, denseMaximumRectangleSize, denseMaxClassRatio);
	pixelPotential->AddFeature(textonFeature);
	pixelPotential->AddFeature(siftFeature);
	pixelPotential->AddFeature(coloursiftFeature);
	pixelPotential->AddFeature(lbpFeature);
	crf->potentials.Add(pixelPotential);

	LBoosting<int> *pixelBoosting = new LBoosting<int>(trainFolder, denseBoostTrainFile, classNo, denseNumRoundsBoosting, denseThetaStart, denseThetaIncrement, denseNumberOfThetas, denseRandomizationFactor, pixelPotential, (int *(LPotential::*)(int, int))&LDenseUnaryPixelPotential::GetTrainBoostingValues, (int *(LPotential::*)(int))&LDenseUnaryPixelPotential::GetEvalBoostingValues);
	crf->learnings.Add(pixelBoosting);
	pixelPotential->learning = pixelBoosting;


	//高阶数据项
	LStatsUnarySegmentPotential *statsPotential = new LStatsUnarySegmentPotential(this, crf, objDomain, trainFolder, statsTrainFile, statsFolder, statsExtension, classNo, statsPrior, statsFactor, cliqueMinLabelRatio, statsAlpha, statsMaxClassRatio);
	statsPotential->AddFeature(textonFeature);
	statsPotential->AddFeature(siftFeature);
	statsPotential->AddFeature(coloursiftFeature);
	statsPotential->AddFeature(locationFeature);
	statsPotential->AddFeature(lbpFeature);
	statsPotential->AddLayer(superpixelLayer[0]);

	//segment boosting
	LBoosting<double> *segmentBoosting = new LBoosting<double>(trainFolder, statsTrainFile, classNo, statsNumberOfBoosts, statsThetaStart, statsThetaIncrement, statsNumberOfThetas, statsRandomizationFactor, statsPotential, (double *(LPotential::*)(int, int))&LStatsUnarySegmentPotential::GetTrainBoostingValues, (double *(LPotential::*)(int))&LStatsUnarySegmentPotential::GetEvalBoostingValues);
	crf->learnings.Add(segmentBoosting);
	statsPotential->learning = segmentBoosting;
	crf->potentials.Add(statsPotential);


	//视差改雷达
	LCrfDomain *lidarDomain = new LCrfDomain(crf, this, classNo, lidarTestFolder, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::RgbToLabel, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::LabelToRgb);
	crf->domains.Add(lidarDomain);
	//视差底层
	LBaseCrfLayer *lidarBaseLayer = new LBaseCrfLayer(crf, lidarDomain, this, 0);
	crf->layers.Add(lidarBaseLayer);

	//视差数据项(已解决)
	LLidarUnaryPixelPotential *pixelLidarPotential = new LLidarUnaryPixelPotential(this, crf, lidarDomain, lidarBaseLayer, lidarClassNo, lidarUnaryFactor);
	crf->potentials.Add(pixelLidarPotential);

	//连接两个模型的基于目标高度分布的连接项
	LHeightUnaryPixelPotential *heightPotential = new LHeightUnaryPixelPotential(this, crf, objDomain, baseLayer, lidarDomain, lidarBaseLayer, trainFolder, crossTrainFile, classNo, crossUnaryWeight, lidarClassNo, crossThreshold);
	crf->potentials.Add(heightPotential);

	//联合平滑项
	crf->potentials.Add(new LJointPairwisePixelPotential(this, crf, objDomain, baseLayer, lidarDomain, lidarBaseLayer, classNo, lidarClassNo, pairwisePrior, pairwiseFactor, pairwiseBeta, pairwiseLWeight, pairwiseUWeight, pairwiseVWeight, lidarPairwiseFactor, lidarPairwiseTruncation, crossPairwiseWeight));

}


//KITTI数据集umm
LKITTIummDataset::LKITTIummDataset()
{
	seed = 10000;
	classNo = 2;	//KITTI数据集就道路和非路两类
	filePermutations = 10000;
	optimizeAverage = 1;
	featuresOnline = 0;

	//类别权重，初始化为全1
	unaryWeighted = 0;
	unaryWeights = new double[classNo];
	for (int i = 0; i < classNo; i++) unaryWeights[i] = 1.0;

	//训练集和测试集比例1:1
	proportionTrain = 0.5;
	proportionTest = 0.5;

	datasetName = "umm";
	imageFolder = "/home/yangfei/Datasets/Data/KITTI/umm/Images/";
	imageExtension = ".png";
	groundTruthFolder = "/home/yangfei/Datasets/Data/KITTI/umm/GroundTruth/";
	groundTruthExtension = ".png";
	lidarFolder = "/home/yangfei/Datasets/Data/KITTI/umm/Lidar/";
	lidarExtension = ".bin";
	calibFolder = "/home/yangfei/Datasets/Data/KITTI/umm/calib/";
	calibExtension = ".txt";
	trainFolder = "/home/yangfei/Datasets/Result/KITTI/umm/Train/";
	testFolder = "/home/yangfei/Datasets/Result/KITTI/umm/Crf/";

	clusterPointsPerKDTreeCluster = 30;
	clusterKMeansMaxChange = 0.01;

	textonNumberOfClusters = 50;		//50个聚类中心
	textonFilterBankRescale = 0.7;
	textonKMeansSubSample = 20;
	textonClusteringTrainFile = "textonclustering.dat";
	textonFolder = "/home/yangfei/Datasets/Result/KITTI/umm/Feature/Texton/";
	textonExtension = ".txn";

	siftClusteringTrainFile = "siftclustering.dat";
	siftKMeansSubSample = 40;
	siftNumberOfClusters = 50;
	siftSizes[0] = 5, siftSizes[1] = 7, siftSizes[2] = 9;
	siftSizeCount = 3;
	siftWindowNumber = 3;
	sift360 = 1;	//360度
	siftAngles = 8;		//HOG有8个bin代表360度的8个方向
	siftFolder = "/home/yangfei/Datasets/Result/KITTI/umm/Feature/Sift/";
	siftExtension = ".sft";

	//加了颜色信息的HOG
	colourSiftClusteringTrainFile = "coloursiftclustering.dat";
	colourSiftKMeansSubSample = 40;
	colourSiftNumberOfClusters = 50;
	colourSiftSizes[0] = 5, colourSiftSizes[1] = 7, colourSiftSizes[2] = 9;
	colourSiftSizeCount = 3;
	colourSiftWindowNumber = 3;
	colourSift360 = 1;
	colourSiftAngles = 8;
	colourSiftFolder = "/home/yangfei/Datasets/Result/KITTI/umm/Feature/ColourSift/";
	colourSiftExtension = ".csf";

	locationBuckets = 20;	//位置数，归一化
	locationFolder = "/home/yangfei/Datasets/Result/KITTI/umm/Feature/Location/";
	locationExtension = ".loc";

	lbpClusteringFile = "lbpclustering.dat";
	lbpFolder = "/home/yangfei/Datasets/Result/KITTI/umm/Feature/Lbp/";
	lbpExtension = ".lbp";
	lbpSize = 11;
	lbpKMeansSubSample = 20;
	lbpNumberOfClusters = 50;

	//第一层
	meanShiftXY[0] = 5.0;
	meanShiftLuv[0] = 3.0;
	meanShiftMinRegion[0] = 20;
	meanShiftFolder[0] = "/home/yangfei/Datasets/Result/KITTI/umm/MeanShift/50x30/";
	//第一层
	meanShiftXY[1] = 5.0;
	meanShiftLuv[1] = 5.0;
	meanShiftMinRegion[1] = 20;
	meanShiftFolder[1] = "/home/yangfei/Datasets/Result/KITTI/umm/MeanShift/50x50/";
	//第二层
	meanShiftXY[2] = 7.0;
	meanShiftLuv[2] = 5.0;
	meanShiftMinRegion[2] = 20;
	meanShiftFolder[2] = "/home/yangfei/Datasets/Result/KITTI/umm/MeanShift/70x50/";
	//第二层
	meanShiftXY[3] = 7.0;
	meanShiftLuv[3] = 6.0;
	meanShiftMinRegion[3] = 20;
	meanShiftFolder[3] = "/home/yangfei/Datasets/Result/KITTI/umm/MeanShift/70x60/";

	meanShiftExtension = ".msh";


	//Boosting参数
	denseNumRoundsBoosting = 5000;
	denseBoostingSubSample = 10;
	denseNumberOfThetas = 20;
	denseThetaStart = 3;
	denseThetaIncrement = 2;
	denseNumberOfRectangles = 100;
	denseMinimumRectangleSize = 5;
	denseMaximumRectangleSize = 200;
	denseRandomizationFactor = 0.003;
	denseBoostTrainFile = "denseboost.dat";
	denseExtension = ".dns";
	denseFolder = "/home/yangfei/Datasets/Result/KITTI/umm/Dense/";

	denseWeight = 1.0;		//1.0->2.0
							//最大类别概率
	denseMaxClassRatio = 0.2;

	//平滑项三个通道的权重
	pairwiseLWeight = 1.0 / 3.0;
	pairwiseUWeight = 1.0 / 3.0;
	pairwiseVWeight = 1.0 / 3.0;
	pairwisePrior = 1.5;		//1.5->0
	pairwiseFactor = 6.0;		//6.0->20
	pairwiseBeta = 16.0;

	//主导标签比例
	cliqueMinLabelRatio = 0.5;
	//阈值
	cliqueThresholdRatio = 0.1;
	//截断参数
	cliqueTruncation = 0.1;		//0.1->0.5

								//Boosting参数
	statsThetaStart = 2;
	statsThetaIncrement = 1;
	statsNumberOfThetas = 15;
	statsNumberOfBoosts = 5000;
	statsRandomizationFactor = 0.1;
	statsFactor = 0.6;		//0.6->1.0
	statsAlpha = 0.05;
	statsPrior = 0.0;
	statsMaxClassRatio = 0.5;		//0.5->0.7
	statsTrainFile = "statsboost.dat";
	statsFolder = "/home/yangfei/Datasets/Result/KITTI/umm/Stats/";
	statsExtension = ".sts";

	//一致性先验项
	consistencyPrior = 0.05;
	//高阶平滑项参数
	pairwiseSegmentBuckets = 8;
	pairwiseSegmentPrior = 0.0;		//0.0—>1.5
	pairwiseSegmentFactor = 2.0;
	pairwiseSegmentBeta = 40.0;

	//3D雷达
	//测试结果保存路径
	lidarTestFolder = "/home/yangfei/Datasets/Result/KITTI/umm/LidarCrf/";
	//深度GT图
	lidarGroundTruthFolder = "/home/yangfei/Datasets/Data/KITTI/umm/GroundTruth/";		//雷达的标签，跟图像一样
	lidarGroundTruthExtension = ".png";
	lidarUnaryFactor = 5.0;//0.7;  //0.1->0.5

							 //视差参数

	lidarPairwiseFactor = 1.0;//0.0005;		//0.00005->0.0005
										//平滑截断概率
	lidarPairwiseTruncation = 10.0;
	//视差类别,很大
	lidarClassNo = 2;


	//模型参数初始化
	crossUnaryWeight = 0.5;
	crossPairwiseWeight = 0.2;//-1e-4;
	crossThreshold = 1e-6;
	//这个是要训练的
	crossTrainFile = "height.dat";

	//初始化，打乱顺序，按比例生成训练集和测试集
	Init();

	//创建目录
	int i;
	ForceDirectory(trainFolder);

	ForceDirectory(testFolder, "Train/");
	ForceDirectory(lidarTestFolder, "Train/");
	ForceDirectory(textonFolder, "Train/");
	ForceDirectory(siftFolder, "Train/");
	ForceDirectory(colourSiftFolder, "Train/");
	ForceDirectory(locationFolder, "Train/");
	ForceDirectory(lbpFolder, "Train/");
	for (i = 0; i < 4; i++)	ForceDirectory(meanShiftFolder[i], "Train/");
	ForceDirectory(denseFolder, "Train/");
	ForceDirectory(statsFolder, "Train/");

	ForceDirectory(testFolder, "Test/");
	ForceDirectory(lidarTestFolder, "Test/");
	ForceDirectory(textonFolder, "Test/");
	ForceDirectory(siftFolder, "Test/");
	ForceDirectory(colourSiftFolder, "Test/");
	ForceDirectory(locationFolder, "Test/");
	ForceDirectory(lbpFolder, "Test/");
	for (i = 0; i < 4; i++)	ForceDirectory(meanShiftFolder[i], "Test/");
	ForceDirectory(denseFolder, "Test/");
	ForceDirectory(statsFolder, "Test/");

	ForceDirectory("/home/yangfei/Datasets/Result/KITTI/umm/Lidar/denseClassifierOut/Train/");
	ForceDirectory("/home/yangfei/Datasets/Result/KITTI/umm/Lidar/statClassifierOut/Train/");
	ForceDirectory("/home/yangfei/Datasets/Result/KITTI/umm/Lidar/denseFeatureEval/Train/");
	ForceDirectory("/home/yangfei/Datasets/Result/KITTI/umm/Lidar/statFeatureEval/Train/");

	ForceDirectory("/home/yangfei/Datasets/Result/KITTI/umm/Lidar/denseClassifierOut/Test/");
	ForceDirectory("/home/yangfei/Datasets/Result/KITTI/umm/Lidar/statClassifierOut/Test/");
	ForceDirectory("/home/yangfei/Datasets/Result/KITTI/umm/Lidar/denseFeatureEval/Test/");
	ForceDirectory("/home/yangfei/Datasets/Result/KITTI/umm/Lidar/statFeatureEval/Test/");

}
void LKITTIummDataset::AddFolder(char *folder, LList<char *> &fileList)
{
	char *fileName, *folderExt;

#ifdef _WIN32	
	_finddata_t info;
	int hnd;
	int done;

	folderExt = new char[strlen(imageFolder) + strlen(folder) + strlen(imageExtension) + 2];
	sprintf(folderExt, "%s%s*%s", imageFolder, folder, imageExtension);

	hnd = (int)_findfirst(folderExt, &info);
	done = (hnd == -1);

	while (!done)
	{
		info.name[strlen(info.name) - strlen(imageExtension)] = 0;
		fileName = new char[strlen(folder) + strlen(info.name) + 1];
		sprintf(fileName, "%s%s", folder, info.name);
		fileList.Add(fileName);
		allImageFiles.Add(fileName);	//同时添加到所有文件中
		done = _findnext(hnd, &info);
	}
	_findclose(hnd);
#else
	char *wholeFolder;
	struct dirent **nameList = NULL;
	int count;

	folderExt = new char[strlen(imageExtension) + 2];
	sprintf(folderExt, "*%s", imageExtension);

	wholeFolder = new char[strlen(imageFolder) + strlen(folder) + 1];
	sprintf(wholeFolder, "%s%s", imageFolder, folder);

	count = scandir(wholeFolder, &nameList, NULL, alphasort);
	if (count >= 0)
	{
		for (int i = 0; i < count; i++)
		{
			if (!fnmatch(folderExt, nameList[i]->d_name, 0))
			{
				nameList[i]->d_name[strlen(nameList[i]->d_name) - strlen(imageExtension)] = 0;
				fileName = new char[strlen(folder) + strlen(nameList[i]->d_name) + 1];
				sprintf(fileName, "%s%s", folder, nameList[i]->d_name);
				fileList.Add(fileName);
				allImageFiles.Add(fileName);
			}
			if (nameList[i] != NULL) free(nameList[i]);
		}
		if (nameList != NULL) free(nameList);
	}
	delete[] wholeFolder;
#endif
	delete[] folderExt;
}
void LKITTIummDataset::Init()
{
	//训练集和验证集都当作训练集
	AddFolder("Train/", trainImageFiles);
	AddFolder("Test/", testImageFiles);
}
void LKITTIummDataset::RgbToLabel(unsigned char *rgb, unsigned char *label)
{
	if (rgb[2] == 255)		//r通道代表有效点
	{
		if (rgb[0] == 255)	//b通道代表道路样本
		{
			label[0] = 1;		//道路
		}
		else
			label[0] = 2;		//非路
	}
	else
		label[0] = 0;		//无效

}
void LKITTIummDataset::LabelToRgb(unsigned char *label, unsigned char *rgb)
{
	rgb[0] = rgb[1] = rgb[2] = 0;
	if (label[0] == 2)
	{
		rgb[2] = 255;		//非路红色
	}
	else
	{
		if (label[0] == 1)
		{
			rgb[2] = 255, rgb[0] = 255;		//道路粉色
		}
	}
}
void LKITTIummDataset::SetCRFStructure(LCrf *crf)
{
	//创建CRF主体对象
	LCrfDomain *objDomain = new LCrfDomain(crf, this, classNo, testFolder, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::RgbToLabel, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::LabelToRgb);
	//添加主体到模型
	crf->domains.Add(objDomain);

	//创建底层
	LBaseCrfLayer *baseLayer = new LBaseCrfLayer(crf, objDomain, this, 0);
	//添加底层到模型
	crf->layers.Add(baseLayer);

	//创建高层和分割
	LPnCrfLayer *superpixelLayer[3];
	LSegmentation2D *segmentation[3];


	//baselayer
	segmentation[0] = new LMeanShiftSegmentation2D(meanShiftXY[1], meanShiftLuv[1], meanShiftMinRegion[1], meanShiftFolder[1], meanShiftExtension);
	//建立在底层上的分割，设置父层为base层以及超像素层
	superpixelLayer[0] = new LPnCrfLayer(crf, objDomain, this, baseLayer, segmentation[0], cliqueTruncation);
	crf->segmentations.Add(segmentation[0]);		//添加分割到crf模型
	crf->layers.Add(superpixelLayer[0]);		//添加高阶层到crf模型


												//添加特征对象
	LTextonFeature *textonFeature = new LTextonFeature(this, trainFolder, textonClusteringTrainFile, textonFolder, textonExtension, textonFilterBankRescale, textonKMeansSubSample, textonNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster);
	LLocationFeature *locationFeature = new LLocationFeature(this, locationFolder, locationExtension, locationBuckets);
	LSiftFeature *siftFeature = new LSiftFeature(this, trainFolder, siftClusteringTrainFile, siftFolder, siftExtension, siftSizeCount, siftSizes, siftWindowNumber, sift360, siftAngles, siftKMeansSubSample, siftNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster, 1);
	LColourSiftFeature *coloursiftFeature = new LColourSiftFeature(this, trainFolder, colourSiftClusteringTrainFile, colourSiftFolder, colourSiftExtension, colourSiftSizeCount, colourSiftSizes, colourSiftWindowNumber, colourSift360, colourSiftAngles, colourSiftKMeansSubSample, colourSiftNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster, 1);
	LLbpFeature *lbpFeature = new LLbpFeature(this, trainFolder, lbpClusteringFile, lbpFolder, lbpExtension, lbpSize, lbpKMeansSubSample, lbpNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster);

	crf->features.Add(textonFeature);
	crf->features.Add(locationFeature);
	crf->features.Add(lbpFeature);
	crf->features.Add(siftFeature);
	crf->features.Add(coloursiftFeature);

	//一阶数据项
	LDenseUnaryPixelPotential *pixelPotential = new LDenseUnaryPixelPotential(this, crf, objDomain, baseLayer, trainFolder, denseBoostTrainFile, denseFolder, denseExtension, classNo, denseWeight, denseBoostingSubSample, denseNumberOfRectangles, denseMinimumRectangleSize, denseMaximumRectangleSize, denseMaxClassRatio);
	pixelPotential->AddFeature(textonFeature);
	pixelPotential->AddFeature(siftFeature);
	pixelPotential->AddFeature(coloursiftFeature);
	pixelPotential->AddFeature(lbpFeature);
	crf->potentials.Add(pixelPotential);

	LBoosting<int> *pixelBoosting = new LBoosting<int>(trainFolder, denseBoostTrainFile, classNo, denseNumRoundsBoosting, denseThetaStart, denseThetaIncrement, denseNumberOfThetas, denseRandomizationFactor, pixelPotential, (int *(LPotential::*)(int, int))&LDenseUnaryPixelPotential::GetTrainBoostingValues, (int *(LPotential::*)(int))&LDenseUnaryPixelPotential::GetEvalBoostingValues);
	crf->learnings.Add(pixelBoosting);
	pixelPotential->learning = pixelBoosting;


	//高阶数据项
	LStatsUnarySegmentPotential *statsPotential = new LStatsUnarySegmentPotential(this, crf, objDomain, trainFolder, statsTrainFile, statsFolder, statsExtension, classNo, statsPrior, statsFactor, cliqueMinLabelRatio, statsAlpha, statsMaxClassRatio);
	statsPotential->AddFeature(textonFeature);
	statsPotential->AddFeature(siftFeature);
	statsPotential->AddFeature(coloursiftFeature);
	statsPotential->AddFeature(locationFeature);
	statsPotential->AddFeature(lbpFeature);
	statsPotential->AddLayer(superpixelLayer[0]);

	//segment boosting
	LBoosting<double> *segmentBoosting = new LBoosting<double>(trainFolder, statsTrainFile, classNo, statsNumberOfBoosts, statsThetaStart, statsThetaIncrement, statsNumberOfThetas, statsRandomizationFactor, statsPotential, (double *(LPotential::*)(int, int))&LStatsUnarySegmentPotential::GetTrainBoostingValues, (double *(LPotential::*)(int))&LStatsUnarySegmentPotential::GetEvalBoostingValues);
	crf->learnings.Add(segmentBoosting);
	statsPotential->learning = segmentBoosting;
	crf->potentials.Add(statsPotential);


	//视差改雷达
	LCrfDomain *lidarDomain = new LCrfDomain(crf, this, classNo, lidarTestFolder, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::RgbToLabel, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::LabelToRgb);
	crf->domains.Add(lidarDomain);
	//视差底层
	LBaseCrfLayer *lidarBaseLayer = new LBaseCrfLayer(crf, lidarDomain, this, 0);
	crf->layers.Add(lidarBaseLayer);

	//视差数据项(已解决)
	LLidarUnaryPixelPotential *pixelLidarPotential = new LLidarUnaryPixelPotential(this, crf, lidarDomain, lidarBaseLayer, lidarClassNo, lidarUnaryFactor);
	crf->potentials.Add(pixelLidarPotential);

	//连接两个模型的基于目标高度分布的连接项
	LHeightUnaryPixelPotential *heightPotential = new LHeightUnaryPixelPotential(this, crf, objDomain, baseLayer, lidarDomain, lidarBaseLayer, trainFolder, crossTrainFile, classNo, crossUnaryWeight, lidarClassNo, crossThreshold);
	crf->potentials.Add(heightPotential);

	//联合平滑项
	crf->potentials.Add(new LJointPairwisePixelPotential(this, crf, objDomain, baseLayer, lidarDomain, lidarBaseLayer, classNo, lidarClassNo, pairwisePrior, pairwiseFactor, pairwiseBeta, pairwiseLWeight, pairwiseUWeight, pairwiseVWeight, lidarPairwiseFactor, lidarPairwiseTruncation, crossPairwiseWeight));

}


//KITTI数据集uu
LKITTIuuDataset::LKITTIuuDataset()
{
	seed = 10000;
	classNo = 2;	//KITTI数据集就道路和非路两类
	filePermutations = 10000;
	optimizeAverage = 1;
	featuresOnline = 0;

	//类别权重，初始化为全1
	unaryWeighted = 0;
	unaryWeights = new double[classNo];
	for (int i = 0; i < classNo; i++) unaryWeights[i] = 1.0;

	//训练集和测试集比例1:1
	proportionTrain = 0.5;
	proportionTest = 0.5;

	datasetName = "uu";
	imageFolder = "/home/yangfei/Datasets/Data/KITTI/uu/Images/";
	imageExtension = ".png";
	groundTruthFolder = "/home/yangfei/Datasets/Data/KITTI/uu/GroundTruth/";
	groundTruthExtension = ".png";
	lidarFolder = "/home/yangfei/Datasets/Data/KITTI/uu/Lidar/";
	lidarExtension = ".bin";
	calibFolder = "/home/yangfei/Datasets/Data/KITTI/uu/calib/";
	calibExtension = ".txt";
	trainFolder = "/home/yangfei/Datasets/Result/KITTI/uu/Train/";
	testFolder = "/home/yangfei/Datasets/Result/KITTI/uu/Crf/";

	clusterPointsPerKDTreeCluster = 30;
	clusterKMeansMaxChange = 0.01;

	textonNumberOfClusters = 50;		//50个聚类中心
	textonFilterBankRescale = 0.7;
	textonKMeansSubSample = 20;
	textonClusteringTrainFile = "textonclustering.dat";
	textonFolder = "/home/yangfei/Datasets/Result/KITTI/uu/Feature/Texton/";
	textonExtension = ".txn";

	siftClusteringTrainFile = "siftclustering.dat";
	siftKMeansSubSample = 40;
	siftNumberOfClusters = 50;
	siftSizes[0] = 5, siftSizes[1] = 7, siftSizes[2] = 9;
	siftSizeCount = 3;
	siftWindowNumber = 3;
	sift360 = 1;	//360度
	siftAngles = 8;		//HOG有8个bin代表360度的8个方向
	siftFolder = "/home/yangfei/Datasets/Result/KITTI/uu/Feature/Sift/";
	siftExtension = ".sft";

	//加了颜色信息的HOG
	colourSiftClusteringTrainFile = "coloursiftclustering.dat";
	colourSiftKMeansSubSample = 40;
	colourSiftNumberOfClusters = 50;
	colourSiftSizes[0] = 5, colourSiftSizes[1] = 7, colourSiftSizes[2] = 9;
	colourSiftSizeCount = 3;
	colourSiftWindowNumber = 3;
	colourSift360 = 1;
	colourSiftAngles = 8;
	colourSiftFolder = "/home/yangfei/Datasets/Result/KITTI/uu/Feature/ColourSift/";
	colourSiftExtension = ".csf";

	locationBuckets = 20;	//位置数，归一化
	locationFolder = "/home/yangfei/Datasets/Result/KITTI/uu/Feature/Location/";
	locationExtension = ".loc";

	lbpClusteringFile = "lbpclustering.dat";
	lbpFolder = "/home/yangfei/Datasets/Result/KITTI/uu/Feature/Lbp/";
	lbpExtension = ".lbp";
	lbpSize = 11;
	lbpKMeansSubSample = 20;
	lbpNumberOfClusters = 50;

	//第一层
	meanShiftXY[0] = 5.0;
	meanShiftLuv[0] = 3.0;
	meanShiftMinRegion[0] = 20;
	meanShiftFolder[0] = "/home/yangfei/Datasets/Result/KITTI/uu/MeanShift/50x30/";
	//第一层
	meanShiftXY[1] = 5.0;
	meanShiftLuv[1] = 5.0;
	meanShiftMinRegion[1] = 20;
	meanShiftFolder[1] = "/home/yangfei/Datasets/Result/KITTI/uu/MeanShift/50x50/";
	//第二层
	meanShiftXY[2] = 7.0;
	meanShiftLuv[2] = 5.0;
	meanShiftMinRegion[2] = 20;
	meanShiftFolder[2] = "/home/yangfei/Datasets/Result/KITTI/uu/MeanShift/70x50/";
	//第二层
	meanShiftXY[3] = 7.0;
	meanShiftLuv[3] = 6.0;
	meanShiftMinRegion[3] = 20;
	meanShiftFolder[3] = "/home/yangfei/Datasets/Result/KITTI/uu/MeanShift/70x60/";

	meanShiftExtension = ".msh";


	//Boosting参数
	denseNumRoundsBoosting = 5000;
	denseBoostingSubSample = 10;
	denseNumberOfThetas = 20;
	denseThetaStart = 3;
	denseThetaIncrement = 2;
	denseNumberOfRectangles = 100;
	denseMinimumRectangleSize = 5;
	denseMaximumRectangleSize = 200;
	denseRandomizationFactor = 0.003;
	denseBoostTrainFile = "denseboost.dat";
	denseExtension = ".dns";
	denseFolder = "/home/yangfei/Datasets/Result/KITTI/uu/Dense/";

	denseWeight = 1.0;		//1.0->2.0
							//最大类别概率
	denseMaxClassRatio = 0.2;

	//平滑项三个通道的权重
	pairwiseLWeight = 1.0 / 3.0;
	pairwiseUWeight = 1.0 / 3.0;
	pairwiseVWeight = 1.0 / 3.0;
	pairwisePrior = 1.5;		//1.5->0
	pairwiseFactor = 6.0;		//6.0->20
	pairwiseBeta = 16.0;

	//主导标签比例
	cliqueMinLabelRatio = 0.5;
	//阈值
	cliqueThresholdRatio = 0.1;
	//截断参数
	cliqueTruncation = 0.1;		//0.1->0.5

								//Boosting参数
	statsThetaStart = 2;
	statsThetaIncrement = 1;
	statsNumberOfThetas = 15;
	statsNumberOfBoosts = 5000;
	statsRandomizationFactor = 0.1;
	statsFactor = 0.6;		//0.6->1.0
	statsAlpha = 0.05;
	statsPrior = 0.0;
	statsMaxClassRatio = 0.5;		//0.5->0.7
	statsTrainFile = "statsboost.dat";
	statsFolder = "/home/yangfei/Datasets/Result/KITTI/uu/Stats/";
	statsExtension = ".sts";

	//一致性先验项
	consistencyPrior = 0.05;
	//高阶平滑项参数
	pairwiseSegmentBuckets = 8;
	pairwiseSegmentPrior = 0.0;		//0.0—>1.5
	pairwiseSegmentFactor = 2.0;
	pairwiseSegmentBeta = 40.0;

	//3D雷达
	//测试结果保存路径
	lidarTestFolder = "/home/yangfei/Datasets/Result/KITTI/uu/LidarCrf/";
	//雷达GT图
	lidarGroundTruthFolder = "/home/yangfei/Datasets/Data/KITTI/uu/GroundTruth/";		//雷达的标签，跟图像一样
	lidarGroundTruthExtension = ".png";
	lidarUnaryFactor = 5.0;//0.7;  //0.1->0.5
						

	lidarPairwiseFactor = 1.0;//0.0005;		//0.00005->0.0005
										//平滑截断概率
	lidarPairwiseTruncation = 10.0;
	//视差类别,很大
	lidarClassNo = 2;


	//模型参数初始化
	crossUnaryWeight = 0.5;
	crossPairwiseWeight = 0.2;//-1e-4;
	crossThreshold = 1e-6;
	//这个是要训练的
	crossTrainFile = "height.dat";

	//初始化，打乱顺序，按比例生成训练集和测试集
	Init();

	//创建目录
	int i;
	ForceDirectory(trainFolder);

	ForceDirectory(testFolder, "Train/");
	ForceDirectory(lidarTestFolder, "Train/");
	ForceDirectory(textonFolder, "Train/");
	ForceDirectory(siftFolder, "Train/");
	ForceDirectory(colourSiftFolder, "Train/");
	ForceDirectory(locationFolder, "Train/");
	ForceDirectory(lbpFolder, "Train/");
	for (i = 0; i < 4; i++)	ForceDirectory(meanShiftFolder[i], "Train/");
	ForceDirectory(denseFolder, "Train/");
	ForceDirectory(statsFolder, "Train/");

	ForceDirectory(testFolder, "Test/");
	ForceDirectory(lidarTestFolder, "Test/");
	ForceDirectory(textonFolder, "Test/");
	ForceDirectory(siftFolder, "Test/");
	ForceDirectory(colourSiftFolder, "Test/");
	ForceDirectory(locationFolder, "Test/");
	ForceDirectory(lbpFolder, "Test/");
	for (i = 0; i < 4; i++)	ForceDirectory(meanShiftFolder[i], "Test/");
	ForceDirectory(denseFolder, "Test/");
	ForceDirectory(statsFolder, "Test/");

	ForceDirectory("/home/yangfei/Datasets/Result/KITTI/uu/Lidar/denseClassifierOut/Train/");
	ForceDirectory("/home/yangfei/Datasets/Result/KITTI/uu/Lidar/statClassifierOut/Train/");
	ForceDirectory("/home/yangfei/Datasets/Result/KITTI/uu/Lidar/denseFeatureEval/Train/");
	ForceDirectory("/home/yangfei/Datasets/Result/KITTI/uu/Lidar/statFeatureEval/Train/");

	ForceDirectory("/home/yangfei/Datasets/Result/KITTI/uu/Lidar/denseClassifierOut/Test/");
	ForceDirectory("/home/yangfei/Datasets/Result/KITTI/uu/Lidar/statClassifierOut/Test/");
	ForceDirectory("/home/yangfei/Datasets/Result/KITTI/uu/Lidar/denseFeatureEval/Test/");
	ForceDirectory("/home/yangfei/Datasets/Result/KITTI/uu/Lidar/statFeatureEval/Test/");

}
void LKITTIuuDataset::AddFolder(char *folder, LList<char *> &fileList)
{
	char *fileName, *folderExt;

#ifdef _WIN32	
	_finddata_t info;
	int hnd;
	int done;

	folderExt = new char[strlen(imageFolder) + strlen(folder) + strlen(imageExtension) + 2];
	sprintf(folderExt, "%s%s*%s", imageFolder, folder, imageExtension);

	hnd = (int)_findfirst(folderExt, &info);
	done = (hnd == -1);

	while (!done)
	{
		info.name[strlen(info.name) - strlen(imageExtension)] = 0;
		fileName = new char[strlen(folder) + strlen(info.name) + 1];
		sprintf(fileName, "%s%s", folder, info.name);
		fileList.Add(fileName);
		allImageFiles.Add(fileName);	//同时添加到所有文件中
		done = _findnext(hnd, &info);
	}
	_findclose(hnd);
#else
	char *wholeFolder;
	struct dirent **nameList = NULL;
	int count;

	folderExt = new char[strlen(imageExtension) + 2];
	sprintf(folderExt, "*%s", imageExtension);

	wholeFolder = new char[strlen(imageFolder) + strlen(folder) + 1];
	sprintf(wholeFolder, "%s%s", imageFolder, folder);

	count = scandir(wholeFolder, &nameList, NULL, alphasort);
	if (count >= 0)
	{
		for (int i = 0; i < count; i++)
		{
			if (!fnmatch(folderExt, nameList[i]->d_name, 0))
			{
				nameList[i]->d_name[strlen(nameList[i]->d_name) - strlen(imageExtension)] = 0;
				fileName = new char[strlen(folder) + strlen(nameList[i]->d_name) + 1];
				sprintf(fileName, "%s%s", folder, nameList[i]->d_name);
				fileList.Add(fileName);
				allImageFiles.Add(fileName);
			}
			if (nameList[i] != NULL) free(nameList[i]);
		}
		if (nameList != NULL) free(nameList);
	}
	delete[] wholeFolder;
#endif
	delete[] folderExt;
}
void LKITTIuuDataset::Init()
{
	//训练集和验证集都当作训练集
	AddFolder("Train/", trainImageFiles);
	AddFolder("Test/", testImageFiles);
}
void LKITTIuuDataset::RgbToLabel(unsigned char *rgb, unsigned char *label)
{
	if (rgb[2] == 255)		//r通道代表有效点
	{
		if (rgb[0] == 255)	//b通道代表道路样本
		{
			label[0] = 1;		//道路
		}
		else
			label[0] = 2;		//非路
	}
	else
		label[0] = 0;		//无效

}
void LKITTIuuDataset::LabelToRgb(unsigned char *label, unsigned char *rgb)
{
	rgb[0] = rgb[1] = rgb[2] = 0;
	if (label[0] == 2)
	{
		rgb[2] = 255;		//非路红色
	}
	else
	{
		if (label[0] == 1)
		{
			rgb[2] = 255, rgb[0] = 255;		//道路粉色
		}
	}
}
void LKITTIuuDataset::SetCRFStructure(LCrf *crf)
{
	//创建CRF主体对象
	LCrfDomain *objDomain = new LCrfDomain(crf, this, classNo, testFolder, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::RgbToLabel, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::LabelToRgb);
	//添加主体到模型
	crf->domains.Add(objDomain);

	//创建底层
	LBaseCrfLayer *baseLayer = new LBaseCrfLayer(crf, objDomain, this, 0);
	//添加底层到模型
	crf->layers.Add(baseLayer);

	//创建高层和分割
	LPnCrfLayer *superpixelLayer[3];
	LSegmentation2D *segmentation[3];


	//baselayer
	segmentation[0] = new LMeanShiftSegmentation2D(meanShiftXY[1], meanShiftLuv[1], meanShiftMinRegion[1], meanShiftFolder[1], meanShiftExtension);
	//建立在底层上的分割，设置父层为base层以及超像素层
	superpixelLayer[0] = new LPnCrfLayer(crf, objDomain, this, baseLayer, segmentation[0], cliqueTruncation);
	crf->segmentations.Add(segmentation[0]);		//添加分割到crf模型
	crf->layers.Add(superpixelLayer[0]);		//添加高阶层到crf模型


												//添加特征对象
	LTextonFeature *textonFeature = new LTextonFeature(this, trainFolder, textonClusteringTrainFile, textonFolder, textonExtension, textonFilterBankRescale, textonKMeansSubSample, textonNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster);
	LLocationFeature *locationFeature = new LLocationFeature(this, locationFolder, locationExtension, locationBuckets);
	LSiftFeature *siftFeature = new LSiftFeature(this, trainFolder, siftClusteringTrainFile, siftFolder, siftExtension, siftSizeCount, siftSizes, siftWindowNumber, sift360, siftAngles, siftKMeansSubSample, siftNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster, 1);
	LColourSiftFeature *coloursiftFeature = new LColourSiftFeature(this, trainFolder, colourSiftClusteringTrainFile, colourSiftFolder, colourSiftExtension, colourSiftSizeCount, colourSiftSizes, colourSiftWindowNumber, colourSift360, colourSiftAngles, colourSiftKMeansSubSample, colourSiftNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster, 1);
	LLbpFeature *lbpFeature = new LLbpFeature(this, trainFolder, lbpClusteringFile, lbpFolder, lbpExtension, lbpSize, lbpKMeansSubSample, lbpNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster);

	crf->features.Add(textonFeature);
	crf->features.Add(locationFeature);
	crf->features.Add(lbpFeature);
	crf->features.Add(siftFeature);
	crf->features.Add(coloursiftFeature);

	//一阶数据项
	LDenseUnaryPixelPotential *pixelPotential = new LDenseUnaryPixelPotential(this, crf, objDomain, baseLayer, trainFolder, denseBoostTrainFile, denseFolder, denseExtension, classNo, denseWeight, denseBoostingSubSample, denseNumberOfRectangles, denseMinimumRectangleSize, denseMaximumRectangleSize, denseMaxClassRatio);
	pixelPotential->AddFeature(textonFeature);
	pixelPotential->AddFeature(siftFeature);
	pixelPotential->AddFeature(coloursiftFeature);
	pixelPotential->AddFeature(lbpFeature);
	crf->potentials.Add(pixelPotential);

	LBoosting<int> *pixelBoosting = new LBoosting<int>(trainFolder, denseBoostTrainFile, classNo, denseNumRoundsBoosting, denseThetaStart, denseThetaIncrement, denseNumberOfThetas, denseRandomizationFactor, pixelPotential, (int *(LPotential::*)(int, int))&LDenseUnaryPixelPotential::GetTrainBoostingValues, (int *(LPotential::*)(int))&LDenseUnaryPixelPotential::GetEvalBoostingValues);
	crf->learnings.Add(pixelBoosting);
	pixelPotential->learning = pixelBoosting;


	//高阶数据项
	LStatsUnarySegmentPotential *statsPotential = new LStatsUnarySegmentPotential(this, crf, objDomain, trainFolder, statsTrainFile, statsFolder, statsExtension, classNo, statsPrior, statsFactor, cliqueMinLabelRatio, statsAlpha, statsMaxClassRatio);
	statsPotential->AddFeature(textonFeature);
	statsPotential->AddFeature(siftFeature);
	statsPotential->AddFeature(coloursiftFeature);
	statsPotential->AddFeature(locationFeature);
	statsPotential->AddFeature(lbpFeature);
	statsPotential->AddLayer(superpixelLayer[0]);

	//segment boosting
	LBoosting<double> *segmentBoosting = new LBoosting<double>(trainFolder, statsTrainFile, classNo, statsNumberOfBoosts, statsThetaStart, statsThetaIncrement, statsNumberOfThetas, statsRandomizationFactor, statsPotential, (double *(LPotential::*)(int, int))&LStatsUnarySegmentPotential::GetTrainBoostingValues, (double *(LPotential::*)(int))&LStatsUnarySegmentPotential::GetEvalBoostingValues);
	crf->learnings.Add(segmentBoosting);
	statsPotential->learning = segmentBoosting;
	crf->potentials.Add(statsPotential);


	//视差改雷达
	LCrfDomain *lidarDomain = new LCrfDomain(crf, this, classNo, lidarTestFolder, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::RgbToLabel, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::LabelToRgb);
	crf->domains.Add(lidarDomain);
	//视差底层
	LBaseCrfLayer *lidarBaseLayer = new LBaseCrfLayer(crf, lidarDomain, this, 0);
	crf->layers.Add(lidarBaseLayer);

	//视差数据项(已解决)
	LLidarUnaryPixelPotential *pixelLidarPotential = new LLidarUnaryPixelPotential(this, crf, lidarDomain, lidarBaseLayer, lidarClassNo, lidarUnaryFactor);
	crf->potentials.Add(pixelLidarPotential);

	//连接两个模型的基于目标高度分布的连接项
	LHeightUnaryPixelPotential *heightPotential = new LHeightUnaryPixelPotential(this, crf, objDomain, baseLayer, lidarDomain, lidarBaseLayer, trainFolder, crossTrainFile, classNo, crossUnaryWeight, lidarClassNo, crossThreshold);
	crf->potentials.Add(heightPotential);

	//联合平滑项
	crf->potentials.Add(new LJointPairwisePixelPotential(this, crf, objDomain, baseLayer, lidarDomain, lidarBaseLayer, classNo, lidarClassNo, pairwisePrior, pairwiseFactor, pairwiseBeta, pairwiseLWeight, pairwiseUWeight, pairwiseVWeight, lidarPairwiseFactor, lidarPairwiseTruncation, crossPairwiseWeight));

}


//KITTI验证集
LKITTIValidationset::LKITTIValidationset()
{
	seed = 10000;
	classNo = 2;	//KITTI数据集就道路和非路两类
	filePermutations = 10000;
	optimizeAverage = 1;
	featuresOnline = 0;

	//类别权重，初始化为全1
	unaryWeighted = 0;
	unaryWeights = new double[classNo];
	for (int i = 0; i < classNo; i++) unaryWeights[i] = 1.0;

	//训练集和测试集比例1:1
	proportionTrain = 0.5;
	proportionTest = 0.5;

	datasetName = "val";
	imageFolder = "/home/yangfei/Datasets/Data/KITTI/val/Images/";
	imageExtension = ".png";
	groundTruthFolder = "/home/yangfei/Datasets/Data/KITTI/val/GroundTruth/";
	groundTruthExtension = ".png";
	lidarFolder = "/home/yangfei/Datasets/Data/KITTI/val/Lidar/";
	lidarExtension = ".bin";
	calibFolder = "/home/yangfei/Datasets/Data/KITTI/val/calib/";
	calibExtension = ".txt";
	trainFolder = "/home/yangfei/Datasets/Result/KITTI/val/Train/";
	testFolder = "/home/yangfei/Datasets/Result/KITTI/val/Crf/";

	clusterPointsPerKDTreeCluster = 30;
	clusterKMeansMaxChange = 0.01;

	textonNumberOfClusters = 50;		//50个聚类中心
	textonFilterBankRescale = 0.7;
	textonKMeansSubSample = 20;
	textonClusteringTrainFile = "textonclustering.dat";
	textonFolder = "/home/yangfei/Datasets/Result/KITTI/val/Feature/Texton/";
	textonExtension = ".txn";

	siftClusteringTrainFile = "siftclustering.dat";
	siftKMeansSubSample = 40;
	siftNumberOfClusters = 50;
	siftSizes[0] = 5, siftSizes[1] = 7, siftSizes[2] = 9;
	siftSizeCount = 3;
	siftWindowNumber = 3;
	sift360 = 1;	//360度
	siftAngles = 8;		//HOG有8个bin代表360度的8个方向
	siftFolder = "/home/yangfei/Datasets/Result/KITTI/val/Feature/Sift/";
	siftExtension = ".sft";

	//加了颜色信息的HOG
	colourSiftClusteringTrainFile = "coloursiftclustering.dat";
	colourSiftKMeansSubSample = 40;
	colourSiftNumberOfClusters = 50;
	colourSiftSizes[0] = 5, colourSiftSizes[1] = 7, colourSiftSizes[2] = 9;
	colourSiftSizeCount = 3;
	colourSiftWindowNumber = 3;
	colourSift360 = 1;
	colourSiftAngles = 8;
	colourSiftFolder = "/home/yangfei/Datasets/Result/KITTI/val/Feature/ColourSift/";
	colourSiftExtension = ".csf";

	locationBuckets = 20;	//位置数，归一化
	locationFolder = "/home/yangfei/Datasets/Result/KITTI/val/Feature/Location/";
	locationExtension = ".loc";

	lbpClusteringFile = "lbpclustering.dat";
	lbpFolder = "/home/yangfei/Datasets/Result/KITTI/val/Feature/Lbp/";
	lbpExtension = ".lbp";
	lbpSize = 11;
	lbpKMeansSubSample = 20;
	lbpNumberOfClusters = 50;

	//第一层
	meanShiftXY[0] = 5.0;
	meanShiftLuv[0] = 3.0;
	meanShiftMinRegion[0] = 20;
	meanShiftFolder[0] = "/home/yangfei/Datasets/Result/KITTI/val/MeanShift/50x30/";
	//第一层
	meanShiftXY[1] = 5.0;
	meanShiftLuv[1] = 5.0;
	meanShiftMinRegion[1] = 20;
	meanShiftFolder[1] = "/home/yangfei/Datasets/Result/KITTI/val/MeanShift/50x50/";
	//第二层
	meanShiftXY[2] = 7.0;
	meanShiftLuv[2] = 5.0;
	meanShiftMinRegion[2] = 20;
	meanShiftFolder[2] = "/home/yangfei/Datasets/Result/KITTI/val/MeanShift/70x50/";
	//第二层
	meanShiftXY[3] = 7.0;
	meanShiftLuv[3] = 6.0;
	meanShiftMinRegion[3] = 20;
	meanShiftFolder[3] = "/home/yangfei/Datasets/Result/KITTI/val/MeanShift/70x60/";

	meanShiftExtension = ".msh";


	//Boosting参数
	denseNumRoundsBoosting = 5000;
	denseBoostingSubSample = 10;
	denseNumberOfThetas = 20;
	denseThetaStart = 3;
	denseThetaIncrement = 2;
	denseNumberOfRectangles = 100;
	denseMinimumRectangleSize = 5;
	denseMaximumRectangleSize = 200;
	denseRandomizationFactor = 0.003;
	denseBoostTrainFile = "denseboost.dat";
	denseExtension = ".dns";
	denseFolder = "/home/yangfei/Datasets/Result/KITTI/val/Dense/";

	denseWeight = 1.0;		//1.0->2.0
							//最大类别概率
	denseMaxClassRatio = 0.2;

	//平滑项三个通道的权重
	pairwiseLWeight = 1.0 / 3.0;
	pairwiseUWeight = 1.0 / 3.0;
	pairwiseVWeight = 1.0 / 3.0;
	pairwisePrior = 1.5;		//1.5->0
	pairwiseFactor = 6.0;		//6.0->20
	pairwiseBeta = 16.0;

	//主导标签比例
	cliqueMinLabelRatio = 0.5;
	//阈值
	cliqueThresholdRatio = 0.1;
	//截断参数
	cliqueTruncation = 0.1;		//0.1->0.5

								//Boosting参数
	statsThetaStart = 2;
	statsThetaIncrement = 1;
	statsNumberOfThetas = 15;
	statsNumberOfBoosts = 5000;
	statsRandomizationFactor = 0.1;
	statsFactor = 0.6;		//0.6->1.0
	statsAlpha = 0.05;
	statsPrior = 0.0;
	statsMaxClassRatio = 0.5;		//0.5->0.7
	statsTrainFile = "statsboost.dat";
	statsFolder = "/home/yangfei/Datasets/Result/KITTI/val/Stats/";
	statsExtension = ".sts";

	//一致性先验项
	consistencyPrior = 0.05;
	//高阶平滑项参数
	pairwiseSegmentBuckets = 8;
	pairwiseSegmentPrior = 0.0;		//0.0—>1.5
	pairwiseSegmentFactor = 2.0;
	pairwiseSegmentBeta = 40.0;

	//3D雷达
	//测试结果保存路径
	lidarTestFolder = "/home/yangfei/Datasets/Result/KITTI/val/LidarCrf/";
	//雷达GT图
	lidarGroundTruthFolder = "/home/yangfei/Datasets/Data/KITTI/val/GroundTruth/";		//雷达的标签，跟图像一样
	lidarGroundTruthExtension = ".png";
	lidarUnaryFactor = 5.0;//0.7;  //0.1->0.5


	lidarPairwiseFactor = 1.0;//0.0005;		//0.00005->0.0005
							  //平滑截断概率
	lidarPairwiseTruncation = 10.0;
	//视差类别,很大
	lidarClassNo = 2;


	//模型参数初始化
	crossUnaryWeight = 0.5;
	crossPairwiseWeight = 0.2;//-1e-4;
	crossThreshold = 1e-6;
	//这个是要训练的
	crossTrainFile = "height.dat";

	//初始化，打乱顺序，按比例生成训练集和测试集
	Init();

	//创建目录
	int i;
	ForceDirectory(trainFolder);
	ForceDirectory(testFolder);
	ForceDirectory(lidarTestFolder);
	ForceDirectory(textonFolder);
	ForceDirectory(siftFolder);
	ForceDirectory(colourSiftFolder);
	ForceDirectory(locationFolder);
	ForceDirectory(lbpFolder);
	for (i = 0; i < 4; i++)	ForceDirectory(meanShiftFolder[i]);
	ForceDirectory(denseFolder);
	ForceDirectory(statsFolder);

	ForceDirectory("/home/yangfei/Datasets/Result/KITTI/val/Lidar/denseClassifierOut/");
	ForceDirectory("/home/yangfei/Datasets/Result/KITTI/val/Lidar/statClassifierOut/");
	ForceDirectory("/home/yangfei/Datasets/Result/KITTI/val/Lidar/denseFeatureEval/");
	ForceDirectory("/home/yangfei/Datasets/Result/KITTI/val/Lidar/statFeatureEval/");
}


void LKITTIValidationset::Init()
{
	int i;
	LoadFolder(imageFolder, imageExtension, allImageFiles);
	//将奇数作为训练集，偶数作为测试集
	for (i=0;i<allImageFiles.GetCount();i++)
	{
		if (i%2)	//奇数
		{
			trainImageFiles.Add(allImageFiles[i]);
		}
		else
		{
			testImageFiles.Add(allImageFiles[i]);
		}
	}

}

void LKITTIValidationset::RgbToLabel(unsigned char * rgb, unsigned char * label)
{
	if (rgb[2] == 255)		//r通道代表有效点
	{
		if (rgb[0] == 255)	//b通道代表道路样本
		{
			label[0] = 1;		//道路
		}
		else
			label[0] = 2;		//非路
	}
	else
		label[0] = 0;		//无效
}

void LKITTIValidationset::LabelToRgb(unsigned char * label, unsigned char * rgb)
{
	rgb[0] = rgb[1] = rgb[2] = 0;
	if (label[0] == 2)
	{
		rgb[2] = 255;		//非路红色
	}
	else
	{
		if (label[0] == 1)
		{
			rgb[2] = 255, rgb[0] = 255;		//道路粉色
		}
	}
}

void LKITTIValidationset::SetCRFStructure(LCrf * crf)
{
	//创建CRF主体对象
	LCrfDomain *objDomain = new LCrfDomain(crf, this, classNo, testFolder, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::RgbToLabel, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::LabelToRgb);
	//添加主体到模型
	crf->domains.Add(objDomain);

	//创建底层
	LBaseCrfLayer *baseLayer = new LBaseCrfLayer(crf, objDomain, this, 0);
	//添加底层到模型
	crf->layers.Add(baseLayer);

	//创建高层和分割
	LPnCrfLayer *superpixelLayer[3];
	LSegmentation2D *segmentation[3];


	//baselayer
	segmentation[0] = new LMeanShiftSegmentation2D(meanShiftXY[1], meanShiftLuv[1], meanShiftMinRegion[1], meanShiftFolder[1], meanShiftExtension);
	//建立在底层上的分割，设置父层为base层以及超像素层
	superpixelLayer[0] = new LPnCrfLayer(crf, objDomain, this, baseLayer, segmentation[0], cliqueTruncation);
	crf->segmentations.Add(segmentation[0]);		//添加分割到crf模型
	crf->layers.Add(superpixelLayer[0]);		//添加高阶层到crf模型


												//添加特征对象
	LTextonFeature *textonFeature = new LTextonFeature(this, trainFolder, textonClusteringTrainFile, textonFolder, textonExtension, textonFilterBankRescale, textonKMeansSubSample, textonNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster);
	LLocationFeature *locationFeature = new LLocationFeature(this, locationFolder, locationExtension, locationBuckets);
	LSiftFeature *siftFeature = new LSiftFeature(this, trainFolder, siftClusteringTrainFile, siftFolder, siftExtension, siftSizeCount, siftSizes, siftWindowNumber, sift360, siftAngles, siftKMeansSubSample, siftNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster, 1);
	LColourSiftFeature *coloursiftFeature = new LColourSiftFeature(this, trainFolder, colourSiftClusteringTrainFile, colourSiftFolder, colourSiftExtension, colourSiftSizeCount, colourSiftSizes, colourSiftWindowNumber, colourSift360, colourSiftAngles, colourSiftKMeansSubSample, colourSiftNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster, 1);
	LLbpFeature *lbpFeature = new LLbpFeature(this, trainFolder, lbpClusteringFile, lbpFolder, lbpExtension, lbpSize, lbpKMeansSubSample, lbpNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster);

	crf->features.Add(textonFeature);
	crf->features.Add(locationFeature);
	crf->features.Add(lbpFeature);
	crf->features.Add(siftFeature);
	crf->features.Add(coloursiftFeature);

	//一阶数据项
	LDenseUnaryPixelPotential *pixelPotential = new LDenseUnaryPixelPotential(this, crf, objDomain, baseLayer, trainFolder, denseBoostTrainFile, denseFolder, denseExtension, classNo, denseWeight, denseBoostingSubSample, denseNumberOfRectangles, denseMinimumRectangleSize, denseMaximumRectangleSize, denseMaxClassRatio);
	pixelPotential->AddFeature(textonFeature);
	pixelPotential->AddFeature(siftFeature);
	pixelPotential->AddFeature(coloursiftFeature);
	pixelPotential->AddFeature(lbpFeature);
	crf->potentials.Add(pixelPotential);

	LBoosting<int> *pixelBoosting = new LBoosting<int>(trainFolder, denseBoostTrainFile, classNo, denseNumRoundsBoosting, denseThetaStart, denseThetaIncrement, denseNumberOfThetas, denseRandomizationFactor, pixelPotential, (int *(LPotential::*)(int, int))&LDenseUnaryPixelPotential::GetTrainBoostingValues, (int *(LPotential::*)(int))&LDenseUnaryPixelPotential::GetEvalBoostingValues);
	crf->learnings.Add(pixelBoosting);
	pixelPotential->learning = pixelBoosting;


	//高阶数据项
	LStatsUnarySegmentPotential *statsPotential = new LStatsUnarySegmentPotential(this, crf, objDomain, trainFolder, statsTrainFile, statsFolder, statsExtension, classNo, statsPrior, statsFactor, cliqueMinLabelRatio, statsAlpha, statsMaxClassRatio);
	statsPotential->AddFeature(textonFeature);
	statsPotential->AddFeature(siftFeature);
	statsPotential->AddFeature(coloursiftFeature);
	statsPotential->AddFeature(locationFeature);
	statsPotential->AddFeature(lbpFeature);
	statsPotential->AddLayer(superpixelLayer[0]);

	//segment boosting
	LBoosting<double> *segmentBoosting = new LBoosting<double>(trainFolder, statsTrainFile, classNo, statsNumberOfBoosts, statsThetaStart, statsThetaIncrement, statsNumberOfThetas, statsRandomizationFactor, statsPotential, (double *(LPotential::*)(int, int))&LStatsUnarySegmentPotential::GetTrainBoostingValues, (double *(LPotential::*)(int))&LStatsUnarySegmentPotential::GetEvalBoostingValues);
	crf->learnings.Add(segmentBoosting);
	statsPotential->learning = segmentBoosting;
	crf->potentials.Add(statsPotential);


	//视差改雷达
	LCrfDomain *lidarDomain = new LCrfDomain(crf, this, classNo, lidarTestFolder, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::RgbToLabel, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::LabelToRgb);
	crf->domains.Add(lidarDomain);
	//视差底层
	LBaseCrfLayer *lidarBaseLayer = new LBaseCrfLayer(crf, lidarDomain, this, 0);
	crf->layers.Add(lidarBaseLayer);

	//视差数据项(已解决)
	LLidarUnaryPixelPotential *pixelLidarPotential = new LLidarUnaryPixelPotential(this, crf, lidarDomain, lidarBaseLayer, lidarClassNo, lidarUnaryFactor);
	crf->potentials.Add(pixelLidarPotential);

	//连接两个模型的基于目标高度分布的连接项
	LHeightUnaryPixelPotential *heightPotential = new LHeightUnaryPixelPotential(this, crf, objDomain, baseLayer, lidarDomain, lidarBaseLayer, trainFolder, crossTrainFile, classNo, crossUnaryWeight, lidarClassNo, crossThreshold);
	crf->potentials.Add(heightPotential);

	//联合平滑项
	crf->potentials.Add(new LJointPairwisePixelPotential(this, crf, objDomain, baseLayer, lidarDomain, lidarBaseLayer, classNo, lidarClassNo, pairwisePrior, pairwiseFactor, pairwiseBeta, pairwiseLWeight, pairwiseUWeight, pairwiseVWeight, lidarPairwiseFactor, lidarPairwiseTruncation, crossPairwiseWeight));

}


//pairwise crf
void LKITTIPairwiseCompare::Init()
{
	int i;
	LoadFolder(imageFolder, imageExtension, allImageFiles);
	//将奇数作为训练集，偶数作为测试集
	for (i = 0;i < allImageFiles.GetCount();i++)
	{
		if (i % 2)	//奇数
		{
			trainImageFiles.Add(allImageFiles[i]);
		}
		else
		{
			testImageFiles.Add(allImageFiles[i]);
		}
	}

}

LKITTIPairwiseCompare::LKITTIPairwiseCompare()
{
	seed = 10000;
	classNo = 2;	//KITTI数据集就道路和非路两类
	filePermutations = 10000;
	optimizeAverage = 1;
	featuresOnline = 0;

	//类别权重，初始化为全1
	unaryWeighted = 0;
	unaryWeights = new double[classNo];
	for (int i = 0; i < classNo; i++) unaryWeights[i] = 1.0;

	//训练集和测试集比例1:1
	proportionTrain = 0.5;
	proportionTest = 0.5;

	datasetName = "val";
	imageFolder = "/home/yangfei/Datasets/Data/KITTI/val/Images/";
	imageExtension = ".png";
	groundTruthFolder = "/home/yangfei/Datasets/Data/KITTI/val/GroundTruth/";
	groundTruthExtension = ".png";
	lidarFolder = "/home/yangfei/Datasets/Data/KITTI/val/Lidar/";
	lidarExtension = ".bin";
	calibFolder = "/home/yangfei/Datasets/Data/KITTI/val/calib/";
	calibExtension = ".txt";
	trainFolder = "/home/yangfei/Datasets/Result/KITTI/val/Train/";
	testFolder = "/home/yangfei/Datasets/Result/KITTI/val/Crf/";

	clusterPointsPerKDTreeCluster = 30;
	clusterKMeansMaxChange = 0.01;

	textonNumberOfClusters = 50;		//50个聚类中心
	textonFilterBankRescale = 0.7;
	textonKMeansSubSample = 20;
	textonClusteringTrainFile = "textonclustering.dat";
	textonFolder = "/home/yangfei/Datasets//home/yangfei/Datasets/Result/KITTI/val/Feature/Texton/";
	textonExtension = ".txn";

	siftClusteringTrainFile = "siftclustering.dat";
	siftKMeansSubSample = 40;
	siftNumberOfClusters = 50;
	siftSizes[0] = 5, siftSizes[1] = 7, siftSizes[2] = 9;
	siftSizeCount = 3;
	siftWindowNumber = 3;
	sift360 = 1;	//360度
	siftAngles = 8;		//HOG有8个bin代表360度的8个方向
	siftFolder = "/home/yangfei/Datasets//home/yangfei/Datasets/Result/KITTI/val/Feature/Sift/";
	siftExtension = ".sft";

	//加了颜色信息的HOG
	colourSiftClusteringTrainFile = "coloursiftclustering.dat";
	colourSiftKMeansSubSample = 40;
	colourSiftNumberOfClusters = 50;
	colourSiftSizes[0] = 5, colourSiftSizes[1] = 7, colourSiftSizes[2] = 9;
	colourSiftSizeCount = 3;
	colourSiftWindowNumber = 3;
	colourSift360 = 1;
	colourSiftAngles = 8;
	colourSiftFolder = "/home/yangfei/Datasets//home/yangfei/Datasets/Result/KITTI/val/Feature/ColourSift/";
	colourSiftExtension = ".csf";

	locationBuckets = 20;	//位置数，归一化
	locationFolder = "/home/yangfei/Datasets/Result/KITTI/val/Feature/Location/";
	locationExtension = ".loc";

	lbpClusteringFile = "lbpclustering.dat";
	lbpFolder = "/home/yangfei/Datasets/Result/KITTI/val/Feature/Lbp/";
	lbpExtension = ".lbp";
	lbpSize = 11;
	lbpKMeansSubSample = 20;
	lbpNumberOfClusters = 50;


	//Boosting参数
	denseNumRoundsBoosting = 5000;
	denseBoostingSubSample = 10;
	denseNumberOfThetas = 20;
	denseThetaStart = 3;
	denseThetaIncrement = 2;
	denseNumberOfRectangles = 100;
	denseMinimumRectangleSize = 5;
	denseMaximumRectangleSize = 200;
	denseRandomizationFactor = 0.003;
	denseBoostTrainFile = "denseboost.dat";
	denseExtension = ".dns";
	denseFolder = "/home/yangfei/Datasets/Result/KITTI/val/Dense/";

	denseWeight = 1.0;		//1.0->2.0
							//最大类别概率
	denseMaxClassRatio = 0.2;

	//平滑项三个通道的权重
	pairwiseLWeight = 1.0 / 3.0;
	pairwiseUWeight = 1.0 / 3.0;
	pairwiseVWeight = 1.0 / 3.0;
	pairwisePrior = 1.5;		//1.5->0
	pairwiseFactor = 6.0;		//6.0->20
	pairwiseBeta = 16.0;


	//初始化，打乱顺序，按比例生成训练集和测试集
	Init();

	//创建目录
	ForceDirectory(trainFolder);
	ForceDirectory(testFolder);

	ForceDirectory(textonFolder);
	ForceDirectory(siftFolder);
	ForceDirectory(colourSiftFolder);
	ForceDirectory(locationFolder);
	ForceDirectory(lbpFolder);

	ForceDirectory(denseFolder);

}

void LKITTIPairwiseCompare::RgbToLabel(unsigned char * rgb, unsigned char * label)
{
	if (rgb[2] == 255)		//r通道代表有效点
	{
		if (rgb[0] == 255)	//b通道代表道路样本
		{
			label[0] = 1;		//道路
		}
		else
			label[0] = 2;		//非路
	}
	else
		label[0] = 0;		//无效
}

void LKITTIPairwiseCompare::LabelToRgb(unsigned char * label, unsigned char * rgb)
{
	rgb[0] = rgb[1] = rgb[2] = 0;
	if (label[0] == 2)
	{
		rgb[2] = 255;		//非路红色
	}
	else
	{
		if (label[0] == 1)
		{
			rgb[2] = 255, rgb[0] = 255;		//道路粉色
		}
	}
}

void LKITTIPairwiseCompare::SetCRFStructure(LCrf * crf)
{
	//创建CRF主体对象
	LCrfDomain *objDomain = new LCrfDomain(crf, this, classNo, testFolder, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::RgbToLabel, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::LabelToRgb);
	//添加主体到模型
	crf->domains.Add(objDomain);

	//创建底层
	LBaseCrfLayer *baseLayer = new LBaseCrfLayer(crf, objDomain, this, 0);
	//添加底层到模型
	crf->layers.Add(baseLayer);
												//添加特征对象
	LTextonFeature *textonFeature = new LTextonFeature(this, trainFolder, textonClusteringTrainFile, textonFolder, textonExtension, textonFilterBankRescale, textonKMeansSubSample, textonNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster);
	LLocationFeature *locationFeature = new LLocationFeature(this, locationFolder, locationExtension, locationBuckets);
	LSiftFeature *siftFeature = new LSiftFeature(this, trainFolder, siftClusteringTrainFile, siftFolder, siftExtension, siftSizeCount, siftSizes, siftWindowNumber, sift360, siftAngles, siftKMeansSubSample, siftNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster, 1);
	LColourSiftFeature *coloursiftFeature = new LColourSiftFeature(this, trainFolder, colourSiftClusteringTrainFile, colourSiftFolder, colourSiftExtension, colourSiftSizeCount, colourSiftSizes, colourSiftWindowNumber, colourSift360, colourSiftAngles, colourSiftKMeansSubSample, colourSiftNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster, 1);
	LLbpFeature *lbpFeature = new LLbpFeature(this, trainFolder, lbpClusteringFile, lbpFolder, lbpExtension, lbpSize, lbpKMeansSubSample, lbpNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster);

	crf->features.Add(textonFeature);
	crf->features.Add(locationFeature);
	crf->features.Add(lbpFeature);
	crf->features.Add(siftFeature);
	crf->features.Add(coloursiftFeature);

	//一阶数据项
	LDenseUnaryPixelPotential *pixelPotential = new LDenseUnaryPixelPotential(this, crf, objDomain, baseLayer, trainFolder, denseBoostTrainFile, denseFolder, denseExtension, classNo, denseWeight, denseBoostingSubSample, denseNumberOfRectangles, denseMinimumRectangleSize, denseMaximumRectangleSize, denseMaxClassRatio);
	pixelPotential->AddFeature(textonFeature);
	pixelPotential->AddFeature(siftFeature);
	pixelPotential->AddFeature(coloursiftFeature);
	pixelPotential->AddFeature(lbpFeature);
	crf->potentials.Add(pixelPotential);

	LBoosting<int> *pixelBoosting = new LBoosting<int>(trainFolder, denseBoostTrainFile, classNo, denseNumRoundsBoosting, denseThetaStart, denseThetaIncrement, denseNumberOfThetas, denseRandomizationFactor, pixelPotential, (int *(LPotential::*)(int, int))&LDenseUnaryPixelPotential::GetTrainBoostingValues, (int *(LPotential::*)(int))&LDenseUnaryPixelPotential::GetEvalBoostingValues);
	crf->learnings.Add(pixelBoosting);
	pixelPotential->learning = pixelBoosting;

	//创建基于8领域的Potts模型的二元势函数
	LEightNeighbourPottsPairwisePixelPotential *pairwisePotentianl = new LEightNeighbourPottsPairwisePixelPotential(this, crf, objDomain, baseLayer, classNo, pairwisePrior, pairwiseFactor, pairwiseBeta, pairwiseLWeight, pairwiseUWeight, pairwiseVWeight);
	//添加平滑项到CRF模型
	crf->potentials.Add(pairwisePotentianl);
}


//high order crf
void LKITTIHighOrderCompare::Init()
{
	int i;
	LoadFolder(imageFolder, imageExtension, allImageFiles);
	//将奇数作为训练集，偶数作为测试集
	for (i = 0;i < allImageFiles.GetCount();i++)
	{
		if (i % 2)	//奇数
		{
			trainImageFiles.Add(allImageFiles[i]);
		}
		else
		{
			testImageFiles.Add(allImageFiles[i]);
		}
	}
}

LKITTIHighOrderCompare::LKITTIHighOrderCompare()
{
	seed = 10000;
	classNo = 2;	//KITTI数据集就道路和非路两类
	filePermutations = 10000;
	optimizeAverage = 1;
	featuresOnline = 0;

	//类别权重，初始化为全1
	unaryWeighted = 0;
	unaryWeights = new double[classNo];
	for (int i = 0; i < classNo; i++) unaryWeights[i] = 1.0;

	//训练集和测试集比例1:1
	proportionTrain = 0.5;
	proportionTest = 0.5;

	datasetName = "val";
	imageFolder = "/home/yangfei/Datasets/Data/KITTI/val/Images/";
	imageExtension = ".png";
	groundTruthFolder = "/home/yangfei/Datasets/Data/KITTI/val/GroundTruth/";
	groundTruthExtension = ".png";
	lidarFolder = "/home/yangfei/Datasets/Data/KITTI/val/Lidar/";
	lidarExtension = ".bin";
	calibFolder = "/home/yangfei/Datasets/Data/KITTI/val/calib/";
	calibExtension = ".txt";
	trainFolder = "/home/yangfei/Datasets/Result/KITTI/val/Train/";
	testFolder = "/home/yangfei/Datasets/Result/KITTI/val/Crf/";

	clusterPointsPerKDTreeCluster = 30;
	clusterKMeansMaxChange = 0.01;

	textonNumberOfClusters = 50;		//50个聚类中心
	textonFilterBankRescale = 0.7;
	textonKMeansSubSample = 20;
	textonClusteringTrainFile = "textonclustering.dat";
	textonFolder = "/home/yangfei/Datasets/Result/KITTI/val/Feature/Texton/";
	textonExtension = ".txn";

	siftClusteringTrainFile = "siftclustering.dat";
	siftKMeansSubSample = 40;
	siftNumberOfClusters = 50;
	siftSizes[0] = 5, siftSizes[1] = 7, siftSizes[2] = 9;
	siftSizeCount = 3;
	siftWindowNumber = 3;
	sift360 = 1;	//360度
	siftAngles = 8;		//HOG有8个bin代表360度的8个方向
	siftFolder = "/home/yangfei/Datasets/Result/KITTI/val/Feature/Sift/";
	siftExtension = ".sft";

	//加了颜色信息的HOG
	colourSiftClusteringTrainFile = "coloursiftclustering.dat";
	colourSiftKMeansSubSample = 40;
	colourSiftNumberOfClusters = 50;
	colourSiftSizes[0] = 5, colourSiftSizes[1] = 7, colourSiftSizes[2] = 9;
	colourSiftSizeCount = 3;
	colourSiftWindowNumber = 3;
	colourSift360 = 1;
	colourSiftAngles = 8;
	colourSiftFolder = "/home/yangfei/Datasets/Result/KITTI/val/Feature/ColourSift/";
	colourSiftExtension = ".csf";

	locationBuckets = 20;	//位置数，归一化
	locationFolder = "/home/yangfei/Datasets/Result/KITTI/val/Feature/Location/";
	locationExtension = ".loc";

	lbpClusteringFile = "lbpclustering.dat";
	lbpFolder = "/home/yangfei/Datasets/Result/KITTI/val/Feature/Lbp/";
	lbpExtension = ".lbp";
	lbpSize = 11;
	lbpKMeansSubSample = 20;
	lbpNumberOfClusters = 50;

	//第一层
	meanShiftXY[0] = 5.0;
	meanShiftLuv[0] = 3.0;
	meanShiftMinRegion[0] = 20;
	meanShiftFolder[0] = "/home/yangfei/Datasets/Result/KITTI/val/MeanShift/50x30/";
	//第一层
	meanShiftXY[1] = 5.0;
	meanShiftLuv[1] = 5.0;
	meanShiftMinRegion[1] = 20;
	meanShiftFolder[1] = "/home/yangfei/Datasets/Result/KITTI/val/MeanShift/50x50/";
	//第二层
	meanShiftXY[2] = 7.0;
	meanShiftLuv[2] = 5.0;
	meanShiftMinRegion[2] = 20;
	meanShiftFolder[2] = "/home/yangfei/Datasets/Result/KITTI/val/MeanShift/70x50/";
	//第二层
	meanShiftXY[3] = 7.0;
	meanShiftLuv[3] = 6.0;
	meanShiftMinRegion[3] = 20;
	meanShiftFolder[3] = "/home/yangfei/Datasets/Result/KITTI/val/MeanShift/70x60/";

	meanShiftExtension = ".msh";


	//Boosting参数
	denseNumRoundsBoosting = 5000;
	denseBoostingSubSample = 10;
	denseNumberOfThetas = 20;
	denseThetaStart = 3;
	denseThetaIncrement = 2;
	denseNumberOfRectangles = 100;
	denseMinimumRectangleSize = 5;
	denseMaximumRectangleSize = 200;
	denseRandomizationFactor = 0.003;
	denseBoostTrainFile = "denseboost.dat";
	denseExtension = ".dns";
	denseFolder = "/home/yangfei/Datasets/Result/KITTI/val/Dense/";

	denseWeight = 1.0;		//1.0->2.0
							//最大类别概率
	denseMaxClassRatio = 0.2;

	//平滑项三个通道的权重
	pairwiseLWeight = 1.0 / 3.0;
	pairwiseUWeight = 1.0 / 3.0;
	pairwiseVWeight = 1.0 / 3.0;
	pairwisePrior = 1.5;		//1.5->0
	pairwiseFactor = 6.0;		//6.0->20
	pairwiseBeta = 16.0;

	//主导标签比例
	cliqueMinLabelRatio = 0.5;
	//阈值
	cliqueThresholdRatio = 0.1;
	//截断参数
	cliqueTruncation = 0.1;		//0.1->0.5

								//Boosting参数
	statsThetaStart = 2;
	statsThetaIncrement = 1;
	statsNumberOfThetas = 15;
	statsNumberOfBoosts = 5000;
	statsRandomizationFactor = 0.1;
	statsFactor = 0.6;		//0.6->1.0
	statsAlpha = 0.05;
	statsPrior = 0.0;
	statsMaxClassRatio = 0.5;		//0.5->0.7
	statsTrainFile = "statsboost.dat";
	statsFolder = "/home/yangfei/Datasets/Result/KITTI/val/Stats/";
	statsExtension = ".sts";

	//一致性先验项
	consistencyPrior = 0.05;
	//高阶平滑项参数
	pairwiseSegmentBuckets = 8;
	pairwiseSegmentPrior = 0.0;		//0.0—>1.5
	pairwiseSegmentFactor = 2.0;
	pairwiseSegmentBeta = 40.0;

	
	//初始化，打乱顺序，按比例生成训练集和测试集
	Init();

	//创建目录
	int i;
	ForceDirectory(trainFolder);
	ForceDirectory(testFolder);

	ForceDirectory(textonFolder);
	ForceDirectory(siftFolder);
	ForceDirectory(colourSiftFolder);
	ForceDirectory(locationFolder);
	ForceDirectory(lbpFolder);

	for (i = 0; i < 4; i++)	ForceDirectory(meanShiftFolder[i]);

	ForceDirectory(denseFolder);
	ForceDirectory(statsFolder);
}

void LKITTIHighOrderCompare::RgbToLabel(unsigned char * rgb, unsigned char * label)
{
	if (rgb[2] == 255)		//r通道代表有效点
	{
		if (rgb[0] == 255)	//b通道代表道路样本
		{
			label[0] = 1;		//道路
		}
		else
			label[0] = 2;		//非路
	}
	else
		label[0] = 0;		//无效
}

void LKITTIHighOrderCompare::LabelToRgb(unsigned char * label, unsigned char * rgb)
{
	rgb[0] = rgb[1] = rgb[2] = 0;
	if (label[0] == 2)
	{
		rgb[2] = 255;		//非路红色
	}
	else
	{
		if (label[0] == 1)
		{
			rgb[2] = 255, rgb[0] = 255;		//道路粉色
		}
	}
}

void LKITTIHighOrderCompare::SetCRFStructure(LCrf * crf)
{
	//创建CRF主体对象
	LCrfDomain *objDomain = new LCrfDomain(crf, this, classNo, testFolder, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::RgbToLabel, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::LabelToRgb);
	//添加主体到模型
	crf->domains.Add(objDomain);

	//创建底层
	LBaseCrfLayer *baseLayer = new LBaseCrfLayer(crf, objDomain, this, 0);
	//添加底层到模型
	crf->layers.Add(baseLayer);

	//创建高层和分割
	LPnCrfLayer *superpixelLayer[3];
	LSegmentation2D *segmentation[3];


	//baselayer
	segmentation[0] = new LMeanShiftSegmentation2D(meanShiftXY[1], meanShiftLuv[1], meanShiftMinRegion[1], meanShiftFolder[1], meanShiftExtension);
	//建立在底层上的分割，设置父层为base层以及超像素层
	superpixelLayer[0] = new LPnCrfLayer(crf, objDomain, this, baseLayer, segmentation[0], cliqueTruncation);
	crf->segmentations.Add(segmentation[0]);		//添加分割到crf模型
	crf->layers.Add(superpixelLayer[0]);		//添加高阶层到crf模型


												//添加特征对象
	LTextonFeature *textonFeature = new LTextonFeature(this, trainFolder, textonClusteringTrainFile, textonFolder, textonExtension, textonFilterBankRescale, textonKMeansSubSample, textonNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster);
	LLocationFeature *locationFeature = new LLocationFeature(this, locationFolder, locationExtension, locationBuckets);
	LSiftFeature *siftFeature = new LSiftFeature(this, trainFolder, siftClusteringTrainFile, siftFolder, siftExtension, siftSizeCount, siftSizes, siftWindowNumber, sift360, siftAngles, siftKMeansSubSample, siftNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster, 1);
	LColourSiftFeature *coloursiftFeature = new LColourSiftFeature(this, trainFolder, colourSiftClusteringTrainFile, colourSiftFolder, colourSiftExtension, colourSiftSizeCount, colourSiftSizes, colourSiftWindowNumber, colourSift360, colourSiftAngles, colourSiftKMeansSubSample, colourSiftNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster, 1);
	LLbpFeature *lbpFeature = new LLbpFeature(this, trainFolder, lbpClusteringFile, lbpFolder, lbpExtension, lbpSize, lbpKMeansSubSample, lbpNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster);

	crf->features.Add(textonFeature);
	crf->features.Add(locationFeature);
	crf->features.Add(lbpFeature);
	crf->features.Add(siftFeature);
	crf->features.Add(coloursiftFeature);

	//一阶数据项
	LDenseUnaryPixelPotential *pixelPotential = new LDenseUnaryPixelPotential(this, crf, objDomain, baseLayer, trainFolder, denseBoostTrainFile, denseFolder, denseExtension, classNo, denseWeight, denseBoostingSubSample, denseNumberOfRectangles, denseMinimumRectangleSize, denseMaximumRectangleSize, denseMaxClassRatio);
	pixelPotential->AddFeature(textonFeature);
	pixelPotential->AddFeature(siftFeature);
	pixelPotential->AddFeature(coloursiftFeature);
	pixelPotential->AddFeature(lbpFeature);
	crf->potentials.Add(pixelPotential);

	LBoosting<int> *pixelBoosting = new LBoosting<int>(trainFolder, denseBoostTrainFile, classNo, denseNumRoundsBoosting, denseThetaStart, denseThetaIncrement, denseNumberOfThetas, denseRandomizationFactor, pixelPotential, (int *(LPotential::*)(int, int))&LDenseUnaryPixelPotential::GetTrainBoostingValues, (int *(LPotential::*)(int))&LDenseUnaryPixelPotential::GetEvalBoostingValues);
	crf->learnings.Add(pixelBoosting);
	pixelPotential->learning = pixelBoosting;

	//创建基于8领域的Potts模型的二元势函数
	LEightNeighbourPottsPairwisePixelPotential *pairwisePotentianl = new LEightNeighbourPottsPairwisePixelPotential(this, crf, objDomain, baseLayer, classNo, pairwisePrior, pairwiseFactor, pairwiseBeta, pairwiseLWeight, pairwiseUWeight, pairwiseVWeight);
	//添加平滑项到CRF模型
	crf->potentials.Add(pairwisePotentianl);

	//高阶数据项
	LStatsUnarySegmentPotential *statsPotential = new LStatsUnarySegmentPotential(this, crf, objDomain, trainFolder, statsTrainFile, statsFolder, statsExtension, classNo, statsPrior, statsFactor, cliqueMinLabelRatio, statsAlpha, statsMaxClassRatio);
	statsPotential->AddFeature(textonFeature);
	statsPotential->AddFeature(siftFeature);
	statsPotential->AddFeature(coloursiftFeature);
	statsPotential->AddFeature(locationFeature);
	statsPotential->AddFeature(lbpFeature);
	statsPotential->AddLayer(superpixelLayer[0]);

	//segment boosting
	LBoosting<double> *segmentBoosting = new LBoosting<double>(trainFolder, statsTrainFile, classNo, statsNumberOfBoosts, statsThetaStart, statsThetaIncrement, statsNumberOfThetas, statsRandomizationFactor, statsPotential, (double *(LPotential::*)(int, int))&LStatsUnarySegmentPotential::GetTrainBoostingValues, (double *(LPotential::*)(int))&LStatsUnarySegmentPotential::GetEvalBoostingValues);
	crf->learnings.Add(segmentBoosting);
	statsPotential->learning = segmentBoosting;
	crf->potentials.Add(statsPotential);
}
