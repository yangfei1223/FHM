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

	//�������
	LMath::SetSeed(seed);

	LoadFolder(imageFolder, imageExtension, allImageFiles);

	//����˳��filePermutations�������������
	printf("Permuting image files..\n");
	if (allImageFiles.GetCount() > 0)
		for (i = 0; i < filePermutations; i++)	//�����������
		{
			index1 = LMath::RandomInt(allImageFiles.GetCount());
			index2 = LMath::RandomInt(allImageFiles.GetCount());
			allImageFiles.Swap(index1, index2);
		}

	//�����ݷֳ������֣�ѵ�����Ͳ��Լ�
	printf("Splitting image files..\n");
	for (i = 0; i < allImageFiles.GetCount(); i++)
	{
		//����˳���ǰһ����Ϊѵ����
		if (i < proportionTrain * allImageFiles.GetCount())
			trainImageFiles.Add(allImageFiles[i]);
		//��һ����Ϊ���Լ�
		else if
			(i < (proportionTrain + proportionTest) * allImageFiles.GetCount()) testImageFiles.Add(allImageFiles[i]);
	}
}

void LDataset::SaveImage(LLabelImage &labelImage, LCrfDomain *domain, char *fileName)
{
	int points = labelImage.GetPoints();
	unsigned char *labelData = labelImage.GetData();
	for (int j = 0; j < points; j++, labelData++) (*labelData)++;	//ȫ����1���ָ���0��ǩ��״̬
	labelImage.Save(fileName, domain);
}

int LDataset::Segmented(char *imageFileName)
{
	return(1);
}

//��ǩ����
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

	//ͼ���ļ���
	imageFolder = "Data/CamVid/Images/";
	//��չ��
	imageExtension = ".png";
	//��ǩ�ļ���
	groundTruthFolder = "Data/CamVid/GroundTruth/";
	//��չ��
	groundTruthExtension = ".png";
	//ѵ���ļ���
	trainFolder = "Result/CamVid/Train/";
	//�����ļ���
	testFolder = "Result/CamVid/Crf/";

	//K-mean���������K-meansӦ�����������������
	clusterPointsPerKDTreeCluster = 30;
	clusterKMeansMaxChange = 0.01;

	//λ��������������
	locationBuckets = 12;
	locationFolder = "Result/CamVid/Feature/Location/";
	locationExtension = ".loc";

	//����������������
	textonNumberOfClusters = 50;	//filter respond�������ĸ���
	textonFilterBankRescale = 0.7;	//�˲����Ĳ�����
	textonKMeansSubSample = 10;		//??
	textonClusteringTrainFile = "textonclustering.dat";
	textonFolder = "Result/CamVid/Feature/Texton/";
	textonExtension = ".txn";

	//sift�����������ã�HOG
	siftClusteringTrainFile = "siftclustering.dat";
	siftKMeansSubSample = 20;
	siftNumberOfClusters = 50;
	siftSizes[0] = 3, siftSizes[1] = 5, siftSizes[2] = 7, siftSizes[3] = 9;		//4���߶�
	siftSizeCount = 4;
	siftWindowNumber = 3;
	sift360 = 1;
	siftAngles = 8;
	siftFolder = "Result/CamVid/Feature/Sift/";
	siftExtension = ".sft";

	//color-sift������������
	colourSiftClusteringTrainFile = "coloursiftclustering.dat";
	colourSiftKMeansSubSample = 20;
	colourSiftNumberOfClusters = 50;
	colourSiftSizes[0] = 3, colourSiftSizes[1] = 5, colourSiftSizes[2] = 7, colourSiftSizes[3] = 9;		//4���߶�
	colourSiftSizeCount = 4;
	colourSiftWindowNumber = 3;
	colourSift360 = 1;
	colourSiftAngles = 8;
	colourSiftFolder = "Result/CamVid/Feature/ColourSift/";
	colourSiftExtension = ".csf";

	//LBP������������
	lbpClusteringFile = "lbpclustering.dat";
	lbpFolder = "Result/CamVid/Feature/Lbp/";
	lbpExtension = ".lbp";
	lbpSize = 11;
	lbpKMeansSubSample = 10;
	lbpNumberOfClusters = 50;

	//AdaBoost��������
	denseNumRoundsBoosting = 500;		//����������Ӧ�þ���������������
	denseBoostingSubSample = 5;		//������
	denseNumberOfThetas = 25;
	denseThetaStart = 3;
	denseThetaIncrement = 2;
	denseNumberOfRectangles = 100;		//���θ���
	denseMinimumRectangleSize = 5;		//��С���δ�С
	denseMaximumRectangleSize = 200;		//�����δ�С
	denseRandomizationFactor = 0.003;		//�������
	denseBoostTrainFile = "denseboost.dat";
	denseExtension = ".dns";
	denseFolder = "Result/CamVid/Dense/";
	denseWeight = 1.0;
	denseMaxClassRatio = 0.1;

	//mean-sift��������
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

	//ƽ�����������
	pairwiseLWeight = 1.0 / 3.0;
	pairwiseUWeight = 1.0 / 3.0;
	pairwiseVWeight = 1.0 / 3.0;
	pairwisePrior = 1.5;
	pairwiseFactor = 6.0;
	pairwiseBeta = 16.0;

	//�ŵ�һЩ����
	cliqueMinLabelRatio = 0.5;	//Ӧ����������ǩ����
	cliqueThresholdRatio = 0.1;
	cliqueTruncation = 0.1;		//�ض���ֵ�������е�alpha��

	//��Щ��ʱ��̫���
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

	//�����ļ���
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

//��ָ���ļ���������ļ���ӵ���Ӧ��list��
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
		allImageFiles.Add(fileName);	//ͬʱ��ӵ������ļ���
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

//����ļ�
void LCamVidDataset::Init()
{
	//ѵ��������֤��������ѵ����
	AddFolder("Train/", trainImageFiles);
	//	AddFolder("Val/", trainImageFiles);
	AddFolder("Test/", testImageFiles);
}

void LCamVidDataset::SetCRFStructure(LCrf *crf)
{
	//����CFR���壬����һЩ������Ϣ
	LCrfDomain *objDomain = new LCrfDomain(crf, this, classNo, testFolder, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::RgbToLabel, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::LabelToRgb);
	//���������Ϣ��CRFģ��
	crf->domains.Add(objDomain);
	//�����ײ�ģ�ͣ����������ز�CRF
	LBaseCrfLayer *baseLayer = new LBaseCrfLayer(crf, objDomain, this, 0);
	//��ӵײ㵽CRFģ��
	crf->layers.Add(baseLayer);

	//�߽ײ㣬������segment�ĸ߽�CRF
	LPnCrfLayer *superpixelLayer[3];
	//�߽ײ�ĳ����ض�
	LSegmentation2D *segmentation[3];

	//�ֱ𴴽��߽ײ�ģ��
	int i;
	for (i = 0; i < 3; i++)
	{
		//���Ƚ����޼ලsegmentation����ȡ��ͬscale��segment���˴���mean-sift�ָ��㷨
		segmentation[i] = new LMeanShiftSegmentation2D(meanShiftXY[i], meanShiftLuv[i], meanShiftMinRegion[i], meanShiftFolder[i], meanShiftExtension);
		//Ȼ�����segment map�����߽ײ�
		superpixelLayer[i] = new LPnCrfLayer(crf, objDomain, this, baseLayer, segmentation[i], cliqueTruncation);
		//��Ӹ�segment map��CRFģ��
		crf->segmentations.Add(segmentation[i]);
		//��Ӹ߽ײ㵽CRFģ��
		crf->layers.Add(superpixelLayer[i]);
	}

	//������ȡ����������
	//TextonBoost����
	LTextonFeature *textonFeature = new LTextonFeature(this, trainFolder, textonClusteringTrainFile, textonFolder, textonExtension, textonFilterBankRescale, textonKMeansSubSample, textonNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster);
	//λ������
	LLocationFeature *locationFeature = new LLocationFeature(this, locationFolder, locationExtension, locationBuckets);
	//Sift������ʵ��������˵����HOG����
	LSiftFeature *siftFeature = new LSiftFeature(this, trainFolder, siftClusteringTrainFile, siftFolder, siftExtension, siftSizeCount, siftSizes, siftWindowNumber, sift360, siftAngles, siftKMeansSubSample, siftNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster, 1);
	//����LUV��ɫ�ռ��Sift����
	LColourSiftFeature *coloursiftFeature = new LColourSiftFeature(this, trainFolder, colourSiftClusteringTrainFile, colourSiftFolder, colourSiftExtension, colourSiftSizeCount, colourSiftSizes, colourSiftWindowNumber, colourSift360, colourSiftAngles, colourSiftKMeansSubSample, colourSiftNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster, 1);
	//LBP���������������
	LLbpFeature *lbpFeature = new LLbpFeature(this, trainFolder, lbpClusteringFile, lbpFolder, lbpExtension, lbpSize, lbpKMeansSubSample, lbpNumberOfClusters, clusterKMeansMaxChange, clusterPointsPerKDTreeCluster);

	//��Ӹ�������CRFģ��
	crf->features.Add(textonFeature);
	crf->features.Add(locationFeature);
	crf->features.Add(lbpFeature);
	crf->features.Add(siftFeature);
	crf->features.Add(coloursiftFeature);

	//����Dense Feature�������������ĵײ�һԪ�ƺ���
	LDenseUnaryPixelPotential *pixelPotential = new LDenseUnaryPixelPotential(this, crf, objDomain, baseLayer, trainFolder, denseBoostTrainFile, denseFolder, denseExtension, classNo, denseWeight, denseBoostingSubSample, denseNumberOfRectangles, denseMinimumRectangleSize, denseMaximumRectangleSize, denseMaxClassRatio);
	//Dense Feature���õ���4������
	pixelPotential->AddFeature(textonFeature);
	pixelPotential->AddFeature(siftFeature);
	pixelPotential->AddFeature(coloursiftFeature);
	pixelPotential->AddFeature(lbpFeature);
	//��������CRFģ��
	crf->potentials.Add(pixelPotential);
	//�������ؼ������Boosting������
	LBoosting<int> *pixelBoosting = new LBoosting<int>(trainFolder, denseBoostTrainFile, classNo, denseNumRoundsBoosting, denseThetaStart, denseThetaIncrement, denseNumberOfThetas, denseRandomizationFactor, pixelPotential, (int *(LPotential::*)(int, int))&LDenseUnaryPixelPotential::GetTrainBoostingValues, (int *(LPotential::*)(int))&LDenseUnaryPixelPotential::GetEvalBoostingValues);
	//��ӷ�������CRFģ��
	crf->learnings.Add(pixelBoosting);
	//ѧϰ�㷨����ΪBoosting
	pixelPotential->learning = pixelBoosting;		//�������ص�Boosting

	//��������8�����Pottsģ�͵Ķ�Ԫ�ƺ���
	LEightNeighbourPottsPairwisePixelPotential *pairwisePotentianl = new LEightNeighbourPottsPairwisePixelPotential(this, crf, objDomain, baseLayer, classNo, pairwisePrior, pairwiseFactor, pairwiseBeta, pairwiseLWeight, pairwiseUWeight, pairwiseVWeight);
	//���ƽ���CRFģ��
	crf->potentials.Add(pairwisePotentianl);

	//����segment��һԪ�ƺ�����ʹ��ͳ�Ƶķ�����ȡ�������ֲ�
	LStatsUnarySegmentPotential *statsPotential = new LStatsUnarySegmentPotential(this, crf, objDomain, trainFolder, statsTrainFile, statsFolder, statsExtension, classNo, statsPrior, statsFactor, cliqueMinLabelRatio, statsAlpha, statsMaxClassRatio);
	//segment���õ���������segment�������ǻ���dense feature�Ĺ�һ��ֱ��ͼ�����������ֲ����
	statsPotential->AddFeature(textonFeature);
	statsPotential->AddFeature(siftFeature);
	statsPotential->AddFeature(coloursiftFeature);
	statsPotential->AddFeature(locationFeature);
	statsPotential->AddFeature(lbpFeature);
	//��ÿ���߽ײ���ӵ��߽��ƺ���
	for (i = 0; i < 3; i++) statsPotential->AddLayer(superpixelLayer[i]);
	//����segment�����Boosting������
	LBoosting<double> *segmentBoosting = new LBoosting<double>(trainFolder, statsTrainFile, classNo, statsNumberOfBoosts, statsThetaStart, statsThetaIncrement, statsNumberOfThetas, statsRandomizationFactor, statsPotential, (double *(LPotential::*)(int, int))&LStatsUnarySegmentPotential::GetTrainBoostingValues, (double *(LPotential::*)(int))&LStatsUnarySegmentPotential::GetEvalBoostingValues);
	//��ӷ�������CRFģ��
	crf->learnings.Add(segmentBoosting);
	//�߽��ƺ�����ѧϰ����ΪBoosting
	statsPotential->learning = segmentBoosting;		//���ڿ��Boosting
	//��Ӹ߽��ƺ�����CRFģ��
	crf->potentials.Add(statsPotential);
}



/////////////////////////////////////////////////////////////////////////////////////////////
//
//				KITTI	Dataet
//
/////////////////////////////////////////////////////////////////////////////////////////////

//KITTI���ݼ����ֲ�ģ��
LKITTIDataset::LKITTIDataset()
{
	seed = 10000;
	classNo = 2;	//KITTI���ݼ��͵�·�ͷ�·����
	filePermutations = 10000;
	optimizeAverage = 1;
	featuresOnline = 0;

	//���Ȩ�أ���ʼ��Ϊȫ1
	unaryWeighted = 0;
	unaryWeights = new double[classNo];
	for (int i = 0; i < classNo; i++) unaryWeights[i] = 1.0;

	//ѵ�����Ͳ��Լ�����1:1
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

	textonNumberOfClusters = 50;		//50����������
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
	sift360 = 1;	//360��
	siftAngles = 8;		//HOG��8��bin����360�ȵ�8������
	siftFolder = "/home/yangfei/Datasets/Result/KITTI/um/Feature/Sift/";
	siftExtension = ".sft";

	//������ɫ��Ϣ��HOG
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

	locationBuckets = 20;	//λ��������һ��
	locationFolder = "/home/yangfei/Datasets/Result/KITTI/um/Feature/Location/";
	locationExtension = ".loc";

	lbpClusteringFile = "lbpclustering.dat";
	lbpFolder = "/home/yangfei/Datasets/Result/KITTI/um/Feature/Lbp/";
	lbpExtension = ".lbp";
	lbpSize = 11;
	lbpKMeansSubSample = 20;
	lbpNumberOfClusters = 50;

	//��һ��
	meanShiftXY[0] = 5.0;
	meanShiftLuv[0] = 3.0;
	meanShiftMinRegion[0] = 20;
	meanShiftFolder[0] = "/home/yangfei/Datasets/Result/KITTI/um/MeanShift/50x30/";
	//��һ��
	meanShiftXY[1] = 5.0;
	meanShiftLuv[1] = 5.0;
	meanShiftMinRegion[1] = 20;
	meanShiftFolder[1] = "/home/yangfei/Datasets/Result/KITTI/um/MeanShift/50x50/";
	//�ڶ���
	meanShiftXY[2] = 7.0;
	meanShiftLuv[2] = 5.0;
	meanShiftMinRegion[2] = 20;
	meanShiftFolder[2] = "/home/yangfei/Datasets/Result/KITTI/um/MeanShift/70x50/";
	//�ڶ���
	meanShiftXY[3] = 7.0;
	meanShiftLuv[3] = 6.0;
	meanShiftMinRegion[3] = 20;
	meanShiftFolder[3] = "/home/yangfei/Datasets/Result/KITTI/um/MeanShift/70x60/";

	meanShiftExtension = ".msh";


	//Boosting����
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
	//���������
	denseMaxClassRatio = 0.2;

	//ƽ��������ͨ����Ȩ��
	pairwiseLWeight = 1.0 / 3.0;
	pairwiseUWeight = 1.0 / 3.0;
	pairwiseVWeight = 1.0 / 3.0;
	pairwisePrior = 1.5;		//1.5->0
	pairwiseFactor = 6.0;		//6.0->20
	pairwiseBeta = 16.0;

	//������ǩ����
	cliqueMinLabelRatio = 0.5;
	//��ֵ
	cliqueThresholdRatio = 0.1;
	//�ضϲ���
	cliqueTruncation = 0.1;		//0.1->0.5

	//Boosting����
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

	//һ����������
	consistencyPrior = 0.05;
	//�߽�ƽ�������
	pairwiseSegmentBuckets = 8;
	pairwiseSegmentPrior = 0.0;		//0.0��>1.5
	pairwiseSegmentFactor = 2.0;
	pairwiseSegmentBeta = 40.0;

	//��ʼ��������˳�򣬰���������ѵ�����Ͳ��Լ�
	Init();

	//����Ŀ¼
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
		allImageFiles.Add(fileName);	//ͬʱ��ӵ������ļ���
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
	//ѵ��������֤��������ѵ����
	AddFolder("Train/", trainImageFiles);
	AddFolder("Test/", testImageFiles);
}
void LKITTIDataset::RgbToLabel(unsigned char *rgb, unsigned char *label)
{
	if (rgb[2]==255)		//rͨ��������Ч��
	{
		if (rgb[0]==255)	//bͨ�������·����
		{
			label[0] = 1;		//��·
		}
		else	
			label[0] = 2;		//��·
	}
	else	
		label[0] = 0;		//��Ч
	
}
void LKITTIDataset::LabelToRgb(unsigned char *label, unsigned char *rgb)
{
	rgb[0] = rgb[1] = rgb[2] = 0;
	if (label[0]==2)
	{
		rgb[2] = 255;		//��·��ɫ
	}
	else
	{
		if (label[0]==1)
		{
			rgb[2] = 255, rgb[0] = 255;		//��·��ɫ
		}
	}	
}
void LKITTIDataset::SetCRFStructure(LCrf *crf)
{
	//����CRF�������
	LCrfDomain *objDomain = new LCrfDomain(crf, this, classNo, testFolder, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::RgbToLabel, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::LabelToRgb);
	//������嵽ģ��
	crf->domains.Add(objDomain);

	//�����ײ�
	LBaseCrfLayer *baseLayer = new LBaseCrfLayer(crf, objDomain, this, 0);
	//��ӵײ㵽ģ��
	crf->layers.Add(baseLayer);

	//�����߲�ͷָ�
	LPnCrfLayer *superpixelLayer[3];
	LSegmentation2D *segmentation[3];


	//baselayer
	segmentation[0] = new LMeanShiftSegmentation2D(meanShiftXY[1], meanShiftLuv[1], meanShiftMinRegion[1], meanShiftFolder[1], meanShiftExtension);
	//�����ڵײ��ϵķָ���ø���Ϊbase���Լ������ز�
	superpixelLayer[0] = new LPnCrfLayer(crf, objDomain, this, baseLayer, segmentation[0], cliqueTruncation);
	crf->segmentations.Add(segmentation[0]);		//��ӷָcrfģ��
	crf->layers.Add(superpixelLayer[0]);		//��Ӹ߽ײ㵽crfģ��

	int i;
	//meansSift�ָ�
	//for (i = 1; i < 2; i++)		//2~3layer
	//{
	//	//��ͬ�߶ȵľ������
	//	segmentation[i] = new LMeanShiftSegmentation2D(meanShiftXY[i], meanShiftLuv[i], meanShiftMinRegion[i], meanShiftFolder[i], meanShiftExtension);
	//	//�����ڵ�һ�㳬���ز��ϵĲ�
	//	superpixelLayer[i] = new LPnCrfLayer(crf, objDomain, this, superpixelLayer[0], segmentation[i], cliqueTruncation);
	//	crf->segmentations.Add(segmentation[i]);
	//	crf->layers.Add(superpixelLayer[i]);
	//}

	//�����������
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

	//һ��������
	LDenseUnaryPixelPotential *pixelPotential = new LDenseUnaryPixelPotential(this, crf, objDomain, baseLayer, trainFolder, denseBoostTrainFile, denseFolder, denseExtension, classNo, denseWeight, denseBoostingSubSample, denseNumberOfRectangles, denseMinimumRectangleSize, denseMaximumRectangleSize, denseMaxClassRatio);
	pixelPotential->AddFeature(textonFeature);
	pixelPotential->AddFeature(siftFeature);
	pixelPotential->AddFeature(coloursiftFeature);
	pixelPotential->AddFeature(lbpFeature);
	crf->potentials.Add(pixelPotential);

	LBoosting<int> *pixelBoosting = new LBoosting<int>(trainFolder, denseBoostTrainFile, classNo, denseNumRoundsBoosting, denseThetaStart, denseThetaIncrement, denseNumberOfThetas, denseRandomizationFactor, pixelPotential, (int *(LPotential::*)(int, int))&LDenseUnaryPixelPotential::GetTrainBoostingValues, (int *(LPotential::*)(int))&LDenseUnaryPixelPotential::GetEvalBoostingValues);
	crf->learnings.Add(pixelBoosting);
	pixelPotential->learning = pixelBoosting;

	//һ��ƽ����
	crf->potentials.Add(new LEightNeighbourPottsPairwisePixelPotential(this, crf, objDomain, baseLayer, classNo, pairwisePrior, pairwiseFactor, pairwiseBeta, pairwiseLWeight, pairwiseUWeight, pairwiseVWeight));

	
	//�߽�������
	LStatsUnarySegmentPotential *statsPotential = new LStatsUnarySegmentPotential(this, crf, objDomain, trainFolder, statsTrainFile, statsFolder, statsExtension, classNo, statsPrior, statsFactor, cliqueMinLabelRatio, statsAlpha, statsMaxClassRatio);
	statsPotential->AddFeature(textonFeature);
	statsPotential->AddFeature(siftFeature);
	statsPotential->AddFeature(coloursiftFeature);
	statsPotential->AddFeature(locationFeature);
	statsPotential->AddFeature(lbpFeature);
	
	//��ӳ����ز�
	for (i = 0; i < 1; i++) statsPotential->AddLayer(superpixelLayer[i]);

	crf->potentials.Add(statsPotential);

	LBoosting<double> *segmentBoosting = new LBoosting<double>(trainFolder, statsTrainFile, classNo, statsNumberOfBoosts, statsThetaStart, statsThetaIncrement, statsNumberOfThetas, statsRandomizationFactor, statsPotential, (double *(LPotential::*)(int, int))&LStatsUnarySegmentPotential::GetTrainBoostingValues, (double *(LPotential::*)(int))&LStatsUnarySegmentPotential::GetEvalBoostingValues);
	crf->learnings.Add(segmentBoosting);
	statsPotential->learning = segmentBoosting;
	
	//�߽�ƽ����
	crf->potentials.Add(new LHistogramPottsPairwiseSegmentPotential(this, crf, objDomain, superpixelLayer[0], classNo, pairwiseSegmentPrior, pairwiseSegmentFactor, pairwiseSegmentBeta, pairwiseSegmentBuckets));

	//һ�������������
	consistencyPrior = 100000;
	//һ���Ը߽��������������һ���Ե�
	LConsistencyUnarySegmentPotential *consistencyPotential = new LConsistencyUnarySegmentPotential(this, crf, objDomain, classNo, consistencyPrior);
	//���ڵ�һ�������ز�
	consistencyPotential->AddLayer(superpixelLayer[0]);
	crf->potentials.Add(consistencyPotential);

	//�������ʱ�����
	LPreferenceCrfLayer *preferenceLayer = new LPreferenceCrfLayer(crf, objDomain, this, baseLayer);
	crf->layers.Add(preferenceLayer);

}



//����Ϊ�ں�ģ��

//KITTI���ݼ�um
LKITTIumDataset::LKITTIumDataset()
{
	seed = 10000;
	classNo = 2;	//KITTI���ݼ��͵�·�ͷ�·����
	filePermutations = 10000;
	optimizeAverage = 1;
	featuresOnline = 0;

	//���Ȩ�أ���ʼ��Ϊȫ1
	unaryWeighted = 0;
	unaryWeights = new double[classNo];
	for (int i = 0; i < classNo; i++) unaryWeights[i] = 1.0;

	//ѵ�����Ͳ��Լ�����1:1
	proportionTrain = 0.5;
	proportionTest = 0.5;

	datasetName = "um";			//���ݼ���
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

	textonNumberOfClusters = 50;		//50����������
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
	sift360 = 1;	//360��
	siftAngles = 8;		//HOG��8��bin����360�ȵ�8������
	siftFolder = "/home/yangfei/Datasets/Result/KITTI/um/Feature/Sift/";
	siftExtension = ".sft";

	//������ɫ��Ϣ��HOG
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

	locationBuckets = 20;	//λ��������һ��
	locationFolder = "/home/yangfei/Datasets/Result/KITTI/um/Feature/Location/";
	locationExtension = ".loc";

	lbpClusteringFile = "lbpclustering.dat";
	lbpFolder = "/home/yangfei/Datasets/Result/KITTI/um/Feature/Lbp/";
	lbpExtension = ".lbp";
	lbpSize = 11;
	lbpKMeansSubSample = 20;
	lbpNumberOfClusters = 50;

	//��һ��
	meanShiftXY[0] = 5.0;
	meanShiftLuv[0] = 3.0;
	meanShiftMinRegion[0] = 20;
	meanShiftFolder[0] = "/home/yangfei/Datasets/Result/KITTI/um/MeanShift/50x30/";
	//��һ��
	meanShiftXY[1] = 5.0;
	meanShiftLuv[1] = 5.0;
	meanShiftMinRegion[1] = 20;
	meanShiftFolder[1] = "/home/yangfei/Datasets/Result/KITTI/um/MeanShift/50x50/";
	//�ڶ���
	meanShiftXY[2] = 7.0;
	meanShiftLuv[2] = 5.0;
	meanShiftMinRegion[2] = 20;
	meanShiftFolder[2] = "/home/yangfei/Datasets/Result/KITTI/um/MeanShift/70x50/";
	//�ڶ���
	meanShiftXY[3] = 7.0;
	meanShiftLuv[3] = 6.0;
	meanShiftMinRegion[3] = 20;
	meanShiftFolder[3] = "/home/yangfei/Datasets/Result/KITTI/um/MeanShift/70x60/";

	meanShiftExtension = ".msh";


	//Boosting����
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
							//���������
	denseMaxClassRatio = 0.2;

	//ƽ��������ͨ����Ȩ��
	pairwiseLWeight = 1.0 / 3.0;
	pairwiseUWeight = 1.0 / 3.0;
	pairwiseVWeight = 1.0 / 3.0;
	pairwisePrior = 1.5;		//1.5->0
	pairwiseFactor = 6.0;		//6.0->20
	pairwiseBeta = 16.0;

	//������ǩ����
	cliqueMinLabelRatio = 0.5;
	//��ֵ
	cliqueThresholdRatio = 0.1;
	//�ضϲ���
	cliqueTruncation = 0.1;		//0.1->0.5

								//Boosting����
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

	//һ����������
	consistencyPrior = 0.05;
	//�߽�ƽ�������
	pairwiseSegmentBuckets = 8;
	pairwiseSegmentPrior = 0.0;		//0.0��>1.5
	pairwiseSegmentFactor = 2.0;
	pairwiseSegmentBeta = 40.0;

	//3D�״�
	//���Խ������·��
	lidarTestFolder = "/home/yangfei/Datasets/Result/KITTI/um/LidarCrf/";
	//���GTͼ
	lidarGroundTruthFolder = "/home/yangfei/Datasets/Data/KITTI/um/GroundTruth/";		//�״�ı�ǩ����ͼ��һ��
	lidarGroundTruthExtension = ".png";
	lidarUnaryFactor = 5.0;//0.7;  //0.1->0.5

	//�Ӳ����

	lidarPairwiseFactor = 1.0;//0.0005;		//0.00005->0.0005
	//ƽ���ضϸ���
	lidarPairwiseTruncation = 10.0;
	//�Ӳ����,�ܴ�
	lidarClassNo = 2;


	//ģ�Ͳ�����ʼ��
	crossUnaryWeight = 0.5;
	crossPairwiseWeight = 0.2;//-1e-4;
	crossThreshold = 1e-6;		
	//�����Ҫѵ����
	crossTrainFile = "height.dat";

	//��ʼ��������˳�򣬰���������ѵ�����Ͳ��Լ�
	Init();

	//����Ŀ¼
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
		allImageFiles.Add(fileName);	//ͬʱ��ӵ������ļ���
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
	//ѵ��������֤��������ѵ����
	AddFolder("Train/", trainImageFiles);
	AddFolder("Test/", testImageFiles);
}
void LKITTIumDataset::RgbToLabel(unsigned char *rgb, unsigned char *label)
{
	if (rgb[2] == 255)		//rͨ��������Ч��
	{
		if (rgb[0] == 255)	//bͨ�������·����
		{
			label[0] = 1;		//��·
		}
		else
			label[0] = 2;		//��·
	}
	else
		label[0] = 0;		//��Ч

}
void LKITTIumDataset::LabelToRgb(unsigned char *label, unsigned char *rgb)
{
	rgb[0] = rgb[1] = rgb[2] = 0;
	if (label[0] == 2)
	{
		rgb[2] = 255;		//��·��ɫ
	}
	else
	{
		if (label[0] == 1)
		{
			rgb[2] = 255, rgb[0] = 255;		//��·��ɫ
		}
	}
}
void LKITTIumDataset::SetCRFStructure(LCrf *crf)
{
	//����CRF�������
	LCrfDomain *objDomain = new LCrfDomain(crf, this, classNo, testFolder, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::RgbToLabel, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::LabelToRgb);
	//������嵽ģ��
	crf->domains.Add(objDomain);

	//�����ײ�
	LBaseCrfLayer *baseLayer = new LBaseCrfLayer(crf, objDomain, this, 0);
	//��ӵײ㵽ģ��
	crf->layers.Add(baseLayer);

	//�����߲�ͷָ�
	LPnCrfLayer *superpixelLayer[3];
	LSegmentation2D *segmentation[3];


	//baselayer
	segmentation[0] = new LMeanShiftSegmentation2D(meanShiftXY[1], meanShiftLuv[1], meanShiftMinRegion[1], meanShiftFolder[1], meanShiftExtension);
	//�����ڵײ��ϵķָ���ø���Ϊbase���Լ������ز�
	superpixelLayer[0] = new LPnCrfLayer(crf, objDomain, this, baseLayer, segmentation[0], cliqueTruncation);
	crf->segmentations.Add(segmentation[0]);		//��ӷָcrfģ��
	crf->layers.Add(superpixelLayer[0]);		//��Ӹ߽ײ㵽crfģ��


	//�����������
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

	//һ��������
	LDenseUnaryPixelPotential *pixelPotential = new LDenseUnaryPixelPotential(this, crf, objDomain, baseLayer, trainFolder, denseBoostTrainFile, denseFolder, denseExtension, classNo, denseWeight, denseBoostingSubSample, denseNumberOfRectangles, denseMinimumRectangleSize, denseMaximumRectangleSize, denseMaxClassRatio);
	pixelPotential->AddFeature(textonFeature);
	pixelPotential->AddFeature(siftFeature);
	pixelPotential->AddFeature(coloursiftFeature);
	pixelPotential->AddFeature(lbpFeature);
	crf->potentials.Add(pixelPotential);

	LBoosting<int> *pixelBoosting = new LBoosting<int>(trainFolder, denseBoostTrainFile, classNo, denseNumRoundsBoosting, denseThetaStart, denseThetaIncrement, denseNumberOfThetas, denseRandomizationFactor, pixelPotential, (int *(LPotential::*)(int, int))&LDenseUnaryPixelPotential::GetTrainBoostingValues, (int *(LPotential::*)(int))&LDenseUnaryPixelPotential::GetEvalBoostingValues);
	crf->learnings.Add(pixelBoosting);
	pixelPotential->learning = pixelBoosting;


	//�߽�������
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


	//�Ӳ���״�
	LCrfDomain *lidarDomain = new LCrfDomain(crf, this, classNo, lidarTestFolder, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::RgbToLabel, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::LabelToRgb);
	crf->domains.Add(lidarDomain);
	//�Ӳ�ײ�
	LBaseCrfLayer *lidarBaseLayer = new LBaseCrfLayer(crf, lidarDomain, this, 0);
	crf->layers.Add(lidarBaseLayer);

	//�Ӳ�������(�ѽ��)
	LLidarUnaryPixelPotential *pixelLidarPotential = new LLidarUnaryPixelPotential(this, crf, lidarDomain, lidarBaseLayer, lidarClassNo, lidarUnaryFactor);
	crf->potentials.Add(pixelLidarPotential);

	//��������ģ�͵Ļ���Ŀ��߶ȷֲ���������
	LHeightUnaryPixelPotential *heightPotential = new LHeightUnaryPixelPotential(this, crf, objDomain, baseLayer, lidarDomain, lidarBaseLayer, trainFolder, crossTrainFile, classNo, crossUnaryWeight, lidarClassNo, crossThreshold);
	crf->potentials.Add(heightPotential);

	//����ƽ����
	crf->potentials.Add(new LJointPairwisePixelPotential(this, crf, objDomain, baseLayer, lidarDomain, lidarBaseLayer, classNo, lidarClassNo, pairwisePrior, pairwiseFactor, pairwiseBeta, pairwiseLWeight, pairwiseUWeight, pairwiseVWeight, lidarPairwiseFactor, lidarPairwiseTruncation, crossPairwiseWeight));

}


//KITTI���ݼ�umm
LKITTIummDataset::LKITTIummDataset()
{
	seed = 10000;
	classNo = 2;	//KITTI���ݼ��͵�·�ͷ�·����
	filePermutations = 10000;
	optimizeAverage = 1;
	featuresOnline = 0;

	//���Ȩ�أ���ʼ��Ϊȫ1
	unaryWeighted = 0;
	unaryWeights = new double[classNo];
	for (int i = 0; i < classNo; i++) unaryWeights[i] = 1.0;

	//ѵ�����Ͳ��Լ�����1:1
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

	textonNumberOfClusters = 50;		//50����������
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
	sift360 = 1;	//360��
	siftAngles = 8;		//HOG��8��bin����360�ȵ�8������
	siftFolder = "/home/yangfei/Datasets/Result/KITTI/umm/Feature/Sift/";
	siftExtension = ".sft";

	//������ɫ��Ϣ��HOG
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

	locationBuckets = 20;	//λ��������һ��
	locationFolder = "/home/yangfei/Datasets/Result/KITTI/umm/Feature/Location/";
	locationExtension = ".loc";

	lbpClusteringFile = "lbpclustering.dat";
	lbpFolder = "/home/yangfei/Datasets/Result/KITTI/umm/Feature/Lbp/";
	lbpExtension = ".lbp";
	lbpSize = 11;
	lbpKMeansSubSample = 20;
	lbpNumberOfClusters = 50;

	//��һ��
	meanShiftXY[0] = 5.0;
	meanShiftLuv[0] = 3.0;
	meanShiftMinRegion[0] = 20;
	meanShiftFolder[0] = "/home/yangfei/Datasets/Result/KITTI/umm/MeanShift/50x30/";
	//��һ��
	meanShiftXY[1] = 5.0;
	meanShiftLuv[1] = 5.0;
	meanShiftMinRegion[1] = 20;
	meanShiftFolder[1] = "/home/yangfei/Datasets/Result/KITTI/umm/MeanShift/50x50/";
	//�ڶ���
	meanShiftXY[2] = 7.0;
	meanShiftLuv[2] = 5.0;
	meanShiftMinRegion[2] = 20;
	meanShiftFolder[2] = "/home/yangfei/Datasets/Result/KITTI/umm/MeanShift/70x50/";
	//�ڶ���
	meanShiftXY[3] = 7.0;
	meanShiftLuv[3] = 6.0;
	meanShiftMinRegion[3] = 20;
	meanShiftFolder[3] = "/home/yangfei/Datasets/Result/KITTI/umm/MeanShift/70x60/";

	meanShiftExtension = ".msh";


	//Boosting����
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
							//���������
	denseMaxClassRatio = 0.2;

	//ƽ��������ͨ����Ȩ��
	pairwiseLWeight = 1.0 / 3.0;
	pairwiseUWeight = 1.0 / 3.0;
	pairwiseVWeight = 1.0 / 3.0;
	pairwisePrior = 1.5;		//1.5->0
	pairwiseFactor = 6.0;		//6.0->20
	pairwiseBeta = 16.0;

	//������ǩ����
	cliqueMinLabelRatio = 0.5;
	//��ֵ
	cliqueThresholdRatio = 0.1;
	//�ضϲ���
	cliqueTruncation = 0.1;		//0.1->0.5

								//Boosting����
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

	//һ����������
	consistencyPrior = 0.05;
	//�߽�ƽ�������
	pairwiseSegmentBuckets = 8;
	pairwiseSegmentPrior = 0.0;		//0.0��>1.5
	pairwiseSegmentFactor = 2.0;
	pairwiseSegmentBeta = 40.0;

	//3D�״�
	//���Խ������·��
	lidarTestFolder = "/home/yangfei/Datasets/Result/KITTI/umm/LidarCrf/";
	//���GTͼ
	lidarGroundTruthFolder = "/home/yangfei/Datasets/Data/KITTI/umm/GroundTruth/";		//�״�ı�ǩ����ͼ��һ��
	lidarGroundTruthExtension = ".png";
	lidarUnaryFactor = 5.0;//0.7;  //0.1->0.5

							 //�Ӳ����

	lidarPairwiseFactor = 1.0;//0.0005;		//0.00005->0.0005
										//ƽ���ضϸ���
	lidarPairwiseTruncation = 10.0;
	//�Ӳ����,�ܴ�
	lidarClassNo = 2;


	//ģ�Ͳ�����ʼ��
	crossUnaryWeight = 0.5;
	crossPairwiseWeight = 0.2;//-1e-4;
	crossThreshold = 1e-6;
	//�����Ҫѵ����
	crossTrainFile = "height.dat";

	//��ʼ��������˳�򣬰���������ѵ�����Ͳ��Լ�
	Init();

	//����Ŀ¼
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
		allImageFiles.Add(fileName);	//ͬʱ��ӵ������ļ���
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
	//ѵ��������֤��������ѵ����
	AddFolder("Train/", trainImageFiles);
	AddFolder("Test/", testImageFiles);
}
void LKITTIummDataset::RgbToLabel(unsigned char *rgb, unsigned char *label)
{
	if (rgb[2] == 255)		//rͨ��������Ч��
	{
		if (rgb[0] == 255)	//bͨ�������·����
		{
			label[0] = 1;		//��·
		}
		else
			label[0] = 2;		//��·
	}
	else
		label[0] = 0;		//��Ч

}
void LKITTIummDataset::LabelToRgb(unsigned char *label, unsigned char *rgb)
{
	rgb[0] = rgb[1] = rgb[2] = 0;
	if (label[0] == 2)
	{
		rgb[2] = 255;		//��·��ɫ
	}
	else
	{
		if (label[0] == 1)
		{
			rgb[2] = 255, rgb[0] = 255;		//��·��ɫ
		}
	}
}
void LKITTIummDataset::SetCRFStructure(LCrf *crf)
{
	//����CRF�������
	LCrfDomain *objDomain = new LCrfDomain(crf, this, classNo, testFolder, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::RgbToLabel, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::LabelToRgb);
	//������嵽ģ��
	crf->domains.Add(objDomain);

	//�����ײ�
	LBaseCrfLayer *baseLayer = new LBaseCrfLayer(crf, objDomain, this, 0);
	//��ӵײ㵽ģ��
	crf->layers.Add(baseLayer);

	//�����߲�ͷָ�
	LPnCrfLayer *superpixelLayer[3];
	LSegmentation2D *segmentation[3];


	//baselayer
	segmentation[0] = new LMeanShiftSegmentation2D(meanShiftXY[1], meanShiftLuv[1], meanShiftMinRegion[1], meanShiftFolder[1], meanShiftExtension);
	//�����ڵײ��ϵķָ���ø���Ϊbase���Լ������ز�
	superpixelLayer[0] = new LPnCrfLayer(crf, objDomain, this, baseLayer, segmentation[0], cliqueTruncation);
	crf->segmentations.Add(segmentation[0]);		//��ӷָcrfģ��
	crf->layers.Add(superpixelLayer[0]);		//��Ӹ߽ײ㵽crfģ��


												//�����������
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

	//һ��������
	LDenseUnaryPixelPotential *pixelPotential = new LDenseUnaryPixelPotential(this, crf, objDomain, baseLayer, trainFolder, denseBoostTrainFile, denseFolder, denseExtension, classNo, denseWeight, denseBoostingSubSample, denseNumberOfRectangles, denseMinimumRectangleSize, denseMaximumRectangleSize, denseMaxClassRatio);
	pixelPotential->AddFeature(textonFeature);
	pixelPotential->AddFeature(siftFeature);
	pixelPotential->AddFeature(coloursiftFeature);
	pixelPotential->AddFeature(lbpFeature);
	crf->potentials.Add(pixelPotential);

	LBoosting<int> *pixelBoosting = new LBoosting<int>(trainFolder, denseBoostTrainFile, classNo, denseNumRoundsBoosting, denseThetaStart, denseThetaIncrement, denseNumberOfThetas, denseRandomizationFactor, pixelPotential, (int *(LPotential::*)(int, int))&LDenseUnaryPixelPotential::GetTrainBoostingValues, (int *(LPotential::*)(int))&LDenseUnaryPixelPotential::GetEvalBoostingValues);
	crf->learnings.Add(pixelBoosting);
	pixelPotential->learning = pixelBoosting;


	//�߽�������
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


	//�Ӳ���״�
	LCrfDomain *lidarDomain = new LCrfDomain(crf, this, classNo, lidarTestFolder, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::RgbToLabel, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::LabelToRgb);
	crf->domains.Add(lidarDomain);
	//�Ӳ�ײ�
	LBaseCrfLayer *lidarBaseLayer = new LBaseCrfLayer(crf, lidarDomain, this, 0);
	crf->layers.Add(lidarBaseLayer);

	//�Ӳ�������(�ѽ��)
	LLidarUnaryPixelPotential *pixelLidarPotential = new LLidarUnaryPixelPotential(this, crf, lidarDomain, lidarBaseLayer, lidarClassNo, lidarUnaryFactor);
	crf->potentials.Add(pixelLidarPotential);

	//��������ģ�͵Ļ���Ŀ��߶ȷֲ���������
	LHeightUnaryPixelPotential *heightPotential = new LHeightUnaryPixelPotential(this, crf, objDomain, baseLayer, lidarDomain, lidarBaseLayer, trainFolder, crossTrainFile, classNo, crossUnaryWeight, lidarClassNo, crossThreshold);
	crf->potentials.Add(heightPotential);

	//����ƽ����
	crf->potentials.Add(new LJointPairwisePixelPotential(this, crf, objDomain, baseLayer, lidarDomain, lidarBaseLayer, classNo, lidarClassNo, pairwisePrior, pairwiseFactor, pairwiseBeta, pairwiseLWeight, pairwiseUWeight, pairwiseVWeight, lidarPairwiseFactor, lidarPairwiseTruncation, crossPairwiseWeight));

}


//KITTI���ݼ�uu
LKITTIuuDataset::LKITTIuuDataset()
{
	seed = 10000;
	classNo = 2;	//KITTI���ݼ��͵�·�ͷ�·����
	filePermutations = 10000;
	optimizeAverage = 1;
	featuresOnline = 0;

	//���Ȩ�أ���ʼ��Ϊȫ1
	unaryWeighted = 0;
	unaryWeights = new double[classNo];
	for (int i = 0; i < classNo; i++) unaryWeights[i] = 1.0;

	//ѵ�����Ͳ��Լ�����1:1
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

	textonNumberOfClusters = 50;		//50����������
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
	sift360 = 1;	//360��
	siftAngles = 8;		//HOG��8��bin����360�ȵ�8������
	siftFolder = "/home/yangfei/Datasets/Result/KITTI/uu/Feature/Sift/";
	siftExtension = ".sft";

	//������ɫ��Ϣ��HOG
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

	locationBuckets = 20;	//λ��������һ��
	locationFolder = "/home/yangfei/Datasets/Result/KITTI/uu/Feature/Location/";
	locationExtension = ".loc";

	lbpClusteringFile = "lbpclustering.dat";
	lbpFolder = "/home/yangfei/Datasets/Result/KITTI/uu/Feature/Lbp/";
	lbpExtension = ".lbp";
	lbpSize = 11;
	lbpKMeansSubSample = 20;
	lbpNumberOfClusters = 50;

	//��һ��
	meanShiftXY[0] = 5.0;
	meanShiftLuv[0] = 3.0;
	meanShiftMinRegion[0] = 20;
	meanShiftFolder[0] = "/home/yangfei/Datasets/Result/KITTI/uu/MeanShift/50x30/";
	//��һ��
	meanShiftXY[1] = 5.0;
	meanShiftLuv[1] = 5.0;
	meanShiftMinRegion[1] = 20;
	meanShiftFolder[1] = "/home/yangfei/Datasets/Result/KITTI/uu/MeanShift/50x50/";
	//�ڶ���
	meanShiftXY[2] = 7.0;
	meanShiftLuv[2] = 5.0;
	meanShiftMinRegion[2] = 20;
	meanShiftFolder[2] = "/home/yangfei/Datasets/Result/KITTI/uu/MeanShift/70x50/";
	//�ڶ���
	meanShiftXY[3] = 7.0;
	meanShiftLuv[3] = 6.0;
	meanShiftMinRegion[3] = 20;
	meanShiftFolder[3] = "/home/yangfei/Datasets/Result/KITTI/uu/MeanShift/70x60/";

	meanShiftExtension = ".msh";


	//Boosting����
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
							//���������
	denseMaxClassRatio = 0.2;

	//ƽ��������ͨ����Ȩ��
	pairwiseLWeight = 1.0 / 3.0;
	pairwiseUWeight = 1.0 / 3.0;
	pairwiseVWeight = 1.0 / 3.0;
	pairwisePrior = 1.5;		//1.5->0
	pairwiseFactor = 6.0;		//6.0->20
	pairwiseBeta = 16.0;

	//������ǩ����
	cliqueMinLabelRatio = 0.5;
	//��ֵ
	cliqueThresholdRatio = 0.1;
	//�ضϲ���
	cliqueTruncation = 0.1;		//0.1->0.5

								//Boosting����
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

	//һ����������
	consistencyPrior = 0.05;
	//�߽�ƽ�������
	pairwiseSegmentBuckets = 8;
	pairwiseSegmentPrior = 0.0;		//0.0��>1.5
	pairwiseSegmentFactor = 2.0;
	pairwiseSegmentBeta = 40.0;

	//3D�״�
	//���Խ������·��
	lidarTestFolder = "/home/yangfei/Datasets/Result/KITTI/uu/LidarCrf/";
	//�״�GTͼ
	lidarGroundTruthFolder = "/home/yangfei/Datasets/Data/KITTI/uu/GroundTruth/";		//�״�ı�ǩ����ͼ��һ��
	lidarGroundTruthExtension = ".png";
	lidarUnaryFactor = 5.0;//0.7;  //0.1->0.5
						

	lidarPairwiseFactor = 1.0;//0.0005;		//0.00005->0.0005
										//ƽ���ضϸ���
	lidarPairwiseTruncation = 10.0;
	//�Ӳ����,�ܴ�
	lidarClassNo = 2;


	//ģ�Ͳ�����ʼ��
	crossUnaryWeight = 0.5;
	crossPairwiseWeight = 0.2;//-1e-4;
	crossThreshold = 1e-6;
	//�����Ҫѵ����
	crossTrainFile = "height.dat";

	//��ʼ��������˳�򣬰���������ѵ�����Ͳ��Լ�
	Init();

	//����Ŀ¼
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
		allImageFiles.Add(fileName);	//ͬʱ��ӵ������ļ���
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
	//ѵ��������֤��������ѵ����
	AddFolder("Train/", trainImageFiles);
	AddFolder("Test/", testImageFiles);
}
void LKITTIuuDataset::RgbToLabel(unsigned char *rgb, unsigned char *label)
{
	if (rgb[2] == 255)		//rͨ��������Ч��
	{
		if (rgb[0] == 255)	//bͨ�������·����
		{
			label[0] = 1;		//��·
		}
		else
			label[0] = 2;		//��·
	}
	else
		label[0] = 0;		//��Ч

}
void LKITTIuuDataset::LabelToRgb(unsigned char *label, unsigned char *rgb)
{
	rgb[0] = rgb[1] = rgb[2] = 0;
	if (label[0] == 2)
	{
		rgb[2] = 255;		//��·��ɫ
	}
	else
	{
		if (label[0] == 1)
		{
			rgb[2] = 255, rgb[0] = 255;		//��·��ɫ
		}
	}
}
void LKITTIuuDataset::SetCRFStructure(LCrf *crf)
{
	//����CRF�������
	LCrfDomain *objDomain = new LCrfDomain(crf, this, classNo, testFolder, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::RgbToLabel, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::LabelToRgb);
	//������嵽ģ��
	crf->domains.Add(objDomain);

	//�����ײ�
	LBaseCrfLayer *baseLayer = new LBaseCrfLayer(crf, objDomain, this, 0);
	//��ӵײ㵽ģ��
	crf->layers.Add(baseLayer);

	//�����߲�ͷָ�
	LPnCrfLayer *superpixelLayer[3];
	LSegmentation2D *segmentation[3];


	//baselayer
	segmentation[0] = new LMeanShiftSegmentation2D(meanShiftXY[1], meanShiftLuv[1], meanShiftMinRegion[1], meanShiftFolder[1], meanShiftExtension);
	//�����ڵײ��ϵķָ���ø���Ϊbase���Լ������ز�
	superpixelLayer[0] = new LPnCrfLayer(crf, objDomain, this, baseLayer, segmentation[0], cliqueTruncation);
	crf->segmentations.Add(segmentation[0]);		//��ӷָcrfģ��
	crf->layers.Add(superpixelLayer[0]);		//��Ӹ߽ײ㵽crfģ��


												//�����������
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

	//һ��������
	LDenseUnaryPixelPotential *pixelPotential = new LDenseUnaryPixelPotential(this, crf, objDomain, baseLayer, trainFolder, denseBoostTrainFile, denseFolder, denseExtension, classNo, denseWeight, denseBoostingSubSample, denseNumberOfRectangles, denseMinimumRectangleSize, denseMaximumRectangleSize, denseMaxClassRatio);
	pixelPotential->AddFeature(textonFeature);
	pixelPotential->AddFeature(siftFeature);
	pixelPotential->AddFeature(coloursiftFeature);
	pixelPotential->AddFeature(lbpFeature);
	crf->potentials.Add(pixelPotential);

	LBoosting<int> *pixelBoosting = new LBoosting<int>(trainFolder, denseBoostTrainFile, classNo, denseNumRoundsBoosting, denseThetaStart, denseThetaIncrement, denseNumberOfThetas, denseRandomizationFactor, pixelPotential, (int *(LPotential::*)(int, int))&LDenseUnaryPixelPotential::GetTrainBoostingValues, (int *(LPotential::*)(int))&LDenseUnaryPixelPotential::GetEvalBoostingValues);
	crf->learnings.Add(pixelBoosting);
	pixelPotential->learning = pixelBoosting;


	//�߽�������
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


	//�Ӳ���״�
	LCrfDomain *lidarDomain = new LCrfDomain(crf, this, classNo, lidarTestFolder, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::RgbToLabel, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::LabelToRgb);
	crf->domains.Add(lidarDomain);
	//�Ӳ�ײ�
	LBaseCrfLayer *lidarBaseLayer = new LBaseCrfLayer(crf, lidarDomain, this, 0);
	crf->layers.Add(lidarBaseLayer);

	//�Ӳ�������(�ѽ��)
	LLidarUnaryPixelPotential *pixelLidarPotential = new LLidarUnaryPixelPotential(this, crf, lidarDomain, lidarBaseLayer, lidarClassNo, lidarUnaryFactor);
	crf->potentials.Add(pixelLidarPotential);

	//��������ģ�͵Ļ���Ŀ��߶ȷֲ���������
	LHeightUnaryPixelPotential *heightPotential = new LHeightUnaryPixelPotential(this, crf, objDomain, baseLayer, lidarDomain, lidarBaseLayer, trainFolder, crossTrainFile, classNo, crossUnaryWeight, lidarClassNo, crossThreshold);
	crf->potentials.Add(heightPotential);

	//����ƽ����
	crf->potentials.Add(new LJointPairwisePixelPotential(this, crf, objDomain, baseLayer, lidarDomain, lidarBaseLayer, classNo, lidarClassNo, pairwisePrior, pairwiseFactor, pairwiseBeta, pairwiseLWeight, pairwiseUWeight, pairwiseVWeight, lidarPairwiseFactor, lidarPairwiseTruncation, crossPairwiseWeight));

}


//KITTI��֤��
LKITTIValidationset::LKITTIValidationset()
{
	seed = 10000;
	classNo = 2;	//KITTI���ݼ��͵�·�ͷ�·����
	filePermutations = 10000;
	optimizeAverage = 1;
	featuresOnline = 0;

	//���Ȩ�أ���ʼ��Ϊȫ1
	unaryWeighted = 0;
	unaryWeights = new double[classNo];
	for (int i = 0; i < classNo; i++) unaryWeights[i] = 1.0;

	//ѵ�����Ͳ��Լ�����1:1
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

	textonNumberOfClusters = 50;		//50����������
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
	sift360 = 1;	//360��
	siftAngles = 8;		//HOG��8��bin����360�ȵ�8������
	siftFolder = "/home/yangfei/Datasets/Result/KITTI/val/Feature/Sift/";
	siftExtension = ".sft";

	//������ɫ��Ϣ��HOG
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

	locationBuckets = 20;	//λ��������һ��
	locationFolder = "/home/yangfei/Datasets/Result/KITTI/val/Feature/Location/";
	locationExtension = ".loc";

	lbpClusteringFile = "lbpclustering.dat";
	lbpFolder = "/home/yangfei/Datasets/Result/KITTI/val/Feature/Lbp/";
	lbpExtension = ".lbp";
	lbpSize = 11;
	lbpKMeansSubSample = 20;
	lbpNumberOfClusters = 50;

	//��һ��
	meanShiftXY[0] = 5.0;
	meanShiftLuv[0] = 3.0;
	meanShiftMinRegion[0] = 20;
	meanShiftFolder[0] = "/home/yangfei/Datasets/Result/KITTI/val/MeanShift/50x30/";
	//��һ��
	meanShiftXY[1] = 5.0;
	meanShiftLuv[1] = 5.0;
	meanShiftMinRegion[1] = 20;
	meanShiftFolder[1] = "/home/yangfei/Datasets/Result/KITTI/val/MeanShift/50x50/";
	//�ڶ���
	meanShiftXY[2] = 7.0;
	meanShiftLuv[2] = 5.0;
	meanShiftMinRegion[2] = 20;
	meanShiftFolder[2] = "/home/yangfei/Datasets/Result/KITTI/val/MeanShift/70x50/";
	//�ڶ���
	meanShiftXY[3] = 7.0;
	meanShiftLuv[3] = 6.0;
	meanShiftMinRegion[3] = 20;
	meanShiftFolder[3] = "/home/yangfei/Datasets/Result/KITTI/val/MeanShift/70x60/";

	meanShiftExtension = ".msh";


	//Boosting����
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
							//���������
	denseMaxClassRatio = 0.2;

	//ƽ��������ͨ����Ȩ��
	pairwiseLWeight = 1.0 / 3.0;
	pairwiseUWeight = 1.0 / 3.0;
	pairwiseVWeight = 1.0 / 3.0;
	pairwisePrior = 1.5;		//1.5->0
	pairwiseFactor = 6.0;		//6.0->20
	pairwiseBeta = 16.0;

	//������ǩ����
	cliqueMinLabelRatio = 0.5;
	//��ֵ
	cliqueThresholdRatio = 0.1;
	//�ضϲ���
	cliqueTruncation = 0.1;		//0.1->0.5

								//Boosting����
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

	//һ����������
	consistencyPrior = 0.05;
	//�߽�ƽ�������
	pairwiseSegmentBuckets = 8;
	pairwiseSegmentPrior = 0.0;		//0.0��>1.5
	pairwiseSegmentFactor = 2.0;
	pairwiseSegmentBeta = 40.0;

	//3D�״�
	//���Խ������·��
	lidarTestFolder = "/home/yangfei/Datasets/Result/KITTI/val/LidarCrf/";
	//�״�GTͼ
	lidarGroundTruthFolder = "/home/yangfei/Datasets/Data/KITTI/val/GroundTruth/";		//�״�ı�ǩ����ͼ��һ��
	lidarGroundTruthExtension = ".png";
	lidarUnaryFactor = 5.0;//0.7;  //0.1->0.5


	lidarPairwiseFactor = 1.0;//0.0005;		//0.00005->0.0005
							  //ƽ���ضϸ���
	lidarPairwiseTruncation = 10.0;
	//�Ӳ����,�ܴ�
	lidarClassNo = 2;


	//ģ�Ͳ�����ʼ��
	crossUnaryWeight = 0.5;
	crossPairwiseWeight = 0.2;//-1e-4;
	crossThreshold = 1e-6;
	//�����Ҫѵ����
	crossTrainFile = "height.dat";

	//��ʼ��������˳�򣬰���������ѵ�����Ͳ��Լ�
	Init();

	//����Ŀ¼
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
	//��������Ϊѵ������ż����Ϊ���Լ�
	for (i=0;i<allImageFiles.GetCount();i++)
	{
		if (i%2)	//����
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
	if (rgb[2] == 255)		//rͨ��������Ч��
	{
		if (rgb[0] == 255)	//bͨ�������·����
		{
			label[0] = 1;		//��·
		}
		else
			label[0] = 2;		//��·
	}
	else
		label[0] = 0;		//��Ч
}

void LKITTIValidationset::LabelToRgb(unsigned char * label, unsigned char * rgb)
{
	rgb[0] = rgb[1] = rgb[2] = 0;
	if (label[0] == 2)
	{
		rgb[2] = 255;		//��·��ɫ
	}
	else
	{
		if (label[0] == 1)
		{
			rgb[2] = 255, rgb[0] = 255;		//��·��ɫ
		}
	}
}

void LKITTIValidationset::SetCRFStructure(LCrf * crf)
{
	//����CRF�������
	LCrfDomain *objDomain = new LCrfDomain(crf, this, classNo, testFolder, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::RgbToLabel, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::LabelToRgb);
	//������嵽ģ��
	crf->domains.Add(objDomain);

	//�����ײ�
	LBaseCrfLayer *baseLayer = new LBaseCrfLayer(crf, objDomain, this, 0);
	//��ӵײ㵽ģ��
	crf->layers.Add(baseLayer);

	//�����߲�ͷָ�
	LPnCrfLayer *superpixelLayer[3];
	LSegmentation2D *segmentation[3];


	//baselayer
	segmentation[0] = new LMeanShiftSegmentation2D(meanShiftXY[1], meanShiftLuv[1], meanShiftMinRegion[1], meanShiftFolder[1], meanShiftExtension);
	//�����ڵײ��ϵķָ���ø���Ϊbase���Լ������ز�
	superpixelLayer[0] = new LPnCrfLayer(crf, objDomain, this, baseLayer, segmentation[0], cliqueTruncation);
	crf->segmentations.Add(segmentation[0]);		//��ӷָcrfģ��
	crf->layers.Add(superpixelLayer[0]);		//��Ӹ߽ײ㵽crfģ��


												//�����������
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

	//һ��������
	LDenseUnaryPixelPotential *pixelPotential = new LDenseUnaryPixelPotential(this, crf, objDomain, baseLayer, trainFolder, denseBoostTrainFile, denseFolder, denseExtension, classNo, denseWeight, denseBoostingSubSample, denseNumberOfRectangles, denseMinimumRectangleSize, denseMaximumRectangleSize, denseMaxClassRatio);
	pixelPotential->AddFeature(textonFeature);
	pixelPotential->AddFeature(siftFeature);
	pixelPotential->AddFeature(coloursiftFeature);
	pixelPotential->AddFeature(lbpFeature);
	crf->potentials.Add(pixelPotential);

	LBoosting<int> *pixelBoosting = new LBoosting<int>(trainFolder, denseBoostTrainFile, classNo, denseNumRoundsBoosting, denseThetaStart, denseThetaIncrement, denseNumberOfThetas, denseRandomizationFactor, pixelPotential, (int *(LPotential::*)(int, int))&LDenseUnaryPixelPotential::GetTrainBoostingValues, (int *(LPotential::*)(int))&LDenseUnaryPixelPotential::GetEvalBoostingValues);
	crf->learnings.Add(pixelBoosting);
	pixelPotential->learning = pixelBoosting;


	//�߽�������
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


	//�Ӳ���״�
	LCrfDomain *lidarDomain = new LCrfDomain(crf, this, classNo, lidarTestFolder, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::RgbToLabel, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::LabelToRgb);
	crf->domains.Add(lidarDomain);
	//�Ӳ�ײ�
	LBaseCrfLayer *lidarBaseLayer = new LBaseCrfLayer(crf, lidarDomain, this, 0);
	crf->layers.Add(lidarBaseLayer);

	//�Ӳ�������(�ѽ��)
	LLidarUnaryPixelPotential *pixelLidarPotential = new LLidarUnaryPixelPotential(this, crf, lidarDomain, lidarBaseLayer, lidarClassNo, lidarUnaryFactor);
	crf->potentials.Add(pixelLidarPotential);

	//��������ģ�͵Ļ���Ŀ��߶ȷֲ���������
	LHeightUnaryPixelPotential *heightPotential = new LHeightUnaryPixelPotential(this, crf, objDomain, baseLayer, lidarDomain, lidarBaseLayer, trainFolder, crossTrainFile, classNo, crossUnaryWeight, lidarClassNo, crossThreshold);
	crf->potentials.Add(heightPotential);

	//����ƽ����
	crf->potentials.Add(new LJointPairwisePixelPotential(this, crf, objDomain, baseLayer, lidarDomain, lidarBaseLayer, classNo, lidarClassNo, pairwisePrior, pairwiseFactor, pairwiseBeta, pairwiseLWeight, pairwiseUWeight, pairwiseVWeight, lidarPairwiseFactor, lidarPairwiseTruncation, crossPairwiseWeight));

}


//pairwise crf
void LKITTIPairwiseCompare::Init()
{
	int i;
	LoadFolder(imageFolder, imageExtension, allImageFiles);
	//��������Ϊѵ������ż����Ϊ���Լ�
	for (i = 0;i < allImageFiles.GetCount();i++)
	{
		if (i % 2)	//����
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
	classNo = 2;	//KITTI���ݼ��͵�·�ͷ�·����
	filePermutations = 10000;
	optimizeAverage = 1;
	featuresOnline = 0;

	//���Ȩ�أ���ʼ��Ϊȫ1
	unaryWeighted = 0;
	unaryWeights = new double[classNo];
	for (int i = 0; i < classNo; i++) unaryWeights[i] = 1.0;

	//ѵ�����Ͳ��Լ�����1:1
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

	textonNumberOfClusters = 50;		//50����������
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
	sift360 = 1;	//360��
	siftAngles = 8;		//HOG��8��bin����360�ȵ�8������
	siftFolder = "/home/yangfei/Datasets//home/yangfei/Datasets/Result/KITTI/val/Feature/Sift/";
	siftExtension = ".sft";

	//������ɫ��Ϣ��HOG
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

	locationBuckets = 20;	//λ��������һ��
	locationFolder = "/home/yangfei/Datasets/Result/KITTI/val/Feature/Location/";
	locationExtension = ".loc";

	lbpClusteringFile = "lbpclustering.dat";
	lbpFolder = "/home/yangfei/Datasets/Result/KITTI/val/Feature/Lbp/";
	lbpExtension = ".lbp";
	lbpSize = 11;
	lbpKMeansSubSample = 20;
	lbpNumberOfClusters = 50;


	//Boosting����
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
							//���������
	denseMaxClassRatio = 0.2;

	//ƽ��������ͨ����Ȩ��
	pairwiseLWeight = 1.0 / 3.0;
	pairwiseUWeight = 1.0 / 3.0;
	pairwiseVWeight = 1.0 / 3.0;
	pairwisePrior = 1.5;		//1.5->0
	pairwiseFactor = 6.0;		//6.0->20
	pairwiseBeta = 16.0;


	//��ʼ��������˳�򣬰���������ѵ�����Ͳ��Լ�
	Init();

	//����Ŀ¼
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
	if (rgb[2] == 255)		//rͨ��������Ч��
	{
		if (rgb[0] == 255)	//bͨ�������·����
		{
			label[0] = 1;		//��·
		}
		else
			label[0] = 2;		//��·
	}
	else
		label[0] = 0;		//��Ч
}

void LKITTIPairwiseCompare::LabelToRgb(unsigned char * label, unsigned char * rgb)
{
	rgb[0] = rgb[1] = rgb[2] = 0;
	if (label[0] == 2)
	{
		rgb[2] = 255;		//��·��ɫ
	}
	else
	{
		if (label[0] == 1)
		{
			rgb[2] = 255, rgb[0] = 255;		//��·��ɫ
		}
	}
}

void LKITTIPairwiseCompare::SetCRFStructure(LCrf * crf)
{
	//����CRF�������
	LCrfDomain *objDomain = new LCrfDomain(crf, this, classNo, testFolder, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::RgbToLabel, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::LabelToRgb);
	//������嵽ģ��
	crf->domains.Add(objDomain);

	//�����ײ�
	LBaseCrfLayer *baseLayer = new LBaseCrfLayer(crf, objDomain, this, 0);
	//��ӵײ㵽ģ��
	crf->layers.Add(baseLayer);
												//�����������
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

	//һ��������
	LDenseUnaryPixelPotential *pixelPotential = new LDenseUnaryPixelPotential(this, crf, objDomain, baseLayer, trainFolder, denseBoostTrainFile, denseFolder, denseExtension, classNo, denseWeight, denseBoostingSubSample, denseNumberOfRectangles, denseMinimumRectangleSize, denseMaximumRectangleSize, denseMaxClassRatio);
	pixelPotential->AddFeature(textonFeature);
	pixelPotential->AddFeature(siftFeature);
	pixelPotential->AddFeature(coloursiftFeature);
	pixelPotential->AddFeature(lbpFeature);
	crf->potentials.Add(pixelPotential);

	LBoosting<int> *pixelBoosting = new LBoosting<int>(trainFolder, denseBoostTrainFile, classNo, denseNumRoundsBoosting, denseThetaStart, denseThetaIncrement, denseNumberOfThetas, denseRandomizationFactor, pixelPotential, (int *(LPotential::*)(int, int))&LDenseUnaryPixelPotential::GetTrainBoostingValues, (int *(LPotential::*)(int))&LDenseUnaryPixelPotential::GetEvalBoostingValues);
	crf->learnings.Add(pixelBoosting);
	pixelPotential->learning = pixelBoosting;

	//��������8�����Pottsģ�͵Ķ�Ԫ�ƺ���
	LEightNeighbourPottsPairwisePixelPotential *pairwisePotentianl = new LEightNeighbourPottsPairwisePixelPotential(this, crf, objDomain, baseLayer, classNo, pairwisePrior, pairwiseFactor, pairwiseBeta, pairwiseLWeight, pairwiseUWeight, pairwiseVWeight);
	//���ƽ���CRFģ��
	crf->potentials.Add(pairwisePotentianl);
}


//high order crf
void LKITTIHighOrderCompare::Init()
{
	int i;
	LoadFolder(imageFolder, imageExtension, allImageFiles);
	//��������Ϊѵ������ż����Ϊ���Լ�
	for (i = 0;i < allImageFiles.GetCount();i++)
	{
		if (i % 2)	//����
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
	classNo = 2;	//KITTI���ݼ��͵�·�ͷ�·����
	filePermutations = 10000;
	optimizeAverage = 1;
	featuresOnline = 0;

	//���Ȩ�أ���ʼ��Ϊȫ1
	unaryWeighted = 0;
	unaryWeights = new double[classNo];
	for (int i = 0; i < classNo; i++) unaryWeights[i] = 1.0;

	//ѵ�����Ͳ��Լ�����1:1
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

	textonNumberOfClusters = 50;		//50����������
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
	sift360 = 1;	//360��
	siftAngles = 8;		//HOG��8��bin����360�ȵ�8������
	siftFolder = "/home/yangfei/Datasets/Result/KITTI/val/Feature/Sift/";
	siftExtension = ".sft";

	//������ɫ��Ϣ��HOG
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

	locationBuckets = 20;	//λ��������һ��
	locationFolder = "/home/yangfei/Datasets/Result/KITTI/val/Feature/Location/";
	locationExtension = ".loc";

	lbpClusteringFile = "lbpclustering.dat";
	lbpFolder = "/home/yangfei/Datasets/Result/KITTI/val/Feature/Lbp/";
	lbpExtension = ".lbp";
	lbpSize = 11;
	lbpKMeansSubSample = 20;
	lbpNumberOfClusters = 50;

	//��һ��
	meanShiftXY[0] = 5.0;
	meanShiftLuv[0] = 3.0;
	meanShiftMinRegion[0] = 20;
	meanShiftFolder[0] = "/home/yangfei/Datasets/Result/KITTI/val/MeanShift/50x30/";
	//��һ��
	meanShiftXY[1] = 5.0;
	meanShiftLuv[1] = 5.0;
	meanShiftMinRegion[1] = 20;
	meanShiftFolder[1] = "/home/yangfei/Datasets/Result/KITTI/val/MeanShift/50x50/";
	//�ڶ���
	meanShiftXY[2] = 7.0;
	meanShiftLuv[2] = 5.0;
	meanShiftMinRegion[2] = 20;
	meanShiftFolder[2] = "/home/yangfei/Datasets/Result/KITTI/val/MeanShift/70x50/";
	//�ڶ���
	meanShiftXY[3] = 7.0;
	meanShiftLuv[3] = 6.0;
	meanShiftMinRegion[3] = 20;
	meanShiftFolder[3] = "/home/yangfei/Datasets/Result/KITTI/val/MeanShift/70x60/";

	meanShiftExtension = ".msh";


	//Boosting����
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
							//���������
	denseMaxClassRatio = 0.2;

	//ƽ��������ͨ����Ȩ��
	pairwiseLWeight = 1.0 / 3.0;
	pairwiseUWeight = 1.0 / 3.0;
	pairwiseVWeight = 1.0 / 3.0;
	pairwisePrior = 1.5;		//1.5->0
	pairwiseFactor = 6.0;		//6.0->20
	pairwiseBeta = 16.0;

	//������ǩ����
	cliqueMinLabelRatio = 0.5;
	//��ֵ
	cliqueThresholdRatio = 0.1;
	//�ضϲ���
	cliqueTruncation = 0.1;		//0.1->0.5

								//Boosting����
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

	//һ����������
	consistencyPrior = 0.05;
	//�߽�ƽ�������
	pairwiseSegmentBuckets = 8;
	pairwiseSegmentPrior = 0.0;		//0.0��>1.5
	pairwiseSegmentFactor = 2.0;
	pairwiseSegmentBeta = 40.0;

	
	//��ʼ��������˳�򣬰���������ѵ�����Ͳ��Լ�
	Init();

	//����Ŀ¼
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
	if (rgb[2] == 255)		//rͨ��������Ч��
	{
		if (rgb[0] == 255)	//bͨ�������·����
		{
			label[0] = 1;		//��·
		}
		else
			label[0] = 2;		//��·
	}
	else
		label[0] = 0;		//��Ч
}

void LKITTIHighOrderCompare::LabelToRgb(unsigned char * label, unsigned char * rgb)
{
	rgb[0] = rgb[1] = rgb[2] = 0;
	if (label[0] == 2)
	{
		rgb[2] = 255;		//��·��ɫ
	}
	else
	{
		if (label[0] == 1)
		{
			rgb[2] = 255, rgb[0] = 255;		//��·��ɫ
		}
	}
}

void LKITTIHighOrderCompare::SetCRFStructure(LCrf * crf)
{
	//����CRF�������
	LCrfDomain *objDomain = new LCrfDomain(crf, this, classNo, testFolder, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::RgbToLabel, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::LabelToRgb);
	//������嵽ģ��
	crf->domains.Add(objDomain);

	//�����ײ�
	LBaseCrfLayer *baseLayer = new LBaseCrfLayer(crf, objDomain, this, 0);
	//��ӵײ㵽ģ��
	crf->layers.Add(baseLayer);

	//�����߲�ͷָ�
	LPnCrfLayer *superpixelLayer[3];
	LSegmentation2D *segmentation[3];


	//baselayer
	segmentation[0] = new LMeanShiftSegmentation2D(meanShiftXY[1], meanShiftLuv[1], meanShiftMinRegion[1], meanShiftFolder[1], meanShiftExtension);
	//�����ڵײ��ϵķָ���ø���Ϊbase���Լ������ز�
	superpixelLayer[0] = new LPnCrfLayer(crf, objDomain, this, baseLayer, segmentation[0], cliqueTruncation);
	crf->segmentations.Add(segmentation[0]);		//��ӷָcrfģ��
	crf->layers.Add(superpixelLayer[0]);		//��Ӹ߽ײ㵽crfģ��


												//�����������
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

	//һ��������
	LDenseUnaryPixelPotential *pixelPotential = new LDenseUnaryPixelPotential(this, crf, objDomain, baseLayer, trainFolder, denseBoostTrainFile, denseFolder, denseExtension, classNo, denseWeight, denseBoostingSubSample, denseNumberOfRectangles, denseMinimumRectangleSize, denseMaximumRectangleSize, denseMaxClassRatio);
	pixelPotential->AddFeature(textonFeature);
	pixelPotential->AddFeature(siftFeature);
	pixelPotential->AddFeature(coloursiftFeature);
	pixelPotential->AddFeature(lbpFeature);
	crf->potentials.Add(pixelPotential);

	LBoosting<int> *pixelBoosting = new LBoosting<int>(trainFolder, denseBoostTrainFile, classNo, denseNumRoundsBoosting, denseThetaStart, denseThetaIncrement, denseNumberOfThetas, denseRandomizationFactor, pixelPotential, (int *(LPotential::*)(int, int))&LDenseUnaryPixelPotential::GetTrainBoostingValues, (int *(LPotential::*)(int))&LDenseUnaryPixelPotential::GetEvalBoostingValues);
	crf->learnings.Add(pixelBoosting);
	pixelPotential->learning = pixelBoosting;

	//��������8�����Pottsģ�͵Ķ�Ԫ�ƺ���
	LEightNeighbourPottsPairwisePixelPotential *pairwisePotentianl = new LEightNeighbourPottsPairwisePixelPotential(this, crf, objDomain, baseLayer, classNo, pairwisePrior, pairwiseFactor, pairwiseBeta, pairwiseLWeight, pairwiseUWeight, pairwiseVWeight);
	//���ƽ���CRFģ��
	crf->potentials.Add(pairwisePotentianl);

	//�߽�������
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
