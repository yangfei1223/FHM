#ifndef __dataset
#define __dataset

#include <stdio.h>
#include "std.h"

class LCrf;
class LCrfDomain;
class LCrfLayer;
class LBaseCrfLayer;
class LPnCrfLayer;
class LPreferenceCrfLayer;
class LLabelImage;

//���ݼ�
class LDataset
{
	private :
		//�ļ�����
		char *GetFolderFileName(const char *imageFile, const char *folder, const char *extension);
		//�ַ�������
		static int SortStr(char *str1, char *str2);
	protected :
		//�����ļ���,�����ݶ�����List��
		void LoadFolder(const char *folder, const char *extension, LList<char *> &list);
	public :
		LDataset();
		virtual ~LDataset();
		virtual void Init();

		const char *datasetName;
		//���ѵ���ļ��������ļ��������ļ�����
		LList<char *> trainImageFiles, testImageFiles, allImageFiles;

		unsigned int seed;
		int classNo, filePermutations, featuresOnline;
		double proportionTrain, proportionTest;

		const char *imageFolder, *imageExtension, *groundTruthFolder, *groundTruthExtension, *lidarFolder,*lidarExtension,*calibFolder,*calibExtension,*trainFolder, *testFolder, *lidarTestFolder;
		int optimizeAverage;

		//K-means����,������������
		int clusterPointsPerKDTreeCluster;
		double clusterKMeansMaxChange;
		//location
		int locationBuckets;
		const char *locationFolder, *locationExtension;
		//TextonBoost
		int textonNumberOfClusters, textonKMeansSubSample;
		double textonFilterBankRescale;
		const char *textonClusteringTrainFile, *textonFolder, *textonExtension;
		//Sift
		int siftKMeansSubSample, siftNumberOfClusters, siftWindowSize, siftWindowNumber, sift360, siftAngles, siftSizeCount, siftSizes[4];
		const char *siftClusteringTrainFile, *siftFolder, *siftExtension;
		//��ɫSift
		int colourSiftKMeansSubSample, colourSiftNumberOfClusters, colourSiftWindowSize, colourSiftWindowNumber, colourSift360, colourSiftAngles, colourSiftSizeCount, colourSiftSizes[4];
		const char *colourSiftClusteringTrainFile, *colourSiftFolder, *colourSiftExtension;
		//LBP
		int lbpSize, lbpKMeansSubSample, lbpNumberOfClusters;
		const char *lbpClusteringFile, *lbpFolder, *lbpExtension;
		//Boosting�������Ĳ���
		int denseNumRoundsBoosting, denseBoostingSubSample, denseNumberOfThetas, denseThetaStart, denseThetaIncrement, denseNumberOfRectangles, denseMinimumRectangleSize, denseMaximumRectangleSize;
		double denseRandomizationFactor, denseWeight, denseMaxClassRatio;
		const char *denseBoostTrainFile, *denseExtension, *denseFolder;
		//Mean-Sift�������������ؾ���
		double meanShiftXY[4], meanShiftLuv[4];
		int meanShiftMinRegion[4];
		const char *meanShiftFolder[4], *meanShiftExtension;
		//һ����������
		double consistencyPrior;
		//segment Boosting����
		int statsThetaStart, statsThetaIncrement, statsNumberOfThetas, statsNumberOfBoosts;
		double statsRandomizationFactor, statsAlpha, statsFactor, statsPrior, statsMaxClassRatio;
		const char *statsTrainFile, *statsExtension, *statsFolder;
		//Ȩ������
		double pairwiseLWeight, pairwiseUWeight, pairwiseVWeight, pairwisePrior, pairwiseFactor, pairwiseBeta;
		double cliqueMinLabelRatio, cliqueThresholdRatio, cliqueTruncation;

		int pairwiseSegmentBuckets;
		double pairwiseSegmentPrior, pairwiseSegmentFactor, pairwiseSegmentBeta;
		//������Ϣ
		const char *cooccurenceTrainFile;
		double cooccurenceWeight;
		//˫Ŀ�Ӿ�
		double lidarUnaryFactor;
		int lidarClassNo;
		double lidarPairwiseFactor, lidarPairwiseTruncation;
		const char *lidarGroundTruthFolder, *lidarGroundTruthExtension;

		double crossUnaryWeight, crossPairwiseWeight, crossThreshold;

		const char *crossTrainFile;
		//K-means����
		double kMeansXyLuvRatio[6];
		const char *kMeansFolder[6], *kMeansExtension;
		int kMeansIterations, kMeansMaxDiff, kMeansDistance[6];

		int unaryWeighted;
		double *unaryWeights;
		//��ǩͼ��ת��
		virtual void RgbToLabel(unsigned char *rgb, unsigned char *label);
		virtual void LabelToRgb(unsigned char *label, unsigned char *rgb);
		//����ͼ��
		virtual void SaveImage(LLabelImage &labelImage, LCrfDomain *domain, char *fileName);
		//����CRFģ��
		virtual void SetCRFStructure(LCrf *crf) {};
		//�޼ලͼ��ָ�
		virtual int Segmented(char *imageFileName);
		//��ȡ��ǩ����
		virtual void GetLabelSet(unsigned char *labelset, char *imageFileName);
};


//��Ŀ��·���ݼ�
class LCamVidDataset : public LDataset
{
	// building 0, tree     1, sky         2, car   3
	// sign     4, road     5, pedestrian  6, fense 7
	// column   8, pavement 9, bicyclist  10,

	private :
		void AddFolder(char *folder, LList<char *> &fileList);
	protected :
		void Init();
	public :
		LCamVidDataset();
		
		void RgbToLabel(unsigned char *rgb, unsigned char *label);
		void LabelToRgb(unsigned char *label, unsigned char *rgb);
		void SetCRFStructure(LCrf *crf);
};

/////////////////////////////////////////////////////////////////////////////////////////////
//
//				KITTI	Dataet
//
/////////////////////////////////////////////////////////////////////////////////////////////

//KITTI��·������ݼ�
class LKITTIDataset : public LDataset
{
	//0 non-road	1 road

private:
	void AddFolder(char *folder, LList<char *> &fileList);
protected:
	void Init();
public:
	LKITTIDataset();
	void RgbToLabel(unsigned char *rgb, unsigned char *label);
	void LabelToRgb(unsigned char *label, unsigned char *rgb);
	void SetCRFStructure(LCrf *crf);
};




//KITTI��·���um���ݼ�
class LKITTIumDataset : public LDataset
{
	//0 non-road	1 road

private:
	void AddFolder(char *folder, LList<char *> &fileList);
protected:
	void Init();
public:
	LKITTIumDataset();
	void RgbToLabel(unsigned char *rgb, unsigned char *label);
	void LabelToRgb(unsigned char *label, unsigned char *rgb);
	void SetCRFStructure(LCrf *crf);
};


//KITTI��·���umm���ݼ�
class LKITTIummDataset : public LDataset
{
	//0 non-road	1 road

private:
	void AddFolder(char *folder, LList<char *> &fileList);
protected:
	void Init();
public:
	LKITTIummDataset();
	void RgbToLabel(unsigned char *rgb, unsigned char *label);
	void LabelToRgb(unsigned char *label, unsigned char *rgb);
	void SetCRFStructure(LCrf *crf);
};


//KITTI��·���uu���ݼ�
class LKITTIuuDataset : public LDataset
{
	//0 non-road	1 road

private:
	void AddFolder(char *folder, LList<char *> &fileList);
protected:
	void Init();
public:
	LKITTIuuDataset();
	void RgbToLabel(unsigned char *rgb, unsigned char *label);
	void LabelToRgb(unsigned char *label, unsigned char *rgb);
	void SetCRFStructure(LCrf *crf);
};


//KITTIѵ������֤ʵ��
class LKITTIValidationset : public LDataset
{
protected:
	void Init();
public:
	LKITTIValidationset();
	void RgbToLabel(unsigned char *rgb, unsigned char *label);
	void LabelToRgb(unsigned char *label, unsigned char *rgb);
	void SetCRFStructure(LCrf *crf);
};


//KITII���ں϶Ա�

//pairewise crf
class LKITTIPairwiseCompare : public LDataset
{
protected:
	void Init();
public:
	LKITTIPairwiseCompare();
	void RgbToLabel(unsigned char *rgb, unsigned char *label);
	void LabelToRgb(unsigned char *label, unsigned char *rgb);
	void SetCRFStructure(LCrf *crf);
};


//high order crf
class LKITTIHighOrderCompare : public LDataset
{
protected:
	void Init();
public:
	LKITTIHighOrderCompare();
	void RgbToLabel(unsigned char *rgb, unsigned char *label);
	void LabelToRgb(unsigned char *label, unsigned char *rgb);
	void SetCRFStructure(LCrf *crf);
};

#endif