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

//数据集
class LDataset
{
	private :
		//文件夹名
		char *GetFolderFileName(const char *imageFile, const char *folder, const char *extension);
		//字符串排序
		static int SortStr(char *str1, char *str2);
	protected :
		//载入文件夹,把数据都载入List中
		void LoadFolder(const char *folder, const char *extension, LList<char *> &list);
	public :
		LDataset();
		virtual ~LDataset();
		virtual void Init();

		const char *datasetName;
		//存放训练文件，测试文件，所有文件名的
		LList<char *> trainImageFiles, testImageFiles, allImageFiles;

		unsigned int seed;
		int classNo, filePermutations, featuresOnline;
		double proportionTrain, proportionTest;

		const char *imageFolder, *imageExtension, *groundTruthFolder, *groundTruthExtension, *lidarFolder,*lidarExtension,*calibFolder,*calibExtension,*trainFolder, *testFolder, *lidarTestFolder;
		int optimizeAverage;

		//K-means参数,用于特征聚类
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
		//颜色Sift
		int colourSiftKMeansSubSample, colourSiftNumberOfClusters, colourSiftWindowSize, colourSiftWindowNumber, colourSift360, colourSiftAngles, colourSiftSizeCount, colourSiftSizes[4];
		const char *colourSiftClusteringTrainFile, *colourSiftFolder, *colourSiftExtension;
		//LBP
		int lbpSize, lbpKMeansSubSample, lbpNumberOfClusters;
		const char *lbpClusteringFile, *lbpFolder, *lbpExtension;
		//Boosting分类器的参数
		int denseNumRoundsBoosting, denseBoostingSubSample, denseNumberOfThetas, denseThetaStart, denseThetaIncrement, denseNumberOfRectangles, denseMinimumRectangleSize, denseMaximumRectangleSize;
		double denseRandomizationFactor, denseWeight, denseMaxClassRatio;
		const char *denseBoostTrainFile, *denseExtension, *denseFolder;
		//Mean-Sift参数，用于像素聚类
		double meanShiftXY[4], meanShiftLuv[4];
		int meanShiftMinRegion[4];
		const char *meanShiftFolder[4], *meanShiftExtension;
		//一致性先验项
		double consistencyPrior;
		//segment Boosting参数
		int statsThetaStart, statsThetaIncrement, statsNumberOfThetas, statsNumberOfBoosts;
		double statsRandomizationFactor, statsAlpha, statsFactor, statsPrior, statsMaxClassRatio;
		const char *statsTrainFile, *statsExtension, *statsFolder;
		//权重设置
		double pairwiseLWeight, pairwiseUWeight, pairwiseVWeight, pairwisePrior, pairwiseFactor, pairwiseBeta;
		double cliqueMinLabelRatio, cliqueThresholdRatio, cliqueTruncation;

		int pairwiseSegmentBuckets;
		double pairwiseSegmentPrior, pairwiseSegmentFactor, pairwiseSegmentBeta;
		//共生信息
		const char *cooccurenceTrainFile;
		double cooccurenceWeight;
		//双目视觉
		double lidarUnaryFactor;
		int lidarClassNo;
		double lidarPairwiseFactor, lidarPairwiseTruncation;
		const char *lidarGroundTruthFolder, *lidarGroundTruthExtension;

		double crossUnaryWeight, crossPairwiseWeight, crossThreshold;

		const char *crossTrainFile;
		//K-means参数
		double kMeansXyLuvRatio[6];
		const char *kMeansFolder[6], *kMeansExtension;
		int kMeansIterations, kMeansMaxDiff, kMeansDistance[6];

		int unaryWeighted;
		double *unaryWeights;
		//标签图像转换
		virtual void RgbToLabel(unsigned char *rgb, unsigned char *label);
		virtual void LabelToRgb(unsigned char *label, unsigned char *rgb);
		//保存图像
		virtual void SaveImage(LLabelImage &labelImage, LCrfDomain *domain, char *fileName);
		//创建CRF模型
		virtual void SetCRFStructure(LCrf *crf) {};
		//无监督图像分割
		virtual int Segmented(char *imageFileName);
		//获取标签配置
		virtual void GetLabelSet(unsigned char *labelset, char *imageFileName);
};


//单目道路数据集
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

//KITTI道路检测数据集
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




//KITTI道路检测um数据集
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


//KITTI道路检测umm数据集
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


//KITTI道路检测uu数据集
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


//KITTI训练集验证实验
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


//KITII非融合对比

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