#ifndef __potential
#define __potential

#include <stdio.h>
#include "feature.h"
#include "dataset.h"
#include "graph.h"

class LLearning;

//�ƺ�������
class LPotential
{
	protected :
		int classNo;
		const char *trainFolder, *trainFile, *evalFolder, *evalExtension;
	public :
		LDataset *dataset;
		LCrf *crf;
		LCrfDomain *domain;
		int nodeOffset, edgeOffset;

		LPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension, int setClassNo);
		virtual ~LPotential() {};

		void Evaluate(LList<char *> &imageFiles, int from = 0, int to = -1);

		virtual int GetNodeCount();
		virtual int GetEdgeCount();

		virtual void Train(LList<char *> &trainImageFiles) {};
		virtual void SaveTraining() {};
		virtual void LoadTraining() {};
		virtual void Evaluate(char *imageFileName) {};
		virtual void Initialize(LLabImage &labImage, char *imageFileName) {};
		virtual void UnInitialize() {};
		virtual void AddCosts(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes) {};
		virtual double GetCost(LCrfDomain *costDomain) { return(0); }
};

//һ��һԪ�ƺ���
class LUnaryPixelPotential : public LPotential
{
	protected :
		double unaryFactor;
		int width, height, unaryCount;
		//������
		double *unaryCosts;
		LBaseCrfLayer *layer;
	public :
		LUnaryPixelPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LBaseCrfLayer *setLayer, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension, int setClassNo, double setUnaryFactor);
		~LUnaryPixelPotential();
		void AddCosts(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);
		void SetLabels();
		virtual double GetCost(LCrfDomain *costDomain);
};


//�״�һ���ƺ���
class LLidarUnaryPixelPotential : public LUnaryPixelPotential
{
private:
	int fileSum;
	//��ʼ��ѵ������
	void InitTrainData(LList<char *> &trainImageFiles);
	//����ʼ��
	void UnInitTrainData();
	//��ʼ����������
	void InitEvalData(char *evalImageFile);
	//����ʼ��
	void UnInitEvalData();
public:
	LLidarUnaryPixelPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LBaseCrfLayer *setLayer, int setClassNo, double setUnaryFactor);
	~LLidarUnaryPixelPotential();
	//û��ѵ��
	void Train(LList<char *> &trainImageFiles);
	//����
	void Evaluate(char *imageFileName);
	void Initialize(LLabImage &labImage, char *imageFileName);
	void UnInitialize();
};

//����DenseFeature��һ��һԪ�ƺ���
class LDenseUnaryPixelPotential : public LUnaryPixelPotential
{
	protected :
		//��״
		struct LShape
		{
			int x, y, width, height;
		};
		int thetaIncrement, thetaStart, numberOfThetas;

	private :
		//����������ͼ��ָ�룬������*ÿ��ά����������һ����άָ��
		LIntegralImage ***integralImages;
		//��С�����ο�
		int minimumRectangleSize, maximumRectangleSize;
		//�������ʵ�
		int subSample, *buckets, fileSum, **featureValues, numberOfShapes, *integralBuckets, *integralPoints;
		//Ȩ��
		double **weights, *classWeights;
		int *targets, totalFeatures;
		//��Ч��
		int **validPointsX, **validPointsY, *pointsSum, numExamples;
		LShape *shapes;
		double maxClassRatio;
		//����
		LList<LFeature *> features;

		//������״�˲�
		int CalculateShapeFilterResponse(int index, int bucket, LShape *shape, int pointX, int pointY);
		//��ʼ��ѵ������
		void InitTrainData(LList<char *> &trainImageFiles);
		void InitTrainData(LLabImage *labImages, LLabelImage *groundTruth);
		//����ʼ��
		void UnInitTrainData();
		//��ʼ����������
		void InitEvalData(char *evalImageFile, int *width, int *height);
		void InitEvalData(LLabImage &labImage);
		//����ʼ��
		void UnInitEvalData();
		//��������ͼ
		void CalculateIntegralImages(LLabImage &labImage, LIntegralImage ***integralImage, int subSample, int *width, int *height, char *imageFileName);
	public :
		//ѧϰ��
		LLearning *learning;

		LDenseUnaryPixelPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LBaseCrfLayer *setLayer, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension, int setClassNo, double setUnaryFactor, int setSubSample, int setNumberOfShapes, int setMinimumRectangleSize, int setMaximumRectangleSize, double setMaxClassRatio);
		~LDenseUnaryPixelPotential();

		void AddFeature(LDenseFeature *feature);

		//ѵ��
		void Train(LList<char *> &trainImageFiles);
		void Train(LLabImage *labImages, LLabelImage *groundTruth, int count);

		void SaveTraining();
		void LoadTraining();
		//����
		void Evaluate(char *imageFileName);

		void Initialize(LLabImage &labImage, char *imageFileName);

		void UnInitialize();

		int *GetTrainBoostingValues(int index, int core);
		int *GetEvalBoostingValues(int index);
		int *GetTrainForestValues(int index, int *pixelIndexes, int count, int core);
		int GetEvalForestValue(int index, int pixelIndex);
		int GetLength(int index);
		int GetSize(int index);
};

//����Ŀ��߶ȵ��ƺ���������Ŀ�����3D�ؽ���������
class LHeightUnaryPixelPotential : public LUnaryPixelPotential
{
protected:
	double threshold;
	//ֱ��ͼ��Ҫ���࣬�����ݸ߶ȷֲ�ֱ��ͼ�����ൽĳ��Ŀ��
	int disparityClassNo;
	//������ģ��
	LCrfDomain *objDomain, *lidarDomain;
	LBaseCrfLayer *lidarLayer, *objLayer;
	int first;
public:
	LHeightUnaryPixelPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setObjDomain, LBaseCrfLayer *setObjLayer, LCrfDomain *setLidarDomain, LBaseCrfLayer *setLidarLayer, const char *setTrainFolder, const char *setTrainFile, int setClassNo, double setUnaryFactor, int setDisparityClassNo, double setThreshold);
	~LHeightUnaryPixelPotential();
	//��Ӵ���
	void AddCosts(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);

	void Initialize(LLabImage &labImage, char *imageFileName);
	//�����Ҫѵ���ģ�����ѵ����ͳ��Ŀ��߶ȷֲ�
	void Train(LList<char *> &trainImageFiles);
	void SaveTraining();
	void LoadTraining();
	void UnInitialize();
	double GetCost(LCrfDomain *costDomain);
};

//һ�׶�Ԫ�ƺ���
class LPairwisePixelPotential : public LPotential
{
	protected :
		LBaseCrfLayer *layer;
	public :
		LPairwisePixelPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LBaseCrfLayer *setLayer, int setClassNo);
};

//����pottsģ�͵�һ�׶�Ԫ�ƺ���������ͬ�޴��ۣ���ͬ�д���
class LPottsPairwisePixelPotential : public LPairwisePixelPotential
{
	protected :
		double *pairwiseCosts;
		int pairwiseCount, *pairwiseIndexes;
	public :
		LPottsPairwisePixelPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LBaseCrfLayer *setLayer, int setClassNo);
		~LPottsPairwisePixelPotential();
		int GetEdgeCount();
		//�Ѵ���ֵ��ӵ�ͼģ����
		void AddCosts(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);
		double GetCost(LCrfDomain *costDomain);
};

//�������һ�׶�Ԫ�ƺ���
class LEightNeighbourPottsPairwisePixelPotential : public LPottsPairwisePixelPotential
{
	protected :
		double pairwisePrior, pairwiseFactor, pairwiseBeta, pairwiseLWeight, pairwiseUWeight, pairwiseVWeight;
		//�������
		double PairwiseDiff(double *lab1, double *lab2, double distance);
	public :
		LEightNeighbourPottsPairwisePixelPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LBaseCrfLayer *setLayer, int setClassNo, double setPairwisePrior, double setPairwiseFactor, double setPairwiseBeta, double setPairwiseLWeight, double setPairwiseUWeight, double setPairwiseVWeight);
		//����ѵ��
		void Train(LList<char *> &trainImageFiles) {};
		void SaveTraining() {};
		void LoadTraining() {};
		void Evaluate(LList<char *> &imageFiles) {};
		void Initialize(LLabImage &labImage, char *imageFileName);
		void UnInitialize();
};

//���ڲ�һ�������Խضϵ�һ�׶�Ԫ�ƺ���
class LLinearTruncatedPairwisePixelPotential : public LPairwisePixelPotential
{
	protected :
		double *pairwiseCosts;
		int pairwiseCount, *pairwiseIndexes;
		double pairwiseFactor;
		double truncation;
	public :
		LLinearTruncatedPairwisePixelPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LBaseCrfLayer *setLayer, int setClassNo, double setPairwiseFactor, double setTruncation);
		~LLinearTruncatedPairwisePixelPotential();
		int GetEdgeCount();
		int GetNodeCount();
		void AddCosts(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);
		double GetCost(LCrfDomain *costDomain);
};

//�����Ӳһ�������Խضϵ�һ�׶�Ԫ�ƺ���
class LDisparityLinearTruncatedPairwisePixelPotential : public LLinearTruncatedPairwisePixelPotential
{
protected:
public:
	LDisparityLinearTruncatedPairwisePixelPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LBaseCrfLayer *setLayer, int setClassNo, double setPairwiseFactor, double setTruncation);
	void Train(LList<char *> &trainImageFiles) {};
	void SaveTraining() {};
	void LoadTraining() {};
	void Evaluate(LList<char *> &imageFiles) {};
	void Initialize(LLabImage &labImage, char *imageFileName);
	void UnInitialize();
};


//����Ŀ�����3D�ؽ��Ķ�Ԫ�ƺ���
class LJointPairwisePixelPotential : public LPairwisePixelPotential
{
protected:
	double *pairwiseCosts;
	int pairwiseCount, *pairwiseIndexes;
	int first;

	double *objCosts, disparityFactor, crossFactor, truncation;
	double pairwisePrior, pairwiseFactor, pairwiseBeta, pairwiseLWeight, pairwiseUWeight, pairwiseVWeight;
	int disparityClassNo;
	double PairwiseDiff(double *lab1, double *lab2, double distance);

	LCrfDomain *objDomain, *lidarDomain;
	LBaseCrfLayer *lidarLayer, *objLayer;
public:
	LJointPairwisePixelPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setObjDomain, LBaseCrfLayer *setObjLayer, LCrfDomain *setLidarDomain, LBaseCrfLayer *setLidarLayer, int setClassNo, int setDisparityClassNo, double setPairwisePrior, double setPairwiseFactor, double setPairwiseBeta, double setPairwiseLWeight, double setPairwiseUWeight, double setPairwiseVWeight, double setDisparityFactor, double setTruncation, double setCrossFactor);
	~LJointPairwisePixelPotential();

	void Train(LList<char *> &trainImageFiles) {};
	void SaveTraining() {};
	void LoadTraining() {};
	void Evaluate(LList<char *> &imageFiles) {};
	void Initialize(LLabImage &labImage, char *imageFileName);
	void UnInitialize();
	int GetEdgeCount();
	int GetNodeCount();
	void AddCosts(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);
	double GetCost(LCrfDomain *costDomain);
};


//����segment��һԪ�ƺ���
class LUnarySegmentPotential : public LPotential
{
	protected :
		double consistencyPrior, segmentFactor;
		LList<LPnCrfLayer *> layers;
	public :
		double *unaryCosts;

		LUnarySegmentPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension, int setClassNo, double setConsistencyPrior, double setSegmentFactor);
		~LUnarySegmentPotential();
		void AddCosts(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);
		void AddLayer(LPnCrfLayer *layer);
		double GetCost(LCrfDomain *costDomain);
};

//����segment��һ���Ե�һԪ�ƺ���
class LConsistencyUnarySegmentPotential : public LUnarySegmentPotential
{
	protected :
	public :
		LConsistencyUnarySegmentPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, int setClassNo, double setConsistencyPrior);

		void Initialize(LLabImage &labImage, char *imageFileName);
		void UnInitialize();

		void Train(LList<char *> &trainImageFiles) {};
		void SaveTraining() {};
		void LoadTraining() {};
};

//�߽ײ����������
class LStatsUnarySegmentPotential : public LUnarySegmentPotential
{
	protected :
		double *clusterProbabilities, minLabelRatio, kMeansMaxChange, alpha, maxClassRatio;
		int colourStatsClusters, pointsPerKdCluster, *buckets, *buckets2;

		int numExamples, totalFeatures, *targets, totalSegments, width, height;
		double **weights, **featureValues;
		int numberOfThetas;
		double *classWeights, **evalData;
		LSegmentation2D *segmentation;
		LList<LDenseFeature *> features;
		int neighbour;
	public :
		LLearning *learning;

		LStatsUnarySegmentPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension, int setClassNo, double setConsistencyPrior, double setSegmentFactor, double setMinLabelRatio, double setAlpha, double setMaxClassRatio, int setNeighbour = 0, LSegmentation2D *setSegmentation = NULL);
		~LStatsUnarySegmentPotential();

		void Train(LList<char *> &trainImageFiles);
		void SaveTraining();
		void LoadTraining();
		void Evaluate(char *imageFileName);
		void Initialize(LLabImage &labImage, char *imageFileName);
		void UnInitialize();

		void AddFeature(LDenseFeature *feature);

		double *GetTrainBoostingValues(int index, int core);
		double *GetEvalBoostingValues(int index);
		double *GetTrainSVMValues(int index, int core);
		double *GetEvalSVMValues(int index);
};

//����segment�Ķ�Ԫ�ƺ���
class LPairwiseSegmentPotential : public LPotential
{
	protected :
		LPnCrfLayer *layer;
	public :
		LPairwiseSegmentPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LPnCrfLayer *setLayer, int setClassNo);
		//����ѵ��
		void Train(LList<char *> &trainImageFiles) {};
		void SaveTraining() {};
		void LoadTraining() {};
};

//����Pottsģ�͵ĸ߽׶�Ԫ�ƺ���
class LPottsPairwiseSegmentPotential : public LPairwiseSegmentPotential
{
	protected :
		double *pairwiseCosts;
		int pairwiseCount, *pairwiseIndexes;
	public :
		LPottsPairwiseSegmentPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LPnCrfLayer *setLayer, int setClassNo);
		~LPottsPairwiseSegmentPotential();
		int GetEdgeCount();
		int GetNodeCount();
		void AddCosts(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);
		double GetCost(LCrfDomain *costDomain);
};

//��������ֱ��ͼ�ĸ߽׶�Ԫ�ƺ���
class LHistogramPottsPairwiseSegmentPotential : public LPottsPairwiseSegmentPotential
{
	protected :
		double pairwisePrior, pairwiseFactor, pairwiseBeta;
		int buckets;
		//����ֱ��ͼŷ�Ͼ���
		double PairwiseDistance(double *h1, double *h2);
	public :
		LHistogramPottsPairwiseSegmentPotential(LDataset *setDataset, LCrf *setCrf, LCrfDomain *setDomain, LPnCrfLayer *setLayer, int setClassNo, double setPairwisePrior, double setPairwiseFactor, double setPairwiseBeta, int setBuckets);
		void Initialize(LLabImage &labImage, char *imageFileName);
		void UnInitialize();
};




#endif