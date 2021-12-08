#ifndef __feature
#define __feature

#include "std.h"
#include "image.h"
#include "clustering.h"
#include "segmentation.h"
#include "lidarproc.h"

//��������
class LFeature
{
	protected :
		//���ݼ�
		LDataset *dataset;
		//�ļ��������ļ���
		const char *trainFolder, *trainFile;
	public :
		//�����ļ�����������չ��
		const char *evalFolder, *evalExtension;

		//���캯��
		LFeature(LDataset *setDataset, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension);
		virtual ~LFeature() {};
		//����
		void Evaluate(LList<char *> &imageFiles, int from = 0, int to = -1);

		//����
		virtual void LoadTraining() = 0;
		//����
		virtual void SaveTraining() = 0;
		//ѵ��
		virtual void Train(LList<char *> &trainImageFiles) = 0;
		virtual int GetBuckets() = 0;
		//�麯������
		virtual void Evaluate(char *imageFileName) {};
};

//����������
class LDenseFeature : public LFeature
{
	public :
		LDenseFeature(LDataset *setDataset, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension);
		//���麯����ʵ��ʵ����������
		virtual void Discretize(LLabImage &labImage, LImage<unsigned short> &image, const char *imageFileName) = 0;
		//ʵ���ϵ�������ȡ����
		virtual void Evaluate(char *imageFileName);
};

//��������
class LTextonFeature : public LDenseFeature
{
	private :
		//��ͬ���˲���
		LList<LFilter2D<double> *> filters;
		//�˲����Լ���������Ĳ���
		int subSample, numberOfClusters, pointsPerKDTreeCluster;
		double filterScale, kMeansMaxChange;
		//������
		LKMeansClustering<double> *clustering;

		//�����˲���
		void CreateFilterList();
	public :
		//���캯��
		LTextonFeature(LDataset *setDataset, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension, double setFilterScale, int setSubSample, int setNumberOfClusters, double setKMeansMaxChange, int setPointsPerKDTreeCluster);

		~LTextonFeature();
		//��ȡ�˲�����������ע��ÿ��ͨ����Ҫ�˲�
		int GetTotalFilterCount();

		//ѵ��
		void Train(LList<char *> &trainImageFiles);
		//��ɢ��
		void Discretize(LLabImage &labImage, LImage<unsigned short> &image, const char *imageFileName);
		//����ѵ������
		void LoadTraining();
		//����ѵ������
		void SaveTraining();
		int GetBuckets();
};

class LSiftFeature : public LDenseFeature
{
	private :
		int subSample, numberOfClusters, pointsPerKDTreeCluster, *windowSize, windowNumber, is360, angles, diff, sizeCount;
		double kMeansMaxChange;
		LKMeansClustering<double> *clustering;
	public :
		LSiftFeature(LDataset *setDataset, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension, int setWindowSize, int setWindowNumber, int set360, int setAngles, int setSubSample = 0, int setNumberOfClusters = 0, double setKMeansMaxChange = 0, int setPointsPerKDTreeCluster = 0, int setSymetric = 1);
		LSiftFeature(LDataset *setDataset, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension, int setSizeCount, int *setWindowSize, int setWindowNumber, int set360, int setAngles, int setSubSample = 0, int setNumberOfClusters = 0, double setKMeansMaxChange = 0, int setPointsPerKDTreeCluster = 0, int setSymetric = 1);
		~LSiftFeature();

		void Train(LList<char *> &trainImageFiles);
		void Discretize(LLabImage &labImage, LImage<unsigned short> &image, const char *imageFileName);
		void LoadTraining();
		void SaveTraining();
		int GetBuckets();
};

class LColourSiftFeature : public LDenseFeature
{
	private :
		int subSample, numberOfClusters, pointsPerKDTreeCluster, *windowSize, windowNumber, is360, angles, diff, sizeCount;
		double kMeansMaxChange;
		LKMeansClustering<double> *clustering;
	public :
		LColourSiftFeature(LDataset *setDataset, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension, int setWindowSize, int setWindowNumber, int set360, int setAngles, int setSubSample = 0, int setNumberOfClusters = 0, double setKMeansMaxChange = 0, int setPointsPerKDTreeCluster = 0, int setSymetric = 1);
		LColourSiftFeature(LDataset *setDataset, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension, int setSizeCount, int *setWindowSize, int setWindowNumber, int set360, int setAngles, int setSubSample = 0, int setNumberOfClusters = 0, double setKMeansMaxChange = 0, int setPointsPerKDTreeCluster = 0, int setSymetric = 1);
		~LColourSiftFeature();

		void Train(LList<char *> &trainImageFiles);
		void Discretize(LLabImage &labImage, LImage<unsigned short> &image, const char *imageFileName);
		void LoadTraining();
		void SaveTraining();
		int GetBuckets();
};

class LLbpFeature : public LDenseFeature
{
	private :
		int subSample, numberOfClusters, pointsPerKDTreeCluster, windowSize;
		double kMeansMaxChange;
		LKMeansClustering<double> *clustering;
	public :
		LLbpFeature(LDataset *setDataset, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension, int setWindowSize, int setSubSample, int setNumberOfClusters, double setKMeansMaxChange, int setPointsPerKDTreeCluster);
		~LLbpFeature();

		void Train(LList<char *> &trainImageFiles);
		void Discretize(LLabImage &labImage, LImage<unsigned short> &image, const char *imageFileName);
		void LoadTraining();
		void SaveTraining();
		int GetBuckets();
};

//λ��������
class LLocationFeature : public LDenseFeature
{
	private :
		//դ����࣬λ������ʹ�õ���դ����෽��
		LLatticeClustering<double> *clustering;
	public :
		//���캯��
		LLocationFeature(LDataset *setDataset, const char *setEvalFolder, const char *setEvalExtension, int setLocationBuckets);
		~LLocationFeature();

		//������ȡ��λ������ֻ��Ҫ�ڲ���ʱ��ȡ
		void Train(LList<char *> &trainImageFiles) { printf("Location need not to train.\n"); };
		//��ɢ��
		void Discretize(LLabImage &labImage, LImage<unsigned short> &image, const char *imageFileName);
		void LoadTraining() {};
		void SaveTraining() {};
		int GetBuckets();
};


//�״�����
class LLidarFeature: public LDenseFeature
{
private:
	int numberOfClusters, pointsPerKDTreeCluster, featureCount;
	double kMeansMaxChange;
	LKMeansClustering<double> *clustering;
	cLidarProcess m_Lidar;
public:
	LLidarFeature(LDataset *setDataset, const char *setTrainFolder, const char *setTrainFile, const char *setEvalFolder, const char *setEvalExtension, int setNumberOfClusters, double setKMeansMaxChange, int setPointsPerKDTreeCluster,int setFeatureCount);
	~LLidarFeature();

	void Train(LList<char *> &trainImageFiles);
	void Discretize(LLabImage &labImage, LImage<unsigned short> &image, const char *imageFileName);
	void LoadTraining();
	void SaveTraining();
	int GetBuckets();
};

#endif