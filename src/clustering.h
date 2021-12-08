#ifndef __clustering
#define __clustering

#include <stdio.h>
#include "std.h"

//���ุ��
template <class T>
class LClustering
{
	protected :
		//�����ļ��к��ļ�
		const char *clusterFolder, *clusterFile;
		//ͨ����
		int bands;
	public :
		//���캯��
		LClustering(const char *setClusterFolder, const char *setClusterFile, int setBands);
		virtual ~LClustering() {};

		//����ھ�
		virtual int NearestNeighbour(T *values) = 0;
		//����
		virtual void Cluster(T *data, int size) = 0;
		//����
		virtual void LoadTraining() = 0;
		virtual void LoadTraining(FILE *f) = 0;
		//����
		virtual void SaveTraining() = 0;
		virtual void SaveTraining(FILE *f) = 0;
		//Ӧ���ǻ�ȡ����������
		virtual int GetClusters() = 0;
};

//դ�����
template <class T>
class LLatticeClustering : public LClustering<T>
{
	private :
		T *minValues, *maxValues;
		//Ӧ���Ǿ������ĵ�
		int *buckets;	
	public :
		LLatticeClustering(int setBands, T *setMinValues, T *setMaxValues, int setBuckets);
		LLatticeClustering(int setBands, T *setMinValues, T *setMaxValues, int *setBuckets);
		~LLatticeClustering();

		int NearestNeighbour(T *values);
		void Cluster(T *data, int size) {};
		void LoadTraining() {};
		void LoadTraining(FILE *f) {};
		void SaveTraining() {};
		void SaveTraining(FILE *f) {};
		int GetClusters();
};

//K-means����
template <class T>
class LKMeansClustering : public LClustering<T>
{
	private :
		//���������е��࣬��û�в�ȡ�̳�
		class LCluster
		{
			private :
				int bands;		//ͨ����
				double *means;		//��ֵ
				double *variances;		//����
			public :
				int count;
				double logMixCoeff, logDetCov;

				LCluster();
				LCluster(int setBands);
				~LCluster();

				void SetBands(int setBands);
				double *GetMeans();
				double *GetVariances();
		};
		//�������ڵ㣬�þ��������о���
		class LKdTreeNode
		{
			public :
				int terminal;		//���ڵ�
				int *indices;		//���ӽڵ�
				double splitValue;		//����ĵ�
				int indiceSize, splitDim;		//���Ӹ����ͣ���
				LKdTreeNode *left, *right;	//���Һ��ӽڵ�

				LKdTreeNode();
				LKdTreeNode(int *setIndices, int setIndiceSize);
				~LKdTreeNode();
				void SetAsNonTerminal(int setSplitDim, double setSplitValue, LKdTreeNode *setLeft, LKdTreeNode *setRight);
		};
		//��������
		class LKdTree
		{
			public :
				LKdTreeNode *root;		//���ڵ�

				//���캯��
				LKdTree();
				LKdTree(T *data, int numberOfClusters, int bands, int pointsPerKDTreeCluster);
				~LKdTree();
				//����ڵ�
				int NearestNeighbour(T *data, int bands, double *values, LKdTreeNode *node, double (*meassure)(double *, double *, int));
		};

		int numberOfClusters, pointsPerKDTreeCluster;		//�������ĸ�����ÿ�����߾������ĵ�
		double kMeansMaxChange;

		double *clusterMeans, *dataMeans, *dataVariances;
		LKdTree *kd;		//������
		double (*meassure)(double *, double *, int);
		int finalClusters, normalize;
		//�����ǩ�����
		double AssignLabels(T *data, int size, LCluster **tempClusters, int *labels);
	public :
		LKMeansClustering(const char *setClusterFolder, const char *setClusterFile, int setBands, int setNumberOfClusters, double setKMeansMaxChange, int setPointsPerKDTreeCluster, int setNormalize = 1);
		~LKMeansClustering();

		//����ھ�
		int NearestNeighbour(T *values);		//����һ��ָ��
		//����
		void Cluster(T *data, int size);
		void FillMeans(double *data);
		//���ļ�����
		void LoadTraining();	
		void LoadTraining(FILE *f);
		//����Ϊ�ļ�
		void SaveTraining();
		void SaveTraining(FILE *f);
		//��ȡ��������
		int GetClusters();
		//��ȡ��ֵ
		double *GetMeans();
};

#endif