#ifndef __clustering
#define __clustering

#include <stdio.h>
#include "std.h"

//聚类父类
template <class T>
class LClustering
{
	protected :
		//聚类文件夹和文件
		const char *clusterFolder, *clusterFile;
		//通道数
		int bands;
	public :
		//构造函数
		LClustering(const char *setClusterFolder, const char *setClusterFile, int setBands);
		virtual ~LClustering() {};

		//最近邻居
		virtual int NearestNeighbour(T *values) = 0;
		//聚类
		virtual void Cluster(T *data, int size) = 0;
		//载入
		virtual void LoadTraining() = 0;
		virtual void LoadTraining(FILE *f) = 0;
		//保存
		virtual void SaveTraining() = 0;
		virtual void SaveTraining(FILE *f) = 0;
		//应该是获取聚类中心数
		virtual int GetClusters() = 0;
};

//栅格聚类
template <class T>
class LLatticeClustering : public LClustering<T>
{
	private :
		T *minValues, *maxValues;
		//应该是聚类中心点
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

//K-means聚类
template <class T>
class LKMeansClustering : public LClustering<T>
{
	private :
		//定义在类中的类，并没有采取继承
		class LCluster
		{
			private :
				int bands;		//通道数
				double *means;		//均值
				double *variances;		//方差
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
		//决策树节点，用决策树进行聚类
		class LKdTreeNode
		{
			public :
				int terminal;		//父节点
				int *indices;		//孩子节点
				double splitValue;		//分离的点
				int indiceSize, splitDim;		//孩子个数和？？
				LKdTreeNode *left, *right;	//左右孩子节点

				LKdTreeNode();
				LKdTreeNode(int *setIndices, int setIndiceSize);
				~LKdTreeNode();
				void SetAsNonTerminal(int setSplitDim, double setSplitValue, LKdTreeNode *setLeft, LKdTreeNode *setRight);
		};
		//决策树类
		class LKdTree
		{
			public :
				LKdTreeNode *root;		//根节点

				//构造函数
				LKdTree();
				LKdTree(T *data, int numberOfClusters, int bands, int pointsPerKDTreeCluster);
				~LKdTree();
				//最近节点
				int NearestNeighbour(T *data, int bands, double *values, LKdTreeNode *node, double (*meassure)(double *, double *, int));
		};

		int numberOfClusters, pointsPerKDTreeCluster;		//聚类中心个数和每个决策聚类树的点
		double kMeansMaxChange;

		double *clusterMeans, *dataMeans, *dataVariances;
		LKdTree *kd;		//聚类树
		double (*meassure)(double *, double *, int);
		int finalClusters, normalize;
		//分配标签（类别）
		double AssignLabels(T *data, int size, LCluster **tempClusters, int *labels);
	public :
		LKMeansClustering(const char *setClusterFolder, const char *setClusterFile, int setBands, int setNumberOfClusters, double setKMeansMaxChange, int setPointsPerKDTreeCluster, int setNormalize = 1);
		~LKMeansClustering();

		//最近邻居
		int NearestNeighbour(T *values);		//返回一个指针
		//聚类
		void Cluster(T *data, int size);
		void FillMeans(double *data);
		//从文件载入
		void LoadTraining();	
		void LoadTraining(FILE *f);
		//保存为文件
		void SaveTraining();
		void SaveTraining(FILE *f);
		//获取聚类中心
		int GetClusters();
		//获取均值
		double *GetMeans();
};

#endif