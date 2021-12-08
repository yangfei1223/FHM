#ifndef __crf
#define __crf

#include "dataset.h"
#include "learning.h"
#include "graph.h"

class LCrfDomain;
class LCrf;
//CRF�㸸��
class LCrfLayer		
{
	protected :
		//Ŀ�������
		int classNo;
		LDataset *dataset;
	public :
		LCrf *crf;		//ָ��crfģ�͵�ָ��
		LCrfDomain *domain;
		//���㣬����һ��
		LCrfLayer *parent;		//����������ÿ���ṹ����һ��ָ����һ����ָ��
		//��ǩ
		unsigned char *labels, *active;
		int nodeOffset;
		int range;		//range��ʲô��˼��
		//���캯��
		LCrfLayer(LCrf *setCrf, LCrfDomain *setDomain, LDataset *setDataset, LCrfLayer *setParent, int setRange);
		//��������
		~LCrfLayer();

		//��ʼ��
		virtual void Initialize(char *imageFileName, int onFly = 0) {};
		//���ñ�ǩ
		virtual void SetLabels() {};
		//����ʼ����Ӧ��������Release����Dump
		virtual void UnInitialize() {};
		//��ȡ������
		virtual int GetNodeCount() = 0;
		//��ȡ����
		virtual int GetEdgeCount() = 0;
		//��ȡ�ɶԵĽڵ���
		virtual int GetPairwiseNodeCount();
		//����ͼģ��
		virtual void BuildGraph(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes) {};
		//���±�ǩ
		virtual int UpdateLabels(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);
		//???
		virtual int BinaryNodes();
};

//�ײ㣬�����ز�
class LBaseCrfLayer : public LCrfLayer
{
	protected :
		int width, height;	//ͼ���͸�
	public :
		//���캯��
		LBaseCrfLayer(LCrf *setCrf, LCrfDomain *setDomain, LDataset *setDataset, int setRange);
		//��������
		~LBaseCrfLayer();

		//��ʼ��
		void Initialize(char *imageFileName, int onFly = 0);
		//����ʼ��
		void UnInitialize();
		//��ȡ�ڵ���
		int GetNodeCount();
		//��ȡ��Ե��
		int GetEdgeCount();
		//����ͼģ��
		void BuildGraph(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);
		//���±�ǩ
		int UpdateLabels(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);
		//???
		int BinaryNodes();
		//��ȡ�ɶԽڵ���
		int GetPairwiseNodeCount();
};

//Pn�߽ײ�
class LPnCrfLayer : public LCrfLayer
{
	protected :
	public :
		int *segmentCounts, **segmentIndexes, segmentCount;		//����ķָ����
		int *baseSegmentCounts, **baseSegmentIndexes;		//����ָ����
		LSegmentation2D *segmentation;
		double **weights, *weightSums, truncation;		//Ȩ�ز���
		const char *segFolder, *segExtension;		//�ļ���

		LPnCrfLayer(LCrf *setCrf, LCrfDomain *setDomain, LDataset *setDataset, LCrfLayer *setParent, LSegmentation2D *setSegmentation, double setTruncation);
		LPnCrfLayer(LCrf *setCrf, LCrfDomain *setDomain, LDataset *setDataset, LCrfLayer *setParent, const char *setSegFolder, const char *setSegExtension, double setTruncation);
		~LPnCrfLayer();

		void Initialize(char *imageFileName, int onFly = 0);
		void SetLabels();
		void UnInitialize();
		int GetNodeCount();
		int GetEdgeCount();
		void BuildGraph(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);
		int UpdateLabels(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);
		int BinaryNodes();
		int GetPairwiseNodeCount();
};

//�����Ȳ㣿
class LPreferenceCrfLayer : public LCrfLayer
{
	protected :
	public :
		LPreferenceCrfLayer(LCrf *setCrf, LCrfDomain *setDomain, LDataset *setDataset, LCrfLayer *setParent);
		~LPreferenceCrfLayer();
		int GetNodeCount();
		int GetEdgeCount();
		void BuildGraph(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);
		void Initialize(char *imageFileName, int onFly = 0);
		void UnInitialize();
};

//CRF����
class LCrfDomain
{
	protected :
	public :
		LCrf *crf;		//ָ��ģ�͵�ָ��
		LDataset *dataset;		//ָ�����ݼ���ָ��
		LBaseCrfLayer *baseLayer;		//ָ��ײ��ָ��
		int classNo;	//����
		const char *testFolder;		//�����ļ���
		//���캯��
		LCrfDomain(LCrf *setCrf, LDataset *setDataset, int setClassNo, const char *setTestFolder, void (LDataset::*setRgbToLabel)(unsigned char *, unsigned char *), void (LDataset::*setLabelToRgb)(unsigned char *, unsigned char *));
		//��ǩת��
		void (LDataset::*rgbToLabel)(unsigned char *, unsigned char *);
		void (LDataset::*labelToRgb)(unsigned char *, unsigned char *);
};

//CRFģ��
class LCrf
{
	private :
		//���ݼ�
		LDataset *dataset;
		//ͼģ��
		Graph<double, double, double> *g;
		//ͼģ�ͽڵ�
		Graph<double, double, double>::node_id *nodes;

	public :
		//�������������,������ģ���л��õ�������ָ��3D�ؽ�������ģ��
		LList<LCrfDomain *> domains;
		//CRF��
		LList<LCrfLayer *> layers;
		//�ƺ���
		LList<LPotential *> potentials;
		//����
		LList<LFeature *> features;
		//�߲��޼ල�ķָ�
		LList<LSegmentation2D *> segmentations;
		//������
		LList<LLearning *> learnings;
		//���ݼ�
		LCrf(LDataset *setDataset);
		~LCrf();
		//�ָ�ԭͼ
		void Segment(char *imageFileName);
		void Segment(LList<char *> &imageFiles, int from = 0, int to = -1);
		//ѵ������
		void TrainFeatures(LList<char *> &imageFiles);
		//��ȡ����
		void EvaluateFeatures(LList<char *> &imageFiles, int from = 0, int to = -1);
		//ѵ���ƺ���
		void TrainPotentials(LList<char *> &imageFiles);
		//�ƺ������ԣ���Ҫ��������ķ����������ݵĲ���
		void EvaluatePotentials(LList<char *> &imageFiles, int from = 0, int to = -1);
		//ģ�ͺ����Ż����
		void Solve(LList<char *> &imageFiles, int from = 0, int to = -1);

		//��ʼ���ͷ���ʼ�������
		void InitSolver(char *imageFileName, LLabelImage &labelImage);
		void UnInitSolver();
		void Solve(char *imageFileName);

		//alpha-expand�㷨
		int Expand(LCrfDomain *domain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes);
		//�������,�������Խ��
		void Confusion(LList<char *> &imageFiles, char *confusionFileName);
		void Confusion(LList<char *> &imageFiles, char *confusionFileName, int maxError);
};

#endif