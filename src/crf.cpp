#include "std.h"
#include "crf.h"
#include <string.h>
#include <sys/time.h>
#ifdef _WIN32
#include <windows.h>
#endif



LCrfLayer::LCrfLayer(LCrf *setCrf, LCrfDomain *setDomain, LDataset *setDataset, LCrfLayer *setParent, int setRange)
{
	crf = setCrf, domain = setDomain, dataset = setDataset;
	classNo = domain->classNo;
	parent = setParent;
	labels = NULL, active = NULL;
	nodeOffset = 0, range = setRange;
}

LCrfLayer::~LCrfLayer()
{
	if (labels != NULL) delete[] labels;
	if (active != NULL) delete[] active;
}

int LCrfLayer::UpdateLabels(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes)
{
	return(0);
}

int LCrfLayer::BinaryNodes()
{
	return(0);
}

int LCrfLayer::GetPairwiseNodeCount()
{
	return(0);
}

LBaseCrfLayer::LBaseCrfLayer(LCrf *setCrf, LCrfDomain *setDomain, LDataset *setDataset, int setRange) : LCrfLayer(setCrf, setDomain, setDataset, NULL, setRange)
{
	width = height = 0;
	domain->baseLayer = this;
}

LBaseCrfLayer::~LBaseCrfLayer()
{
}

//�ײ��ʼ��
void LBaseCrfLayer::Initialize(char *imageFileName, int onFly)
{
#ifdef MULTITHREAD
	EnterCriticalSection();
#endif
	//��ͼ��
	char *fileName;
	fileName = GetFileName(dataset->imageFolder, imageFileName, dataset->imageExtension);
	LRgbImage rgbImage(fileName);
	delete[] fileName;
#ifdef MULTITHREAD
	LeaveCriticalSection();
#endif
	width = rgbImage.GetWidth(), height = rgbImage.GetHeight();

	//��ǩ��ʼ��Ϊȫ0
	if (labels != NULL) delete[] labels;
	labels = new unsigned char[width * height];
	memset(labels, 0, width * height * sizeof(unsigned char));
	if (active != NULL) delete[] active;
	active = new unsigned char[width * height];
}

void LBaseCrfLayer::UnInitialize()
{
	if (labels != NULL) delete[] labels;
	labels = NULL;
	if (active != NULL) delete[] active;
	active = NULL;
}

int LBaseCrfLayer::GetNodeCount()
{
	if (!range) return(width * height);
	else return(width * height * range);
}

int LBaseCrfLayer::GetPairwiseNodeCount()
{
	return(width * height);
}

int LBaseCrfLayer::GetEdgeCount()
{
	if (!range) return(0);
	else return(width * height * range);
}

int LBaseCrfLayer::BinaryNodes()
{
	return(1);
}

//�ײ�ͼģ�ͽ���
void LBaseCrfLayer::BuildGraph(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes)
{
	int i, j;
	//����ǵ�ǰģ�͵Ĳ㣬ѭ��
	if (domain == costDomain) for (i = 0; i < width * height; i++)
	{
		//�ж�rangeֵ��baselayer range=0
		if (!range)		//range=0
		{
			if (labels[i] != label)		//��ǰ���صı�ǩ�����ڵ�ǰ��ǩ
			{
				//��ǩ�Ƿ񼤻�
				active[i] = 0;		//���ü���״̬Ϊ0
				nodes[i + nodeOffset] = g->add_node();		//��ӽڵ�
			}
			else active[i] = 1;		//��ǰ���ر�ǩ���ڵ�ǰ��ǩ�����ü���״̬Ϊ1
		}
		//range!=0
		else
		{
			//�ڷ�Χ�����ü���״̬Ϊ0
			if ((labels[i] < label) || (labels[i] >= label + range)) active[i] = 0;
			else active[i] = 1;
			//ÿ��range���һ���ڵ�
			for (j = 0; j < range; j++) nodes[i * range + j + nodeOffset] = g->add_node();		//��ӽڵ�
		}
	}
}

//�ײ�������ǩ
int LBaseCrfLayer::UpdateLabels(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes)
{
	int i, change = 0;
	if (domain == costDomain)
		for (i = 0; i < width * height; i++)
		{
			if (!range)
			{
				if ((!active[i]) && (g->what_segment(nodes[i + nodeOffset]) == Graph<double, double, double>::SINK)) labels[i] = label, change = 1;
			}
			else
			{
				int j = 0;
				while ((j < range) && (g->what_segment(nodes[i * range + j + nodeOffset]) == Graph<double, double, double>::SINK)) j++;
				if ((j > 0) && (j - 1 + label != labels[i])) labels[i] = j - 1 + label, change = 1;
			}
		}
	return(change);
}

LPnCrfLayer::LPnCrfLayer(LCrf *setCrf, LCrfDomain *setDomain, LDataset *setDataset, LCrfLayer *setParent, LSegmentation2D *setSegmentation, double setTruncation) : LCrfLayer(setCrf, setDomain, setDataset, setParent, 0)
{
	segmentCounts = NULL;
	segmentIndexes = NULL;
	weights = NULL;
	weightSums = NULL;
	segmentation = setSegmentation;
	truncation = setTruncation;
	segFolder = NULL;
	segmentCount = 0;
}

LPnCrfLayer::LPnCrfLayer(LCrf *setCrf, LCrfDomain *setDomain, LDataset *setDataset, LCrfLayer *setParent, const char *setSegFolder, const char *setSegExtension, double setTruncation) : LCrfLayer(setCrf, setDomain, setDataset, setParent, 0)
{
	segmentCounts = NULL;
	segmentIndexes = NULL;
	baseSegmentCounts = NULL;
	baseSegmentIndexes = NULL;
	weights = NULL;
	weightSums = NULL;
	segmentation = NULL;
	truncation = setTruncation;
	segFolder = setSegFolder;
	segExtension = setSegExtension;
	segmentCount = 0;
}

LPnCrfLayer::~LPnCrfLayer()
{
	int i;
	if (segmentIndexes != NULL)
	{
		for (i = 0; i < segmentCount; i++) if (segmentIndexes[i] != NULL) delete[] segmentIndexes[i];
		delete[] segmentIndexes;
	}
	if (weights != NULL)
	{
		for (i = 0; i < segmentCount; i++) if (weights[i] != NULL) delete[] weights[i];
		delete[] weights;
	}
	if (segmentCounts != NULL) delete[] segmentCounts;
	if (weightSums != NULL) delete[] weightSums;
}

//�߽ײ��ʼ�����յ��ע֮һ
void LPnCrfLayer::Initialize(char *imageFileName, int onFly)
{
	int i, j;

	//������ڷָ�
	if (segmentation != NULL)
	{
		LSegmentImage segmentImage;
		if (!onFly)		//���ļ�����
		{
			char *fileName;
			fileName = GetFileName(segmentation->folder, imageFileName, segmentation->extension);
			segmentImage.Load(fileName);
			delete[] fileName;
		}
		else	//��ʱ�ָ�
		{
#ifdef MULTITHREAD
			EnterCriticalSection();
#endif
			//����ͼ��
			char *fileName;
			fileName = GetFileName(dataset->imageFolder, imageFileName, dataset->imageExtension);
			//LUV�ռ�
			LLuvImage luvImage(fileName);
			delete[] fileName;
#ifdef MULTITHREAD
			LeaveCriticalSection();
#endif
			//���շָ�ָ�ͼ��
			segmentation->Segment(luvImage, segmentImage);
		}

		//�ָ�ͼ���������������ص�����
		int points = segmentImage.GetPoints();		//�������ظ�������ͼ���Сwidth*height

		segmentCount = 0;	//�޼ල�ָ��������
		//ָ�����ݵ�ָ��
		int *segmentData = segmentImage.GetData();
		//ð��һ�֣��ҵ�������±�+1�õ��ָ�ͼ��ľ������ĸ��������ڵ���
		for (j = 0; j < points; j++, segmentData++) if (*segmentData + 1 > segmentCount) segmentCount = *segmentData + 1;

		//�����ָ��������һ�㳬���ز�
		baseSegmentCounts = new int[segmentCount];		//ͳ��ÿ��������������
		memset(baseSegmentCounts, 0, segmentCount * sizeof(int));		//��ʼ��0

		//ָ���ٴ�ָ��ָ�ͼ
		segmentData = segmentImage.GetData();
		//��ֱ��ͼͳ��ÿ���ڵ����ظ���
		for (j = 0; j < points; j++, segmentData++) baseSegmentCounts[*segmentData]++;

		//��ά���飬�ָ�������ָ��ÿ������ÿ�����ص�ָ��
		baseSegmentIndexes = new int *[segmentCount];
		//����ռ�
		for (j = 0; j < segmentCount; j++) baseSegmentIndexes[j] = new int[baseSegmentCounts[j]];

		int *segmentationIndex;		//��¼ÿ�����ĵ�ǰ�������Ǹ��м����
		segmentationIndex = new int[segmentCount];		//�����
		memset(segmentationIndex, 0, segmentCount * sizeof(int));		//��ʼ��ȫ0

		//ָ������ָ��segmentData
		segmentData = segmentImage.GetData();

		//����ÿ���ָԪ���±꣬����������
		for (j = 0; j < points; j++, segmentData++)
		{
			//����ÿ�����ص�����
			baseSegmentIndexes[*segmentData][segmentationIndex[*segmentData]] = j;
			segmentationIndex[*segmentData]++;		//����
		}
		if (segmentationIndex != NULL) delete[] segmentationIndex;		//�ͷ��ڴ�

		//������������ز���߸���ָ���Ϊ0��BinaryNodes()Ĭ�Ϸ���0��BaseLayer����1��PnLayer����2
		if ((parent->BinaryNodes() == 1) || (!((LPnCrfLayer *)parent)->segmentCount))
		{
			//���������ز�
			segmentCounts = baseSegmentCounts;		//���������������
			segmentIndexes = baseSegmentIndexes;		//��������

			//Ȩ�ظ���
			weightSums = new double[segmentCount];		//ÿ������Ȩ��
			//��ά���飬ָ��Ȩ��
			weights = new double *[segmentCount];		//ÿ�����ص�Ȩ��

			//����Ȩ��
			for (j = 0; j < segmentCount; j++)
			{
				weightSums[j] = (double)segmentCounts[j];	//ÿ������Ȩ�ص�����������ظ���
				weights[j] = new double[segmentCounts[j]];		//����ռ�
				for (i = 0; i < segmentCounts[j]; i++) weights[j][i] = 1.0;		//����Ȩ�ض���1.0
			}
		}
		else
		{
			//�����ǳ����ز㣬���������ز�
			//��ȡ��������
			int parentCount = ((LPnCrfLayer *)parent)->segmentCount;	//����ķָ���
			int *parentCounts = ((LPnCrfLayer *)parent)->baseSegmentCounts;		//����ÿ������������
			int **parentIndexes = ((LPnCrfLayer *)parent)->baseSegmentIndexes;		//����ÿ��������������

			//��ǰ����Ը���ķָ���
			int *counts = new int[segmentCount * parentCount];
			memset(counts, 0, segmentCount * parentCount * sizeof(int));	//��ʼ��ȫ0

			segmentData = segmentImage.GetData();		//ָ��ָ�ͼ��ָ��

			//ͳ�Ʊ����븸��Ķ�Ӧ��ϵ��������������е�������������������
			//������ÿ������еı���������ĳɷ�
			for (i = 0; i < parentCount; i++)		//����ָ���
			{
				for (j = 0; j < parentCounts[i]; j++)		//����ÿ�����������
				{
					//��ȡ��������
					int index = ((LPnCrfLayer *)parent)->baseSegmentIndexes[i][j];
					//�ڸ���ͱ�������
					counts[i * segmentCount + segmentData[index]]++;
				}
			}
			segmentCounts = new int[segmentCount];		//����ÿ����������ͳ��
			segmentIndexes = new int *[segmentCount];		//����ÿ��������������
			weightSums = new double[segmentCount];		//�������Ȩ��
			weights = new double *[segmentCount];		//��������Ȩ��

			//���ݱ�������Ȩ��
			for (i = 0; i < segmentCount; i++)		//��ǰ��ÿ����
			{
				//ͳ�Ʊ���ÿ������ڸ�����ÿ��������ཻ�ķ�0����
				int nonzero = 0;
				for (j = 0; j < parentCount; j++) if (counts[j * segmentCount + i] > 0) nonzero++;
				segmentCounts[i] = nonzero;		//�����и���ĳ����ظ���
				weightSums[i] = (double)baseSegmentCounts[i];		//����������Ȩ��

				segmentIndexes[i] = new int[nonzero];		//��������
				weights[i] = new double[nonzero];

				int index = 0;
				for (j = 0; j < parentCount; j++) if (counts[j * segmentCount + i] > 0)
				{
					segmentIndexes[i][index] = j;		//�ڸ����е�����
					weights[i][index] = counts[j * segmentCount + i];		//�Ը����еĳ����ظ�����ΪȨ��
					index++;
				}

			}
			if (counts != NULL) delete[] counts;
		}
	}
	//��������ڣ����ļ�
	else
	{
		char *fileName;
		FILE *f;

		fileName = GetFileName(segFolder, imageFileName, segExtension);
		f = fopen(fileName, "rb");
		fread(&segmentCount, sizeof(int), 1, f);

		baseSegmentCounts = new int[segmentCount];
		baseSegmentIndexes = new int *[segmentCount];

		for (i = 0; i < segmentCount; i++)
		{
			fread(&baseSegmentCounts[i], sizeof(int), 1, f);
			baseSegmentIndexes[i] = new int[baseSegmentCounts[i]];
			if (baseSegmentCounts[i] != 0) fread(baseSegmentIndexes[i], sizeof(int), baseSegmentCounts[i], f);
		}
		fclose(f);

		segmentCounts = baseSegmentCounts;
		segmentIndexes = baseSegmentIndexes;

		weightSums = new double[segmentCount];
		weights = new double *[segmentCount];

		for (j = 0; j < segmentCount; j++)
		{
			weightSums[j] = (double)segmentCounts[j];
			weights[j] = new double[segmentCounts[j]];
		}
		for (j = 0; j < segmentCount; j++) for (i = 0; i < segmentCounts[j]; i++) weights[j][i] = 1.0;
	}
	if (labels != NULL) delete[] labels;
	labels = new unsigned char[segmentCount];
	if (active != NULL) delete[] active;
	active = new unsigned char[segmentCount];
}

int LPnCrfLayer::GetNodeCount()
{
	return(2 * segmentCount);
}

int LPnCrfLayer::GetPairwiseNodeCount()
{
	return(segmentCount);
}

int LPnCrfLayer::GetEdgeCount()
{
	return(2 * parent->GetPairwiseNodeCount() + segmentCount);
}

int LPnCrfLayer::BinaryNodes()
{
	return(2);
}

void LPnCrfLayer::UnInitialize()
{
	int i;

	if ((baseSegmentIndexes != segmentIndexes) && (segmentIndexes != NULL))
	{
		for (i = 0; i < segmentCount; i++) if (segmentIndexes[i] != NULL) delete[] segmentIndexes[i];
		delete[] segmentIndexes;
	}
	if ((baseSegmentCounts != segmentCounts) && (segmentCounts != NULL)) delete[] segmentCounts;

	if (baseSegmentIndexes != NULL)
	{
		for (i = 0; i < segmentCount; i++) if (baseSegmentIndexes[i] != NULL) delete[] baseSegmentIndexes[i];
		delete[] baseSegmentIndexes;
	}
	if (baseSegmentCounts != NULL) delete[] baseSegmentCounts;

	if (weights != NULL)
	{
		for (i = 0; i < segmentCount; i++) if (weights[i] != NULL) delete[] weights[i];
		delete[] weights;
	}
	if (weightSums != NULL) delete[] weightSums;
	if (labels != NULL) delete[] labels;
	if (active != NULL) delete[] active;

	baseSegmentCounts = NULL;
	baseSegmentIndexes = NULL;
	segmentCounts = NULL;
	segmentIndexes = NULL;
	weightSums = NULL;
	weights = NULL;
	labels = NULL;
	active = NULL;
	segmentCount = 0;
}

void LPnCrfLayer::SetLabels()
{
	unsigned char *parentLabels = parent->labels;
	int i, j;

	//���ݸ���ı�ǩ������������ǩ
	double *dominant = new double[classNo];

	for (i = 0; i < segmentCount; i++)
	{
		labels[i] = classNo;
		memset(dominant, 0, classNo * sizeof(double));
		//ÿ����ǩ�����ظ���
		for (j = 0; j < segmentCounts[i]; j++) dominant[parentLabels[segmentIndexes[i][j]]] += weights[i][j];
		//���ݽض���ֵ��ֵ��ǩ����ĳ����ǩ������������ĳһ��ֵ
		for (j = 0; j < classNo; j++) if (dominant[j] > weightSums[i] * (1 - truncation)) labels[i] = j;	//�ضϸ���
	}
	delete[] dominant;
}


//�߽ײ㽨��ͼģ��
void LPnCrfLayer::BuildGraph(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes)
{
	int i;
	//����ǵ�ǰģ�͵Ĳ㣬ÿ��������ѭ��
	if (domain == costDomain) for (i = 0; i < segmentCount; i++)
	{
		if (labels[i] != label)		//��������ڵ�ǰ��ǩ
		{
			active[i] = 0;		//������
			nodes[2 * i + nodeOffset] = g->add_node();		//ÿ�������������ڵ�
			//�����ǩ�������ɱ�ǩ���ڶ����ڵ����
			if (labels[i] != classNo) nodes[2 * i + 1 + nodeOffset] = g->add_node();
		}
		else active[i] = 1;		//������ڵ�ǰ��ǩ�����ü���
	}
}

int LPnCrfLayer::UpdateLabels(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes)
{
	int i, change = 0;

	if (domain == costDomain) for (i = 0; i < segmentCount; i++) if ((!active[i]))
	{
		if (g->what_segment(nodes[2 * i + nodeOffset]) == Graph<double, double, double>::SINK) labels[i] = label, change = 1;
		else if ((labels[i] != classNo) && (g->what_segment(nodes[2 * i + nodeOffset]) != g->what_segment(nodes[2 * i + nodeOffset + 1]))) labels[i] = classNo, change = 1;
	}
	return(change);
}

LPreferenceCrfLayer::LPreferenceCrfLayer(LCrf *setCrf, LCrfDomain *setDomain, LDataset *setDataset, LCrfLayer *setParent) : LCrfLayer(setCrf, setDomain, setDataset, setParent, 0)
{
}

LPreferenceCrfLayer::~LPreferenceCrfLayer()
{
}

int LPreferenceCrfLayer::GetNodeCount()
{
	return(classNo);
}

int LPreferenceCrfLayer::GetEdgeCount()
{
	return(0);
}

void LPreferenceCrfLayer::Initialize(char *imageFileName, int onFly)
{
	if (active != NULL) delete[] active;
	active = new unsigned char[classNo];
}

void LPreferenceCrfLayer::UnInitialize()
{
	if (active != NULL) delete[] active;
	active = NULL;
}

//���Ȳ�ͼģ�ͽ���
void LPreferenceCrfLayer::BuildGraph(LCrfDomain *costDomain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes)
{
	if (domain == costDomain)
	{
		//����ĵ����Ŀ
		int i, nodeCount = parent->GetPairwiseNodeCount();
		memset(active, 1, classNo * sizeof(unsigned char));		//ȫ������

		for (i = 0; i < nodeCount; i++) active[parent->labels[i]] = 0;		//���ֲ�����

		for (i = 0; i < classNo; i++)
		{
			if (((i == label) && (active[i])) || ((i != label) && (!active[i]))) nodes[i + nodeOffset] = g->add_node();
		}
	}
}

LCrfDomain::LCrfDomain(LCrf *setCrf, LDataset *setDataset, int setClassNo, const char *setTestFolder, void (LDataset::*setRgbToLabel)(unsigned char *, unsigned char *), void (LDataset::*setLabelToRgb)(unsigned char *, unsigned char *))
{
	crf = setCrf, dataset = setDataset;
	rgbToLabel = setRgbToLabel;
	labelToRgb = setLabelToRgb;
	classNo = setClassNo;
	testFolder = setTestFolder;
	baseLayer = NULL;
}

//����CRFģ�Ͷ��󣬹��캯������ֻ�����ݼ��������ݲ�ͬ���ݼ�����ģ��
LCrf::LCrf(LDataset *setDataset)
{
	//�������ݼ�
	dataset = setDataset;
}

LCrf::~LCrf()
{
	int i;
	for (i = 0; i < layers.GetCount(); i++) delete layers[i];
	for (i = 0; i < potentials.GetCount(); i++) delete potentials[i];
	for (i = 0; i < features.GetCount(); i++) delete features[i];
	for (i = 0; i < learnings.GetCount(); i++) delete learnings[i];
}

void LCrf::TrainFeatures(LList<char *> &imageFiles)
{
	//ѵ�����ĸ���
	for (int i = 0; i < features.GetCount(); i++)
	{
		printf("Training feature %d..\n", i);
		features[i]->Train(imageFiles);	//ѵ��ͼ��,xѵ����ͬ����
	}
}

void LCrf::EvaluateFeatures(LList<char *> &imageFiles, int from, int to)
{
	if (to == -1) to = imageFiles.GetCount();		//��ȡ�ļ���

	int i;
	//��������
	for (i = 0; i < features.GetCount(); i++) features[i]->LoadTraining();
	for (i = 0; i < features.GetCount(); i++)
	{
		printf("Evaluating feature %d..\n", i);
		features[i]->Evaluate(imageFiles, from, to);		//�ֱ���ȡ��ͬ����
	}
}

void LCrf::TrainPotentials(LList<char *> &imageFiles)
{
	int i;
	//��������
	for (i = 0; i < features.GetCount(); i++) features[i]->LoadTraining();
	//ѵ��
	for (i = 0; i < potentials.GetCount(); i++)
	{
		if (i==1)
		{
			printf("Training potential %d..\n", i);
			potentials[i]->Train(imageFiles);
		}
		
	}
}

void LCrf::EvaluatePotentials(LList<char *> &imageFiles, int from, int to)
{
	if (to == -1) to = imageFiles.GetCount();

	int i;

	//���������Լ�������
	for (i = 0; i < features.GetCount(); i++) features[i]->LoadTraining();
	for (i = 0; i < learnings.GetCount(); i++) learnings[i]->LoadTraining();
	for (i = 0; i < potentials.GetCount(); i++)
	{
		if (i==1)
		{
			printf("Evaluating potentials %d..\n", i);
			//�����ƺ���
			potentials[i]->LoadTraining();
			potentials[i]->Evaluate(imageFiles, from, to);
		}
		
	}
}

#ifdef MULTITHREAD

struct LSegmentParams
{
	LCrf *crf;
	char *fileName;
};

thread_return SegmentThread(void *par)
{
	LSegmentParams *params = (LSegmentParams *)par;
	params->crf->Segment(params->fileName);
	return(thread_defoutput);
}
#endif

void LCrf::Segment(LList<char *> &imageFiles, int from, int to)
{
	if (to == -1) to = imageFiles.GetCount();		//���û�У�����ȫ����
#ifdef MULTITHREAD
	int i;
	int processors = GetProcessors(), running = 0;
	thread_type *threads;
	LSegmentParams *params;

	threads = new thread_type[processors];
	memset(threads, 0, processors * sizeof(thread_type));
	params = new LSegmentParams[processors];
	InitializeCriticalSection();

	int index = 0;

	index = from;

	for (i = 0; i < processors; i++) if (index < to)
	{
		params[i].crf = this, params[i].fileName = imageFiles[index];

		threads[i] = NewThread(SegmentThread, &params[i]);
		if (threads[i] != 0)
		{
			printf("Segmenting image %d..\n", index);
			index++;
			running++;
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

				if (index < to)
				{
					params[i].fileName = imageFiles[index];
					threads[i] = NewThread(SegmentThread, &params[i]);

					if (threads[i] != 0)
					{
						printf("Segmenting image %d..\n", index);
						index++;
						running++;
					}
				}
			}
		}
		Sleep(0);
	}
	DeleteCriticalSection();
	delete[] threads;
	delete[] params;
#else
	//�ָ�����ͼ��
	for (int j = from; j < to; j++)
	{
		printf("Segmenting image %d %s..\n", j, imageFiles[j]);
		Segment(imageFiles[j]);
	}
#endif
}

void LCrf::Segment(char *imageFileName)
{
	int i;
	//����ͼ��ָ���
	LSegmentImage segmentImage;

#ifdef MULTITHREAD
	EnterCriticalSection();
#endif
	char *fileName;
	//��ȡͼ���ļ���
	fileName = GetFileName(dataset->imageFolder, imageFileName, dataset->imageExtension);//�ļ��У��ļ�������չ��

	//��ȡͼ��ת����LUV��ɫ�ռ�
	LLuvImage luvImage(fileName);
	delete[] fileName;
#ifdef MULTITHREAD
	LeaveCriticalSection();
#endif
	//��ͬ�߶ȣ����ָ�ͼ��
	for (i = 0; i < segmentations.GetCount(); i++)
	{
		//�����ļ���
		char *fileName;
		fileName = GetFileName(segmentations[i]->folder, imageFileName, segmentations[i]->extension);
		segmentations[i]->Segment(luvImage, segmentImage);
		if (segmentImage.GetPoints() > 0) segmentImage.LImage<int>::Save(fileName);

		delete[] fileName;
	}
}

void LCrf::InitSolver(char *imageFileName, LLabelImage &labelImage)
{
#ifdef MULTITHREAD
	EnterCriticalSection();
#endif
	//����Labͼ��
	char *fileName;
	fileName = GetFileName(dataset->imageFolder, imageFileName, dataset->imageExtension);
	LLabImage labImage(fileName);
	delete[] fileName;


#ifdef MULTITHREAD
	LeaveCriticalSection();
#endif
	//���ñ�ǩͼ���С
	labelImage.SetResolution(labImage.GetWidth(), labImage.GetHeight());

	//�ڵ���������
	int nodeCount = 0, edgeCount = 0, i;

	//��ʼ��ÿһ��
	for (i = 0; i < layers.GetCount(); i++) layers[i]->Initialize(imageFileName);
	//��ʼ���ƺ���
	for (i = 0; i < potentials.GetCount(); i++) potentials[i]->Initialize(labImage, imageFileName);
	//��ÿһ�����ó�ʼ��ǩ
	for (i = 0; i < layers.GetCount(); i++) layers[i]->SetLabels();

	//���ǳ���ģ��ƺ����Ǿ����
	//��ȡÿһ��Ľڵ����ͱ���
	for (i = 0; i < layers.GetCount(); i++)
	{
		layers[i]->nodeOffset = nodeCount;		//���ýڵ�ƫ��

		nodeCount += layers[i]->GetNodeCount();
		edgeCount += layers[i]->GetEdgeCount();
	}
	//��ȡÿ���ƺ����Ľڵ����ͱ���
	for (i = 0; i < potentials.GetCount(); i++)
	{
		potentials[i]->nodeOffset = nodeCount;

		nodeCount += potentials[i]->GetNodeCount();
		edgeCount += potentials[i]->GetEdgeCount();
	}

	//�½�һ����������ͼ
	g = new Graph<double, double, double>(nodeCount, edgeCount);
	//�½��ڵ�
	nodes = new Graph<double, double, double>::node_id[nodeCount];
}

//����ʼ�������
void LCrf::UnInitSolver()
{
	delete g;
	delete[] nodes;

	int i;
	for (i = 0; i < layers.GetCount(); i++) layers[i]->UnInitialize();
	for (i = 0; i < potentials.GetCount(); i++) potentials[i]->UnInitialize();
}
extern float t_spend;
//CRFģ�����
void LCrf::Solve(char *imageFileName)
{
	LLabelImage labelImage;
	//��ʼ�������
	InitSolver(imageFileName, labelImage);

	struct timeval tv1,tv2;
	gettimeofday(&tv1,NULL);
	//���ò�����������������
	int maxIterations = 4, iter, freeSteps = 0, index, i;		//maxiter 4->8

	int totalLabels = 0;
	for (i = 0; i < domains.GetCount(); i++) totalLabels += domains[i]->classNo - ((domains[i]->baseLayer->range) ? domains[i]->baseLayer->range + 1 : 0);

	//��������������������
	for (iter = 0; (iter < maxIterations) && (freeSteps < totalLabels); iter++)
	{
		//ÿ��crf����
		for (i = 0; i < domains.GetCount(); i++)
		{
			//ÿ����ǩ
			for (index = 0; (index < domains[i]->classNo - ((domains[i]->baseLayer->range) ? domains[i]->baseLayer->range + 1 : 0)) && (freeSteps < totalLabels); index++)
			{
				//freeSteps�����ж�����
				//ÿһ����alpha-expansion�㷨������ǩ������ƶ�����Ч����Ч������1
				if (!Expand(domains[i], index, g, nodes)) freeSteps++;
				//�����Ч������Ч����Ϊ1
				else freeSteps = 1;
				//����ͼ
				g->reset();
			}
		}
	}
	gettimeofday(&tv2,NULL);
	float t_spend_frame=(tv2.tv_sec+tv2.tv_usec/1e+6)-(tv1.tv_sec+tv1.tv_usec/1e+6);
	t_spend+=t_spend_frame;
	//������ͼ��
	for (i = 0; i < domains.GetCount(); i++)
	{
		//�ײ�����ս��
		memcpy(labelImage.GetData(), domains[i]->baseLayer->labels, labelImage.GetPoints() * sizeof(unsigned char));

		char *fileName;
		fileName = GetFileName(domains[i]->testFolder, imageFileName, dataset->groundTruthExtension);
#ifdef MULTITHREAD
		EnterCriticalSection();
#endif
		//������ͼ��
		dataset->SaveImage(labelImage, domains[i], fileName);
#ifdef MULTITHREAD
		LeaveCriticalSection();
#endif
		delete[] fileName;
	}
	//����ʼ��
	UnInitSolver();
	}

//graph-cut based move-making�㷨�������߸Ľ���alpha-expansion�㷨
int LCrf::Expand(LCrfDomain *domain, unsigned char label, Graph<double, double, double> *g, Graph<double, double, double>::node_id *nodes)
{
	int i;
	for (i = 0; i < layers.GetCount(); i++) layers[i]->BuildGraph(domain, label, g, nodes);
	for (i = 0; i < potentials.GetCount(); i++) potentials[i]->AddCosts(domain, label, g, nodes);
	//ͼ���㷨��ÿһ��expansion����ͼ��Ѱ����ѷָ�
	g->maxflow();
	//�ƶ��Ƿ���Ч
	int change = 0;
	for (i = 0; i < layers.GetCount(); i++)
		//����alpha-expansion�ƶ��Ƿ���Ч
		if (layers[i]->UpdateLabels(domain, label, g, nodes))
			change = 1;		//�����һ��ı�ǩ���ˣ�����Ч
	//���ر�־λ
	return(change);
}

#ifdef MULTITHREAD
struct LSolveParams
{
	LCrf *crf;
	char *fileName;
};

thread_return SolveThread(void *par)
{
	LSolveParams *params = (LSolveParams *)par;
	params->crf->Solve(params->fileName);
	return(thread_defoutput);
}
#endif

void LCrf::Solve(LList<char *> &imageFiles, int from, int to)
{
	if (to == -1) to = imageFiles.GetCount();
#ifdef MULTITHREAD
	int i, ind = from;
	int processors = GetProcessors(), running = 0;
	thread_type *threads;
	LSolveParams *params;
	LCrf **crfs;

	threads = new thread_type[processors];
	memset(threads, 0, processors * sizeof(thread_type));
	params = new LSolveParams[processors];
	crfs = new LCrf *[processors];
	InitializeCriticalSection();

	for (i = 0; i < processors; i++)
	{
		crfs[i] = new LCrf(dataset);
		dataset->SetCRFStructure(crfs[i]);
		for (int j = 0; j < crfs[i]->features.GetCount(); j++) crfs[i]->features[j]->LoadTraining();
		for (int j = 0; j < crfs[i]->potentials.GetCount(); j++) crfs[i]->potentials[j]->LoadTraining();
	}

	for (i = 0; i < processors; i++) if (ind < to)
	{
		params[i].crf = crfs[i], params[i].fileName = imageFiles[ind];
		threads[i] = NewThread(SolveThread, &params[i]);
		if (threads[i] != 0)
		{
			printf("Solving image %d..\n", ind);
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
					threads[i] = NewThread(SolveThread, &params[i]);
					if (threads[i] != 0)
					{
						printf("Solving image %d..\n", ind);
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
	int i;
	//��������
	for (i = 0; i < features.GetCount(); i++) features[i]->LoadTraining();
	//�����ƺ���
	for (i = 0; i < potentials.GetCount(); i++) potentials[i]->LoadTraining();
	for (i = from; i < to; i++)
	{
		printf("Solving image %d %s..\n", i, imageFiles[i]);
		Solve(imageFiles[i]);
	}
#endif
}

//�ɻ����ݴ���
void LCrf::Confusion(LList<char *> &imageFiles, char *confusionFileName)
{
	int *pixTotalClass, *pixOkClass, *confusion, *pixLabel, i;
	const char *saveFolder = dataset->testFolder;

	pixTotalClass = new int[dataset->classNo];
	pixOkClass = new int[dataset->classNo];
	pixLabel = new int[dataset->classNo];
	confusion = new int[dataset->classNo * dataset->classNo];
	memset(pixTotalClass, 0, dataset->classNo * sizeof(int));
	memset(pixOkClass, 0, dataset->classNo * sizeof(int));
	memset(pixLabel, 0, dataset->classNo * sizeof(int));
	memset(confusion, 0, dataset->classNo * dataset->classNo * sizeof(int));

	unsigned int pixTotal = 0, pixOk = 0;

	for (i = 0; i < imageFiles.GetCount(); i++)
	{
		printf("Estimating image %d %s..\n", i, imageFiles[i]);
		char *fileName, *fileName2;
		fileName = GetFileName(saveFolder, imageFiles[i], dataset->groundTruthExtension);
		fileName2 = GetFileName(dataset->groundTruthFolder, imageFiles[i], dataset->groundTruthExtension);

		LLabelImage labels(fileName, dataset, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::RgbToLabel), groundTruth(fileName2, dataset, (void (LDataset::*)(unsigned char *, unsigned char *))&LDataset::RgbToLabel);

		unsigned char *labelData = labels.GetData();
		unsigned char *gtData = groundTruth.GetData();
		int points = groundTruth.GetPoints();

		for (int l = 0; l < points; l++, gtData++, labelData++)
		{
			if (*gtData != 0)
			{
				pixTotal++;
				pixTotalClass[*gtData - 1]++;
				pixLabel[*labelData - 1]++;
				if (*gtData == *labelData) pixOk++, pixOkClass[*gtData - 1]++;
				confusion[(*gtData - 1) * dataset->classNo + *labelData - 1]++;
			}
		}
		delete[] fileName;
		delete[] fileName2;
	}
	double average = (double)0.0, waverage = 0.0;

	for (i = 0; i < dataset->classNo; i++)
	{
		average += (pixTotalClass[i] == 0) ? 0 : pixOkClass[i] / (double)pixTotalClass[i];
		waverage += (pixTotalClass[i] + pixLabel[i] - pixOkClass[i] == 0) ? 0 : pixOkClass[i] / (double)(pixTotalClass[i] + pixLabel[i] - pixOkClass[i]);
	}
	average /= dataset->classNo, waverage /= dataset->classNo;

	FILE *ff;
	ff = fopen(confusionFileName, "w");
	for (int q = 0; q < dataset->classNo; q++)
	{
		for (int w = 0; w < dataset->classNo; w++) fprintf(ff, "%.3f ", (pixTotalClass[q] == 0) ? 0 : (confusion[q * dataset->classNo + w] / (double)pixTotalClass[q]));
		fprintf(ff, "\n");
	}
	fprintf(ff, "\n");
	for (int q = 0; q < dataset->classNo; q++) fprintf(ff, "%.3f ", (pixTotalClass[q] + pixLabel[q] - pixOkClass[q] == 0) ? 0 : pixOkClass[q] / (double)(pixTotalClass[q] + pixLabel[q] - pixOkClass[q]));
	fprintf(ff, "\n");
	for (int q = 0; q < dataset->classNo; q++) fprintf(ff, "%.3f ", (pixTotalClass[q] == 0) ? 0 : pixOkClass[q] / (double)pixTotalClass[q]);
	fprintf(ff, "\n");

	fprintf(ff, "overall %.4f, average %.4f, waverage %.4f\n", (pixTotal != 0) ? pixOk / (double)pixTotal : (double)0.0, average, waverage);
	fclose(ff);

	delete[] pixTotalClass;
	delete[] pixLabel;
	delete[] pixOkClass;
	delete[] confusion;
}
