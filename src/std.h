#ifndef __std
#define __std

//���߳�
//#define MULTITHREAD
#define MAXTHREAD 64

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <algorithm>

#include <opencv2/opencv.hpp>

#ifdef _WIN32
#include <windows.h>
#include <conio.h>

#else
#include <unistd.h>
#include <pthread.h>
#endif


#ifndef _WIN32
class LCrfDomain;
class LDataset;
class LGreyImage;
class LRgbImage;
class LLuvImage;
class LLabImage;
class LLabelImage;
class LCostImage;
class LSegmentImage;
class LCrf;
class LCrfLayer;
class LBaseCrfLayer;
class LPnCrfLayer;
class LPreferenceCrfLayer;
#endif


//��ѧ����
namespace LMath
{
	static const double pi = (double)3.1415926535897932384626433832795;
	static const double positiveInfinity = (double)1e50;
	static const double negativeInfinity = (double)-1e50;
	static const double almostZero = (double)1e-12;

	void SetSeed(unsigned int seed);
	unsigned int RandomInt();
	unsigned int RandomInt(unsigned int maxval);
	unsigned int RandomInt(unsigned int minval, unsigned int maxval);
	double RandomReal();
	double RandomGaussian(double mi, double var);

	double SquareEuclidianDistance(double *v1, double *v2, int size);
	double KLDivergence(double *histogram, double *referenceHistogram, int size, double threshold);
	double GetAngle(double x, double y);
};

//����ģ��
template <class T>
class LList
{
	private :
		int count, capacity;	//Ԫ�ظ���������
		T *items;	//Ԫ��ָ��

		//�����С
		void Resize(int size);
		//����
		void QuickSort(int from, int to, int (*sort)(T, T));
	public :
		LList();
		~LList();
		
		//������
		T &operator[](int index);
		//���
		T &Add(T value);
		//����
		T &Insert(T value, int index);
		//ɾ��
		void Delete(int index);
		//����
		void Swap(int index1, int index2);
		//����
		void Sort(int (*sort)(T, T));
		//����Ԫ��ָ��
		T *GetArray();
		//����Ԫ�ظ���
		int GetCount();
		//���
		void Clear();
};

#ifdef _WIN32
#define thread_type HANDLE
#define thread_return DWORD
#define thread_defoutput 1
#else
#define thread_type pthread_t
#define thread_return void *
#define thread_defoutput NULL
#endif

//#ifdef _WIN32
//������
void _error(char *str);
//��ȡ�ļ���
char *GetFileName(const char *folder, const char *name, const char *extension);
//�����ļ���
void ForceDirectory(const char *dir);
void ForceDirectory(const char *dir, const char *subdir);
//��ȡ������ID
int GetProcessors();
//��ʼ���ٽ���
void InitializeCriticalSection();
//ɾ���ٽ���
void DeleteCriticalSection();
//�����ٽ���
void EnterCriticalSection();
//�˳��ٽ���
void LeaveCriticalSection();
//�������߳�
thread_type NewThread(thread_return (*routine)(void *), void *param);
//�߳����
int ThreadFinished(thread_type thread);
//�ر��߳�
void CloseThread(thread_type *thread);
//#endif

#endif