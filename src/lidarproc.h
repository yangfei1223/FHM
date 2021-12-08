//
// Created by yangfei on 17-4-26.
//
/////////////////////////////////////////////////////////////////////////////////
//	Here, data contains 4 * num values, where the first 3 values correspond to
//	x, y and z, and the last value is the reflectance information.All scans
//	are stored row - aligned, meaning that the first 4 values correspond to the
//	first measurement.Since each scan might potentially have a different
//	number of points, this must be determined from the file size when reading
//	the file, where 1e6 is a good enough upper bound on the number of values :
/////////////////////////////////////////////////////////////////////////////////
#ifndef SEGCRF_LIDARPROC_H
#define SEGCRF_LIDARPROC_H

#include "./std.h"
#include <eigen3/Eigen/Eigen>
using namespace std;
//using namespace Eigen;
using namespace cv;
#define MAX_NUM				   1000000	//�����������ֵ
#define GRID_RESOLUTION        0.2		//դ��ֱ���
#define GRID_WIDTH             800		//դ����
#define GRID_HEIGHT            800		//դ��߶�

class cLidarProcess
{
private:
	Eigen::Matrix4f m_P;	//�ڲξ���
	Eigen::Matrix4f m_R0_rect;		//��������
	Eigen::Matrix4f m_Tr_velo_to_cam;		//���RT����
	int m_numPoint;		//ÿ֡���ݵ���Ч�����
	float *m_LidarData;
public:
	int m_width;	//ͼ���
	int m_height;	//ͼ���
	Mat m_Img;      //ͼ��
	Mat m_Grid;     //�״��դ��ͼ
	Mat m_Label;
	Mat m_Weights;		//����Ȩ��
	Mat m_Valid2Img;        //ӳ�䵽ͼ�����Ч��
	Mat m_Lidar2Img;    //ӳ�䵽ͼ��ƽ��ĵ���ԭʼ����
	Mat m_Curvature;     //curvature����
	Mat m_MaxHeightDiff;
	Mat m_HeightVar;

public:
	//���캯��
	cLidarProcess();
	//��������
	~cLidarProcess();
	//��ʼ��
	void Init();
	//�ͷ�
	void Dump();
	//����
	void Debug();
private:
	//��ͼ��
	void ReadImage(char *filename);
	//��ȡ��ǩ
	void ReadLabel(char *filename);
	//��ȡ�궨����
	int ReadCalibData(char *filename);
	//�����������
	int ReadPointCloudData(char *filename);
	//����ת��Ϊդ��
	void TransLidarPoint2Grid();
	//����ӳ�䵽ͼ��
	void TransLidarPoint2Img();

	//����һ���㣬Ѱ�ҹ̶��뾶��Χ�ڴ�N����
	void FindNearestNeighborsofPoint(Eigen::Vector3f &p, int px, int py, float R, Eigen::MatrixXf &output);
	//����Э�������
	void ComputeCovMatrix(Eigen::Vector3f &p, Eigen::MatrixXf &input, Eigen::Matrix3f &covMat);
	//����Э������������ֵ
	float ComputeEigenValofcovMat(Eigen::Matrix3f &covMat);
	//�������߶Ȳ�
	float ComputeMaxHeightDiff(Eigen::MatrixXf &input);
	//����Curvature����
	void Compute3DFeature();

public:
	//ִ��
	void DoNext(char *image_filename, char *gt_filename, char *calib_filename, char *lidar_filename);
	//ִ��2,ֻ��ͼ�������ת���������㷨
	void DoNext(char *image_filename, char *calib_filename, char *lidar_filename);
	void TestLidarFeature(char *image_filename, char *calib_filename, char *lidar_filename,char *res_filename, char *filename);
};


#endif //SEGCRF_LIDARPROC_H