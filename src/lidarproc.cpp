//
// Created by yangfei on 17-4-26.
//

#include "lidarproc.h"
//#include "bmpFile.h"

cLidarProcess::cLidarProcess()
{
	m_numPoint = 0;
	Init();
}
cLidarProcess::~cLidarProcess()
{
	Dump();
}

//��ʼ���ڴ�
void cLidarProcess::Init()
{
	m_P = Eigen::MatrixXf::Zero(4, 4);
	m_R0_rect = Eigen::MatrixXf::Zero(4, 4);
	m_Tr_velo_to_cam = Eigen::MatrixXf::Zero(4, 4);
	m_LidarData = new float[MAX_NUM * sizeof(float)];
}

//�ͷ��ڴ�
void cLidarProcess::Dump()
{
	if (m_LidarData)	delete[] m_LidarData;
}

void cLidarProcess::Debug()
{
	Mat _8bImg;
//	Rmw_Write24BitImg2BmpFile(m_Img.data, m_width, m_height, "Debug/project.bmp");
	m_MaxHeightDiff.convertTo(_8bImg, CV_8U, 256);
//	Rmw_Write8BitImg2BmpFile(_8bImg.data, m_width, m_height, "Debug/heightdiff.bmp");
	m_HeightVar.convertTo(_8bImg,CV_8U,256);
//	Rmw_Write8BitImg2BmpFile(_8bImg.data, m_width, m_height, "Debug/heightvar.bmp");
	m_Curvature.convertTo(_8bImg,CV_8U,256);
//	Rmw_Write8BitImg2BmpFile(_8bImg.data, m_width, m_height, "Debug/curvature.bmp");

}

void cLidarProcess::ReadImage(char *filename)
{
	m_Img = imread(filename);
	Size size = m_Img.size();
	m_width = size.width;
	m_height = size.height;
}

void cLidarProcess::ReadLabel(char *filename)
{
	int i, j;
	Mat gt_img = imread(filename);
	Mat label(gt_img.size(), CV_8U, Scalar::all(0));
	for (i = 0; i < gt_img.rows; i++)
	{
		for (j = 0; j < gt_img.cols; j++)
		{
			if (gt_img.at<Vec3b>(i, j)[0] > 0)        //bͨ������0Ϊ1
				label.at<uchar>(i, j) = 1;
		}

	}
	label.copyTo(m_Label);
}


int cLidarProcess::ReadCalibData(char *filename)
{
	FILE *fp = NULL;
	if ((fp = fopen(filename, "rt")) == NULL)
	{
		printf("Can't Find Calib File !\n");
		return -1;
	}
	char index[256];
	char str[256];
	float temp;
	fseek(fp, 0, SEEK_SET);       //ָ���ļ�ͷ
								  //�����ɫ�����ӳ�����P2
	sprintf(index, "P2:");
	while (1)
	{
		fscanf(fp, "%s", str);
		if (!strcmp(str, index))
			break;
	}
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			fscanf(fp, "%f", &temp);
			m_P(i, j) = temp;
		}
	}
	m_P(3, 3) = 1.0;
	//����ת��������R0_rect
	sprintf(index, "R0_rect:");
	while (1)
	{
		fscanf(fp, "%s", str);
		if (!strcmp(str, index))
			break;
	}
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			fscanf(fp, "%f", &temp);
			m_R0_rect(i, j) = temp;
		}
	}
	m_R0_rect(3, 3) = 1.0;
	//���״ﵽ���RT����Tr_velo_to_cam
	sprintf(index, "Tr_velo_to_cam:");
	while (1)
	{
		fscanf(fp, "%s", str);
		if (!strcmp(str, index))
			break;
	}
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			fscanf(fp, "%f", &temp);
			m_Tr_velo_to_cam(i, j) = temp;
		}
	}
	m_Tr_velo_to_cam(3, 3) = 1.0;
	fclose(fp);
	return 0;
}

//�����������
int cLidarProcess::ReadPointCloudData(char *filename)
{
	FILE *fp = NULL;
	m_numPoint = MAX_NUM;       //������
	fp = fopen(filename, "rb");
	if (fp == NULL)
	{
		printf("Can not open CloadPoint file !\n");
		return -1;
	}
	m_numPoint = fread(m_LidarData, sizeof(float), m_numPoint, fp) / 4;		//��Ч�����
	printf("==============m_numPoint= %d ==============\n", m_numPoint);
	fclose(fp);
	return 0;
}

//����������ת��Ϊդ��
void cLidarProcess::TransLidarPoint2Grid()
{
	int i, x, y;
	float *p;
	Mat Grid(GRID_HEIGHT, GRID_WIDTH, CV_8U, Scalar(0));
	for (i = 0, p = m_LidarData; i < m_numPoint; i++, p += 4)
	{
		x = GRID_WIDTH / 2 - (*(p + 1) / GRID_RESOLUTION);		//x,yΪդ������
		y = GRID_HEIGHT / 2 - (*p / GRID_RESOLUTION);
		if (x >= 0 && x < GRID_WIDTH && y >= 0 && y < GRID_HEIGHT)
		{
			Grid.at<uchar>(y, x) = 255;
		}
	}
	Grid.copyTo(m_Grid);
}

//����������ӳ�䵽ͼ������ϵ��ÿ���㱣��ԭʼ����
void cLidarProcess::TransLidarPoint2Img()
{
	int i, x, y;
	float *p;
	Eigen::Vector4f X;	//ת��ǰ���״�����
	Eigen::Vector4f Y;	//ת�����ͼ������
	Mat Valid2Img(m_height, m_width, CV_8U, Scalar(0));
	Mat Lidar2Img(m_height, m_width, CV_32FC4, Scalar::all(0));
	for (i = 0, p = m_LidarData; i < m_numPoint; i++, p += 4)
	{
		X << *p, *(p + 1), *(p + 2), 1.0;		//��������
		if (X(0) < 80 && X(0) > 5)
		{
			Y = m_P*m_R0_rect*m_Tr_velo_to_cam*X;	//����ת��
													//ת������������
			x = int(Y(0) / Y(2));     //������
			y = int(Y(1) / Y(2));     //������
			if (x >= 0 && x < m_width && y >= 0 && y < m_height)
			{
				Valid2Img.at<uchar>(y, x) = 255;
				Lidar2Img.at<Vec4f>(y, x)[0] = p[0];
				Lidar2Img.at<Vec4f>(y, x)[1] = p[1];
				Lidar2Img.at<Vec4f>(y, x)[2] = p[2];
				Lidar2Img.at<Vec4f>(y, x)[3] = p[3];
				m_Img.at<Vec3b>(y, x)[1] = 255;
			}
		}
	}
	Valid2Img.copyTo(m_Valid2Img);      //��Ч���־����ɨ����,ӳ���Mask
	Lidar2Img.copyTo(m_Lidar2Img);      //ת��ͼ��������״�㣬4��ͨ�����ֱ�Ϊx,y,z,ref
}


float cLidarProcess::ComputeMaxHeightDiff(Eigen::MatrixXf &input)
{
	float H_min, H_max, H_diff;
	H_min = input.col(2).minCoeff();
	H_max = input.col(2).maxCoeff();
	H_diff = H_max - H_min;
	return H_diff;
}



//�������������������Э�������
void cLidarProcess::ComputeCovMatrix(Eigen::Vector3f &p, Eigen::MatrixXf &input, Eigen::Matrix3f &covMat)
{
	int i;
	float mu;
	long numNN = input.rows();        //�����
									  //��muֵ
	Eigen::RowVector3f pi;     //��ǰ��
	Eigen::RowVectorXf w(numNN);      //Ȩ��
							   //��ȡ��������ֵ
	for (i = 0; i < numNN; i++)
	{
		pi = input.row(i);
		pi = p.transpose() - pi;
		w(i) = pi.norm();     //wi��ʱ����p-pi��ģ
	}
	mu = w.mean();
	for (i = 0; i < numNN; i++)
	{
		if (w(i) >= mu)
		{
			w(i) = exp(-(w(i) / mu)*(w(i) / mu));
		}
		else
		{
			w(i) = 1.0;
		}

	}
	Eigen::RowVector3f meanVec = input.colwise().mean();     //ÿһ�����ֵ���õ�һ��3ά������
	Eigen::MatrixXf zeroMeanMat = input.rowwise() - meanVec;     //0��ֵ����
	Eigen::MatrixXf wMat(3, numNN);
	wMat.rowwise() = w;
	covMat = zeroMeanMat.transpose().cwiseProduct(wMat)*zeroMeanMat;
}

//����Э������󣬼�������ֵ
float cLidarProcess::ComputeEigenValofcovMat(Eigen::Matrix3f &covMat)
{
	float response;
	Eigen::EigenSolver<Eigen::Matrix3f> es(covMat);
	Eigen::Matrix3f D = es.pseudoEigenvalueMatrix();
	Eigen::Matrix3f V = es.pseudoEigenvectors();
	response = D.maxCoeff() / D.sum();
	return response;

}

void cLidarProcess::FindNearestNeighborsofPoint(Eigen::Vector3f &p, int px, int py, float R, Eigen::MatrixXf &output)
{
	int x, y, x1, y1, x2, y2;
	Eigen::MatrixXf NN(m_width*m_height, 3);        //�㹻��ľ���
	int count = 0;
	Eigen::Vector3f pi;
	Eigen::Vector3f diff;
	float dist;
	//�������ڴ�С
	int halfWin = ((py / 100) + 1) * 5;     //��ɱ��С�򴰿ڣ�����Ӧ͸��ͶӰ�ĵ�·ģ��
	x1 = max(0, px - halfWin);
	y1 = max(0, py - halfWin);
	x2 = min(m_width - 1, px + halfWin);
	y2 = min(m_height - 1, py + halfWin);
	for (y = y1; y <= y2; y++)
	{
		for (x = x1; x <= x2; x++)
		{
			if (m_Valid2Img.at<uchar>(y, x))
			{
				pi << m_Lidar2Img.at<Vec4f>(y, x)[0], m_Lidar2Img.at<Vec4f>(y, x)[1], m_Lidar2Img.at<Vec4f>(y, x)[2];
				diff = p - pi;  //������
				dist = diff.norm();       //ģ�������������
				if (dist >= 0 && dist <= R)
				{
					NN.row(count++) = pi.transpose();
				}
			}
		}
	}
	output = NN.block(0, 0, count, 3);       //ȡ����Ч����
}



void cLidarProcess::Compute3DFeature()
{
	int i, j;
	float curv, hdiff;
	Eigen::Vector3f p;
	Eigen::Matrix3f covMat;
	Eigen::MatrixXf NN;
	//��ʼ����������
	Mat Curvature(m_height, m_width, CV_32F, Scalar::all(0));
	Mat MaxHeightDiff(m_height, m_width, CV_32F, Scalar::all(0));
	Mat HeightVar(m_height, m_width, CV_32F, Scalar::all(0));
	Mat Weights(m_height,m_width, CV_32F, Scalar::all(0));
	for (i = 0; i < m_height; i++)
	{
		for (j = 0; j < m_width; j++)
		{
			if (m_Valid2Img.at<uchar>(i, j))
			{
				//                cout<<"i= "<<i<<" "<<"j= "<<j<<endl;
				p << m_Lidar2Img.at<Vec4f>(i, j)[0], m_Lidar2Img.at<Vec4f>(i, j)[1], m_Lidar2Img.at<Vec4f>(i, j)[2];
				//                cout<<"p="<<endl<<p<<endl;
				FindNearestNeighborsofPoint(p, j, i, 0.3, NN);
				//                cout<<"NN="<<endl<<NN<<endl;
				if (NN.rows() > 1)     //������
				{
					ComputeCovMatrix(p, NN, covMat);
					//                    cout<<"covMat="<<covMat<<endl;
					curv = ComputeEigenValofcovMat(covMat);
					//                    cout<<"curv="<<curv<<endl;
					hdiff = ComputeMaxHeightDiff(NN);
					Curvature.at<float>(i, j) = curv;
					MaxHeightDiff.at<float>(i, j) = hdiff;
					HeightVar.at<float>(i, j) = covMat(2, 2);
					Weights.at<float>(i, j) = NN.rows();
				}

			}
		}
	}
	Curvature.copyTo(m_Curvature);
	MaxHeightDiff.copyTo(m_MaxHeightDiff);
	HeightVar.copyTo(m_HeightVar);
	Weights.copyTo(m_Weights);
}

//ִ��
void cLidarProcess::DoNext(char *image_filename, char *gt_filename, char *calib_filename, char *lidar_filename)
{
	printf("%s\n", image_filename);
	ReadImage(image_filename);
	ReadLabel(gt_filename);
	//    imwrite("label.png",m_Label);
	//��ȡ�궨����
	ReadCalibData(calib_filename);
	//cout << m_P << endl << m_R0_rect << endl << m_Tr_velo_to_cam << endl;
	//��ȡ��������
	ReadPointCloudData(lidar_filename);
	//������ת��Ϊդ��ͼ
	TransLidarPoint2Grid();
	//����������ӳ�䵽ͼ������
	TransLidarPoint2Img();
	    /*imshow("cp",m_Valid2Img);
	    waitKey(0);*/
	//��������
	Compute3DFeature();
	  /*  imshow("curv",m_Curvature);
	    waitKey(0);
	    imshow("height",m_MaxHeightDiff);
	    waitKey(0);
	    imshow("var",m_HeightVar);
	    waitKey(0);*/
}

void cLidarProcess::DoNext(char *image_filename, char *calib_filename, char *lidar_filename)
{
	printf("%s\n", image_filename);
	ReadImage(image_filename);
	ReadCalibData(calib_filename);
	//cout << m_P << endl << m_R0_rect << endl << m_Tr_velo_to_cam << endl;
	//��ȡ��������
	ReadPointCloudData(lidar_filename);
	//����������ӳ�䵽ͼ������
	TransLidarPoint2Img();
	//imshow("mask", m_Valid2Img);
	//waitKey(0);
}

void cLidarProcess::TestLidarFeature(char * image_filename, char * calib_filename, char * lidar_filename,char *res_filename,char *filename)
{
	int i, j;
	
	printf("%s\n", image_filename);
	ReadImage(image_filename);
	ReadCalibData(calib_filename);
	//��ȡ��������
	ReadPointCloudData(lidar_filename);
	//����������ӳ�䵽ͼ������
	TransLidarPoint2Img();
	Mat res(m_height, m_width, CV_8UC3, Scalar::all(0));
	Mat prob(m_height,m_width,CV_32F,Scalar(0));
	FILE *fp = fopen(res_filename, "rt");
	for (i=0;i<m_height;i++)
	{
		for(j=0;j<m_width;j++)
		{
			if (m_Valid2Img.at<uchar>(i,j)>0)
			{
				float logit = 0;
				fscanf(fp, "%f", &logit);
                prob.at<float>(i,j)=logit;
				Vec3b color;
				if (logit>0.5)
				{
					color = Vec3b(0,255,0);
				}
				else
				{
					color = Vec3b(0, 0, 255);
				}
				res.at<Vec3b>(i, j) = color;
			}
//			else
//			{
//				prob.at<float>(i,j)=0.5;
//			}

		}
	}
    fclose(fp);
	char writename[256];
//	sprintf(writename,"/home/yangfei/Workspace/HFM/PointNetSeg/test_images_rg/%s.png",filename);
//    imwrite(writename,res);
//    sprintf(writename,"/home/yangfei/Workspace/HFM/PointNetSeg/test_images/%s.png",filename);

    sprintf(writename,"/home/yangfei/Workspace/HFM/LidarSeg/train_images_sparse/%s.png",filename);
    prob.convertTo(prob,CV_8U,255,0);
    imwrite(writename,prob);
//	imshow(image_filename,res);
//	imshow(image_filename,prob);
//	waitKey(0);
//	destroyAllWindows();
}


