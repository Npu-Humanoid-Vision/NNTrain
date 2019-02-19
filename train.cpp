#include <bits/stdc++.h>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

#define POS_LABLE 1
#define NEG_LABLE 0

#ifdef __linux__
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#endif

#ifdef __WIN32
#include <io.h>
#include <windows.h>
#endif

// 获得某文件夹下所有图片的名字
void GetImgNames(string root_path, std::vector<std::string>& names) {
#ifdef __linux__
    struct dirent* filename;
    DIR* dir;
    dir = opendir(root_path.c_str());  
    if(NULL == dir) {  
        return;  
    }  

    int iName=0;
    while((filename = readdir(dir)) != NULL) {  
        if( strcmp( filename->d_name , "." ) == 0 ||
            strcmp( filename->d_name , "..") == 0)
            continue;

        string t_s(filename->d_name);
        names.push_back(t_s);
    }
#endif

#ifdef __WIN32
    intptr_t hFile = 0;
    struct _finddata_t fileinfo;
    string p;

    hFile = _findfirst(p.assign(root_path).append("/*").c_str(), &fileinfo);

    if (hFile != -1) {
        do {
            if (strcmp(fileinfo.name, ".") == 0 || strcmp(fileinfo.name, "..") == 0) {
                continue;
            }
            names.push_back(fileinfo.name); 
        } while (_findnext(hFile, &fileinfo) == 0);
    }
#endif
}


void GetXsSampleData(const string folder_path, int lable, 
            cv::Mat& train_data, cv::Mat& train_data_lables) {

    // get the image names
    std::vector<std::string> image_names;
    GetImgNames(folder_path, image_names);

    // define hog descriptor 
    cv::HOGDescriptor hog_des(Size(32, 32), Size(8, 8), Size(4, 4), Size(4, 4), 9);

    // read images and compute
    for (auto i = image_names.begin(); i != image_names.end(); i++) {
        string t_path = folder_path + (*i);
        cv::Mat t_image = cv::imread(t_path);
        std::vector<float> t_descrip_vec;
        hog_des.compute(t_image, t_descrip_vec);


        // copy t_descrip_vec to train_data
        cv::Mat t_mat = cv::Mat(1, t_descrip_vec.size(), CV_32FC1);
        for (auto j = 0; j < t_descrip_vec.size(); j++) {
            t_mat.at<float>(0, j) = t_descrip_vec[j];
        }
        train_data.push_back(t_mat);
        train_data_lables.push_back(cv::Mat((cv::Mat_<float>(1, 2) << lable, 1-lable)));
    }
}

int main(int argc, char const *argv[]) {
    // for train time
    double begin;

    string pos_root_path = "../../BackUpSource/Ball/Train/Preproc/Pos/";
    string neg_root_path = "../../BackUpSource/Ball/Train/Preproc/Neg/";
    cv::Mat train_data;
    cv::Mat train_data_lables;
    GetXsSampleData(pos_root_path, POS_LABLE, train_data, train_data_lables);
    GetXsSampleData(neg_root_path, NEG_LABLE, train_data, train_data_lables);
    cout<<train_data.size()<<' '<<train_data_lables.size()<<endl;

#ifdef __WIN32 
    // define nn's structure
    cv::Mat nn_structure =  (cv::Mat_<int>(1, 4) << 1764, 48, 24, 2);
    CvANN_MLP trainer(nn_structure, CvANN_MLP::SIGMOID_SYM);
    CvANN_MLP_TrainParams train_params;
    trainer.train(train_data, train_data_lables, cv::Mat());
    trainer.save("test.xml");
#endif

#ifdef __linux__
#endif
    return 0;
}
