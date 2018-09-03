#include <iostream>
#include <fstream>

#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "net.h"
using namespace std;

static int find_max_index(const float * prob, int n)
{
        float max_f = -999999999.0f;
        int res = -1;
        for (int i = 0; i < n; i++)
        {
                if (prob[i] > max_f)
                {
                        max_f = prob[i];
                        res = i;
                }
        }
        return res;
}

static int detect_squeezenet(const cv::Mat& bgr, std::vector<float>& cls_scores)
{
        int res = 0;
        ncnn::Net squeezenet;
        res = squeezenet.load_param("E:\\pythonCode\\PythonTest\\pytorch\\hand\\hand20.param");
        if (res)
        {
                printf("load param failed\n");
                return -1;
        }
        res = squeezenet.load_model("E:\\pythonCode\\PythonTest\\pytorch\\hand\\hand20.bin");
        if (res)
        {
                printf("load bin failed\n");
                return -1;
        }

        ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_GRAY, bgr.cols, bgr.rows, 28, 28);

        for (int i = 0; i < in.c*in.w*in.h; i++)
        {
                in[i] = in[i] / 255.0;
        }

        ncnn::Extractor ex = squeezenet.create_extractor();
        ex.set_light_mode(true);

        res = ex.input("0", in);
        if (res)
        {
                printf("input failed\n");
                return -1;
        }

        ncnn::Mat out;
        res = ex.extract("14", out);
        if (res)
        {
                printf("output failed\n");
                return -1;
        }

        cls_scores.resize(out.c);
        for (int j = 0; j < out.c; j++)
        {
                const float* prob = (const float*)((char*)out.data + out.cstep * j);
                printf("result(%d)\n", find_max_index(prob, out.w));
                cls_scores[j] = prob[0];
        }

        return 0;
}

int main(int argc, char** argv)
{
        if (argc<2)
        {
                printf("please input image path\n");
                return -1;
        }
        const char* imagepath = argv[1];

        cv::Mat m = cv::imread(imagepath, CV_LOAD_IMAGE_COLOR);
        if (m.empty())
        {
                fprintf(stderr, "cv::imread %s failed\n", imagepath);
                return -1;
        }
        cv::Mat gray;
        cvtColor(m, gray, CV_BGR2GRAY);
        //vector<string> labels;
        //read_label("./label.txt", labels);
        std::vector<float> cls_scores;
        detect_squeezenet(gray, cls_scores);
        //detect_squeezenet2(cls_scores);

        //print_topk(cls_scores, 3, labels);

        return 0;
}