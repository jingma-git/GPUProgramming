#include <cstdio>
#include <cstdlib>
#include <opencv2/opencv.hpp>

void test4()
{
    int **a;
    a = (int **)malloc(3 * sizeof(int *));
    if (a == NULL)
    {
        printf("Could not allocate memory!\n");
        return;
    }
    for (int i = 0; i < 3; i++)
    {
        a[i] = (int *)malloc(100 * sizeof(int));
        if (a[i] == NULL)
        {
            printf("Out of memory\n");
            return;
        }

        for (int j = 0; j < 100; j++)
        {
            a[i][j] = i * 100 + j;
            printf("a[%d][%d]=%d\n", i, j, a[i][j]);
        }
    }
}

void testLoadImg()
{
    cv::Mat img = cv::imread("resources/lena_gray.jpg", cv::IMREAD_GRAYSCALE);
    unsigned char *img_ptr = img.ptr<unsigned char>();
    cv::Mat img2(img.rows, img.cols, CV_8U, img_ptr);
    cv::imwrite("resources/out.jpg", img2);
}

int main()
{
    testLoadImg();
    // test4();
    return 0;
}