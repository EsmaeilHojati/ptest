
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>
//#include <opencv2/imgproc.hpp>
#include <chrono>
#include <stdio.h>
#include <math.h>   

#ifndef NDEBUG
#define ASSERT_MSG(Condition, Msg) \
      AssertMessage(#Condition, Condition, __FILE__, __LINE__, #Msg)
#define ASSERT(Condition) \
      AssertMessage(#Condition, Condition, __FILE__, __LINE__, "")

void AssertMessage(const char* ConditionStr, bool Condition, const char* FileName, int Line, const char* Msg)
{
    if (!Condition)
    {
        std::cerr << "Assert failed:\t" << Msg << "\n"
            << "Expected:\t" << ConditionStr << "\n"
            << "Source:\t\t" << FileName << ", line " << Line << "\n";
        abort();
    }
}

#else
#define ASSERT(Condition) 
#define ASSERT_MSG(Condition, Msg) 
#endif






cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}


template <typename T>
class Matrix
   {

   private:
      T m_data;
   };

template <typename T>
class Kernel
   {

   private:
      T m_data;
   };


template <typename T>
class Image: public Matrix<T>
   {
   public:
      void Conv2D(void)
         {
         //some stuff/;
         }
      void Conv1D(void);
   private:
      T test;
   };

template <typename T>
void Image<T>::Conv1D(void)
   {
   //other stuff
   }

using namespace std;
using namespace cv;

typedef std::vector<Point>(*pFunc) (void);

enum eNoiseMode {NoNoise, RandomNoise};

struct RandomImage
   {
   Mat Image;
   std::vector<Point2f> Corners;
   };

class Sqa
   {
   public:
      Sqa(pFunc Func = nullptr, bool SaveTestFails=false, bool PrintReport=false);
      ~Sqa();
      bool RunRandomTests(int TestNumber, 
                          int MaxImageSize, 
                          int MaxRectCount, 
                          eNoiseMode NoiseMode);

      int  RunBenchmark(int IterationNumber);

      RandomImage GenerateRandomImage(int ImageSize, int RectCount);
   private:

      pFunc m_UserFunc;
      bool m_PrintReport;
      bool m_SaveTestFails;
   };

Sqa::Sqa(pFunc Func, bool SaveTestFails, bool PrintReport):
   m_UserFunc(Func),
   m_SaveTestFails(SaveTestFails),
   m_PrintReport(PrintReport)
   {
   }

Sqa::~Sqa()
   {}

bool Sqa::RunRandomTests(int TestNumber, int MaxImageSize, int MaxRectCount, eNoiseMode NoiseMode)
{
   ASSERT_MSG(m_UserFunc, "Cannot run random tests without User Function.");
   //We want to be able to debug several time, so we use a fixed random seed 
   srand(1);
   for(auto TestIndex = 0; TestIndex < TestNumber; TestIndex++)
      {
      int ImageSize = rand() % MaxImageSize;
      int RectCount = rand() % MaxRectCount;

      RandomImage TempImage = GenerateRandomImage(ImageSize, RectCount);
       m_UserFunc();
      TempImage.Image


   }
   return false;
}

template <typename T>
T GetMedian(std::vector<T> InputVector)
   {
   ASSERT(InputVector.size() > 1 );
   std::sort(InputVector.begin(), InputVector.end(), std::greater<T>());
   return InputVector[static_cast<int>(InputVector.size() / 2)];
   }

int  Sqa::RunBenchmark(int IterationNumber)
   {
   ASSERT_MSG(m_UserFunc, "Cannot benchmark without User Function.");
   if(!m_UserFunc || IterationNumber <= 0) return 0;
   auto StartTime = std::chrono::high_resolution_clock::now();
   for(auto i = 0; i < IterationNumber; i++)
      {
      m_UserFunc();
      }
   auto EndTime = std::chrono::high_resolution_clock::now();
   auto Duration = std::chrono::duration_cast<std::chrono::microseconds>( EndTime - StartTime ).count();
   int PerIterDuration =  Duration / IterationNumber;
   return PerIterDuration;
   }

RandomImage Sqa::GenerateRandomImage(int ImageSize, int RectCount)
   {
   assert(ImageSize > 0);
   assert(RectCount > 0);
   RandomImage TempImg;
   
   int DrawingOffset = 10 ;
   int RectPerRow = static_cast<int>(sqrt(RectCount)) ;
   int MaxRectSize = static_cast<int> ((ImageSize / RectPerRow) / 2);
   int DrawingSlotSize = static_cast<int> ((ImageSize / RectPerRow));

   TempImg.Image = Mat::zeros(ImageSize+DrawingOffset, ImageSize+DrawingOffset, CV_8UC1);
   for(int i = 0; i < RectPerRow; i++)
      for(int j = 0; j < RectPerRow; j++)
      {
      Size RectSize(max(rand() % MaxRectSize , 10),
                    max(rand() % MaxRectSize , 10));
      Point RectCenter( i * DrawingSlotSize + DrawingSlotSize / 2 + DrawingOffset,
                        j * DrawingSlotSize + DrawingSlotSize / 2 + DrawingOffset );
      int RectAngle = rand() % 180;

      RotatedRect TempRotatedRect(RectCenter,RectSize,RectAngle);
      Point2f TempRectCorners[4];
      TempRotatedRect.points(TempRectCorners);
      int npt[] = { 4 };
      Point RectCorners[4];
      for(auto i = 0; i < 4; i++)
         {
         RectCorners[i] = TempRectCorners[i];
         TempImg.Corners.push_back(RectCorners[i]);
         }
      //transform(RectCorners, RectCorners + 4, TempRectCorners, RectCorners, [&](Point2f PointF) { return Point(PointF);});
      const Point* ppt[1] = { RectCorners };
      //transform(RectCorners, RectCorners + 4, TempRectCorners, RectCorners, [&](Point2f PointF) { return Point(PointF.x, PointF.y);});

        polylines( TempImg.Image,
                  ppt,
                  npt,
                  1,
                  true,
                  Scalar( 255, 255, 255 ));

      }
   return TempImg;
   }


std::vector<Point> MyFunc()
   {
   std::vector<Point> temp;
   for(auto i = 1; i < 1000; i++)
      temp.push_back(Point(i, i * 2));
   return temp;
   }

int main()
{
    //Mat C = (Mat_<double>(3,3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
    //cout << "C = " << endl << " " << C << endl << endl;
   Sqa SqaModule(MyFunc,false,false);
   RandomImage RI;
   namedWindow("image", WINDOW_NORMAL);

   cout<<SqaModule.RunBenchmark(10000);

   for(auto i = 0; i < 10; i++)
      {
      RI = SqaModule.GenerateRandomImage(500, 10);
      //Mat img = imread("C:/temp/imcpy.bmp");
      imshow("image", RI.Image);
      waitKey(0);
      }

    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

/*
https://gist.github.com/OmarAflak/aca9d0dc8d583ff5a5dc16ca5cdda86a
https://homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm
https://muthu.co/harris-corner-detector-implementation-in-python/
https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#gac1fc3598018010880e370e2f709b4345
https://github.com/opencv/opencv/blob/master/modules/imgproc/src/corner.cpp

https://github.com/alexanderb14/Harris-Corner-Detector/blob/master/harris.cpp

https://stackoverflow.com/questions/8204645/implementing-gaussian-blur-how-to-calculate-convolution-matrix-kernel

*/
