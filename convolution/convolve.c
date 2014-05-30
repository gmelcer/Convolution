//
// File:       transpose.c
//
// Abstract:   
//             
//             
//
//            
//
//

//
// Version:    <1.0>
//
//
// Copyright ( C ) 2014 George W. Melcer All Rights Reserved.
//  
////////////////////////////////////////////////////////////////////////////////////////////////////


//clang -framework OpenCL `pkg-config --cflags --libs opencv` convolve.c -o convolve && ./convolve

//gcc -std=c99 `pkg-config --cflags --libs opencv` -framework OpenCL   -o convolve convolve.c  && ./convolve


#include <libc.h>
#include <stdbool.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <OpenCL/opencl.h>
#include <mach/mach_time.h>
#include <math.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

/////////////////////////////////////////////////////////////////////////////


#define BLUE 0
#define GREEN 1
#define RED 2

#define CHANNELS 3

#define FILTER_SIZE 9
#define KERNEL_NAME "convolve.cl"


#define STD_DEVIATION 3.5f // STD_DEVIATION^2= Variance...This sets the standard deviation to compute the kernel
#define RADIUS 15  //(Setting a Radius of 7 translates to a [1x15] kernel)

static int width      = 256;
static int height     = 4096;

/////////////////////////////////////////////////////////////////////////////

static char *
load_program_source(const char *filename)
{
    struct stat statbuf;
    FILE        *fh;
    char        *source;
    
    fh = fopen(filename, "r");
    if (fh == 0)
        return 0;
    
    stat(filename, &statbuf);
    source = (char *) malloc(statbuf.st_size + 1);
    fread(source, statbuf.st_size, 1, fh);
    source[statbuf.st_size] = '\0';
    
    return source;
}

/////////////////////////////////////////////////////////////////////////////
void setPixel(float data[], int x, int y, int channel, int numOfChannels, int step, float value )
{
    data[y*step+x*numOfChannels+channel] =value;  //set subpixel (RGB) value in OpenCV data-structure
}


float getPixel(float data[], int x, int y, int channel, int numOfChannels, int step )
{
    return data[y*step+x*numOfChannels+channel]; //get subpixel (RGB) value in OpenCV data-structure
}

float* computeKernel(int radius, float* kernel, int* rtnSize)
{
    int x, y, i,  size;
    double coefficient;
    float sum;
    
    size = (radius*2) + 1; //computes the size of the kernel based off the radius
    kernel = (float *)malloc(sizeof(float)*size); //allocates memory to store the kernel
    
    coefficient = -1/(2*STD_DEVIATION*STD_DEVIATION); //creates the coefficent
    for(x=-radius; x<=radius; x++)
        kernel[x+radius] = exp(coefficient * (x*x)); //evaluates gausian function
    sum = 0;
    for ( i=0; i< size; i++)
        sum += kernel[i];
    
    for ( i = 0; i < size; i++) //normalizes the kernel
    {
        kernel[i] /= sum;
    }
    
    *rtnSize = size; //returns the kernel size
    return kernel;
}



int main(int argc, char **argv)
{
    
    cl_device_id     device_id;
    cl_context       context;
    cl_kernel        kernel;
    cl_command_queue queue;
    cl_program       program;
    cl_mem           src, dst,filterKernel;
    int filterSize=9;
    int step,channels, i,j, dy, dx, size, err;
    float* newKernel, *data, *inputRGB[CHANNELS], *outputRGB[CHANNELS];
    uint64_t t0, t1, t2; //for calculating execution time 
    struct mach_timebase_info info; //for determining execution time
    size_t global[2], local[2]; //OpenCL workgroup dimensions for local and global
    static size_t MaxWorkGroupSize;
    static int WorkGroupSize[2];
    static int WorkGroupItems = 32;
    float edge1D[] = {  -2, 4, -2};
    float raised[] = {0, 0, -2, 0,2, 0, 1,0,0}; //works really well!




    newKernel = computeKernel(RADIUS, newKernel, &size);  //computes Guassian kernel
    
    IplImage* img = cvLoadImage( "fishfarts.exr" , CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH); //load image via OpenCV
    height    = img->height;
    width     = img->width;
    step      = img->widthStep/sizeof(float);
    channels  = img->nChannels;
    data      = (float *)img->imageData;
    

    //Allocate image buffers
    //
    for(i=0;i<CHANNELS;i++) //allocates memory for input/output data buffers
    {
        inputRGB[i] = (float *)malloc(sizeof(float)* width*height);
        outputRGB[i] = (float *)malloc(sizeof(float)* width*height*2);
    }
    
    
    //Load openCV image data into buffers to pass into OpenCL
    //
    for(dy =0; dy<height; dy++) //linearize the pixel data into three discrete buffers from OpenCV data-structure
        for(dx =0; dx<width; dx++)
        {
            
            inputRGB[RED][dy*width+dx] = getPixel(data, dx, dy, RED, channels, step);
            inputRGB[GREEN][dy*width+dx] = getPixel(data, dx, dy, GREEN, channels, step);
            inputRGB[BLUE][dy*width+dx] = getPixel(data, dx, dy, BLUE, channels, step);
            
        }

    // Connect to a GPU compute device
    //
    int gpu = 1;
    err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to create a device group!\n");
        return EXIT_FAILURE;
    }
  
    // Create a compute context 
    //
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }

    // Create a command queue
    //
    queue = clCreateCommandQueue(context, device_id, 0, &err);
    if (!queue)
    {
        printf("Error: Failed to create a command queue!\n");
        return EXIT_FAILURE;
    }

    // Load the compute program from disk into a cstring buffer
    //
    char *source = load_program_source(KERNEL_NAME);
    if(!source)
    {
        printf("Error: Failed to load compute program from file!\n");
        return EXIT_FAILURE;    
    }

    // Create the compute program from the source buffer
    //
    program = clCreateProgramWithSource(context, 1, (const char **) & source, NULL, &err);
    if (!program || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }

    // Build the program executable
    //
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return EXIT_FAILURE;
    }

    // Create the compute kernel from within the program
    //
    kernel = clCreateKernel(program, "convolute", &err);
    if (!kernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        return EXIT_FAILURE;
    }

    // Create the input array on the device
    //
    src = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * width * height, NULL, NULL);
    if (!src)
    {
        printf("Error: Failed to allocate source array!\n");
        return EXIT_FAILURE;
    }
    
    filterKernel = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * filterSize, NULL, NULL);
    if (!src)
    {
        printf("Error: Failed to allocate source array!\n");
        return EXIT_FAILURE;
    }
    
    
    // Fill the input array with the host allocated random data
    //
    err = clEnqueueWriteBuffer(queue, src, true, 0, sizeof(float) * width * height, inputRGB[0], 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        return EXIT_FAILURE;
    }
    
    
    // Fill the input array with the host allocated random data
    //
    err = clEnqueueWriteBuffer(queue, filterKernel, true, 0, sizeof(float) * filterSize , newKernel, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        return EXIT_FAILURE;
    }

    // Create the output array on the device
    //
    dst = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * width * (height), NULL, NULL);
    if (!dst)
    {
        printf("Error: Failed to allocate destination array!\n");
        return EXIT_FAILURE;
    }

    // Set the kernel arguments prior to execution
    //
    err  = clSetKernelArg(kernel,  0, sizeof(cl_mem), &dst);
    err |= clSetKernelArg(kernel,  1, sizeof(cl_mem), &src);
    err |= clSetKernelArg(kernel,  2, sizeof(cl_mem) , &filterKernel);
//    err |= clSetKernelArg(kernel,  3, sizeof(int) , &width);
  //  err |= clSetKernelArg(kernel,  4, sizeof(int) , &height);
   // err |= clSetKernelArg(kernel,  5, sizeof(int) , &filterSize);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments!\n");
        return EXIT_FAILURE;
    }


    // Get the maximum work group size for executing the kernel on the device
    //
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &MaxWorkGroupSize, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }
    
#if (DEBUG_INFO)
    printf("MaxWorkGroupSize: %d\n", MaxWorkGroupSize);
    printf("WorkGroupItems: %d\n", WorkGroupItems);
#endif
    
    WorkGroupSize[0] = (MaxWorkGroupSize > 1) ? (MaxWorkGroupSize / WorkGroupItems) : MaxWorkGroupSize;
    WorkGroupSize[1] = MaxWorkGroupSize / WorkGroupSize[0];


    // Determine the global and local dimensions for the execution
    //
    global[0] = width ;
    global[1] = height;
    local[0] = WorkGroupSize[0];
    local[1] = WorkGroupSize[1];    
    
    // Execute once without timing to guarantee data is on the device
    //
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, NULL);
    if (err)
    {
        printf("Error: Failed to execute kernel!\n");
        return EXIT_FAILURE;
    }
    clFinish(queue);


    cvNamedWindow( "Original", CV_WINDOW_AUTOSIZE );
	cvShowImage("Original", img);   //displays the original image
    cvWaitKey(0); //pauses until the user presses a key



    printf("Performing Convolution on [%d x %d] Image...\n", width, height);


    err = CL_SUCCESS;
    t0 = t1 = mach_absolute_time();
    err |= clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, NULL);

    clFinish(queue);
    t2 = mach_absolute_time();
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to execute kernel!\n");
        return EXIT_FAILURE;
    }

    // Calculate the total bandwidth that was obtained on the device for all memory transfers
    //
    mach_timebase_info(&info);
    double t = 1e-9 * (t2 - t1) * info.numer / info.denom;
    printf("Convolution Performed in %fns.\n", t );
    printf("Bandwidth Achieved = %f GB/sec\n", 2e-9 * sizeof(float) * width * height  / t);


    float *h_result = (float *) malloc(sizeof(float) * width * (height)); //temp buffer to read-back OpenCL data


    // Read back the results that were computed on the device
    //
    err = clEnqueueReadBuffer( queue, dst, true, 0, sizeof(float) * width * (height), h_result, 0, NULL, NULL );
    if (err)
    {
        printf("Error: Failed to read back results from the device!\n");
        return EXIT_FAILURE;
    }
    

    
    //Returns filtered openCL data back to OpenCV data-structure
    //
    for(dy =0; dy<height; dy++) 
        for(dx =0; dx<width; dx++)
        {
            
            setPixel(  data   , dx, dy, RED, channels, step,  h_result[dy*width+dx] );
            setPixel(  data   , dx, dy, GREEN, channels, step, h_result[dy*width+dx] );
            setPixel(  data   , dx, dy, BLUE, channels, step,   h_result[dy*width+dx] );
        }

    
    
    //Opens a OpenCV Window with filter image
    //
    cvNamedWindow( "Filtered Result", CV_WINDOW_AUTOSIZE );
	cvShowImage("Filtered Result", img);  //displays the filtered image
	cvWaitKey(0);
    


    //Memory Deallocation
    //
    for(i=0;i<CHANNELS; i++) //frees all the buffers used
    {
        free(inputRGB[i]);
        free(outputRGB[i]);
    }


    free(h_result);
    free(newKernel);

	cvDestroyWindow( "Filtered Result" );
	cvDestroyWindow( "Original" );
    
    clReleaseMemObject(src);
    clReleaseMemObject(dst);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);


    
    return 0;
}
