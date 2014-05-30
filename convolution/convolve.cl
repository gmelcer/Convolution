//
// File:       transpose.c
//
// Abstract:   This example shows how to efficiently perform a transpose of a matrix composed
//             of M x N power-of-two elements for GPU architectures which require specific
//             memory addressing to avoid memory bank conflicts. 
//
//             Transposing large power-of-two matrices naively can easily cause bank 
//             conflicts which can severly affect the performance.
//
//             With appropriate padding and choice of local block size, good performance 
//             can be ensured.
//
//             In this example 64 work items are issued per work-group which individually 
//             operate small 32x2 sections to fill a 32x32 sub-matrix (over 8 iterations). 
//             The final 32 x 32 sub-matrix is transposed locally using local memory 
//             with one column padding to avoid bank conflicts.   Performing the transpose 
//             in local memory allows the reads and writes to global memory to be coalesced.
//
//             The extra column padding is used to offset the write addresses, so that
//             they don't conflict with the read requests. 
//
//             Using a padding of 32 (or any odd multiple of GROUP_DIMX = 32) ensures that
//             the reads and writes for each element in global memory will be offset and 
//             not operate on the same memory bank/channel/port.  
//
//             This is important for the global memory write operations, since the column 
//             major indices are non-sequential and can cause global memory bank conflicts.
//
//             Global memory read requests will operate on sequential indices for the 
//             row-major elements, and will not conflict.
//
// Version:    <1.0>
//
// Disclaimer: IMPORTANT:  This Apple software is supplied to you by Apple Inc. ("Apple")
//             in consideration of your agreement to the following terms, and your use,
//             installation, modification or redistribution of this Apple software
//             constitutes acceptance of these terms.  If you do not agree with these
//             terms, please do not use, install, modify or redistribute this Apple
//             software.
//
//             In consideration of your agreement to abide by the following terms, and
//             subject to these terms, Apple grants you a personal, non - exclusive
//             license, under Apple's copyrights in this original Apple software ( the
//             "Apple Software" ), to use, reproduce, modify and redistribute the Apple
//             Software, with or without modifications, in source and / or binary forms;
//             provided that if you redistribute the Apple Software in its entirety and
//             without modifications, you must retain this notice and the following text
//             and disclaimers in all such redistributions of the Apple Software. Neither
//             the name, trademarks, service marks or logos of Apple Inc. may be used to
//             endorse or promote products derived from the Apple Software without specific
//             prior written permission from Apple.  Except as expressly stated in this
//             notice, no other rights or licenses, express or implied, are granted by
//             Apple herein, including but not limited to any patent rights that may be
//             infringed by your derivative works or by other works in which the Apple
//             Software may be incorporated.
//
//             The Apple Software is provided by Apple on an "AS IS" basis.  APPLE MAKES NO
//             WARRANTIES, EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION THE IMPLIED
//             WARRANTIES OF NON - INFRINGEMENT, MERCHANTABILITY AND FITNESS FOR A
//             PARTICULAR PURPOSE, REGARDING THE APPLE SOFTWARE OR ITS USE AND OPERATION
//             ALONE OR IN COMBINATION WITH YOUR PRODUCTS.
//
//             IN NO EVENT SHALL APPLE BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL OR
//             CONSEQUENTIAL DAMAGES ( INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
//             SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
//             INTERRUPTION ) ARISING IN ANY WAY OUT OF THE USE, REPRODUCTION, MODIFICATION
//             AND / OR DISTRIBUTION OF THE APPLE SOFTWARE, HOWEVER CAUSED AND WHETHER
//             UNDER THEORY OF CONTRACT, TORT ( INCLUDING NEGLIGENCE ), STRICT LIABILITY OR
//             OTHERWISE, EVEN IF APPLE HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Copyright ( C ) 2008 Apple Inc. All Rights Reserved.
//
////////////////////////////////////////////////////////////////////////////////////////////////////


#define IMAGE_W  1024
#define IMAGE_H 640
#define HALF_FILTER_SIZE 9

__kernel void convolute(
                         __global float * output,
                        __global float * input,
                        __global float * filter)

{

//    int HALF_FILTER_SIZE = 4;
//    float *input =h_data;//  (float *) malloc(sizeof(float) * width * (height + PADDING));
//    float *output = (float *) malloc(sizeof(float) * width * (height + PADDING));

    int dy = get_global_id(1);
    int dx = get_global_id(0);
    int width = IMAGE_W;
    int height = IMAGE_H;
	int my = dx + dy * width;

            
            if (
                dx< HALF_FILTER_SIZE ||
                dx > width - HALF_FILTER_SIZE - 1 ||
                dy < HALF_FILTER_SIZE ||
                dy > height - HALF_FILTER_SIZE - 1
                )
            {
                //return;
                output[my] = input[my];
            }
            
            else
            {
                // perform convolution
                int fIndex = 0;
                output[my] = 0.0;
                
                for (int r = -HALF_FILTER_SIZE; r <= HALF_FILTER_SIZE; r++)
                {
                    int curRow = my + r * (width * 1);
                    for (int c = -HALF_FILTER_SIZE; c <= HALF_FILTER_SIZE; c++)
                    {
                        int offset = c * 1;
                        
                        output[ my   ] +=input[ curRow + offset   ] * filter[ fIndex   ];
                        
                        fIndex += 1;
                        if(fIndex> 8)
                            fIndex=0;
                        
                    }
                    
                }
                //        output[my] = 1/output[my];
            }
    
    
}