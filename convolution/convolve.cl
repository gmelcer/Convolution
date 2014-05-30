//
// File:       convolve.cl
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