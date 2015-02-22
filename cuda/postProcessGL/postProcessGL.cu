/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// Utilities and system includes

#include <helper_cuda.h>

#ifndef USE_TEXTURE_RGBA8UI
texture<float4, 2, cudaReadModeElementType> inTex;
#else
texture<uchar4, 2, cudaReadModeElementType> inTex;
#endif

// clamp x to range [a, b]
__device__ float clamp(float x, float a, float b)
{
    return max(a, min(b, x));
}

__device__ int clamp(int x, int a, int b)
{
    return max(a, min(b, x));
}

// convert floating point rgb color to 8-bit integer
__device__ int rgbToInt(float r, float g, float b)
{
    r = clamp(r, 0.0f, 255.0f);
    g = clamp(g, 0.0f, 255.0f);
    b = clamp(b, 0.0f, 255.0f);
    return (int(b)<<16) | (int(g)<<8) | int(r);
}

// get pixel from 2D image, with clamping to border
__device__ uchar4 getPixel(int x, int y, int imgw, int imgh, uchar4* buffer)
{
    y = clamp(y, 0, imgh-1);
    x = clamp(x, 0, imgw-1);
    return buffer[y*imgw + x];
}

// get pixel from 2D image, with clamping to border
__device__ void setPixel(int x, int y, int imgw, uchar4* buffer)
{
#ifndef USE_TEXTURE_RGBA8UI
    float4 res = tex2D(inTex, x, y);
    uchar4 ucres = make_uchar4(res.x*255.0f, res.y*255.0f, res.z*255.0f, res.w*255.0f);
#else
    uchar4 ucres = tex2D(inTex, x, y);
#endif

    int old_c = 30;
    int new_c = 1;
    int total = old_c + new_c;
    buffer[y*imgw + x].x = 255 - (total*255 - new_c*ucres.x - old_c*buffer[y*imgw + x].x)/total;
    buffer[y*imgw + x].y = 255 - (total*255 - new_c*ucres.y - old_c*buffer[y*imgw + x].y)/total;
    buffer[y*imgw + x].z = 255 - (total*255 - new_c*ucres.z - old_c*buffer[y*imgw + x].z)/total;
    buffer[y*imgw + x].w = 255 - (total*255 - new_c*ucres.w - old_c*buffer[y*imgw + x].w)/total;
}

// macros to make indexing shared memory easier
#define SMEM(X, Y) sdata[(Y)*tilew+(X)]

/*
    2D convolution using shared memory
    - operates on 8-bit RGB data stored in 32-bit int
    - assumes kernel radius is less than or equal to block size
    - not optimized for performance
     _____________
    |   :     :   |
    |_ _:_____:_ _|
    |   |     |   |
    |   |     |   |
    |_ _|_____|_ _|
  r |   :     :   |
    |___:_____:___|
      r    bw   r
    <----tilew---->
*/

__global__ void
cudaProcess(unsigned int *g_odata, uchar4 * motion_buffer, int imgw, int imgh,
            int tilew, int r, float threshold, float highlight)
{
    extern __shared__ uchar4 sdata[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int x = blockIdx.x*bw + tx;
    int y = blockIdx.y*bh + ty;


    // perform motion blur
    
    setPixel(x, y, imgw, motion_buffer);
    
    __syncthreads();
    // copy tile to shared memory
    // center region
    SMEM(r + tx, r + ty) = getPixel(x, y, imgw, imgh, motion_buffer);

    // borders
    if (threadIdx.x < r)
    {
        // left
        SMEM(tx, r + ty) = getPixel(x - r, y, imgw, imgh, motion_buffer);
        // right
        SMEM(r + bw + tx, r + ty) = getPixel(x + bw, y, imgw, imgh, motion_buffer);
    }

    if (threadIdx.y < r)
    {
        // top
        SMEM(r + tx, ty) = getPixel(x, y - r, imgw, imgh, motion_buffer);
        // bottom
        SMEM(r + tx, r + bh + ty) = getPixel(x, y + bh, imgw, imgh, motion_buffer);
    }

    // load corners
    if ((threadIdx.x < r) && (threadIdx.y < r))
    {
        // tl
        SMEM(tx, ty) = getPixel(x - r, y - r, imgw, imgh, motion_buffer);
        // bl
        SMEM(tx, r + bh + ty) = getPixel(x - r, y + bh, imgw, imgh, motion_buffer);
        // tr
        SMEM(r + bw + tx, ty) = getPixel(x + bh, y - r, imgw, imgh, motion_buffer);
        // br
        SMEM(r + bw + tx, r + bh + ty) = getPixel(x + bw, y + bh, imgw, imgh, motion_buffer);
    }

    // wait for loads to complete
    __syncthreads();

    // perform convolution
    float rsum = 0.0f;
    float gsum = 0.0f;
    float bsum = 0.0f;
    float samples = 0.0f;

    for (int dy=-r; dy<=r; dy++)
    {
        for (int dx=-r; dx<=r; dx++)
        {
            uchar4 pixel = SMEM(r+tx+dx, r+ty+dy);
            // only sum pixels within disc-shaped kernel
            float l = dx*dx + dy*dy;

            if (l <= r*r)
            {
                float r = float(pixel.x);
                float g = float(pixel.y);
                float b = float(pixel.z);
                // brighten highlights
                float lum = (r + g + b) / (255*3);

                if (lum > threshold)
                {
                    r *= highlight;
                    g *= highlight;
                    b *= highlight;
                }

                rsum += r;
                gsum += g;
                bsum += b;
                samples += 1.0f;
            }
        }
    }

    rsum /= samples;
    gsum /= samples;
    bsum /= samples;
    // ABGR
    g_odata[y*imgw+x] = rgbToInt(rsum, gsum, bsum);
    //g_odata[y*imgw+x] = rgbToInt(x,y,0);
}

extern "C" void
launch_cudaProcess(dim3 grid, dim3 block, int sbytes,
                   cudaArray *g_data_array, uchar4* motion_buffer, unsigned int *g_odata,
                   int imgw, int imgh, int tilew,
                   int radius, float threshold, float highlight)
{
    checkCudaErrors(cudaBindTextureToArray(inTex, g_data_array));

    struct cudaChannelFormatDesc desc;
    checkCudaErrors(cudaGetChannelDesc(&desc, g_data_array));



        cudaProcess<<< grid, block, sbytes >>>(g_odata, motion_buffer, imgw, imgh,
                                               block.x+(2*radius), radius, 0.8f, 4.0f);

}
