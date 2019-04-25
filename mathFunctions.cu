#include "mathFunctions.h"
#include <iostream>
#include "cudaUtility.h"



// -------------------------- Depthwise Layer ----------------------------------
template <typename Dtype>
__global__ void Depthwise(const int nthreads,
                          const Dtype* const bottom_data, const int num, const int channels,
                          const int height, const int width,const int conved_height,
                          const int conved_width,const int kernel_h, const int kernel_w,
                          const int stride_h, const int stride_w, const int pad_h, const int pad_w,
                          Dtype* const top_data,const Dtype* const weight,const Dtype* const bias,const bool bias_term_) {
    CUDA_KERNEL_LOOP(index, nthreads) {

        const int pw = index % conved_width;
        const int ph = (index / conved_width) % conved_height;
        const int c = (index / conved_width / conved_height) % channels;
        const int n = index / conved_width / conved_height / channels;
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = min(hstart + kernel_h, height + pad_h);
        int wend = min(wstart + kernel_w, width + pad_w);

        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        hend = min(hend, height);
        wend = min(wend, width);
        Dtype aveval = 0;
        const Dtype* const bottom_slice =
                bottom_data + (n * channels + c) * height * width;
        const Dtype* const weight_slice =
                weight + c * kernel_h * kernel_w;

        int khstart=hend<kernel_h?kernel_h-hend:0;
        int kwstart=wend<kernel_w?kernel_w-wend:0;
        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {

                aveval += bottom_slice[h * width + w]*weight_slice[(khstart+h-hstart) * kernel_w + (kwstart+w-wstart)];
            }
        }
        if(bias_term_) {
            aveval+=bias[c];
        }
        top_data[index] = aveval;
    }
}

cudaError_t ConvDepthwise(const int nthreads,
                          const float* const bottom_data, const int num, const int channels,
                          const int height, const int width,const int conved_height,
                          const int conved_width,const int kernel_h, const int kernel_w,
                          const int stride_h, const int stride_w, const int pad_h, const int pad_w,
                          float* const top_data,const float* const weight,const float* const bias,const bool bias_term_, cudaStream_t stream)
{
    Depthwise<float><<<TENSORRT_GET_BLOCKS(nthreads), TENSORRT_CUDA_NUM_THREADS,0,stream>>>(nthreads, bottom_data, num, channels, height, width, conved_height, conved_width, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, top_data, weight, bias, bias_term_);
    return cudaPeekAtLastError();
}


// -------------------------- Softmax Layer ----------------------------------
template<typename Dtype>
__global__ void kernel_channel_max(const int num, const int channels,
                                   const int spatial_dim, const Dtype *data, Dtype *out) {
    CUDA_KERNEL_LOOP(index, num * spatial_dim) {
        int n = index / spatial_dim;
        int s = index % spatial_dim;
        Dtype maxval = -1000;
        for (int c = 0; c < channels; ++c) {
            maxval = max(data[(n * channels + c) * spatial_dim + s], maxval);
        }
        out[index] = maxval;
    }
}

template<typename Dtype>
__global__ void kernel_channel_subtract(const int count,
                                        const int num, const int channels,
                                        const int spatial_dim, const Dtype *channel_max, Dtype *data) {
    CUDA_KERNEL_LOOP(index, count) {
        int n = index / channels / spatial_dim;
        int s = index % spatial_dim;
        data[index] -= channel_max[n * spatial_dim + s];
    }
}

template<typename Dtype>
__global__ void kernel_exp(const int count, const Dtype *data, Dtype *out) {
    CUDA_KERNEL_LOOP(index, count) {
        out[index] = exp(data[index]);
    }
}

template<typename Dtype>
__global__ void kernel_channel_sum(const int num, const int channels,
                                   const int spatial_dim, const Dtype *data, Dtype *channel_sum) {
    CUDA_KERNEL_LOOP(index, num * spatial_dim) {
        int n = index / spatial_dim;
        int s = index % spatial_dim;
        Dtype sum = 0;
        for (int c = 0; c < channels; ++c) {
            sum += data[(n * channels + c) * spatial_dim + s];
        }
        channel_sum[index] = sum;
    }
}

template<typename Dtype>
__global__ void kernel_channel_div(const int count,
                                   const int num, const int channels,
                                   const int spatial_dim, const Dtype *channel_sum, Dtype *data) {
    CUDA_KERNEL_LOOP(index, count) {
        int n = index / channels / spatial_dim;
        int s = index % spatial_dim;
        data[index] /= channel_sum[n * spatial_dim + s];
    }
}

cudaError_t
SoftmaxLayer(const float *bottom_data, int count, int channels, int outer_num_, int inner_num_, float *scale_data,
             float *top_data, cudaStream_t stream) {
    kernel_channel_max<float> << < TENSORRT_GET_BLOCKS(outer_num_ * inner_num_), TENSORRT_CUDA_NUM_THREADS, 0,
            stream >> > (outer_num_, channels, inner_num_, top_data, scale_data);
    kernel_channel_subtract<float> << < TENSORRT_GET_BLOCKS(count), TENSORRT_CUDA_NUM_THREADS, 0, stream >> >
                                                                                                  (count, outer_num_, channels, inner_num_, scale_data, top_data);
    kernel_exp<float> << < TENSORRT_GET_BLOCKS(count), TENSORRT_CUDA_NUM_THREADS, 0, stream >> >
                                                                                     (count, top_data, top_data);
    kernel_channel_sum<float> << < TENSORRT_GET_BLOCKS(outer_num_ * inner_num_), TENSORRT_CUDA_NUM_THREADS, 0,
            stream >> > (outer_num_, channels, inner_num_, top_data, scale_data);
    kernel_channel_div<float> << < TENSORRT_GET_BLOCKS(count), TENSORRT_CUDA_NUM_THREADS, 0, stream >> >
                                                                                             (count, outer_num_, channels, inner_num_, scale_data, top_data);
    return cudaPeekAtLastError();
}


// -------------------------- Concat Layer ----------------------------------
template<typename Dtype>
__global__ void Concat(const int nthreads, const Dtype *in_data,
                       const bool forward, const int num_concats, const int concat_size,
                       const int top_concat_axis, const int bottom_concat_axis,
                       const int offset_concat_axis, Dtype *out_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        const int total_concat_size = concat_size * bottom_concat_axis;
        const int concat_num = index / total_concat_size;
        const int concat_index = index % total_concat_size;
        const int top_index = concat_index + (concat_num * top_concat_axis + offset_concat_axis) * concat_size;
        if (forward) {
            out_data[top_index] = in_data[index];
        } else {
            out_data[index] = in_data[top_index];
        }
    }
}

cudaError_t ConcatLayer(int nthreads, const float *bottom_data, bool kForward, int num_concats_, int concat_input_size_,
                        int top_concat_axis, int bottom_concat_axis, int offset_concat_axis, float *top_data,
                        cudaStream_t stream) {
    Concat<float> << < TENSORRT_GET_BLOCKS(nthreads), TENSORRT_CUDA_NUM_THREADS, 0, stream >> >
                                                                                    (nthreads, bottom_data, kForward, num_concats_, concat_input_size_, top_concat_axis, bottom_concat_axis, offset_concat_axis, top_data);
    return cudaPeekAtLastError();
}


