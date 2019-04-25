#include "pluginImplement.h"
#include <vector>
#include <algorithm>
//#include "mathFunctions.h"

void trt_copy(const int N, const float *X, float *Y) {
    if (X != Y) {
        CUDA_CHECK(cudaMemcpy(Y, X, sizeof(float) * N, cudaMemcpyDefault));
    }
}


/**********************************************************************************/
// Depthwise Conv Plugin Layer
/**********************************************************************************/
DepthwisePlugin::DepthwisePlugin(const Weights *weights, int nbWeights, int nbOutputChannels, DepthwiseParam dwp): mNbOutputChannels(nbOutputChannels)
{
    assert(nbWeights == 2);
    mKernelWeights = copyToDevice(weights[0].values, weights[0].count);
    mBiasWeights = copyToDevice(weights[1].values, weights[1].count);
    assert(mBiasWeights.count == 0 || mBiasWeights.count == nbOutputChannels);
    mNbInputChannels = nbOutputChannels;

    depthwise_param = dwp;
}

DepthwisePlugin::DepthwisePlugin(DepthwiseParam dwp, const void* buffer, size_t size)
{
    depthwise_param = dwp;

    const char* d = reinterpret_cast<const char*>(buffer), *a = d;
    input_c = read<int>(d);
    input_h = read<int>(d);
    input_w = read<int>(d);
    dim_h_w = (input_h + 2*depthwise_param.pad - (depthwise_param.dilation * (depthwise_param.kernel_size - 1) + 1)) / depthwise_param.stride + 1;
    input_count = input_c * dim_h_w * dim_h_w;
    mNbInputChannels = read<int>(d);
    int weightCount = read<int>(d);
    int biasCount = read<int>(d);
    mKernelWeights = deserializeToDevice(d, weightCount);
    mBiasWeights = deserializeToDevice(d, biasCount);
    assert(d == a + size);
}

DepthwisePlugin::~DepthwisePlugin()
{
    cudaFree(const_cast<void*>(mKernelWeights.values));
    cudaFree(const_cast<void*>(mBiasWeights.values));
}

Dims DepthwisePlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    assert(index == 0 && nbInputDims == 1 && inputs[0].nbDims == 3);
    dim_h_w = (inputs[0].d[1] + 2*depthwise_param.pad - (depthwise_param.dilation * (depthwise_param.kernel_size - 1) + 1)) / depthwise_param.stride + 1;
    return DimsCHW(inputs[0].d[0], dim_h_w, dim_h_w);
}

int DepthwisePlugin::initialize()
{
    return 0;
}

void DepthwisePlugin::terminate()
{

}

int DepthwisePlugin::enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream)
{
    int inputs_size = 1;
    float *top_data = reinterpret_cast<float*>(outputs[0]);

    for (int i = 0; i < inputs_size; ++i) {
        const float *bottom_data = reinterpret_cast<const float*>(inputs[i]);
        const int channels_ = input_c;
        const int height_ = input_h;
        const int width_ = input_w;

        const int kernel_h_ = depthwise_param.kernel_size;
        const int kernel_w_ = depthwise_param.kernel_size;
        const int stride_h_ = depthwise_param.stride;
        const int stride_w_ = depthwise_param.stride;
        const int pad_h_ = depthwise_param.pad;
        const int pad_w_ = depthwise_param.pad;

        const int conved_height = dim_h_w;
        const int conved_weight = dim_h_w;

        const bool bias_term_ = depthwise_param.bias_term;

        if (bias_term_) {
            ConvDepthwise(input_count, bottom_data, 1, channels_,
                          height_, width_,conved_height,conved_weight,kernel_h_,
                          kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data,reinterpret_cast<const float*>(mKernelWeights.values),reinterpret_cast<const float*>(mBiasWeights.values),bias_term_, stream);
        } else {
            ConvDepthwise(input_count, bottom_data, 1, channels_,
                          height_, width_,conved_height,conved_weight,kernel_h_,
                          kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data,reinterpret_cast<const float*>(mKernelWeights.values),0,bias_term_, stream);
        }
    }

    return 0;
}

size_t DepthwisePlugin::getSerializationSize()
{
    return 6*sizeof(int) + mKernelWeights.count * sizeof(float) + mBiasWeights.count*sizeof(float);
}

void DepthwisePlugin::serialize(void* buffer)
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, dimsBottomData.c());
    write(d, dimsBottomData.h());
    write(d, dimsBottomData.w());
    write(d, mNbInputChannels);
    write(d, (int)mKernelWeights.count);
    write(d, (int)mBiasWeights.count);
    serializeFromDevice(d, mKernelWeights);
    serializeFromDevice(d, mBiasWeights);
    assert(d == a + getSerializationSize());
}

void DepthwisePlugin::configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int)
{
    dimsBottomData = DimsCHW{inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]};
}



/**********************************************************************************/
// Softmax Plugin Layer
/**********************************************************************************/
SoftmaxPlugin::SoftmaxPlugin(const void *buffer, size_t size) {
    assert(size == (3 * sizeof(int)));
    const int *d = reinterpret_cast<const int *>(buffer);
    mInputC = d[0];
    mInputH = d[1];
    mInputW = d[2];
    dimsBottomData = DimsCHW{d[0], d[1], d[2]};
}

Dims SoftmaxPlugin::getOutputDimensions(int index, const Dims *inputs, int nbInputDims) {
    assert(nbInputDims == 1);
    //printf("Softmax inputs[0].d[0] is %d\n",inputs[0].d[0]);
    //printf("Softmax inputs[0].d[1] is %d\n",inputs[0].d[1]);
    //printf("Softmax inputs[0].d[2] is %d\n",inputs[0].d[2]);
    //mInputC = inputs[0].d[0]; mInputH = inputs[0].d[1]; mInputW = inputs[0].d[2];
    return DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
}

int SoftmaxPlugin::initialize() {
    count = mInputC * mInputH * mInputW;    // 1*2252*2
    //outer_num_ = 1917;
    outer_num_ = dimsBottomData.h();        // 2252
    inner_num_ = 1;                         // 1
    channels = mInputW;                     // 2
    CUDA_CHECK(cudaMalloc(&scale_data, count * sizeof(float))); // 中间变量，存储
    return 0;
}

void SoftmaxPlugin::terminate() {
    CUDA_CHECK(cudaFree(scale_data));
}


int SoftmaxPlugin::enqueue(int batchSize, const void *const *inputs, void **outputs, void *, cudaStream_t stream) {
    //printf("mInputC = %d mInputH = %d mInputW = %d\n",mInputC, mInputH, mInputW);
    //printf("start to enqueue softmax");
    const float *bottom_data = reinterpret_cast<const float *>(inputs[0]);
    float *top_data = reinterpret_cast<float *>(outputs[0]);
    trt_copy(count, bottom_data, top_data);
    SoftmaxLayer(bottom_data, count, channels, outer_num_, inner_num_, scale_data, top_data, stream);
    return 0;
}

size_t SoftmaxPlugin::getSerializationSize() {
    return 3 * sizeof(int);
}

void SoftmaxPlugin::serialize(void *buffer) {
    int *d = reinterpret_cast<int *>(buffer);
    d[0] = dimsBottomData.c();
    d[1] = dimsBottomData.h();
    d[2] = dimsBottomData.w();
}

void SoftmaxPlugin::configure(const Dims *inputs, int nbInputs, const Dims *outputs, int nbOutputs, int) {
    //mInputC = inputs[0].d[0]; mInputH = inputs[0].d[1]; mInputW = inputs[0].d[2];
    dimsBottomData = DimsCHW{inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]};
}


/**********************************************************************************/
// Concat Plugin Layer
/**********************************************************************************/
ConcatPlugin::ConcatPlugin(int axis, const void *buffer, size_t size) {
    assert(size == (15 * sizeof(int)));
    const int *d = reinterpret_cast<const int *>(buffer);

    dimsFc = DimsCHW{d[0], d[1], d[2]};
    dimsConv6 = DimsCHW{d[3], d[4], d[5]};
    dimsConv7 = DimsCHW{d[6], d[7], d[8]};
    dimsConv8 = DimsCHW{d[9], d[10], d[11]};
    dimsConv9 = DimsCHW{d[12], d[13], d[14]};

    _axis = axis;
}

Dims ConcatPlugin::getOutputDimensions(int index, const Dims *inputs, int nbInputDims) {
    assert(nbInputDims == 5);

    if (_axis == 1) {
        top_concat_axis = inputs[0].d[0] + inputs[1].d[0] + inputs[2].d[0] + inputs[3].d[0] + inputs[4].d[0];
        return DimsCHW(top_concat_axis, 1, 1);
    } else if (_axis == 2) {
        top_concat_axis = inputs[0].d[1] + inputs[1].d[1] + inputs[2].d[1] + inputs[3].d[1] + inputs[4].d[1];
        return DimsCHW(2, top_concat_axis, 1);
    } else {
        return DimsCHW(0, 0, 0);
    }
}

int ConcatPlugin::initialize() {
    inputs_size = 5;    // 5个bottom层

    if (_axis == 1)     // c
    {
        top_concat_axis = dimsFc.c() + dimsConv6.c() + dimsConv7.c() + dimsConv8.c() + dimsConv9.c();
        bottom_concat_axis[0] = dimsFc.c();
        bottom_concat_axis[1] = dimsConv6.c();
        bottom_concat_axis[2] = dimsConv7.c();
        bottom_concat_axis[3] = dimsConv8.c();
        bottom_concat_axis[4] = dimsConv9.c();

        concat_input_size_[0] = dimsFc.h() * dimsFc.w();
        concat_input_size_[1] = dimsConv6.h() * dimsConv6.w();
        concat_input_size_[2] = dimsConv7.h() * dimsConv7.w();
        concat_input_size_[3] = dimsConv8.h() * dimsConv8.w();
        concat_input_size_[4] = dimsConv9.h() * dimsConv9.w();

        num_concats_[0] = dimsFc.c();
        num_concats_[1] = dimsConv6.c();
        num_concats_[2] = dimsConv7.c();
        num_concats_[3] = dimsConv8.c();
        num_concats_[4] = dimsConv9.c();
    }
    else if (_axis == 2)    // h
    {
        top_concat_axis = dimsFc.h() + dimsConv6.h() + dimsConv7.h() + dimsConv8.h() + dimsConv9.h();
        bottom_concat_axis[0] = dimsFc.h();
        bottom_concat_axis[1] = dimsConv6.h();
        bottom_concat_axis[2] = dimsConv7.h();
        bottom_concat_axis[3] = dimsConv8.h();
        bottom_concat_axis[4] = dimsConv9.h();

        concat_input_size_[0] = dimsFc.w();
        concat_input_size_[1] = dimsConv6.w();
        concat_input_size_[2] = dimsConv7.w();
        concat_input_size_[3] = dimsConv8.w();
        concat_input_size_[4] = dimsConv9.w();

        num_concats_[0] = dimsFc.c() * dimsFc.h();
        num_concats_[1] = dimsConv6.c() * dimsConv6.h();
        num_concats_[2] = dimsConv7.c() * dimsConv7.h();
        num_concats_[3] = dimsConv8.c() * dimsConv8.h();
        num_concats_[4] = dimsConv9.c() * dimsConv9.h();

    }
    else  //_param.concat_axis == 3 , w
    {
        top_concat_axis = dimsFc.w() + dimsConv6.w() + dimsConv7.w() + dimsConv8.w() + dimsConv9.w();
        bottom_concat_axis[0] = dimsFc.w();
        bottom_concat_axis[1] = dimsConv6.w();
        bottom_concat_axis[2] = dimsConv7.w();
        bottom_concat_axis[3] = dimsConv8.w();
        bottom_concat_axis[4] = dimsConv9.w();

        concat_input_size_[0] = 1;
        concat_input_size_[1] = 1;
        concat_input_size_[2] = 1;
        concat_input_size_[3] = 1;
        concat_input_size_[4] = 1;

        return 0;
    }

    return 0;
}

void ConcatPlugin::terminate() {
    //CUDA_CHECK(cudaFree(scale_data));
    delete[] bottom_concat_axis;
    delete[] concat_input_size_;
    delete[] num_concats_;
}


int ConcatPlugin::enqueue(int batchSize, const void *const *inputs, void **outputs, void *, cudaStream_t stream) {
    float *top_data = reinterpret_cast<float *>(outputs[0]);
    int offset_concat_axis = 0;
    const bool kForward = true;
    for (int i = 0; i < inputs_size; ++i) {
        const float *bottom_data = reinterpret_cast<const float *>(inputs[i]);

        const int nthreads = num_concats_[i] * concat_input_size_[i];
        //const int nthreads = bottom_concat_size * num_concats_[i];
        ConcatLayer(nthreads, bottom_data, kForward, num_concats_[i], concat_input_size_[i], top_concat_axis,
                    bottom_concat_axis[i], offset_concat_axis, top_data, stream);

        offset_concat_axis += bottom_concat_axis[i];
    }

    return 0;
}

size_t ConcatPlugin::getSerializationSize() {
    return 15 * sizeof(int);
}

void ConcatPlugin::serialize(void *buffer) {
    int *d = reinterpret_cast<int *>(buffer);
    d[0] = dimsFc.c();
    d[1] = dimsFc.h();
    d[2] = dimsFc.w();
    d[3] = dimsConv6.c();
    d[4] = dimsConv6.h();
    d[5] = dimsConv6.w();
    d[6] = dimsConv7.c();
    d[7] = dimsConv7.h();
    d[8] = dimsConv7.w();
    d[9] = dimsConv8.c();
    d[10] = dimsConv8.h();
    d[11] = dimsConv8.w();
    d[12] = dimsConv9.c();
    d[13] = dimsConv9.h();
    d[14] = dimsConv9.w();
}

void ConcatPlugin::configure(const Dims *inputs, int nbInputs, const Dims *outputs, int nbOutputs, int) {
    dimsFc = DimsCHW{inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]};
    dimsConv6 = DimsCHW{inputs[1].d[0], inputs[1].d[1], inputs[1].d[2]};
    dimsConv7 = DimsCHW{inputs[2].d[0], inputs[2].d[1], inputs[2].d[2]};
    dimsConv8 = DimsCHW{inputs[3].d[0], inputs[3].d[1], inputs[3].d[2]};
    dimsConv9 = DimsCHW{inputs[4].d[0], inputs[4].d[1], inputs[4].d[2]};
}


/**********************************************************************************/
// Concatenation Plugin Layer
/**********************************************************************************/
ConcatenationPlugin::ConcatenationPlugin(int axis, const void *buffer, size_t size) {
    assert(size == (6 * sizeof(int)));
    const int *d = reinterpret_cast<const int *>(buffer);

    dimsA = DimsCHW{d[0], d[1], d[2]};
    dimsB = DimsCHW{d[3], d[4], d[5]};

    _axis_two = axis;
}

Dims ConcatenationPlugin::getOutputDimensions(int index, const Dims *inputs, int nbInputDims) {
    assert(nbInputDims == 2);

    if (_axis_two == 1) {
        top_concat_axis_two = inputs[0].d[0] + inputs[1].d[0];
        return DimsCHW(top_concat_axis_two, inputs[0].d[1], inputs[0].d[1]); // (C1+C2)×10×10  (C1+C2)×5×5  (C1+C2)×3×3
    }
    else {
        return DimsCHW(0, 0, 0);
    }
}

int ConcatenationPlugin::initialize() {
    inputs_size_two = 2;    // 2个bottom层

    if (_axis_two == 1)     // c
    {
        top_concat_axis_two = dimsA.c() + dimsB.c();
        bottom_concat_axis_two[0] = dimsA.c();
        bottom_concat_axis_two[1] = dimsB.c();

        concat_input_size_two[0] = dimsA.h() * dimsA.w();
        concat_input_size_two[1] = dimsB.h() * dimsB.w();

        num_concats_two[0] = dimsA.c();
        num_concats_two[1] = dimsB.c();
    }
    else
    {
        return 0;
    }

    return 0;
}

void ConcatenationPlugin::terminate() {
    //CUDA_CHECK(cudaFree(scale_data));
    delete[] bottom_concat_axis_two;
    delete[] concat_input_size_two;
    delete[] num_concats_two;
}


int ConcatenationPlugin::enqueue(int batchSize, const void *const *inputs, void **outputs, void *, cudaStream_t stream) {
    float *top_data = reinterpret_cast<float *>(outputs[0]);
    int offset_concat_axis = 0;
    const bool kForward = true;
    for (int i = 0; i < inputs_size_two; ++i) {
        const float *bottom_data = reinterpret_cast<const float *>(inputs[i]);

        const int nthreads = num_concats_two[i] * concat_input_size_two[i];
        //const int nthreads = bottom_concat_size * num_concats_[i];
        ConcatLayer(nthreads, bottom_data, kForward, num_concats_two[i], concat_input_size_two[i], top_concat_axis_two,
                    bottom_concat_axis_two[i], offset_concat_axis, top_data, stream);

        offset_concat_axis += bottom_concat_axis_two[i];
    }

    return 0;
}

size_t ConcatenationPlugin::getSerializationSize() {
    return 6 * sizeof(int);
}

void ConcatenationPlugin::serialize(void *buffer) {
    int *d = reinterpret_cast<int *>(buffer);
    d[0] = dimsA.c();
    d[1] = dimsA.h();
    d[2] = dimsA.w();
    d[3] = dimsB.c();
    d[4] = dimsB.h();
    d[5] = dimsB.w();

}

void ConcatenationPlugin::configure(const Dims *inputs, int nbInputs, const Dims *outputs, int nbOutputs, int) {
    dimsA = DimsCHW{inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]};
    dimsB = DimsCHW{inputs[1].d[0], inputs[1].d[1], inputs[1].d[2]};
}


/**********************************************************************************/
// PluginFactory
/**********************************************************************************/
nvinfer1::IPlugin *PluginFactory::createPlugin(const char *layerName, const nvinfer1::Weights *weights, int nbWeights) {

    assert(isPlugin(layerName));

    // depthwise convolution layer
    if (!strcmp(layerName, "conv1/dw"))
    {
        static const int NB_OUTPUT_CHANNELS = 32;

        assert(mConv1_dw_layer.get() == nullptr);
        mConv1_dw_layer = std::unique_ptr<DepthwisePlugin>(new DepthwisePlugin(weights, nbWeights, NB_OUTPUT_CHANNELS, {1,1,3,1,true}));
        return mConv1_dw_layer.get();
    }
    else if (!strcmp(layerName, "conv2/dw"))
    {
        static const int NB_OUTPUT_CHANNELS = 64;

        assert(mConv2_dw_layer.get() == nullptr);
        mConv2_dw_layer = std::unique_ptr<DepthwisePlugin>(new DepthwisePlugin(weights, nbWeights, NB_OUTPUT_CHANNELS, {1,1,3,2,true}));
        return mConv2_dw_layer.get();
    }
    else if (!strcmp(layerName, "conv3/dw"))
    {
        static const int NB_OUTPUT_CHANNELS = 128;

        assert(mConv3_dw_layer.get() == nullptr);
        mConv3_dw_layer = std::unique_ptr<DepthwisePlugin>(new DepthwisePlugin(weights, nbWeights, NB_OUTPUT_CHANNELS, {1,1,3,1,true}));
        return mConv3_dw_layer.get();
    }
    else if (!strcmp(layerName, "conv4/dw"))
    {
        static const int NB_OUTPUT_CHANNELS = 128;
        assert(mConv4_dw_layer.get() == nullptr);
        mConv4_dw_layer = std::unique_ptr<DepthwisePlugin>(new DepthwisePlugin(weights, nbWeights, NB_OUTPUT_CHANNELS, {1,1,3,2,true}));
        return mConv4_dw_layer.get();
    }
    else if (!strcmp(layerName, "conv5/dw"))
    {
        static const int NB_OUTPUT_CHANNELS = 256;

        assert(mConv5_dw_layer.get() == nullptr);
        mConv5_dw_layer = std::unique_ptr<DepthwisePlugin>(new DepthwisePlugin(weights, nbWeights, NB_OUTPUT_CHANNELS, {1,1,3,1,false}));
        return mConv5_dw_layer.get();
    }
    else if (!strcmp(layerName, "conv6/dw"))
    {
        static const int NB_OUTPUT_CHANNELS = 256;

        assert(mConv6_dw_layer.get() == nullptr);
        mConv6_dw_layer = std::unique_ptr<DepthwisePlugin>(new DepthwisePlugin(weights, nbWeights, NB_OUTPUT_CHANNELS, {1,1,3,2,true}));
        return mConv6_dw_layer.get();
    }
    else if (!strcmp(layerName, "conv7/dw"))
    {
        static const int NB_OUTPUT_CHANNELS = 512;

        assert(mConv7_dw_layer.get() == nullptr);
        mConv7_dw_layer = std::unique_ptr<DepthwisePlugin>(new DepthwisePlugin(weights, nbWeights, NB_OUTPUT_CHANNELS, {1,1,3,1,true}));
        return mConv7_dw_layer.get();
    }
    else if (!strcmp(layerName, "conv8/dw"))
    {
        static const int NB_OUTPUT_CHANNELS = 512;

        assert(mConv8_dw_layer.get() == nullptr);
        mConv8_dw_layer = std::unique_ptr<DepthwisePlugin>(new DepthwisePlugin(weights, nbWeights, NB_OUTPUT_CHANNELS, {1,1,3,1,true}));
        return mConv8_dw_layer.get();
    }
    else if (!strcmp(layerName, "conv9/dw"))
    {
        static const int NB_OUTPUT_CHANNELS = 512;

        assert(mConv9_dw_layer.get() == nullptr);
        mConv9_dw_layer = std::unique_ptr<DepthwisePlugin>(new DepthwisePlugin(weights, nbWeights, NB_OUTPUT_CHANNELS, {1,1,3,1,true}));
        return mConv9_dw_layer.get();
    }
    else if (!strcmp(layerName, "conv10/dw"))
    {
        static const int NB_OUTPUT_CHANNELS = 512;

        assert(mConv10_dw_layer.get() == nullptr);
        mConv10_dw_layer = std::unique_ptr<DepthwisePlugin>(new DepthwisePlugin(weights, nbWeights, NB_OUTPUT_CHANNELS, {1,1,3,1,true}));
        return mConv10_dw_layer.get();
    }
    else if (!strcmp(layerName, "conv11/dw"))
    {

        static const int NB_OUTPUT_CHANNELS = 512;

        assert(mConv11_dw_layer.get() == nullptr);
        mConv11_dw_layer = std::unique_ptr<DepthwisePlugin>(new DepthwisePlugin(weights, nbWeights, NB_OUTPUT_CHANNELS, {1,1,3,1,true}));
        return mConv11_dw_layer.get();
    }
    // permute layer
    else if (!strcmp(layerName, "A_mbox_loc_perm")) {
        assert(mA_mbox_loc_perm_layer.get() == nullptr);
        mA_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(
                createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mA_mbox_loc_perm_layer.get();
    }
    else if (!strcmp(layerName, "A_mbox_conf_perm")) {
        assert(mA_mbox_conf_perm_layer.get() == nullptr);
        mA_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(
                createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mA_mbox_conf_perm_layer.get();
    } else if (!strcmp(layerName, "B_mbox_loc_perm")) {
        assert(mB_mbox_loc_perm_layer.get() == nullptr);
        mB_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(
                createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mB_mbox_loc_perm_layer.get();
    } else if (!strcmp(layerName, "B_mbox_conf_perm")) {
        assert(mB_mbox_conf_perm_layer.get() == nullptr);
        mB_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(
                createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mB_mbox_conf_perm_layer.get();
    } else if (!strcmp(layerName, "C_mbox_loc_perm")) {
        assert(mC_mbox_loc_perm_layer.get() == nullptr);
        mC_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(
                createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mC_mbox_loc_perm_layer.get();
    } else if (!strcmp(layerName, "C_mbox_conf_perm")) {
        assert(mC_mbox_conf_perm_layer.get() == nullptr);
        mC_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(
                createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mC_mbox_conf_perm_layer.get();
    } else if (!strcmp(layerName, "D_mbox_loc_perm")) {
        assert(mD_mbox_loc_perm_layer.get() == nullptr);
        mD_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(
                createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mD_mbox_loc_perm_layer.get();
    } else if (!strcmp(layerName, "D_mbox_conf_perm")) {
        assert(mD_mbox_conf_perm_layer.get() == nullptr);
        mD_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(
                createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mD_mbox_conf_perm_layer.get();
    } else if (!strcmp(layerName, "E_mbox_loc_perm")) {
        assert(mE_mbox_loc_perm_layer.get() == nullptr);
        mE_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(
                createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mE_mbox_loc_perm_layer.get();
    } else if (!strcmp(layerName, "E_mbox_conf_perm")) {
        assert(mE_mbox_conf_perm_layer.get() == nullptr);
        mE_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(
                createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mE_mbox_conf_perm_layer.get();
    }
    // priorbox layer
    else if (!strcmp(layerName, "A_mbox_priorbox")) {
        assert(mA_mbox_priorbox_layer.get() == nullptr);
        float min_size = 60.0, max_size = 115.0, aspect_ratio[2] = {1.0, 2.0};
        mA_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(createSSDPriorBoxPlugin(
                        {&min_size, &max_size, aspect_ratio, 1, 1, 2, true, false, {0.1, 0.1, 0.2, 0.2}, 0, 0, 16.0,
                         16.0, 0.5}), nvPluginDeleter);
        return mA_mbox_priorbox_layer.get();
    } else if (!strcmp(layerName, "B_mbox_priorbox")) {
        assert(mB_mbox_priorbox_layer.get() == nullptr);
        float min_size = 115.0, max_size = 170.0, aspect_ratio[3] = {1.0, 2.0, 3.0};
        mB_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(createSSDPriorBoxPlugin(
                {&min_size, &max_size, aspect_ratio, 1, 1, 3, true, false, {0.1, 0.1, 0.2, 0.2}, 0, 0, 30.0,
                 30.0, 0.5}), nvPluginDeleter);
        return mB_mbox_priorbox_layer.get();
    } else if (!strcmp(layerName, "C_mbox_priorbox")) {
        assert(mC_mbox_priorbox_layer.get() == nullptr);
        float min_size = 170.0, max_size = 225.0, aspect_ratio[3] = {1.0, 2.0, 3.0};
        mC_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(createSSDPriorBoxPlugin(
                {&min_size, &max_size, aspect_ratio, 1, 1, 3, true, false, {0.1, 0.1, 0.2, 0.2}, 0, 0, 60.0,
                 60.0, 0.5}), nvPluginDeleter);
        return mC_mbox_priorbox_layer.get();
    } else if (!strcmp(layerName, "D_mbox_priorbox")) {
        assert(mD_mbox_priorbox_layer.get() == nullptr);
        float min_size = 225.0, max_size = 280.0, aspect_ratio[3] = {1.0, 2.0, 3.0};
        mD_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> (createSSDPriorBoxPlugin(
                        {&min_size, &max_size, aspect_ratio, 1, 1, 3, true, false, {0.1, 0.1, 0.2, 0.2}, 0, 0, 100.0,
                         100.0, 0.5}), nvPluginDeleter);
        return mD_mbox_priorbox_layer.get();
    } else if (!strcmp(layerName, "E_mbox_priorbox")) {
        assert(mE_mbox_priorbox_layer.get() == nullptr);
        float min_size = 280.0, max_size = 320.0, aspect_ratio[2] = {1.0, 2.0};
        mE_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> (createSSDPriorBoxPlugin(
                        {&min_size, &max_size, aspect_ratio, 1, 1, 2, true, false, {0.1, 0.1, 0.2, 0.2}, 0, 0, 300.0,
                         300.0, 0.5}), nvPluginDeleter);
        return mE_mbox_priorbox_layer.get();
    }
    // concat layer
    else if (!strcmp(layerName, "concat6")) {
        assert(mconcat6_layer.get() == nullptr);
        mconcat6_layer = std::unique_ptr<ConcatenationPlugin>(new ConcatenationPlugin(1));
        return mconcat6_layer.get();
    }    else if (!strcmp(layerName, "concat7")) {
        assert(mconcat7_layer.get() == nullptr);
        mconcat7_layer = std::unique_ptr<ConcatenationPlugin>(new ConcatenationPlugin(1));
        return mconcat7_layer.get();
    }    else if (!strcmp(layerName, "concat8")) {
        assert(mconcat8_layer.get() == nullptr);
        mconcat8_layer = std::unique_ptr<ConcatenationPlugin>(new ConcatenationPlugin(1));
        return mconcat8_layer.get();
    }
    else if (!strcmp(layerName, "mbox_loc_ours")) {
        assert(mmbox_loc_ours_layer.get() == nullptr);
        mmbox_loc_ours_layer = std::unique_ptr<ConcatPlugin>(new ConcatPlugin(1));
        return mmbox_loc_ours_layer.get();
    } else if (!strcmp(layerName, "mbox_conf_ours")) {
        assert(mmbox_conf_ours_layer.get() == nullptr);
        mmbox_conf_ours_layer = std::unique_ptr<ConcatPlugin>(new ConcatPlugin(1));
        return mmbox_conf_ours_layer.get();
    } else if (!strcmp(layerName, "mbox_priorbox_ours")) {
        assert(mmbox_priorbox_ours_layer.get() == nullptr);
        mmbox_priorbox_ours_layer = std::unique_ptr<ConcatPlugin>(new ConcatPlugin(2));
        return mmbox_priorbox_ours_layer.get();
    }
    // flatten layer
    else if (!strcmp(layerName, "A_mbox_loc_flat")) {
        assert(mA_mbox_loc_flat_layer.get() == nullptr);
        mA_mbox_loc_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return mA_mbox_loc_flat_layer.get();
    } else if (!strcmp(layerName, "A_mbox_conf_flat")) {
        assert(mA_mbox_conf_flat_layer.get() == nullptr);
        mA_mbox_conf_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return mA_mbox_conf_flat_layer.get();
    } else if (!strcmp(layerName, "B_mbox_loc_flat")) {
        assert(mB_mbox_loc_flat_layer.get() == nullptr);
        mB_mbox_loc_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return mB_mbox_loc_flat_layer.get();
    } else if (!strcmp(layerName, "B_mbox_conf_flat")) {
        assert(mB_mbox_conf_flat_layer.get() == nullptr);
        mB_mbox_conf_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return mB_mbox_conf_flat_layer.get();
    } else if (!strcmp(layerName, "C_mbox_loc_flat")) {
        assert(mC_mbox_loc_flat_layer.get() == nullptr);
        mC_mbox_loc_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return mC_mbox_loc_flat_layer.get();
    } else if (!strcmp(layerName, "C_mbox_conf_flat")) {
        assert(mC_mbox_conf_flat_layer.get() == nullptr);
        mC_mbox_conf_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return mC_mbox_conf_flat_layer.get();
    } else if (!strcmp(layerName, "D_mbox_loc_flat")) {
        assert(mD_mbox_loc_flat_layer.get() == nullptr);
        mD_mbox_loc_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return mD_mbox_loc_flat_layer.get();
    } else if (!strcmp(layerName, "D_mbox_conf_flat")) {
        assert(mD_mbox_conf_flat_layer.get() == nullptr);
        mD_mbox_conf_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return mD_mbox_conf_flat_layer.get();
    } else if (!strcmp(layerName, "E_mbox_loc_flat")) {
        assert(mE_mbox_loc_flat_layer.get() == nullptr);
        mE_mbox_loc_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return mE_mbox_loc_flat_layer.get();
    } else if (!strcmp(layerName, "E_mbox_conf_flat")) {
        assert(mE_mbox_conf_flat_layer.get() == nullptr);
        mE_mbox_conf_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return mE_mbox_conf_flat_layer.get();
    } else if (!strcmp(layerName, "mbox_conf_flatten_ours")) {
        assert(mmbox_conf_flatten_ours_layer.get() == nullptr);
        mmbox_conf_flatten_ours_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return mmbox_conf_flatten_ours_layer.get();
    }
    // reshape layer
    else if (!strcmp(layerName, "mbox_conf_reshape_ours")) {
        assert(mmbox_conf_reshape_ours_layer.get() == nullptr);
        assert(nbWeights == 0 && weights == nullptr);
        mmbox_conf_reshape_ours_layer = std::unique_ptr<Reshape<2>>(new Reshape<2>());
        return mmbox_conf_reshape_ours_layer.get();
    }
    // softmax layer
    else if (!strcmp(layerName, "mbox_conf_softmax_ours")) {
        assert(mmbox_conf_softmax_ours_layer == nullptr);
        assert(nbWeights == 0 && weights == nullptr);
        mmbox_conf_softmax_ours_layer = std::unique_ptr<SoftmaxPlugin>(new SoftmaxPlugin());
        return mmbox_conf_softmax_ours_layer.get();
    }
    // detection_out layer
    else if (!strcmp(layerName, "detection_out_ours")) {
        assert(mdetection_out_ours_layer.get() == nullptr);
        mdetection_out_ours_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDDetectionOutputPlugin({true, false, 0, 2, 100, 20, 0.4, 0.35, CodeType_t::CENTER_SIZE}),
                 nvPluginDeleter);
        return mdetection_out_ours_layer.get();
    }
    // others
    else {
        std::cout << layerName << std::endl;
        assert(0);
        return nullptr;
    }
}

IPlugin *PluginFactory::createPlugin(const char *layerName, const void *serialData, size_t serialLength) {
    assert(isPlugin(layerName));

    // depthwise convolution layer
    if (!strcmp(layerName, "conv1/dw"))
    {
        assert(mConv1_dw_layer.get() == nullptr);
        mConv1_dw_layer = std::unique_ptr<DepthwisePlugin>(new DepthwisePlugin({1,1,3,1,true},serialData, serialLength));
        return mConv1_dw_layer.get();
    }
    else if (!strcmp(layerName, "conv2/dw"))
    {
        assert(mConv2_dw_layer.get() == nullptr);
        mConv2_dw_layer = std::unique_ptr<DepthwisePlugin>(new DepthwisePlugin({1,1,3,2,true},serialData, serialLength));
        return mConv2_dw_layer.get();
    }
    else if (!strcmp(layerName, "conv3/dw"))
    {
        assert(mConv3_dw_layer.get() == nullptr);
        mConv3_dw_layer = std::unique_ptr<DepthwisePlugin>(new DepthwisePlugin({1,1,3,1,true},serialData, serialLength));
        return mConv3_dw_layer.get();
    }
    else if (!strcmp(layerName, "conv4/dw"))
    {
        assert(mConv4_dw_layer.get() == nullptr);
        mConv4_dw_layer = std::unique_ptr<DepthwisePlugin>(new DepthwisePlugin({1,1,3,2,true},serialData, serialLength));
        return mConv4_dw_layer.get();
    }
    else if (!strcmp(layerName, "conv5/dw"))
    {
        assert(mConv5_dw_layer.get() == nullptr);
        mConv5_dw_layer = std::unique_ptr<DepthwisePlugin>(new DepthwisePlugin({1,1,3,1,true},serialData, serialLength));
        return mConv5_dw_layer.get();
    }
    else if (!strcmp(layerName, "conv6/dw"))
    {
        assert(mConv6_dw_layer.get() == nullptr);
        mConv6_dw_layer = std::unique_ptr<DepthwisePlugin>(new DepthwisePlugin({1,1,3,2,true},serialData, serialLength));
        return mConv6_dw_layer.get();
    }
    else if (!strcmp(layerName, "conv7/dw"))
    {
        assert(mConv7_dw_layer.get() == nullptr);
        mConv7_dw_layer = std::unique_ptr<DepthwisePlugin>(new DepthwisePlugin({1,1,3,1,true},serialData, serialLength));
        return mConv7_dw_layer.get();
    }
    else if (!strcmp(layerName, "conv8/dw"))
    {
        assert(mConv8_dw_layer.get() == nullptr);
        mConv8_dw_layer = std::unique_ptr<DepthwisePlugin>(new DepthwisePlugin({1,1,3,1,true},serialData, serialLength));
        return mConv8_dw_layer.get();
    }
    else if (!strcmp(layerName, "conv9/dw"))
    {
        assert(mConv9_dw_layer.get() == nullptr);
        mConv9_dw_layer = std::unique_ptr<DepthwisePlugin>(new DepthwisePlugin({1,1,3,1,true},serialData, serialLength));
        return mConv9_dw_layer.get();
    }
    else if (!strcmp(layerName, "conv10/dw"))
    {
        assert(mConv10_dw_layer.get() == nullptr);
        mConv10_dw_layer = std::unique_ptr<DepthwisePlugin>(new DepthwisePlugin({1,1,3,1,true},serialData, serialLength));
        return mConv10_dw_layer.get();
    }
    else if (!strcmp(layerName, "conv11/dw"))
    {
        assert(mConv11_dw_layer.get() == nullptr);
        mConv11_dw_layer = std::unique_ptr<DepthwisePlugin>(new DepthwisePlugin({1,1,3,1,true},serialData, serialLength));
        return mConv11_dw_layer.get();
    }
    // permute layer
    else if (!strcmp(layerName, "A_mbox_loc_perm")) {
        assert(mA_mbox_loc_perm_layer.get() == nullptr);
        mA_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(
                createSSDPermutePlugin(serialData, serialLength), nvPluginDeleter);
        return mA_mbox_loc_perm_layer.get();
    } else if (!strcmp(layerName, "A_mbox_conf_perm")) {
        assert(mA_mbox_conf_perm_layer.get() == nullptr);
        mA_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(
                createSSDPermutePlugin(serialData, serialLength), nvPluginDeleter);
        return mA_mbox_conf_perm_layer.get();
    } else if (!strcmp(layerName, "B_mbox_loc_perm")) {
        assert(mB_mbox_loc_perm_layer.get() == nullptr);
        mB_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(
                createSSDPermutePlugin(serialData, serialLength), nvPluginDeleter);
        return mB_mbox_loc_perm_layer.get();
    } else if (!strcmp(layerName, "B_mbox_conf_perm")) {
        assert(mB_mbox_conf_perm_layer.get() == nullptr);
        mB_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(
                createSSDPermutePlugin(serialData, serialLength), nvPluginDeleter);
        return mB_mbox_conf_perm_layer.get();
    } else if (!strcmp(layerName, "C_mbox_loc_perm")) {
        assert(mC_mbox_loc_perm_layer.get() == nullptr);
        mC_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(
                createSSDPermutePlugin(serialData, serialLength), nvPluginDeleter);
        return mC_mbox_loc_perm_layer.get();
    } else if (!strcmp(layerName, "C_mbox_conf_perm")) {
        assert(mC_mbox_conf_perm_layer.get() == nullptr);
        mC_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(
                createSSDPermutePlugin(serialData, serialLength), nvPluginDeleter);
        return mC_mbox_conf_perm_layer.get();
    } else if (!strcmp(layerName, "D_mbox_loc_perm")) {
        assert(mD_mbox_loc_perm_layer.get() == nullptr);
        mD_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(
                createSSDPermutePlugin(serialData, serialLength), nvPluginDeleter);
        return mD_mbox_loc_perm_layer.get();
    } else if (!strcmp(layerName, "D_mbox_conf_perm")) {
        assert(mD_mbox_conf_perm_layer.get() == nullptr);
        mD_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(
                createSSDPermutePlugin(serialData, serialLength), nvPluginDeleter);
        return mD_mbox_conf_perm_layer.get();
    } else if (!strcmp(layerName, "E_mbox_loc_perm")) {
        assert(mE_mbox_loc_perm_layer.get() == nullptr);
        mE_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(
                createSSDPermutePlugin(serialData, serialLength), nvPluginDeleter);
        return mE_mbox_loc_perm_layer.get();
    } else if (!strcmp(layerName, "E_mbox_conf_perm")) {
        assert(mE_mbox_conf_perm_layer.get() == nullptr);
        mE_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(
                createSSDPermutePlugin(serialData, serialLength), nvPluginDeleter);
        return mE_mbox_conf_perm_layer.get();
    }
    // priorbox layer
    else if (!strcmp(layerName, "A_mbox_priorbox")) {
        assert(mA_mbox_priorbox_layer.get() == nullptr);
        mA_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(
                createSSDPriorBoxPlugin(serialData, serialLength), nvPluginDeleter);
        return mA_mbox_priorbox_layer.get();
    } else if (!strcmp(layerName, "B_mbox_priorbox")) {
        assert(mB_mbox_priorbox_layer.get() == nullptr);
        mB_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(
                createSSDPriorBoxPlugin(serialData, serialLength), nvPluginDeleter);
        return mB_mbox_priorbox_layer.get();
    } else if (!strcmp(layerName, "C_mbox_priorbox")) {
        assert(mC_mbox_priorbox_layer.get() == nullptr);
        mC_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(
                createSSDPriorBoxPlugin(serialData, serialLength), nvPluginDeleter);
        return mC_mbox_priorbox_layer.get();
    } else if (!strcmp(layerName, "D_mbox_priorbox")) {
        assert(mD_mbox_priorbox_layer.get() == nullptr);
        mD_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(
                createSSDPriorBoxPlugin(serialData, serialLength), nvPluginDeleter);
        return mD_mbox_priorbox_layer.get();
    } else if (!strcmp(layerName, "E_mbox_priorbox")) {
        assert(mE_mbox_priorbox_layer.get() == nullptr);
        mE_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(
                createSSDPriorBoxPlugin(serialData, serialLength), nvPluginDeleter);
        return mE_mbox_priorbox_layer.get();
    }
    // concat layer
    else if (!strcmp(layerName, "concat6")) {
        assert(mconcat6_layer.get() == nullptr);
        mconcat6_layer = std::unique_ptr<ConcatenationPlugin>(new ConcatenationPlugin(1, serialData, serialLength));
        return mconcat6_layer.get();
    } else if (!strcmp(layerName, "concat7")) {
        assert(mconcat7_layer.get() == nullptr);
        mconcat7_layer = std::unique_ptr<ConcatenationPlugin>(new ConcatenationPlugin(1, serialData, serialLength));
        return mconcat7_layer.get();
    } else if (!strcmp(layerName, "concat8")) {
        assert(mconcat8_layer.get() == nullptr);
        mconcat8_layer = std::unique_ptr<ConcatenationPlugin>(new ConcatenationPlugin(1, serialData, serialLength));
        return mconcat8_layer.get();
    }
    else if (!strcmp(layerName, "mbox_loc_ours")) {
        assert(mmbox_loc_ours_layer.get() == nullptr);
        mmbox_loc_ours_layer = std::unique_ptr<ConcatPlugin>(new ConcatPlugin(1, serialData, serialLength));
        return mmbox_loc_ours_layer.get();
    } else if (!strcmp(layerName, "mbox_conf_ours")) {
        assert(mmbox_conf_ours_layer.get() == nullptr);
        mmbox_conf_ours_layer = std::unique_ptr<ConcatPlugin>(new ConcatPlugin(1, serialData, serialLength));
        return mmbox_conf_ours_layer.get();
    } else if (!strcmp(layerName, "mbox_priorbox_ours")) {
        assert(mmbox_priorbox_ours_layer.get() == nullptr);
        mmbox_priorbox_ours_layer = std::unique_ptr<ConcatPlugin>(new ConcatPlugin(2, serialData, serialLength));
        return mmbox_priorbox_ours_layer.get();
    }
    // flatten layer
    else if (!strcmp(layerName, "A_mbox_loc_flat")) {
        assert(mA_mbox_loc_flat_layer.get() == nullptr);
        mA_mbox_loc_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData, serialLength));
        return mA_mbox_loc_flat_layer.get();
    } else if (!strcmp(layerName, "A_mbox_conf_flat")) {
        assert(mA_mbox_conf_flat_layer.get() == nullptr);
        mA_mbox_conf_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData, serialLength));
        return mA_mbox_conf_flat_layer.get();
    } else if (!strcmp(layerName, "B_mbox_loc_flat")) {
        assert(mB_mbox_loc_flat_layer.get() == nullptr);
        mB_mbox_loc_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData, serialLength));
        return mB_mbox_loc_flat_layer.get();
    } else if (!strcmp(layerName, "B_mbox_conf_flat")) {
        assert(mB_mbox_conf_flat_layer.get() == nullptr);
        mB_mbox_conf_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData, serialLength));
        return mB_mbox_conf_flat_layer.get();
    } else if (!strcmp(layerName, "C_mbox_loc_flat")) {
        assert(mC_mbox_loc_flat_layer.get() == nullptr);
        mC_mbox_loc_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData, serialLength));
        return mC_mbox_loc_flat_layer.get();
    } else if (!strcmp(layerName, "C_mbox_conf_flat")) {
        assert(mC_mbox_conf_flat_layer.get() == nullptr);
        mC_mbox_conf_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData, serialLength));
        return mC_mbox_conf_flat_layer.get();
    } else if (!strcmp(layerName, "D_mbox_loc_flat")) {
        assert(mD_mbox_loc_flat_layer.get() == nullptr);
        mD_mbox_loc_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData, serialLength));
        return mD_mbox_loc_flat_layer.get();
    } else if (!strcmp(layerName, "D_mbox_conf_flat")) {
        assert(mD_mbox_conf_flat_layer.get() == nullptr);
        mD_mbox_conf_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData, serialLength));
        return mD_mbox_conf_flat_layer.get();
    } else if (!strcmp(layerName, "E_mbox_loc_flat")) {
        assert(mE_mbox_loc_flat_layer.get() == nullptr);
        mE_mbox_loc_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData, serialLength));
        return mE_mbox_loc_flat_layer.get();
    } else if (!strcmp(layerName, "E_mbox_conf_flat")) {
        assert(mE_mbox_conf_flat_layer.get() == nullptr);
        mE_mbox_conf_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData, serialLength));
        return mE_mbox_conf_flat_layer.get();
    } else if (!strcmp(layerName, "mbox_conf_flatten_ours")) {
        assert(mmbox_conf_flatten_ours_layer.get() == nullptr);
        mmbox_conf_flatten_ours_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData, serialLength));
        return mmbox_conf_flatten_ours_layer.get();
    }
    // reshape layer
    else if (!strcmp(layerName, "mbox_conf_reshape_ours")) {
        assert(mmbox_conf_reshape_ours_layer == nullptr);
        mmbox_conf_reshape_ours_layer = std::unique_ptr<Reshape<2>>(new Reshape<2>(serialData, serialLength));
        return mmbox_conf_reshape_ours_layer.get();
    }
    // softmax layer
    else if (!strcmp(layerName, "mbox_conf_softmax_ours")) {
        assert(mmbox_conf_softmax_ours_layer == nullptr);
        mmbox_conf_softmax_ours_layer = std::unique_ptr<SoftmaxPlugin>(new SoftmaxPlugin(serialData, serialLength));
        return mmbox_conf_softmax_ours_layer.get();
    }
    // detection_out layer
    else if (!strcmp(layerName, "detection_out_ours")) {
        assert(mdetection_out_ours_layer.get() == nullptr);
        mdetection_out_ours_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(
                createSSDDetectionOutputPlugin(serialData, serialLength), nvPluginDeleter);
        return mdetection_out_ours_layer.get();
    }
    // others
    else {
        assert(0);
        return nullptr;
    }
}

bool PluginFactory::isPlugin(const char *name) {
    return (
               !strcmp(name, "conv1/dw")                // depthwise convolution layer
            || !strcmp(name, "conv2/dw")
            || !strcmp(name, "conv3/dw")
            || !strcmp(name, "conv4/dw")
            || !strcmp(name, "conv5/dw")
            || !strcmp(name, "conv6/dw")
            || !strcmp(name, "conv7/dw")
            || !strcmp(name, "conv8/dw")
            || !strcmp(name, "conv9/dw")
            || !strcmp(name, "conv10/dw")
            || !strcmp(name, "conv11/dw")
            || !strcmp(name, "A_mbox_loc_perm")         // permute layer
            || !strcmp(name, "A_mbox_conf_perm")
            || !strcmp(name, "B_mbox_loc_perm")
            || !strcmp(name, "B_mbox_conf_perm")
            || !strcmp(name, "C_mbox_loc_perm")
            || !strcmp(name, "C_mbox_conf_perm")
            || !strcmp(name, "D_mbox_loc_perm")
            || !strcmp(name, "D_mbox_conf_perm")
            || !strcmp(name, "E_mbox_loc_perm")
            || !strcmp(name, "E_mbox_conf_perm")
            || !strcmp(name, "A_mbox_priorbox")         // priorbox layer
            || !strcmp(name, "B_mbox_priorbox")
            || !strcmp(name, "C_mbox_priorbox")
            || !strcmp(name, "D_mbox_priorbox")
            || !strcmp(name, "E_mbox_priorbox")
            || !strcmp(name, "concat6")                 //  concatenation layer
            || !strcmp(name, "concat7")
            || !strcmp(name, "concat8")
            || !strcmp(name, "mbox_loc_ours")           // concat layer
            || !strcmp(name, "mbox_conf_ours")
            || !strcmp(name, "mbox_priorbox_ours")
            || !strcmp(name, "mbox_conf_reshape_ours")  // reshape layer
            || !strcmp(name, "A_mbox_loc_flat")         // flatten layer
            || !strcmp(name, "A_mbox_conf_flat")
            || !strcmp(name, "B_mbox_loc_flat")
            || !strcmp(name, "B_mbox_conf_flat")
            || !strcmp(name, "C_mbox_loc_flat")
            || !strcmp(name, "C_mbox_conf_flat")
            || !strcmp(name, "D_mbox_loc_flat")
            || !strcmp(name, "D_mbox_conf_flat")
            || !strcmp(name, "E_mbox_loc_flat")
            || !strcmp(name, "E_mbox_conf_flat")
            || !strcmp(name, "mbox_conf_flatten_ours")
            || !strcmp(name, "mbox_conf_softmax_ours")  // softmax layer
            || !strcmp(name, "detection_out_ours"));    // detection_out layer
}

void PluginFactory::destroyPlugin() {

    // depthwise convolution layer
    mConv1_dw_layer.release();
    mConv1_dw_layer = nullptr;
    mConv2_dw_layer.release();
    mConv2_dw_layer = nullptr;
    mConv3_dw_layer.release();
    mConv3_dw_layer = nullptr;
    mConv4_dw_layer.release();
    mConv4_dw_layer = nullptr;
    mConv5_dw_layer.release();
    mConv5_dw_layer = nullptr;
    mConv6_dw_layer.release();
    mConv6_dw_layer = nullptr;
    mConv7_dw_layer.release();
    mConv7_dw_layer = nullptr;
    mConv8_dw_layer.release();
    mConv8_dw_layer = nullptr;
    mConv9_dw_layer.release();
    mConv9_dw_layer = nullptr;
    mConv10_dw_layer.release();
    mConv10_dw_layer = nullptr;
    mConv11_dw_layer.release();
    mConv11_dw_layer = nullptr;
    // permute layer
    mA_mbox_loc_perm_layer.release();
    mA_mbox_loc_perm_layer = nullptr;
    mA_mbox_conf_perm_layer.release();
    mA_mbox_conf_perm_layer = nullptr;
    mB_mbox_loc_perm_layer.release();
    mB_mbox_loc_perm_layer = nullptr;
    mB_mbox_conf_perm_layer.release();
    mB_mbox_conf_perm_layer = nullptr;
    mC_mbox_loc_perm_layer.release();
    mC_mbox_loc_perm_layer = nullptr;
    mC_mbox_conf_perm_layer.release();
    mC_mbox_conf_perm_layer = nullptr;
    mD_mbox_loc_perm_layer.release();
    mD_mbox_loc_perm_layer = nullptr;
    mD_mbox_conf_perm_layer.release();
    mD_mbox_conf_perm_layer = nullptr;
    mE_mbox_loc_perm_layer.release();
    mE_mbox_loc_perm_layer = nullptr;
    mE_mbox_conf_perm_layer.release();
    mE_mbox_conf_perm_layer = nullptr;
    // priorbox layer
    mA_mbox_priorbox_layer.release();
    mA_mbox_priorbox_layer = nullptr;
    mB_mbox_priorbox_layer.release();
    mB_mbox_priorbox_layer = nullptr;
    mC_mbox_priorbox_layer.release();
    mC_mbox_priorbox_layer = nullptr;
    mD_mbox_priorbox_layer.release();
    mD_mbox_priorbox_layer = nullptr;
    mE_mbox_priorbox_layer.release();
    mE_mbox_priorbox_layer = nullptr;
    // concat layer
    mconcat6_layer.release();
    mconcat6_layer = nullptr;
    mconcat7_layer.release();
    mconcat7_layer = nullptr;
    mconcat8_layer.release();
    mconcat8_layer = nullptr;
    mmbox_loc_ours_layer.release();
    mmbox_loc_ours_layer = nullptr;
    mmbox_conf_ours_layer.release();
    mmbox_conf_ours_layer = nullptr;
    mmbox_priorbox_ours_layer.release();
    mmbox_priorbox_ours_layer = nullptr;
    // reshape layer
    mmbox_conf_reshape_ours_layer.release();
    mmbox_conf_reshape_ours_layer = nullptr;
    // flatten layer
    mA_mbox_loc_flat_layer.release();
    mA_mbox_loc_flat_layer = nullptr;
    mA_mbox_conf_flat_layer.release();
    mA_mbox_conf_flat_layer = nullptr;
    mB_mbox_loc_flat_layer.release();
    mB_mbox_loc_flat_layer = nullptr;
    mB_mbox_conf_flat_layer.release();
    mB_mbox_conf_flat_layer = nullptr;
    mC_mbox_loc_flat_layer.release();
    mC_mbox_loc_flat_layer = nullptr;
    mC_mbox_conf_flat_layer.release();
    mC_mbox_conf_flat_layer = nullptr;
    mD_mbox_loc_flat_layer.release();
    mD_mbox_loc_flat_layer = nullptr;
    mD_mbox_conf_flat_layer.release();
    mD_mbox_conf_flat_layer = nullptr;
    mE_mbox_loc_flat_layer.release();
    mE_mbox_loc_flat_layer = nullptr;
    mE_mbox_conf_flat_layer.release();
    mE_mbox_conf_flat_layer = nullptr;
    mmbox_conf_flatten_ours_layer.release();
    mmbox_conf_flatten_ours_layer = nullptr;
    // softmax layer
    mmbox_conf_softmax_ours_layer.release();
    mmbox_conf_softmax_ours_layer = nullptr;
    // detection output layer
    mdetection_out_ours_layer.release();
    mdetection_out_ours_layer = nullptr;
}
