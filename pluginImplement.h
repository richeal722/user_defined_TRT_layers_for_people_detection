#ifndef __PLUGIN_LAYER_H__
#define __PLUGIN_LAYER_H__

#include <cassert>
#include <iostream>
#include <cudnn.h>
#include <cstring>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "mathFunctions.h"

/*
#define CHECK(status)                                                                                           \
    {                                                                                                                           \
        if (status != 0)                                                                                                \
        {                                                                                                                               \
            std::cout << "Cuda failure: " << cudaGetErrorString(status) \
                      << " at line " << __LINE__                                                        \
                      << std::endl;                                                                     \
            abort();                                                                                                    \
        }                                                                                                                          \
    }
*/

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;


/**********************************************************************************/
// Depthwise Plugin Layer
/**********************************************************************************/
struct DepthwiseParam
{
    int pad;
    int dilation;
    int kernel_size;
    int stride;
    bool bias_term;
};

class DepthwisePlugin : public IPlugin
{
public:
    DepthwisePlugin(const Weights *weights, int nbWeights, int nbOutputChannels, DepthwiseParam dwp);
    DepthwisePlugin(DepthwiseParam dwp, const void* buffer, size_t size);
    ~DepthwisePlugin();

    inline int getNbOutputs() const override {return 1;};
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override ;
    int initialize() override;
    inline void terminate() override;

    inline size_t getWorkspaceSize(int) const override { return 0; };
    int enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream) override;

    size_t getSerializationSize() override;
    void serialize(void* buffer) override;

    void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int) override;

private:
    template<typename T> void write(char*& buffer, const T& val)
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template<typename T> T read(const char*& buffer)
    {
        T val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
        return val;
    }

    Weights copyToDevice(const void* hostData, size_t count)
    {
        void* deviceData;
        //printf("%ld\n",count);
        CUDA_CHECK(cudaMalloc(&deviceData, count * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(deviceData, hostData, count * sizeof(float), cudaMemcpyHostToDevice));
        return Weights{DataType::kFLOAT, deviceData, int64_t(count)};
    }

    void serializeFromDevice(char*& hostBuffer, Weights deviceWeights)
    {
        cudaMemcpy(hostBuffer, deviceWeights.values, deviceWeights.count * sizeof(float), cudaMemcpyDeviceToHost);
        hostBuffer += deviceWeights.count * sizeof(float);
    }

    Weights deserializeToDevice(const char*& hostBuffer, size_t count)
    {
        Weights w =  copyToDevice(hostBuffer, count);
        hostBuffer += count * sizeof(float);
        return w;
    }

    DimsCHW dimsBottomData;
    DepthwiseParam depthwise_param;
    int dim_h_w;
    int input_c, input_h, input_w;
    int input_count;
    Weights mKernelWeights;
    Weights mBiasWeights;
    int mNbInputChannels, mNbOutputChannels;
};



/**********************************************************************************/
// Reshape Plugin Layer
/**********************************************************************************/
//SSD Reshape layer : shape{0,-1,2}
template<int OutC>
class Reshape : public IPlugin {
public:
    Reshape() {}

    Reshape(const void *buffer, size_t size) {
        assert(size == sizeof(mCopySize));
        mCopySize = *reinterpret_cast<const size_t *>(buffer);
    }

    int getNbOutputs() const override {
        return 1;
    }

    Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) override {
        assert(nbInputDims == 1);
        assert(index == 0);
        assert(inputs[index].nbDims == 3);
        assert((inputs[0].d[0]) * (inputs[0].d[1]) % OutC == 0);
        //return DimsCHW(OutC, inputs[0].d[0] * inputs[0].d[1] / OutC, inputs[0].d[2]);//faster rcnn : shape{2,-1,0}
        return DimsCHW(1, inputs[0].d[0] * inputs[0].d[2] / OutC, OutC);    // shape{0,-1,2}  1*2252*2
    }

    int initialize() override {
        return 0;
    }

    void terminate() override {

    }

    size_t getWorkspaceSize(int) const override {
        return 0;
    }

    // currently it is not possible for a plugin to execute "in place". Therefore we memcpy the data from the input to the output buffer
    int enqueue(int batchSize, const void *const *inputs, void **outputs, void *, cudaStream_t stream) override {
        CUDA_CHECK(cudaMemcpyAsync(outputs[0], inputs[0], mCopySize * batchSize, cudaMemcpyDeviceToDevice, stream));
        return 0;
    }

    size_t getSerializationSize() override {
        return sizeof(mCopySize);
    }

    void serialize(void *buffer) override {
        *reinterpret_cast<size_t *>(buffer) = mCopySize;
    }

    void configure(const Dims *inputs, int nbInputs, const Dims *outputs, int nbOutputs, int) override {
        mCopySize = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2] * sizeof(float);
    }

protected:
    size_t mCopySize;
};


/**********************************************************************************/
// Flatten Plugin Layer
/**********************************************************************************/
class FlattenLayer : public IPlugin {
public:
    FlattenLayer() {}

    FlattenLayer(const void *buffer, size_t size) {
        assert(size == 3 * sizeof(int));
        const int *d = reinterpret_cast<const int *>(buffer);
        _size = d[0] * d[1] * d[2];
        dimBottom = DimsCHW{d[0], d[1], d[2]};
    }

    inline int getNbOutputs() const override { return 1; };

    Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) override {
        assert(1 == nbInputDims);
        assert(0 == index);
        assert(3 == inputs[index].nbDims);
        _size = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2];
        return DimsCHW(_size, 1, 1);
    }

    int initialize() override {
        return 0;
    }

    inline void terminate() override {

    }

    inline size_t getWorkspaceSize(int) const override { return 0; }

    int enqueue(int batchSize, const void *const *inputs, void **outputs, void *, cudaStream_t stream) override {
        CUDA_CHECK(cudaMemcpyAsync(outputs[0], inputs[0], batchSize * _size * sizeof(float), cudaMemcpyDeviceToDevice,
                              stream));
        return 0;
    }

    size_t getSerializationSize() override {
        return 3 * sizeof(int);
    }

    void serialize(void *buffer) override {
        int *d = reinterpret_cast<int *>(buffer);
        d[0] = dimBottom.c();
        d[1] = dimBottom.h();
        d[2] = dimBottom.w();
    }

    void configure(const Dims *inputs, int nbInputs, const Dims *outputs, int nbOutputs, int) override {
        dimBottom = DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
    }

protected:
    DimsCHW dimBottom;
    int _size;
};


/**********************************************************************************/
// Softmax Plugin Layer
/**********************************************************************************/
class SoftmaxPlugin : public IPlugin {
public:
    SoftmaxPlugin() {};

    SoftmaxPlugin(const void *buffer, size_t size);

    inline int getNbOutputs() const override { return 1; };

    Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) override;

    int initialize() override;

    inline void terminate() override;

    inline size_t getWorkspaceSize(int) const override { return 0; };

    int enqueue(int batchSize, const void *const *inputs, void **outputs, void *, cudaStream_t stream) override;

    size_t getSerializationSize() override;

    void serialize(void *buffer) override;

    void configure(const Dims *inputs, int nbInputs, const Dims *outputs, int nbOutputs, int) override;

protected:
    DimsCHW dimsBottomData;
    int mInputC, mInputH, mInputW;
    float *scale_data;
    int count, outer_num_, inner_num_, channels;
};


/**********************************************************************************/
// Concat Plugin Layer
/**********************************************************************************/
class ConcatPlugin : public IPlugin {
public:
    ConcatPlugin(int axis) { _axis = axis; };

    ConcatPlugin(int axis, const void *buffer, size_t size);

    inline int getNbOutputs() const override { return 1; };

    Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) override;

    int initialize() override;

    inline void terminate() override;

    inline size_t getWorkspaceSize(int) const override { return 0; };

    int enqueue(int batchSize, const void *const *inputs, void **outputs, void *, cudaStream_t stream) override;

    size_t getSerializationSize() override;

    void serialize(void *buffer) override;

    void configure(const Dims *inputs, int nbInputs, const Dims *outputs, int nbOutputs, int) override;

protected:
    DimsCHW dimsFc, dimsConv6, dimsConv7, dimsConv8, dimsConv9;
    int inputs_size;
    int top_concat_axis;//top 层 concat后的维度
    int *bottom_concat_axis = new int[9];//记录每个bottom层concat维度的shape
    int *concat_input_size_ = new int[9];
    int *num_concats_ = new int[9];
    int _axis;
};


/**********************************************************************************/
// Concatenation Plugin Layer for Concat6, Concat7, Concat8
/**********************************************************************************/
class ConcatenationPlugin : public IPlugin {
public:
    ConcatenationPlugin(int axis) { _axis_two = axis; };

    ConcatenationPlugin(int axis, const void *buffer, size_t size);

    inline int getNbOutputs() const override { return 1; };

    Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) override;

    int initialize() override;

    inline void terminate() override;

    inline size_t getWorkspaceSize(int) const override { return 0; };

    int enqueue(int batchSize, const void *const *inputs, void **outputs, void *, cudaStream_t stream) override;

    size_t getSerializationSize() override;

    void serialize(void *buffer) override;

    void configure(const Dims *inputs, int nbInputs, const Dims *outputs, int nbOutputs, int) override;

protected:
    DimsCHW dimsA, dimsB;
    int inputs_size_two;
    int top_concat_axis_two;    //top 层 concat后的维度
    int *bottom_concat_axis_two = new int[9];//记录每个bottom层concat维度的shape
    int *concat_input_size_two = new int[9];
    int *num_concats_two = new int[9];
    int _axis_two;
};


/**********************************************************************************/
// PluginFactory
/**********************************************************************************/
class PluginFactory : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactory {
public:
    virtual nvinfer1::IPlugin *
    createPlugin(const char *layerName, const nvinfer1::Weights *weights, int nbWeights) override;

    IPlugin *createPlugin(const char *layerName, const void *serialData, size_t serialLength) override;

    void (*nvPluginDeleter)(INvPlugin *) { [](INvPlugin *ptr) { ptr->destroy(); }};

    bool isPlugin(const char *name) override;

    void destroyPlugin();


    // depthwise convolution layer
    std::unique_ptr<DepthwisePlugin> mConv1_dw_layer{ nullptr};
    std::unique_ptr<DepthwisePlugin> mConv2_dw_layer{ nullptr};
    std::unique_ptr<DepthwisePlugin> mConv3_dw_layer{ nullptr};
    std::unique_ptr<DepthwisePlugin> mConv4_dw_layer{ nullptr};
    std::unique_ptr<DepthwisePlugin> mConv5_dw_layer{ nullptr};
    std::unique_ptr<DepthwisePlugin> mConv6_dw_layer{ nullptr};
    std::unique_ptr<DepthwisePlugin> mConv7_dw_layer{ nullptr};
    std::unique_ptr<DepthwisePlugin> mConv8_dw_layer{ nullptr};
    std::unique_ptr<DepthwisePlugin> mConv9_dw_layer{ nullptr};
    std::unique_ptr<DepthwisePlugin> mConv10_dw_layer{ nullptr};
    std::unique_ptr<DepthwisePlugin> mConv11_dw_layer{ nullptr};
    // permute layer
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mA_mbox_loc_perm_layer{nullptr, nvPluginDeleter};
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mA_mbox_conf_perm_layer{nullptr, nvPluginDeleter};
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mB_mbox_loc_perm_layer{nullptr, nvPluginDeleter};
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mB_mbox_conf_perm_layer{nullptr, nvPluginDeleter};
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mC_mbox_loc_perm_layer{nullptr, nvPluginDeleter};
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mC_mbox_conf_perm_layer{nullptr, nvPluginDeleter};
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mD_mbox_loc_perm_layer{nullptr, nvPluginDeleter};
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mD_mbox_conf_perm_layer{nullptr, nvPluginDeleter};
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mE_mbox_loc_perm_layer{nullptr, nvPluginDeleter};
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mE_mbox_conf_perm_layer{nullptr, nvPluginDeleter};
    // priorbox layer
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mA_mbox_priorbox_layer{nullptr, nvPluginDeleter};
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mB_mbox_priorbox_layer{nullptr, nvPluginDeleter};
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mC_mbox_priorbox_layer{nullptr, nvPluginDeleter};
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mD_mbox_priorbox_layer{nullptr, nvPluginDeleter};
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mE_mbox_priorbox_layer{nullptr, nvPluginDeleter};
   // concat layer
    std::unique_ptr<ConcatenationPlugin> mconcat6_layer{nullptr};       // ConcatenationPlugin
    std::unique_ptr<ConcatenationPlugin> mconcat7_layer{nullptr};
    std::unique_ptr<ConcatenationPlugin> mconcat8_layer{nullptr};
    std::unique_ptr<ConcatPlugin> mmbox_loc_ours_layer{nullptr};        // ConcatPlugin
    std::unique_ptr<ConcatPlugin> mmbox_conf_ours_layer{nullptr};
    std::unique_ptr<ConcatPlugin> mmbox_priorbox_ours_layer{nullptr};
    // reshape layer
    std::unique_ptr<Reshape<2>> mmbox_conf_reshape_ours_layer{nullptr};
    // flatten layer
    std::unique_ptr<FlattenLayer> mA_mbox_loc_flat_layer{nullptr};
    std::unique_ptr<FlattenLayer> mA_mbox_conf_flat_layer{nullptr};
    std::unique_ptr<FlattenLayer> mB_mbox_loc_flat_layer{nullptr};
    std::unique_ptr<FlattenLayer> mB_mbox_conf_flat_layer{nullptr};
    std::unique_ptr<FlattenLayer> mC_mbox_loc_flat_layer{nullptr};
    std::unique_ptr<FlattenLayer> mC_mbox_conf_flat_layer{nullptr};
    std::unique_ptr<FlattenLayer> mD_mbox_loc_flat_layer{nullptr};
    std::unique_ptr<FlattenLayer> mD_mbox_conf_flat_layer{nullptr};
    std::unique_ptr<FlattenLayer> mE_mbox_loc_flat_layer{nullptr};
    std::unique_ptr<FlattenLayer> mE_mbox_conf_flat_layer{nullptr};
    std::unique_ptr<FlattenLayer> mmbox_conf_flatten_ours_layer{nullptr};
    // softmax layer
    std::unique_ptr<SoftmaxPlugin> mmbox_conf_softmax_ours_layer{nullptr};
    // detection output layer
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mdetection_out_ours_layer{nullptr, nvPluginDeleter};
};

#endif
