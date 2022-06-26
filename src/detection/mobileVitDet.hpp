/*
 * @Author: error: git config user.name && git config user.email & please set dead value or install git
 * @Date: 2022-06-19 23:34:31
 * @LastEditors: error: git config user.name && git config user.email & please set dead value or install git
 * @LastEditTime: 2022-06-25 19:45:37
 * @FilePath: /root/trt2022_src/mobilenet/ml-cvnets/Detection/mobileVitDet.hpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#ifndef SIMPLE_YOLO_HPP
#define SIMPLE_YOLO_HPP

/*
  简单的yolo接口，容易集成但是高性能
*/

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>

namespace MobileVitDet
{

    using namespace std;

    enum class Mode : int 
    {
        FP32,
        FP16
    };

    struct Box
    {
        float left, top, right, bottom, confidence;
        int class_label;

        Box() = default;

        Box(float left, float top, float right, float bottom, float confidence, int class_label)
        :left(left), top(top), right(right), bottom(bottom), confidence(confidence), class_label(class_label){}
    };

    typedef std::vector<Box> BoxArray;

    class Infer
    {
    public:
        virtual void commit(const cv::Mat& image, BoxArray& box) = 0;
        virtual void commits(const vector<cv::Mat>& images, vector<BoxArray>& box) = 0;
        virtual void commits(const vector<std::string>& image_filenames, vector<BoxArray>& box) = 0;
        // virtual size_t getMaxBatchSize() = 0;
    };

    const char* trt_version();
    const char* mode_string(Mode type);
    void set_device(int device_id);

    /* 
        模型编译
        max batch size：为最大可以允许的batch数量
        source_onnx：仅仅只支持onnx格式输入
        saveto：储存的tensorRT模型，用于后续的加载
        max workspace size：最大工作空间大小，一般给1GB，在嵌入式可以改为256MB，单位是byte
        int8 images folder：对于Mode为INT8时，需要提供图像数据进行标定，请提供文件夹，会自动检索下面的jpg/jpeg/tiff/png/bmp
        int8_entropy_calibrator_cache_file：对于int8模式下，熵文件可以缓存，避免二次加载数据，可以跨平台使用，是一个txt文件
    */
    // 1GB = 1<<30
    bool compile(Mode mode, unsigned int max_batch_size,
                const string& source_onnx,
                const string& saveto,
                size_t max_workspace_size = 1<<30);

    shared_ptr<Infer> create_infer(const string& engine_file, int gpuid, float confidence_threshold=0.25f, float nms_threshold=0.5f);
    
    static void inference(int deviceid, const string& engine_file, MobileVitDet::Mode mode, const string& model_name);

}; // namespace SimpleYolo

#endif