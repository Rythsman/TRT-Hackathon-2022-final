#include "mobileVitDet.hpp"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <fstream>
#include <memory>
#include <string>
#include <future>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <queue>

#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdarg.h>

#include "trt_tensor.hpp"
#include "ilogger.hpp"
#include "json.hpp"


static const char* cocolabels[] = {
	"person", "bicycle", "car", "motorcycle", "airplane",
	"bus", "train", "truck", "boat", "traffic light", "fire hydrant",
	"stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
	"sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
	"umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
	"snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
	"skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
	"cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
	"orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
	"chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
	"laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
	"oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
	"scissors", "teddy bear", "hair drier", "toothbrush"
};

namespace MobileVitDet
{

    
    using namespace nvinfer1;
    using namespace cv;

    #define CURRENT_DEVICE_ID   -1
    #define GPU_BLOCK_THREADS  512
    #define KernelPositionBlock											\
        int position = (blockDim.x * blockIdx.x + threadIdx.x);		    \
        if (position >= (edge)) return;

    #define checkCudaRuntime(call) check_runtime(call, #call, __LINE__, __FILE__)
    static bool check_runtime(cudaError_t e, const char* call, int line, const char *file);

    #define checkCudaKernel(...)                                                                         \
        __VA_ARGS__;                                                                                     \
        do{cudaError_t cudaStatus = cudaPeekAtLastError();                                               \
        if (cudaStatus != cudaSuccess){                                                                  \
            INFOE("launch failed: %s", cudaGetErrorString(cudaStatus));                                  \
        }} while(0);

    #define Assert(op)					 \
        do{                              \
            bool cond = !(!(op));        \
            if(!cond){                   \
                INFOF("Assert failed, " #op);  \
            }                                  \
        }while(false)

    #define CURRENT_LOG_LEVEL       LogLevel::Info

    enum class NormType : int{
        None      = 0,
        MeanStd   = 1,
        AlphaBeta = 2
    };

    enum class ChannelType : int{
        None          = 0,
        SwapRB        = 1
    };

    struct Norm
    {
        float mean[3];
        float std[3];
        float alpha, beta;
        NormType type = NormType::None;
        ChannelType channel_type = ChannelType::None;

        // out = (x * alpha - mean) / std
        static Norm mean_std(const float mean[3], const float std[3], float alpha = 1/255.0f, ChannelType channel_type=ChannelType::None);

        // out = x * alpha + beta
        static Norm alpha_beta(float alpha, float beta = 0, ChannelType channel_type=ChannelType::None);

        // None
        static Norm None();
    };

    Norm Norm::mean_std(const float mean[3], const float std[3], float alpha, ChannelType channel_type){

        Norm out;
        out.type  = NormType::MeanStd;
        out.alpha = alpha;
        out.channel_type = channel_type;
        memcpy(out.mean, mean, sizeof(out.mean));
        memcpy(out.std,  std,  sizeof(out.std));
        return out;
    }

    Norm Norm::alpha_beta(float alpha, float beta, ChannelType channel_type){

        Norm out;
        out.type = NormType::AlphaBeta;
        out.alpha = alpha;
        out.beta = beta;
        out.channel_type = channel_type;
        return out;
    }

    Norm Norm::None(){
        return Norm();
    }
    
    class AutoDevice{
    public:
        AutoDevice(int device_id = 0){
            cudaGetDevice(&old_);
            checkCudaRuntime(cudaSetDevice(device_id));
        }

        virtual ~AutoDevice(){
            checkCudaRuntime(cudaSetDevice(old_));
        }
    
    private:
        int old_ = -1;
    };
    
    enum class LogLevel : int
    {
        Debug   = 5,
        Verbose = 4,
        Info    = 3,
        Warning = 2,
        Error   = 1,
        Fatal   = 0
    };

    static void __log_func(const char* file, int line, LogLevel level, const char* fmt, ...);
    inline int upbound(int n, int align = 32){return (n + align - 1) / align * align;}

    static bool check_runtime(cudaError_t e, const char* call, int line, const char *file){
        if (e != cudaSuccess) {
            INFOE("CUDA Runtime error %s # %s, code = %s [ %d ] in file %s:%d", call, cudaGetErrorString(e), cudaGetErrorName(e), e, file, line);
            return false;
        }
        return true;
    }

    #define TRT_STR(v)  #v
    #define TRT_VERSION_STRING(major, minor, patch, build)   TRT_STR(major) "." TRT_STR(minor) "." TRT_STR(patch) "." TRT_STR(build)
    const char* trt_version(){
        return TRT_VERSION_STRING(NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, NV_TENSORRT_BUILD);
    }

    static bool check_device_id(int device_id){
        int device_count = -1;
        checkCudaRuntime(cudaGetDeviceCount(&device_count));
        if(device_id < 0 || device_id >= device_count){
            INFOE("Invalid device id: %d, count = %d", device_id, device_count);
            return false;
        }
        return true;
    }

    static bool exists(const string& path){
        return access(path.c_str(), R_OK) == 0;
    }

    static const char* level_string(LogLevel level){
        switch (level){
            case LogLevel::Debug: return "debug";
            case LogLevel::Verbose: return "verbo";
            case LogLevel::Info: return "info";
            case LogLevel::Warning: return "warn";
            case LogLevel::Error: return "error";
            case LogLevel::Fatal: return "fatal";
            default: return "unknow";
        }
    }

    template<typename _T>
    static string join_dims(const vector<_T>& dims){
        stringstream output;
        char buf[64];
        const char* fmts[] = {"%d", " x %d"};
        for(int i = 0; i < dims.size(); ++i){
            snprintf(buf, sizeof(buf), fmts[i != 0], dims[i]);
            output << buf;
        }
        return output.str();
    }

    static bool save_file(const string& file, const void* data, size_t length)
    {

        FILE* f = fopen(file.c_str(), "wb");
        if (!f) return false;

        if (data && length > 0){
            if (fwrite(data, 1, length, f) != length){
                fclose(f);
                return false;
            }
        }
        fclose(f);
        return true;
    }

    static bool save_file(const string& file, const vector<uint8_t>& data)
    {
        return save_file(file, data.data(), data.size());
    }

    static string file_name(const string& path, bool include_suffix)
    {

        if (path.empty()) return "";

        int p = path.rfind('/');

        p += 1;

        //include suffix
        if (include_suffix)
            return path.substr(p);

        int u = path.rfind('.');
        if (u == -1)
            return path.substr(p);

        if (u <= p) u = path.size();
        return path.substr(p, u - p);
    }
    
    vector<string> glob_image_files(const string& directory)
    {
        vector<string> files, output;
        set<string> pattern_set{"jpg", "png", "bmp", "jpeg", "tiff"};

        if(directory.empty()){
            INFOE("Glob images from folder failed, folder is empty");
            return output;
        }

        try{
			vector<cv::String> files_;
			files_.reserve(10000);
            cv::glob(directory + "/*", files_, true);
			files.insert(files.end(), files_.begin(), files_.end());
        }catch(...){
            INFOE("Glob %s failed", directory.c_str());
            return output;
        }

        for(int i = 0; i < files.size(); ++i){
            auto& file = files[i];
            int p = file.rfind(".");
            if(p == -1) continue;

            auto suffix = file.substr(p+1);
            std::transform(suffix.begin(), suffix.end(), suffix.begin(), [](char c){
                if(c >= 'A' && c <= 'Z')
                    c -= 'A' + 'a';
                return c;
            });
            if(pattern_set.find(suffix) != pattern_set.end())
                output.push_back(file);
        }
        return output;
    }

    static void __log_func(const char* file, int line, LogLevel level, const char* fmt, ...)
    {
        if(level > CURRENT_LOG_LEVEL)
            return;

        va_list vl;
        va_start(vl, fmt);
        
        char buffer[2048];
        string filename = file_name(file, true);
        int n = snprintf(buffer, sizeof(buffer), "[%s][%s:%d]:", level_string(level), filename.c_str(), line);
        vsnprintf(buffer + n, sizeof(buffer) - n, fmt, vl);

        fprintf(stdout, "%s\n", buffer);
        if (level == LogLevel::Fatal) {
            fflush(stdout);
            abort();
        }
    }

    static dim3 grid_dims(int numJobs) {
        int numBlockThreads = numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
        return dim3(((numJobs + numBlockThreads - 1) / (float)numBlockThreads));
    }

    static dim3 block_dims(int numJobs) {
        return numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
    }

    static int get_device(int device_id){
        if(device_id != CURRENT_DEVICE_ID){
            check_device_id(device_id);
            return device_id;
        }

        checkCudaRuntime(cudaGetDevice(&device_id));
        return device_id;
    }

    void set_device(int device_id) {
        if (device_id == -1)
            return;

        checkCudaRuntime(cudaSetDevice(device_id));
    }

    struct ImageItem{
        string image_file;
        BoxArray detections;
    };

    vector<ImageItem> scan_dataset(const string& images_root){

        vector<ImageItem> output;
        auto image_files = iLogger::find_files(images_root, "*.jpg");

        for(int i = 0; i < image_files.size(); ++i){
            auto& image_file = image_files[i];

            if(!iLogger::exists(image_file)){
                INFOW("Not found: %s", image_file.c_str());
                continue;
            }

            ImageItem item;
            item.image_file = image_file;
            output.emplace_back(item);
        }
        return output;
    }

    bool save_to_json(const vector<ImageItem>& images, const string& file) {

        int to_coco90_class_map[] = {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
            35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
            64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
        };
        Json::Value predictions(Json::arrayValue);
        for(int i = 0; i < images.size(); ++i){
            auto& image = images[i];
            auto file_name = iLogger::file_name(image.image_file, false);
            int image_id = atoi(file_name.c_str());

            auto& boxes = image.detections;
            for(auto& box : boxes){
                Json::Value jitem;
                jitem["image_id"] = image_id;
                jitem["category_id"] = to_coco90_class_map[box.class_label];
                jitem["score"] = box.confidence;

                auto& bbox = jitem["bbox"];
                bbox.append(box.left);
                bbox.append(box.top);
                bbox.append(box.right - box.left);
                bbox.append(box.bottom - box.top);
                predictions.append(jitem);
            }
        }
        return iLogger::save_file(file, predictions.toStyledString());
    }

    /////////////////////////////CUDA kernels////////////////////////////////////////////////

    const int NUM_BOX_ELEMENT = 7;      // left, top, right, bottom, confidence, class, keepflag
    static __device__ void affine_project(float* matrix, float x, float y, float* ox, float* oy){
        *ox = matrix[0] * x + matrix[1] * y + matrix[2];
        *oy = matrix[3] * x + matrix[4] * y + matrix[5];
    }


    static __global__ void decode_kernel(float* predict, int img_height, int img_width, int num_bboxes, int num_classes, float confidence_threshold, float* invert_affine_matrix, float* parray, int max_objects){  

        int position = blockDim.x * blockIdx.x + threadIdx.x;
        if (position >= num_bboxes) return;
        if((int)*parray >= max_objects) return;
        float* pitem     = predict + (5 + num_classes) * position;
        float objectness = 1 - pitem[4];
        if(objectness < confidence_threshold)
            return;

        float* class_confidence = pitem + 5;
        float confidence        = 0.0f;
        int label               = 0;

        float cx         = *pitem++ * img_width;
        float cy         = *pitem++ * img_height;
        float width      = *pitem++ * img_width;
        float height     = *pitem++ * img_height;
        float left   = cx - width * 0.5f;
        float top    = cy - height * 0.5f;
        float right  = cx + width * 0.5f;
        float bottom = cy + height * 0.5f;
        if(nullptr != invert_affine_matrix) {
            affine_project(invert_affine_matrix, left,  top,    &left,  &top);
            affine_project(invert_affine_matrix, right, bottom, &right, &bottom);
        }

        for(int i = 0; i < num_classes; ++i, ++class_confidence){
            if(*class_confidence > confidence_threshold) {
                confidence = *class_confidence;  
                label = i;

                int index = atomicAdd(parray, 1);
                if(index >= max_objects)
                    return;

                float* pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
                *pout_item++ = left;
                *pout_item++ = top;
                *pout_item++ = right;
                *pout_item++ = bottom;
                *pout_item++ = confidence;
                *pout_item++ = label;
                *pout_item++ = 1;
            }
        }
    }

    static __device__ float box_iou(
        float aleft, float atop, float aright, float abottom, 
        float bleft, float btop, float bright, float bbottom
    ){

        float cleft 	= max(aleft, bleft);
        float ctop 		= max(atop, btop);
        float cright 	= min(aright, bright);
        float cbottom 	= min(abottom, bbottom);
        
        float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
        if(c_area == 0.0f)
            return 0.0f;
        
        float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
        float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
        return c_area / (a_area + b_area - c_area);
    }
    
    static __global__ void fast_nms_kernel(float* bboxes, int max_objects, float threshold)
    {
        int position = (blockDim.x * blockIdx.x + threadIdx.x);
        int count = min((int)*bboxes, max_objects);
        if (position >= count) 
            return;
        
        // left, top, right, bottom, confidence, class, keepflag
        float* pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;
        for(int i = 0; i < count; ++i){
            float* pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
            if(i == position || pcurrent[5] != pitem[5]) continue;

            if(pitem[4] >= pcurrent[4]){
                if(pitem[4] == pcurrent[4] && i < position)
                    continue;

                float iou = box_iou(
                    pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                    pitem[0],    pitem[1],    pitem[2],    pitem[3]
                );

                if(iou > threshold){
                    pcurrent[6] = 0;
                    return;
                }
            }
        }
    } 

    static void decode_kernel_invoker(float* predict, int img_height, int img_width, int num_bboxes, int num_classes, float confidence_threshold, float nms_threshold, float* invert_affine_matrix, float* parray, int max_objects, cudaStream_t stream){
        
        auto grid = grid_dims(num_bboxes);
        auto block = block_dims(num_bboxes);
        checkCudaKernel(decode_kernel<<<grid, block, 0, stream>>>(predict, img_height, img_width, num_bboxes, num_classes, confidence_threshold, invert_affine_matrix, parray, max_objects));

        grid = grid_dims(max_objects);
        block = block_dims(max_objects);
        checkCudaKernel(fast_nms_kernel<<<grid, block, 0, stream>>>(parray, max_objects, nms_threshold));
    }

    static __global__ void resize_bilinear_and_normalize_kernel(
		uint8_t* src, int src_line_size, int src_width, int src_height, float* dst, int dst_width, int dst_height, 
		uint8_t const_value_st, float sx, float sy, Norm norm, int edge
	){
		int position = blockDim.x * blockIdx.x + threadIdx.x;
		if (position >= edge) return;

		int dx      = position % dst_width;
		int dy      = position / dst_width;
		float src_x = (dx + 0.5) * sx - 0.5f;
		float src_y = (dy + 0.5) * sy - 0.5f;
		float c0, c1, c2;

		if(src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height){
			// out of range
			c0 = const_value_st;
			c1 = const_value_st;
			c2 = const_value_st;
		}else{
			int y_low = floorf(src_y);
			int x_low = floorf(src_x);
			int y_high = y_low + 1;
			int x_high = x_low + 1;

			uint8_t const_value[] = {const_value_st, const_value_st, const_value_st};
			float ly    = src_y - y_low;
			float lx    = src_x - x_low;
			float hy    = 1 - ly;
			float hx    = 1 - lx;
			float w1    = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
			float* pdst = dst + dy * dst_width + dx * 3;
			uint8_t* v1 = const_value;
			uint8_t* v2 = const_value;
			uint8_t* v3 = const_value;
			uint8_t* v4 = const_value;
			if(y_low >= 0){
				if (x_low >= 0)
					v1 = src + y_low * src_line_size + x_low * 3;

				if (x_high < src_width)
					v2 = src + y_low * src_line_size + x_high * 3;
			}
			
			if(y_high < src_height){
				if (x_low >= 0)
					v3 = src + y_high * src_line_size + x_low * 3;

				if (x_high < src_width)
					v4 = src + y_high * src_line_size + x_high * 3;
			}

			c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0];
			c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1];
			c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2];
		}

		if(norm.channel_type == ChannelType::SwapRB){
			float t = c2;
			c2 = c0;  c0 = t;
		}

		if(norm.type == NormType::MeanStd){
			c0 = (c0 * norm.alpha - norm.mean[0]) / norm.std[0];
			c1 = (c1 * norm.alpha - norm.mean[1]) / norm.std[1];
			c2 = (c2 * norm.alpha - norm.mean[2]) / norm.std[2];
		}else if(norm.type == NormType::AlphaBeta){
			c0 = c0 * norm.alpha + norm.beta;
			c1 = c1 * norm.alpha + norm.beta;
			c2 = c2 * norm.alpha + norm.beta;
		}

		int area = dst_width * dst_height;
		float* pdst_c0 = dst + dy * dst_width + dx;
		float* pdst_c1 = pdst_c0 + area;
		float* pdst_c2 = pdst_c1 + area;
		*pdst_c0 = c0;
		*pdst_c1 = c1;
		*pdst_c2 = c2;
	}



    void resize_bilinear_and_normalize(
		uint8_t* src, int src_line_size, int src_width, int src_height, float* dst, int dst_width, int dst_height,
		uint8_t const_value, const Norm& norm,
		cudaStream_t stream) {
		
		int jobs   = dst_width * dst_height;
		auto grid  = grid_dims(jobs);
		auto block = block_dims(jobs);
		
		checkCudaKernel(resize_bilinear_and_normalize_kernel << <grid, block, 0, stream >> > (
			src, src_line_size,
			src_width, src_height, dst,
			dst_width, dst_height, const_value, src_width/(float)dst_width, src_height/(float)dst_height, norm, jobs
		));
	}


    /////////////////////////////////class TRTInferImpl////////////////////////////////////////////////
    class Logger : public ILogger 
    {
    public:
        virtual void log(Severity severity, const char* msg) noexcept override {

            if (severity == Severity::kINTERNAL_ERROR) {
                INFOE("NVInfer INTERNAL_ERROR: %s", msg);
                abort();
            }else if (severity == Severity::kERROR) {
                INFOE("NVInfer: %s", msg);
            }
            else  if (severity == Severity::kWARNING) {
                INFOW("NVInfer: %s", msg);
            }
            else  if (severity == Severity::kINFO) {
                INFOD("NVInfer: %s", msg);
            }
            else {
                INFOD("%s", msg);
            }
        }
    };
    static Logger gLogger;

    template<typename _T>
    static void destroy_nvidia_pointer(_T* ptr) 
    {
        if (ptr) ptr->destroy();
    }

    class EngineContext 
    {
    public:
        virtual ~EngineContext() { destroy(); }

        void set_stream(cudaStream_t stream)
        {
            if(owner_stream_){
                if (stream_) {cudaStreamDestroy(stream_);}
                owner_stream_ = false;
            }
            stream_ = stream;
        }

        bool build_model(const void* pdata, size_t size) 
        {
            destroy();

            if(pdata == nullptr || size == 0)
                return false;

            owner_stream_ = true;
            checkCudaRuntime(cudaStreamCreate(&stream_));
            if(stream_ == nullptr)
                return false;

            runtime_ = shared_ptr<IRuntime>(createInferRuntime(gLogger), destroy_nvidia_pointer<IRuntime>);
            if (runtime_ == nullptr)
                return false;

            engine_ = shared_ptr<ICudaEngine>(runtime_->deserializeCudaEngine(pdata, size, nullptr), destroy_nvidia_pointer<ICudaEngine>);
            if (engine_ == nullptr)
                return false;

            //runtime_->setDLACore(0);
            context_ = shared_ptr<IExecutionContext>(engine_->createExecutionContext(), destroy_nvidia_pointer<IExecutionContext>);
            return context_ != nullptr;
        }

    private:
        void destroy() {
            context_.reset();
            engine_.reset();
            runtime_.reset();

            if(owner_stream_){
                if (stream_) {cudaStreamDestroy(stream_);}
            }
            stream_ = nullptr;
        }

    public:
        cudaStream_t stream_ = nullptr;
        bool owner_stream_ = false;
        shared_ptr<IExecutionContext> context_;
        shared_ptr<ICudaEngine> engine_;
        shared_ptr<IRuntime> runtime_ = nullptr;
    };

    class TRTInferImpl
    {
    public:
        virtual ~TRTInferImpl();
        bool load(const std::string& file);
        bool load_from_memory(const void* pdata, size_t size);
        void destroy();
        void forward(bool sync);
        int get_max_batch_size();
        cudaStream_t get_stream();
        void set_stream(cudaStream_t stream);
        void synchronize();
        size_t get_device_memory_size();
        std::shared_ptr<MixMemory> get_workspace();
        std::shared_ptr<Tensor> input(int index = 0);
        std::string get_input_name(int index = 0);
        std::shared_ptr<Tensor> output(int index = 0);
        std::string get_output_name(int index = 0);
        std::shared_ptr<Tensor> tensor(const std::string& name);
        bool is_output_name(const std::string& name);
        bool is_input_name(const std::string& name);
        void set_input (int index, std::shared_ptr<Tensor> tensor);
        void set_output(int index, std::shared_ptr<Tensor> tensor);
        std::shared_ptr<std::vector<uint8_t>> serial_engine();

        void print();

        int num_output();
        int num_input();
        int device();

    private:
        void build_engine_input_and_outputs_mapper();

    private:
        std::vector<std::shared_ptr<Tensor>> inputs_;
        std::vector<std::shared_ptr<Tensor>> outputs_;
        std::vector<int> inputs_map_to_ordered_index_;
        std::vector<int> outputs_map_to_ordered_index_;
        std::vector<std::string> inputs_name_;
        std::vector<std::string> outputs_name_;
        std::vector<std::shared_ptr<Tensor>> orderdBlobs_;
        std::map<std::string, int> blobsNameMapper_;
        std::shared_ptr<EngineContext> context_;
        std::vector<void*> bindingsPtr_;
        std::shared_ptr<MixMemory> workspace_;
        int device_ = 0;
    };

    ////////////////////////////////////////////////////////////////////////////////////
    TRTInferImpl::~TRTInferImpl(){
        destroy();
    }

    void TRTInferImpl::destroy() 
    {
        int old_device = 0;
        checkCudaRuntime(cudaGetDevice(&old_device));
        checkCudaRuntime(cudaSetDevice(device_));
        this->context_.reset();
        this->blobsNameMapper_.clear();
        this->outputs_.clear();
        this->inputs_.clear();
        this->inputs_name_.clear();
        this->outputs_name_.clear();
        checkCudaRuntime(cudaSetDevice(old_device));
    }

    void TRTInferImpl::print()
    {
        if(!context_){
            INFOW("Infer print, nullptr.");
            return;
        }

        INFO("Infer %p detail", this);
        INFO("\tMax Batch Size: %d", this->get_max_batch_size());
        INFO("\tInputs: %d", inputs_.size());
        for(int i = 0; i < inputs_.size(); ++i){
            auto& tensor = inputs_[i];
            auto& name = inputs_name_[i];
            INFO("\t\t%d.%s : shape {%s}", i, name.c_str(), tensor->shape_string());
        }

        INFO("\tOutputs: %d", outputs_.size());
        for(int i = 0; i < outputs_.size(); ++i){
            auto& tensor = outputs_[i];
            auto& name = outputs_name_[i];
            INFO("\t\t%d.%s : shape {%s}", i, name.c_str(), tensor->shape_string());
        }
    }

    std::shared_ptr<std::vector<uint8_t>> TRTInferImpl::serial_engine() {
        auto memory = this->context_->engine_->serialize();
        auto output = make_shared<std::vector<uint8_t>>((uint8_t*)memory->data(), (uint8_t*)memory->data()+memory->size());
        memory->destroy();
        return output;
    }

    bool TRTInferImpl::load_from_memory(const void* pdata, size_t size) {

        if (pdata == nullptr || size == 0)
            return false;

        context_.reset(new EngineContext());

        // build model
        if (!context_->build_model(pdata, size)) {
            context_.reset();
            return false;
        }

        workspace_.reset(new MixMemory());
        cudaGetDevice(&device_);
        build_engine_input_and_outputs_mapper();
        return true;
    }

    static std::vector<uint8_t> load_file(const string& file){

        ifstream in(file, ios::in | ios::binary);
        if (!in.is_open())
            return {};

        in.seekg(0, ios::end);
        size_t length = in.tellg();

        std::vector<uint8_t> data;
        if (length > 0) {
            in.seekg(0, ios::beg);
            data.resize(length);
            in.read((char*)&data[0], length);
        }
        in.close();
        return data;
    }

    bool TRTInferImpl::load(const std::string& file) {

        auto data = load_file(file);
        if (data.empty())
            return false;

        context_.reset(new EngineContext());

        //build model
        if (!context_->build_model(data.data(), data.size())) {
            context_.reset();
            return false;
        }

        workspace_.reset(new MixMemory());
        cudaGetDevice(&device_);
        build_engine_input_and_outputs_mapper();
        return true;
    }

    size_t TRTInferImpl::get_device_memory_size() {
        EngineContext* context = (EngineContext*)this->context_.get();
        return context->context_->getEngine().getDeviceMemorySize();
    }

    void TRTInferImpl::build_engine_input_and_outputs_mapper() {
        
        EngineContext* context = (EngineContext*)this->context_.get();
        int nbBindings = context->engine_->getNbBindings();
        int max_batchsize = context->engine_->getMaxBatchSize();

        inputs_.clear();
        inputs_name_.clear();
        outputs_.clear();
        outputs_name_.clear();
        orderdBlobs_.clear();
        bindingsPtr_.clear();
        blobsNameMapper_.clear();
        for (int i = 0; i < nbBindings; ++i) {

            auto dims = context->engine_->getBindingDimensions(i);
            auto type = context->engine_->getBindingDataType(i);
            const char* bindingName = context->engine_->getBindingName(i);
            dims.d[0] = max_batchsize;
            auto newTensor = make_shared<Tensor>(dims.nbDims, dims.d);
            newTensor->set_stream(this->context_->stream_);
            newTensor->set_workspace(this->workspace_);
            if (context->engine_->bindingIsInput(i)) {
                //if is input
                inputs_.push_back(newTensor);
                inputs_name_.push_back(bindingName);
                inputs_map_to_ordered_index_.push_back(orderdBlobs_.size());
            }
            else {
                //if is output
                outputs_.push_back(newTensor);
                outputs_name_.push_back(bindingName);
                outputs_map_to_ordered_index_.push_back(orderdBlobs_.size());
            }
            blobsNameMapper_[bindingName] = i;
            orderdBlobs_.push_back(newTensor);
        }
        bindingsPtr_.resize(orderdBlobs_.size());
    }

    void TRTInferImpl::set_stream(cudaStream_t stream){
        this->context_->set_stream(stream);

        for(auto& t : orderdBlobs_)
            t->set_stream(stream);
    }

    cudaStream_t TRTInferImpl::get_stream() {
        return this->context_->stream_;
    }

    int TRTInferImpl::device() {
        return device_;
    }

    void TRTInferImpl::synchronize() {
        checkCudaRuntime(cudaStreamSynchronize(context_->stream_));
    }

    bool TRTInferImpl::is_output_name(const std::string& name){
        return std::find(outputs_name_.begin(), outputs_name_.end(), name) != outputs_name_.end();
    }

    bool TRTInferImpl::is_input_name(const std::string& name){
        return std::find(inputs_name_.begin(), inputs_name_.end(), name) != inputs_name_.end();
    }

    void TRTInferImpl::forward(bool sync) 
    {
        EngineContext* context = (EngineContext*)context_.get();
        int inputBatchSize = inputs_[0]->size(0);
        for(int i = 0; i < context->engine_->getNbBindings(); ++i){
            auto dims = context->engine_->getBindingDimensions(i);
            auto type = context->engine_->getBindingDataType(i);
            dims.d[0] = inputBatchSize;
            if(context->engine_->bindingIsInput(i)){
                context->context_->setBindingDimensions(i, dims);
            }
        }

        for (int i = 0; i < outputs_.size(); ++i) {
            outputs_[i]->resize_single_dim(0, inputBatchSize);
            outputs_[i]->to_gpu(false);
        }

        for (int i = 0; i < orderdBlobs_.size(); ++i)
            bindingsPtr_[i] = orderdBlobs_[i]->gpu();

        void** bindingsptr = bindingsPtr_.data();
        //bool execute_result = context->context_->enqueue(inputBatchSize, bindingsptr, context->stream_, nullptr);
        bool execute_result = context->context_->enqueueV2(bindingsptr, context->stream_, nullptr);
        if(!execute_result){
            auto code = cudaGetLastError();
            INFOF("execute fail, code %d[%s], message %s", code, cudaGetErrorName(code), cudaGetErrorString(code));
        }

        if (sync) 
        {
            synchronize();
        }
    }

    std::shared_ptr<MixMemory> TRTInferImpl::get_workspace() {
        return workspace_;
    }

    int TRTInferImpl::num_input() {
        return this->inputs_.size();
    }

    int TRTInferImpl::num_output() {
        return this->outputs_.size();
    }

    void TRTInferImpl::set_input (int index, std::shared_ptr<Tensor> tensor)
    {
        Assert(index >= 0 && index < inputs_.size());
        this->inputs_[index] = tensor;

        int order_index = inputs_map_to_ordered_index_[index];
        this->orderdBlobs_[order_index] = tensor;
    }

    void TRTInferImpl::set_output(int index, std::shared_ptr<Tensor> tensor)
    {
        Assert(index >= 0 && index < outputs_.size());
        this->outputs_[index] = tensor;

        int order_index = outputs_map_to_ordered_index_[index];
        this->orderdBlobs_[order_index] = tensor;
    }

    std::shared_ptr<Tensor> TRTInferImpl::input(int index) 
    {
        Assert(index >= 0 && index < inputs_name_.size());
        return this->inputs_[index];
    }

    std::string TRTInferImpl::get_input_name(int index)
    {
        Assert(index >= 0 && index < inputs_name_.size());
        return inputs_name_[index];
    }

    std::shared_ptr<Tensor> TRTInferImpl::output(int index) {
        Assert(index >= 0 && index < outputs_.size());
        return outputs_[index];
    }

    std::string TRTInferImpl::get_output_name(int index){
        Assert(index >= 0 && index < outputs_name_.size());
        return outputs_name_[index];
    }

    int TRTInferImpl::get_max_batch_size() {
        Assert(this->context_ != nullptr);
        return this->context_->engine_->getMaxBatchSize();
    }

    std::shared_ptr<Tensor> TRTInferImpl::tensor(const std::string& name) {
        Assert(this->blobsNameMapper_.find(name) != this->blobsNameMapper_.end());
        return orderdBlobs_[blobsNameMapper_[name]];
    }

    std::shared_ptr<TRTInferImpl> load_infer(const string& file) {
        
        std::shared_ptr<TRTInferImpl> infer(new TRTInferImpl());
        if (!infer->load(file))
            infer.reset();
        return infer;
    }

    ///////////////////////////////////class MobileVitSSDInferImpl//////////////////////////////////////

    class MobileVitSSDInferImpl : public Infer
    {
    public:

        struct Job
        {
            cv::Mat input;
            int ori_width;
            int ori_height;
            BoxArray output;
            shared_ptr<Tensor> mono_tensor;
        };


        /** 要求在MobileVitSSDInferImpl里面执行stop，而不是在基类执行stop **/
        virtual ~MobileVitSSDInferImpl()
        {
            // stop();
            stream_ = nullptr;
            INFO("Engine destroy.");
        }

        bool startup(const string& file, int gpuid, float confidence_threshold, float nms_threshold)
        {

            normalize_ = Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::None);
            confidence_threshold_ = confidence_threshold;
            nms_threshold_        = nms_threshold;
            file_                 = file;
            gpuid_                = gpuid;
            return true;
        }

        void worker(vector<Job>& fetch_jobs, shared_ptr<TRTInferImpl>& engine)
        {
            const int MAX_IMAGE_BBOX  = 1024;
            // Tensor affin_matrix_device;
            Tensor output_array_device;
            int max_batch_size = engine->get_max_batch_size();
            auto input         = engine->tensor("input");
            auto output        = engine->tensor("output");
            int num_classes    = output->size(2) - 5;
            
            stream_            = engine->get_stream();
            gpu_               = gpuid_;

            int infer_batch_size = fetch_jobs.size();
            assert(infer_batch_size <= max_batch_size);
            input->resize_single_dim(0, infer_batch_size).to_gpu();

            // 这里的 1 + MAX_IMAGE_BBOX结构是，counter + bboxes ...
            ///！表示每个img中包含多少个检测框
            output_array_device.resize(infer_batch_size, 1 + MAX_IMAGE_BBOX * NUM_BOX_ELEMENT).to_gpu();
            
            for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch)
            {
                auto& job  = fetch_jobs[ibatch];
                auto& mono = job.mono_tensor;
                input->copy_from_gpu(input->offset(ibatch), mono->gpu(), mono->count());
            }
            engine->forward(true);
            for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch)
            {    
                auto& job                 = fetch_jobs[ibatch];
                float* image_based_output = output->gpu<float>(ibatch);
                float* output_array_ptr   = output_array_device.gpu<float>(ibatch);
                checkCudaRuntime(cudaMemsetAsync(output_array_ptr, 0, sizeof(int), stream_));
                decode_kernel_invoker(image_based_output, job.ori_height, job.ori_width, output->size(1), num_classes, confidence_threshold_, nms_threshold_, nullptr, output_array_ptr, MAX_IMAGE_BBOX, stream_);
            }

            output_array_device.to_cpu();
            for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch)
            {
                float* parray = output_array_device.cpu<float>(ibatch);
                int count     = min(MAX_IMAGE_BBOX, (int)*parray);
                auto& job     = fetch_jobs[ibatch];
                for(int i = 0; i < count; ++i)
                {
                    float* pbox  = parray + 1 + i * NUM_BOX_ELEMENT;
                    int label    = pbox[5];
                    int keepflag = pbox[6];
                    if(keepflag == 1) {
                        job.output.emplace_back(pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], label);
                    }
                }
            }

        }

        // for debug
        void showProcessRes(Job& job)
        {
            static int i = 0;
            auto& tensor = job.mono_tensor;
            if (tensor)
            {
                string preProcessFile = iLogger::format("%s_%d_preprocess.npz", "image", i);
                tensor->save_to_file(preProcessFile);
                ++i;
            }
        }

        virtual bool preprocess(Job& job, const Mat& image)
        {

            AutoDevice auto_device(gpu_);
            auto& tensor = job.mono_tensor;

            if(tensor == nullptr){
                // not init
                tensor = make_shared<Tensor>();
                tensor->set_workspace(make_shared<MixMemory>());
            }

            job.ori_height = image.rows;
            job.ori_width = image.cols;
            
            tensor->set_stream(stream_);
            tensor->resize(1, 3, input_height_, input_width_);

            size_t size_image      = image.cols * image.rows * 3;
  
            auto workspace         = tensor->get_workspace();
            uint8_t* gpu_workspace        = (uint8_t*)workspace->gpu(size_image);
            uint8_t* image_device         = gpu_workspace;

            uint8_t* cpu_workspace        = (uint8_t*)workspace->cpu(size_image);
            uint8_t* image_host           = cpu_workspace;

            memcpy(image_host, image.data, size_image);
            checkCudaRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream_));
 
            resize_bilinear_and_normalize(
                image_device, image.cols * 3, image.cols, image.rows, tensor->gpu<float>(), input_width_, input_height_,
                114, normalize_,
                stream_);

            return true;
        }

        virtual void commits(const vector<Mat>& inputs, vector<BoxArray>& box) override
        {
            int batch_size = (int)inputs.size();
            vector<Job> jobs(batch_size);
            box.resize(batch_size);

            ///! 构建 engine
            set_device(gpuid_);
            static shared_ptr<TRTInferImpl> engine = load_infer(file_);  ///< 即使多次commits engine也只会初始化一次
            if(engine == nullptr)
            {
                INFOE("Engine %s load failed", file_.c_str());
                return;
            }

            engine->print();

            auto input         = engine->tensor("input");
            input_width_       = input->size(3);
            input_height_      = input->size(2);

            for (int i = 0; i < batch_size; ++i)
            {
                Job &job = jobs[i];
                preprocess(job, inputs[i]);
            }

            // 推理
            worker(jobs, engine);

            // 获取返回box
            for (int i = 0; i < batch_size; ++i)
            {
                box[i] = jobs[i].output;
            }
        }

        virtual void commits(const vector<std::string>& image_filenames, vector<BoxArray>& box) override
        {
            int data_size = (int)image_filenames.size();
            ///! 构建 engine
            set_device(gpuid_);
            static shared_ptr<TRTInferImpl> engine = load_infer(file_);  ///< 即使多次commits engine也只会初始化一次
            if(engine == nullptr)
            {
                INFOE("Engine %s load failed", file_.c_str());
                return;
            }
            auto input         = engine->tensor("input");
            input_width_       = input->size(3);
            input_height_      = input->size(2);

            int max_batch_size = engine->get_max_batch_size();

            int total_batch_count = (max_batch_size - 1 + data_size) / max_batch_size;
            INFO("total_batch_count: %d", total_batch_count);
            box.resize(data_size);
            for(int i = 0; i < data_size; i += max_batch_size) 
            {
                int start_index = i;
                int end_index = min(start_index + max_batch_size - 1, data_size - 1);

                vector<Job> jobs(end_index - start_index + 1);

                // 对每张图片进行预处理
                for (int j = start_index; j <= end_index; ++j)
                {
                    Job &job = jobs[j - start_index];
                    preprocess(job, cv::imread(image_filenames[j]));
                }

                // 推理
                worker(jobs, engine);
                // 获取返回box
                for (int j = start_index; j <= end_index; ++j)
                {
                    box[j] = jobs[j - start_index].output;
                }
            }
            
        }

        virtual void commit(const Mat& image, BoxArray& box) override
        {

            // 构建 engine
            set_device(gpuid_);
            static shared_ptr<TRTInferImpl> engine = load_infer(file_);  ///< 即使多次commits engine也只会初始化一次
            if(engine == nullptr)
            {
                INFOE("Engine %s load failed", file_.c_str());
                return;
            }

            auto input         = engine->tensor("input");
            input_width_       = input->size(3);
            input_height_      = input->size(2);

            vector<Job> jobs(1);
            preprocess(jobs[0], image);
            // 推理
            worker(jobs, engine);
            // 获取返回box
            box = jobs[0].output;
        }

    private:
        int input_width_            = 0;
        int input_height_           = 0;
        int gpu_                    = 0;
        float confidence_threshold_ = 0;
        float nms_threshold_        = 0;
        cudaStream_t stream_       = nullptr;
        Norm normalize_;
        int gpuid_                  = 0;
        string file_;
    };

    shared_ptr<Infer> create_infer(const string& engine_file, int gpuid, float confidence_threshold, float nms_threshold)
    {
        shared_ptr<MobileVitSSDInferImpl> instance(new MobileVitSSDInferImpl());
        if(!instance->startup(engine_file, gpuid, confidence_threshold, nms_threshold)){
            instance.reset();
        }
        return instance;
    }

    //////////////////////////////////////Compile Model/////////////////////////////////////////////////////////////

    const char* mode_string(Mode type) 
    {
        switch (type) {
        case Mode::FP32:
            return "FP32";
        case Mode::FP16:
            return "FP16";
        default:
            return "UnknowCompileMode";
        }
    }

    bool compile(Mode mode, unsigned int max_batch_size,
                    const string& source_onnx,
                    const string& saveto,
                    size_t max_workspace_size) 
    {

        INFO("Compile %s %s.", mode_string(mode), source_onnx.c_str());
        shared_ptr<IBuilder> builder(createInferBuilder(gLogger), destroy_nvidia_pointer<IBuilder>);
        if (builder == nullptr) {
            INFOE("Can not create builder.");
            return false;
        }

        shared_ptr<IBuilderConfig> config(builder->createBuilderConfig(), destroy_nvidia_pointer<IBuilderConfig>);
        if (mode == Mode::FP16) {
            if (!builder->platformHasFastFp16()) {
                INFOW("Platform not have fast fp16 support");
            }
            config->setFlag(BuilderFlag::kFP16);
        }

        shared_ptr<INetworkDefinition> network;
        shared_ptr<nvonnxparser::IParser> onnxParser;
        const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        network = shared_ptr<INetworkDefinition>(builder->createNetworkV2(explicitBatch), destroy_nvidia_pointer<INetworkDefinition>);

        // from onnx is not markOutput
        onnxParser.reset(nvonnxparser::createParser(*network, gLogger), destroy_nvidia_pointer<nvonnxparser::IParser>);
        if (onnxParser == nullptr) {
            INFOE("Can not create parser.");
            return false;
        }

        if (!onnxParser->parseFromFile(source_onnx.c_str(), 1)) {
            INFOE("Can not parse OnnX file: %s", source_onnx.c_str());
            return false;
        }
        
        auto inputTensor = network->getInput(0);
        auto inputDims = inputTensor->getDimensions();

        INFO("Input shape is %s", join_dims(vector<int>(inputDims.d, inputDims.d + inputDims.nbDims)).c_str());
        INFO("Set max batch size = %d", max_batch_size);
        INFO("Set max workspace size = %.2f MB", max_workspace_size / 1024.0f / 1024.0f);

        int net_num_input = network->getNbInputs();
        INFO("Network has %d inputs:", net_num_input);
        vector<string> input_names(net_num_input);
        for(int i = 0; i < net_num_input; ++i){
            auto tensor = network->getInput(i);
            auto dims = tensor->getDimensions();
            auto dims_str = join_dims(vector<int>(dims.d, dims.d+dims.nbDims));
            INFO("      %d.[%s] shape is %s", i, tensor->getName(), dims_str.c_str());
            input_names[i] = tensor->getName();
        }

        int net_num_output = network->getNbOutputs();
        INFO("Network has %d outputs:", net_num_output);
        for(int i = 0; i < net_num_output; ++i){
            auto tensor = network->getOutput(i);
            auto dims = tensor->getDimensions();
            auto dims_str = join_dims(vector<int>(dims.d, dims.d+dims.nbDims));
            INFO("      %d.[%s] shape is %s", i, tensor->getName(), dims_str.c_str());
        }

        int net_num_layers = network->getNbLayers();
        INFO("Network has %d layers", net_num_layers);		
        builder->setMaxBatchSize(max_batch_size);
        config->setMaxWorkspaceSize(max_workspace_size);

        auto profile = builder->createOptimizationProfile();
        for(int i = 0; i < net_num_input; ++i)
        {
            auto input = network->getInput(i);
            auto input_dims = input->getDimensions();
            input_dims.d[0] = 1;
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);
            input_dims.d[0] = max_batch_size;
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
        }
        config->addOptimizationProfile(profile);

        INFO("Building engine...");
        auto time_start = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count();
        shared_ptr<ICudaEngine> engine(builder->buildEngineWithConfig(*network, *config), destroy_nvidia_pointer<ICudaEngine>);
        if (engine == nullptr) {
            INFOE("engine is nullptr");
            return false;
        }

        auto time_end = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count();
        INFO("Build done %lld ms !", time_end - time_start);
        
        // serialize the engine, then close everything down
        shared_ptr<IHostMemory> seridata(engine->serialize(), destroy_nvidia_pointer<IHostMemory>);
        return save_file(saveto, seridata->data(), seridata->size());
    }

    static void inference(vector<ImageItem>& images, int deviceid, const string& engine_file, MobileVitDet::Mode mode, const string& model_name) {
        
        
        auto engine = MobileVitDet::create_infer(engine_file, deviceid, 0.15f, 0.5f);
        if(engine == nullptr){
            INFOE("Engine is nullptr");
            return;
        }

        
        int nimages = images.size();
        vector<BoxArray> image_results(nimages);
        std::vector<std::string> filenames;
        for(auto& item: images) {
            filenames.push_back(item.image_file);
        }
        std::vector<BoxArray> boxes;
        engine->commits(filenames, boxes);

        for(int i = 0; i < (int)images.size(); ++i) {
            images[i].detections = boxes[i];
        }
    }

    static void inference(int deviceid, const string& engine_file, MobileVitDet::Mode mode, const string& model_name, const string& filePath) {
        auto engine = MobileVitDet::create_infer(engine_file, deviceid, 0.15f, 0.5f);
        if(engine == nullptr) {
            INFOE("Engine is nullptr");
            return;
        }

        ///! read input
        auto files = iLogger::find_files(filePath, "*.jpg;*.jpeg;*.png;*.gif;*.tif");
        vector<cv::Mat> images;
        for(int i = 0; i < files.size(); ++i){
            auto image = cv::imread(files[i]);
            images.emplace_back(image);
        }

        vector<MobileVitDet::BoxArray> boxes_array;

        // warmup
        vector<cv::Mat> tmp_images;
        for(int i = 0; i < 32; ++i){
            auto image = cv::imread(files[0]);
            tmp_images.push_back(image);
        }
        for(int i = 0; i < 10; ++i)
            engine->commits(tmp_images, boxes_array);
        boxes_array.clear();
        
        /////////////////////////////////////////////////////////
        const int ntest = 50;
        boxes_array.clear();
        auto begin_timer = iLogger::timestamp_now_float();
        
        for(int i  = 0; i < ntest; ++i)
            engine->commits(tmp_images, boxes_array);

        float inference_average_time = (iLogger::timestamp_now_float() - begin_timer) / ntest / tmp_images.size();

        auto mode_name = MobileVitDet::mode_string(mode);
        INFO("%s average: %.2f ms / image, FPS: %.2f", engine_file.c_str(), inference_average_time, 1000 / inference_average_time);
        tmp_images.clear();

        string root = iLogger::format("%s_%s_result", model_name.c_str(), mode_name);
        iLogger::rmtree(root);
        iLogger::mkdir(root);

        boxes_array.clear();
        engine->commits(images, boxes_array);
        for(int i = 0; i < boxes_array.size(); ++i){

            auto& image = images[i];
            auto boxes  = boxes_array[i];
            
            for(auto& obj : boxes){
                uint8_t b, g, r;
                tie(b, g, r) = iLogger::random_color(obj.class_label);
                cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 5);

                auto name    = cocolabels[obj.class_label];
                auto caption = iLogger::format("%s %.2f", name, obj.confidence);
                int width    = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
                cv::rectangle(image, cv::Point(obj.left-3, obj.top-33), cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
                cv::putText(image, caption, cv::Point(obj.left, obj.top-5), 0, 1, cv::Scalar::all(0), 2, 16);
            }

            string file_name = iLogger::file_name(files[i], false);
            string save_path = iLogger::format("%s/%s.jpg", root.c_str(), file_name.c_str());
            INFO("Save to %s, %d object, average time %.2f ms", save_path.c_str(), boxes.size(), inference_average_time);
            cv::imwrite(save_path, image);
        }
    }
    
};




int main(int argc, char* argv[])
{
    ///! load onnx && compile tensorRT engine
    int deviceid = 0;

    if (argc != 4)
        INFOF("请输入3个参数：1. 是否使用fp16；2.是否测试精度；3.精度测试数据集路径");

    MobileVitDet::Mode mode = MobileVitDet::Mode::FP32;
    if (!strcmp(argv[1], "1"))
        mode = MobileVitDet::Mode::FP16;
    auto modeName =  MobileVitDet::mode_string(mode);
    MobileVitDet::set_device(deviceid);

    std::string onnx_file("./mobilevit_s_det.onnx");
    std::string trt_file = iLogger::format("./mobilevit_s_det_%s.plan", modeName);
    int max_batch_size = 32;

    if (!iLogger::exists(trt_file))
    {
        MobileVitDet::compile(mode, max_batch_size, onnx_file, trt_file, (1 << 30));  // < 这里max_workspace设置这么小没关系吗？设置成2<<30就无法生成engin了
    }

    // show result
    std::string model_name("./mobilevit_s_det");
    MobileVitDet::inference(deviceid, trt_file, mode, model_name, "./images");

    ///! readimage && preprocess && infer (need opencv) && decode output
    if (!strcmp(argv[2], "1"))
    {
        std::string curPath(argv[3]);
        curPath += "/images/val2017";
        // auto images = MobileVitDet::scan_dataset("/root/trt2022_src/mobilenet/dataset/fast-ai-coco/images/val2017");
        auto images = MobileVitDet::scan_dataset(curPath);
        MobileVitDet::inference(images, deviceid, trt_file, mode, model_name);
        std::string json_path = iLogger::format("./mobilevit_ssd_det_%s.json", modeName);
        MobileVitDet::save_to_json(images, json_path);
    }

    return 0;

}