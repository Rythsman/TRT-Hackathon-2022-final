
#include "trt_tensor.hpp"
#include <algorithm>
#include <cuda_runtime.h>
#include "cuda_tools.hpp"
#include <cuda_fp16.h>

using namespace std;

namespace TRT{

	float float16_to_float(float16 value){
		return __half2float(*reinterpret_cast<__half*>(&value));
	}

	float16 float_to_float16(float value){
		auto val = __float2half(value);
		return *reinterpret_cast<float16*>(&val);
	}

	int data_type_size(DataType dt){
		switch (dt) {
			case DataType::Float: return sizeof(float);
			case DataType::Float16: return sizeof(float16);
			default: {
				INFOE("Not support dtype: %d", dt);
				return -1;
			}
		}
	}

	MixMemory::MixMemory(void* cpu, size_t cpu_size, void* gpu, size_t gpu_size){
		reference_data(cpu, cpu_size, gpu, gpu_size);		
	}

	void MixMemory::reference_data(void* cpu, size_t cpu_size, void* gpu, size_t gpu_size){
		release_all();
		
		if(cpu == nullptr || cpu_size == 0){
			cpu = nullptr;
			cpu_size = 0;
		}

		if(gpu == nullptr || gpu_size == 0){
			gpu = nullptr;
			gpu_size = 0;
		}

		this->cpu_ = cpu;
		this->cpu_size_ = cpu_size;
		this->gpu_ = gpu;
		this->gpu_size_ = gpu_size;

		this->owner_cpu_ = !(cpu && cpu_size > 0);
		this->owner_gpu_ = !(gpu && gpu_size > 0);
	}

	MixMemory::~MixMemory() {
		release_all();
	}

	void* MixMemory::gpu(size_t size) {

		if (gpu_size_ < size) {
			release_gpu();

			gpu_size_ = size;
			checkCudaRuntime(cudaMalloc(&gpu_, size));
			checkCudaRuntime(cudaMemset(gpu_, 0, size));
		}
		return gpu_;
	}

	void* MixMemory::cpu(size_t size) {

		if (cpu_size_ < size) {
			release_cpu();

			cpu_size_ = size;
			checkCudaRuntime(cudaMallocHost(&cpu_, size));
			Assert(cpu_ != nullptr);
			memset(cpu_, 0, size);
		}
		return cpu_;
	}

	void MixMemory::release_cpu() {
		if (cpu_) {
			if(owner_cpu_){
				checkCudaRuntime(cudaFreeHost(cpu_));
			}
			cpu_ = nullptr;
		}
		cpu_size_ = 0;
	}

	void MixMemory::release_gpu() {
		if (gpu_) {
			if(owner_gpu_){
				checkCudaRuntime(cudaFree(gpu_));
			}
			gpu_ = nullptr;
		}
		gpu_size_ = 0;
	}

	void MixMemory::release_all() {
		release_cpu();
		release_gpu();
	}

	const char* data_head_string(DataHead dh){
		switch(dh){
			case DataHead::Init: return "Init";
			case DataHead::Device: return "Device";
			case DataHead::Host: return "Host";
			default: return "Unknow";
		}
	}

	const char* data_type_string(DataType dt){
		switch(dt){
			case DataType::Float: return "Float32";
			case DataType::Float16: return "Float16";
			default: return "Unknow";
		}
	}

	Tensor::Tensor(int n, int c, int h, int w, DataType dtype, shared_ptr<MixMemory> data) {
		this->dtype_ = dtype;
		setup_data(data);
		resize(n, c, h, w);
	}

	Tensor::Tensor(const std::vector<int>& dims, DataType dtype, shared_ptr<MixMemory> data){
		this->dtype_ = dtype;
		setup_data(data);
		resize(dims);
	}

	Tensor::Tensor(int ndims, const int* dims, DataType dtype, shared_ptr<MixMemory> data) {
		this->dtype_ = dtype;
		setup_data(data);
		resize(ndims, dims);
	}

	Tensor::Tensor(DataType dtype, shared_ptr<MixMemory> data){
		shape_string_[0] = 0;
		dtype_ = dtype;
		setup_data(data);
	}

	Tensor::~Tensor() {
		release();
	}

	Tensor& Tensor::compute_shape_string(){

		// clean string
		shape_string_[0] = 0;

		char* buffer = shape_string_;
		size_t buffer_size = sizeof(shape_string_);
		for(int i = 0; i < shape_.size(); ++i){

			int size = 0;
			if(i < shape_.size() - 1)
				size = snprintf(buffer, buffer_size, "%d x ", shape_[i]);
			else
				size = snprintf(buffer, buffer_size, "%d", shape_[i]);

			buffer += size;
			buffer_size -= size;
		}
		return *this;
	}

	void Tensor::reference_data(const vector<int>& shape, void* cpu_data, size_t cpu_size, void* gpu_data, size_t gpu_size, DataType dtype){

		dtype_ = dtype;
		data_->reference_data(cpu_data, cpu_size, gpu_data, gpu_size);
		setup_data(data_);
		resize(shape);
	}

	void Tensor::setup_data(shared_ptr<MixMemory> data){
		
		data_ = data;
		if(data_ == nullptr){
			data_ = make_shared<MixMemory>();
		}

		head_ = DataHead::Init;
		if(data_->cpu()){
			head_ = DataHead::Host;
		}

		if(data_->gpu()){
			head_ = DataHead::Device;
		}
	}

	shared_ptr<Tensor> Tensor::clone() const{
		auto new_tensor = make_shared<Tensor>(shape_, dtype_);
		if(head_ == DataHead::Init)
			return new_tensor;
		
		if(head_ == DataHead::Host){
			memcpy(new_tensor->cpu(), this->cpu(), this->bytes_);
		}else if(head_ == DataHead::Device){
			checkCudaRuntime(cudaMemcpyAsync(new_tensor->gpu(), this->gpu(), bytes_, cudaMemcpyDeviceToDevice, stream_));
		}
		return new_tensor;
	}

	Tensor& Tensor::copy_from_gpu(size_t offset, const void* src, size_t num_element){

		if(head_ == DataHead::Init)
			to_gpu(false);

		size_t offset_location = offset * element_size();
		if(offset_location >= bytes_){
			INFOE("Offset location[%lld] >= bytes_[%lld], out of range", offset_location, bytes_);
			return *this;
		}

		size_t copyed_bytes = num_element * element_size();
		size_t remain_bytes = bytes_ - offset_location;
		if(copyed_bytes > remain_bytes){
			INFOE("Copyed bytes[%lld] > remain bytes[%lld], out of range", copyed_bytes, remain_bytes);
			return *this;
		}

		if(head_ == DataHead::Device){
			checkCudaRuntime(cudaMemcpyAsync(gpu<unsigned char>() + offset_location, src, copyed_bytes, cudaMemcpyDeviceToDevice, stream_));
		}else if(head_ == DataHead::Host){
			checkCudaRuntime(cudaMemcpyAsync(cpu<unsigned char>() + offset_location, src, copyed_bytes, cudaMemcpyDeviceToHost, stream_));
		}else{
			INFOE("Unsupport head type %d", head_);
		}
		return *this;
	}

	Tensor& Tensor::copy_from_cpu(size_t offset, const void* src, size_t num_element){

		if(head_ == DataHead::Init)
			to_cpu(false);

		size_t offset_location = offset * element_size();
		if(offset_location >= bytes_){
			INFOE("Offset location[%lld] >= bytes_[%lld], out of range", offset_location, bytes_);
			return *this;
		}

		size_t copyed_bytes = num_element * element_size();
		size_t remain_bytes = bytes_ - offset_location;
		if(copyed_bytes > remain_bytes){
			INFOE("Copyed bytes[%lld] > remain bytes[%lld], out of range", copyed_bytes, remain_bytes);
			return *this;
		}

		if(head_ == DataHead::Device){
			checkCudaRuntime(cudaMemcpyAsync((char*)data_->gpu() + offset_location, src, copyed_bytes, cudaMemcpyHostToDevice, stream_));
		}else if(head_ == DataHead::Host){
			//checkCudaRuntime(cudaMemcpyAsync((char*)data_->cpu() + offset_location, src, copyed_bytes, cudaMemcpyHostToHost, stream_));
			memcpy((char*)data_->cpu() + offset_location, src, copyed_bytes);
		}else{
			INFOE("Unsupport head type %d", head_);
		}
		return *this;
	}

	Tensor& Tensor::release() {
		data_->release_all();
		shape_.clear();
		bytes_ = 0;
		head_ = DataHead::Init;
		return *this;
	}

	bool Tensor::empty() const{
		return data_->cpu() == nullptr && data_->gpu() == nullptr;
	}

	int Tensor::count(int start_axis) const {

		if(start_axis >= 0 && start_axis < shape_.size()){
			int size = 1;
			for (int i = start_axis; i < shape_.size(); ++i) 
				size *= shape_[i];
			return size;
		}else{
			return 0;
		}
	}

	Tensor& Tensor::resize(const std::vector<int>& dims) {
		return resize(dims.size(), dims.data());
	}

	int Tensor::numel() const{
		int value = shape_.empty() ? 0 : 1;
		for(int i = 0; i < shape_.size(); ++i){
			value *= shape_[i];
		}
		return value;
	}

	Tensor& Tensor::resize_single_dim(int idim, int size){

		Assert(idim >= 0 && idim < shape_.size());

		auto new_shape = shape_;
		new_shape[idim] = size;
		return resize(new_shape);
	}

	Tensor& Tensor::resize(int ndims, const int* dims) {

		vector<int> setup_dims(ndims);
		for(int i = 0; i < ndims; ++i){
			int dim = dims[i];
			if(dim == -1){
				Assert(ndims == shape_.size());
				dim = shape_[i];
			}
			setup_dims[i] = dim;
		}
		this->shape_ = setup_dims;

		// strides = element_size
		this->strides_.resize(setup_dims.size());
		
		size_t prev_size  = element_size();
		size_t prev_shape = 1;
		for(int i = (int)strides_.size() - 1; i >= 0; --i){
			if(i + 1 < strides_.size()){
				prev_size  = strides_[i+1];
				prev_shape = shape_[i+1];
			}
			strides_[i] = prev_size * prev_shape;
		}

		this->adajust_memory_by_update_dims_or_type();
		this->compute_shape_string();
		return *this;
	}

	Tensor& Tensor::adajust_memory_by_update_dims_or_type(){
		
		int needed_size = this->numel() * element_size();
		if(needed_size > this->bytes_){
			head_ = DataHead::Init;
		}
		this->bytes_ = needed_size;
		return *this;
	}

	Tensor& Tensor::synchronize(){ 
		checkCudaRuntime(cudaStreamSynchronize(stream_));
		return *this;
	}

	Tensor& Tensor::to_gpu(bool copy) {

		if (head_ == DataHead::Device)
			return *this;

		head_ = DataHead::Device;
		data_->gpu(bytes_);

		if (copy && data_->cpu() != nullptr) {
			checkCudaRuntime(cudaMemcpyAsync(data_->gpu(), data_->cpu(), bytes_, cudaMemcpyHostToDevice, stream_));
		}
		return *this;
	}
	
	Tensor& Tensor::to_cpu(bool copy) {

		if (head_ == DataHead::Host)
			return *this;

		head_ = DataHead::Host;
		data_->cpu(bytes_);

		if (copy && data_->gpu() != nullptr) {
			checkCudaRuntime(cudaMemcpyAsync(data_->cpu(), data_->gpu(), bytes_, cudaMemcpyDeviceToHost, stream_));
			checkCudaRuntime(cudaStreamSynchronize(stream_));
		}
		return *this;
	}

	Tensor& Tensor::to_float() {

		if (type() == DataType::Float)
			return *this;

		if (type() != DataType::Float16) {
			INFOF("not implement function");
		}

		auto c = count();
		float* convert_memory = (float*)malloc(c * data_type_size(DataType::Float));
		float* dst = convert_memory;
		float16* src = cpu<float16>();

		for (int i = 0; i < c; ++i)
			*dst++ = float16_to_float(*src++);

		this->dtype_ = DataType::Float;
		adajust_memory_by_update_dims_or_type();
		memcpy(cpu(), convert_memory, bytes_);
		free(convert_memory);
		return *this;
	}

	Tensor& Tensor::to_half() {

		if (type() == DataType::Float16)
			return *this;

		if (type() != DataType::Float) {
			INFOF("not implement function");
		}

		auto c = count();
		float16* convert_memory = (float16*)malloc(c * data_type_size(DataType::Float16));
		float16* dst = convert_memory;
		float* src = cpu<float>();

		for (int i = 0; i < c; ++i) 
			*dst++ = float_to_float16(*src++);

		this->dtype_ = DataType::Float16;
		adajust_memory_by_update_dims_or_type();
		memcpy(cpu(), convert_memory, bytes_);
		free(convert_memory);
		return *this;
	}

	Tensor& Tensor::set_to(float value) {
		int c = count();
		if (dtype_ == DataType::Float) {
			float* ptr = cpu<float>();
			for (int i = 0; i < c; ++i)
				*ptr++ = value;
		}
		else {
			float16* ptr = cpu<float16>();
			auto val = float_to_float16(value);
			for (int i = 0; i < c; ++i)
				*ptr++ = val;
		}
		return *this;
	}

	int Tensor::offset_array(size_t size, const int* index_array) const{

		Assert(size <= shape_.size());
		int value = 0;
		for(int i = 0; i < shape_.size(); ++i){

			if(i < size)
				value += index_array[i];

			if(i + 1 < shape_.size())
				value *= shape_[i+1];
		}
		return value;
	}

	int Tensor::offset_array(const std::vector<int>& index_array) const{
		return offset_array(index_array.size(), index_array.data());
	}


	bool Tensor::save_to_file(const std::string& file) const{

		auto tmp_file = std::string(file);
		
		if(file.find("npz", file.length() - 3) != file.npos) {
			std::cout << "npz loaded" << std::endl;
			std::vector<size_t> tmp_shape;
			for(auto item: shape_) tmp_shape.push_back(item);

			if(dtype_ == DataType::Float) {
				cnpy::npz_save(tmp_file, "data", static_cast<const float*>(cpu()), tmp_shape);
			} else {
				cnpy::npz_save(tmp_file, "data", static_cast<const float16*>(cpu()), tmp_shape);
			}
			return access(file.c_str(), F_OK) == 0;
		} else if(file.find("npy", file.length() - 3) != file.npos) {
			std::cout << "npy loaded" << std::endl;
			std::vector<size_t> tmp_shape;
			for(auto item: shape_) tmp_shape.push_back(item);

			if(dtype_ == DataType::Float) {
				cnpy::npy_save(tmp_file, static_cast<const float*>(cpu()), tmp_shape);
			} else {
				cnpy::npy_save(tmp_file, static_cast<const float16*>(cpu()), tmp_shape);
			}
			return access(file.c_str(), F_OK) == 0;
		}
		std::cout << "bin loaded" << std::endl;
		if(empty()) return false;

		FILE* f = fopen(file.c_str(), "wb");
		if(f == nullptr) return false;

		int ndims = this->ndims();
		unsigned int head[3] = {0xFCCFE2E2, ndims, static_cast<unsigned int>(dtype_)};
		fwrite(head, 1, sizeof(head), f);
		fwrite(shape_.data(), 1, sizeof(shape_[0]) * shape_.size(), f);
		fwrite(cpu(), 1, bytes_, f);
		fclose(f);
		return true;
	}

}; // TRTTensor