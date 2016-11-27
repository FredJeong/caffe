#include <vector>

#include "caffe/common.hpp"
#include "caffe/util/winograd.hpp"
#include "caffe/layers/wino_conv_layer.hpp"

namespace caffe {


template <typename Dtype>
void WinoConvolutionLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseConvolutionLayer<Dtype>::LayerSetUp(bottom, top);

 // LENA
	wino_tile_ = this->layer_param_.convolution_param().wino_tile(); // base = 2/4/6  , AddOpt = 12/14/16
	wino_zero_idx_ = this->layer_param_.convolution_param().wino_zero_idx(); // base = 2/4/6  , AddOpt = 12/14/16

	vector<int> wino_shape(4);
	wino_shape[0] = this->group_; 
	wino_shape[1] = (wino_tile_ + 2) * (wino_tile_ + 2); 
	wino_shape[2] = this->num_output_; 
	wino_shape[3] = this->channels_ ; //  / this->group_; 
  
  //for (int i = 0; i < 4; i++) printf("wino_shape[%d] %d\n", i, wino_shape[i]);


	wino_blob.Reshape(wino_shape);     

	const int num_inputs = this->channels_; 
	const int num_outputs = this->num_output_ ; 
	
	const Dtype* weight = this->blobs_[0]->gpu_data(); //   + g * this->weight_offset_ ;
	Dtype* wino_weight = wino_blob.mutable_gpu_data(); //   + g * this->wino_weight_offset_  ; 

	winoWeight_gpu(num_inputs, num_outputs, weight, wino_weight, wino_tile_); 

}


template <typename Dtype>
void WinoConvolutionLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
	BaseConvolutionLayer<Dtype>::Reshape(bottom, top);
	bottom_offset_ = this->bottom_dim_ / this->group_;
	top_offset_ = this->top_dim_ / this->group_;
  
  	const int batchs = 1; // this->num_; 
	const int num_inputs = this->channels_; 
	const int num_outputs = this->num_output_ ; 
	
	const int height = this->conv_input_shape_.cpu_data()[1] ; 
	const int width =  this->conv_input_shape_.cpu_data()[2] ; 
	const int height_pad = this->pad_.cpu_data()[0]; 
	const int width_pad = this->pad_.cpu_data()[1]; 
	const int height_out = this->output_shape_[0]; 
	const int width_out = this->output_shape_[1];

	int tileW = (width_out + wino_tile_ - 1 ) / wino_tile_; 
	int tileH = (height_out + wino_tile_ -1 ) / wino_tile_;
	
	std::vector<int> shape_temp(1);
	shape_temp[0] = batchs * num_inputs * (height + height_pad * 2) * (width + width_pad * 2);
	p_blob.Reshape(shape_temp);
	//printf("p_shape %d\n", shape_temp[0]);
	shape_temp[0] = batchs * num_inputs * tileH * tileW * (wino_tile_ + 2) * (wino_tile_ + 2);
	v_blob.Reshape(shape_temp);
	//printf("v_shape %d\n", shape_temp[0]);
	shape_temp[0] = batchs * num_outputs * tileH * tileW * (wino_tile_ + 2) * (wino_tile_ + 2);
	m_blob.Reshape(shape_temp);
	//printf("m_shape %d\n", shape_temp[0]);
	wino_weight_offset_ = num_inputs * num_outputs * (wino_tile_ + 2) * (wino_tile_ + 2); 


	const Dtype* weight = this->blobs_[0]->gpu_data(); //   + g * this->weight_offset_ ;
	Dtype* wino_weight = wino_blob.mutable_gpu_data(); //   + g * this->wino_weight_offset_  ; 

	winoWeight_gpu(num_inputs, num_outputs, weight, wino_weight, wino_tile_); 


}


template <typename Dtype>
void WinoConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void WinoConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void WinoConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

#ifndef CPU_ONLY

template <typename Dtype>
void WinoConvolutionLayer<Dtype>::forward_gpu_wino(const Dtype* input,
    const Dtype* weights, Dtype* output, const Dtype *wino_weights, bool skip_im2col) {
	const Dtype* u_matrix = wino_weights; 

	const int batchs = 1; // this->num_; 
	const int num_inputs = this->channels_ / this->group_; 
	const int num_outputs = this->num_output_ ; 
	
	const int height = this->conv_input_shape_.cpu_data()[1] ; 
	const int width =  this->conv_input_shape_.cpu_data()[2] ; 
	const int height_pad = this->pad_.cpu_data()[0]; 
	const int width_pad = this->pad_.cpu_data()[1]; 
	const int height_out = this->output_shape_[0]; 
	const int width_out = this->output_shape_[1];

	int tileW = (width_out + wino_tile_ - 1 ) / wino_tile_; 
	int tileH = (height_out + wino_tile_ -1 ) / wino_tile_;

	Dtype* p_matrix = p_blob.mutable_gpu_data();
	Dtype* v_matrix = v_blob.mutable_gpu_data();
	Dtype* m_matrix = m_blob.mutable_gpu_data(); 
	
	padSrc_gpu(batchs, num_inputs, height, width, height_pad, width_pad, input, p_matrix); 
	winoSrc_gpu(batchs, num_inputs, tileH, tileW, height + height_pad *2, width + width_pad * 2, p_matrix, v_matrix, wino_tile_); 
	winoMulti_gpu(batchs, num_inputs, num_outputs, tileH, tileW, u_matrix, v_matrix, m_matrix, wino_tile_); 
	winoDst_gpu(batchs, num_outputs, tileH, tileW, height_out, width_out, m_matrix, output, wino_tile_); 
}



template <typename Dtype>
void WinoConvolutionLayer<Dtype>::weight_gpu_wino(const Dtype* input,
    const Dtype* output, Dtype* weights) {


///////////////////////////////




}


template <typename Dtype>
void WinoConvolutionLayer<Dtype>::backward_gpu_wino(const Dtype* output,
    const Dtype* weights, Dtype* input) {

//////////////////////////


}


#endif 

#ifdef CPU_ONLY
STUB_GPU(WinoConvolutionLayer);
#endif

INSTANTIATE_CLASS(WinoConvolutionLayer);
REGISTER_LAYER_CLASS(WinoConvolution);

}  // namespace caffe
