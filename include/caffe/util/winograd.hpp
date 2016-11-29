#ifndef _CAFFE_UTIL_WINOGRAD_HPP_
#define _CAFFE_UTIL_WINOGRAD_HPP_


namespace caffe {

template <typename Dtype>
void winoWeight_gpu(const int num_inputs, const int num_outputs, 
					const Dtype *weight, Dtype *wino_weight, const int wino_tile); 


template <typename Dtype>
void padSrc_gpu(const int batchs, const int num_inputs, const int height, const int width, 
				const int height_pad, const int width_pad,
				const Dtype *input, Dtype *input_pad, const int c_height, const int c_width);


template <typename Dtype>
void winoSrc_gpu(const int batchs, const int num_inputs, const int tileH, const int tileW, 
				const int height, const int width, // include padding 
				const Dtype *m_matrix, Dtype *v_matrix, const int wino_tile_size); 


template <typename Dtype>
void winoMulti_gpu(const int batchs, const int num_inputs, const int num_outputs, const int tileH, const int tileW, 
					const Dtype *u_matrix, Dtype *v_matrix, Dtype *m_matrix, const int wino_tile_size); 

template <typename Dtype>
void winoDst_gpu(const int batchs, const int num_outputs, const int tileH, const int tileW, const int height, const int width,
				 Dtype *m_matrix, Dtype *output, const int wino_tile_size); 

template <typename Dtype>
void trimDst_gpu(const int batchs, const int num_outputs, const int c_height, const int c_width,
        const int height, const int width, const Dtype *output_pad, Dtype *output);


} // namespace caffe 

#endif // CAFFE_UTIL_WINOGRAD_HPP_
