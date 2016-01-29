#include <algorithm>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BatchNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BatchNormParameter param = this->layer_param_.batch_norm_param();
  moving_average_fraction_ = param.moving_average_fraction();
  use_global_stats_ = this->phase_ == TEST;
  if (param.has_use_global_stats())
    use_global_stats_ = param.use_global_stats();
  channels_ = bottom[0]->channels();
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(3);
    this->blobs_[0].reset(new Blob<Dtype>(1, channels_, 1, 1));
    this->blobs_[1].reset(new Blob<Dtype>(1, channels_, 1, 1));
    this->blobs_[2].reset(new Blob<Dtype>(1, 1, 1, 1));
    for (int i = 0; i < 3; ++i) {
      caffe_set(this->blobs_[i]->count(), Dtype(0),
                this->blobs_[i]->mutable_cpu_data());
    }
  }
}

template <typename Dtype>
void BatchNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
  CHECK_EQ(bottom[0]->channels(), channels_);
  mean_.Reshape(1, bottom[0]->channels(), 1, 1);
  variance_.Reshape(1, bottom[0]->channels(), 1, 1);
  temp_.ReshapeLike(*bottom[0]);
  num_sum_.Reshape(bottom[0]->num(), 1, 1, 1);
//   if (bottom[0]->width() != sum_multiplier_.width() ||
//       bottom[0]->height() != sum_multiplier_.height()) {
    sum_multiplier_.Reshape(1, 1, bottom[0]->height(), bottom[0]->width());
    if (sum_multiplier_.cpu_data()[sum_multiplier_.count() - 1] != Dtype(1)) {
      Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
      caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
    }
//   }
//   if (bottom[0]->num() != num_by_chans_.num() ||
//       bottom[0]->channels() != num_by_chans_.channels()) {
    num_by_chans_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
    if (num_sum_.cpu_data()[num_sum_.count() - 1] != Dtype(1)) {
      caffe_set(num_sum_.count(), Dtype(1), num_sum_.mutable_cpu_data());
    }
//   }
}

template <typename Dtype>
void BatchNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int spatial_dim = bottom[0]->height() * bottom[0]->width();
  // XXX this should not be here
  Dtype eps = 1e-5;

  // elementwise square
  // XXX how does this compare to caffe_mul?
  caffe_powx(bottom[0]->count(), bottom_data, Dtype(2),
             temp_.mutable_cpu_data());

  // computes variance using var(X) = E(X^2) - (EX)^2
  // mean of bottom and bottom ** 2
  if (use_global_stats_) {
    const Dtype scale_factor = 1 / this->blobs_[2]->cpu_data()[0];
    caffe_cpu_scale(variance_.count(), scale_factor,
        this->blobs_[0]->cpu_data(), mean_.mutable_cpu_data());
    caffe_cpu_scale(variance_.count(), scale_factor,
        this->blobs_[1]->cpu_data(), variance_.mutable_cpu_data());
  } else {
    caffe_set(mean_.count(), static_cast<Dtype>(0),
              mean_.mutable_cpu_data());
    caffe_set(variance_.count(), static_cast<Dtype>(0),
              variance_.mutable_cpu_data());
    for (int n = 0; n < num; ++n) {
      caffe_cpu_gemv<Dtype>(CblasNoTrans, channels, spatial_dim,
          1. / (num * spatial_dim), bottom_data + bottom[0]->offset(n),
          sum_multiplier_.cpu_data(), 1., mean_.mutable_cpu_data());
      caffe_cpu_gemv<Dtype>(CblasNoTrans, channels, spatial_dim,
          1. / (num * spatial_dim), temp_.cpu_data() + temp_.offset(n),
          sum_multiplier_.cpu_data(), 1., variance_.mutable_cpu_data());
    }
    this->blobs_[2]->mutable_cpu_data()[0] *= moving_average_fraction_;
    this->blobs_[2]->mutable_cpu_data()[0] += 1;
    const Dtype alpha = 1;
    caffe_cpu_axpby(mean_.count(), alpha, mean_.cpu_data(),
        moving_average_fraction_, this->blobs_[0]->mutable_cpu_data());
    caffe_cpu_axpby(variance_.count(), alpha, variance_.cpu_data(),
        moving_average_fraction_, this->blobs_[1]->mutable_cpu_data());
  }
  // elementwise square of mean
  caffe_powx(mean_.count(), mean_.cpu_data(), Dtype(2),
             temp_.mutable_cpu_data());

  caffe_sub(mean_.count(), variance_.cpu_data(), temp_.cpu_data(),
            variance_.mutable_cpu_data());  // variance

  // normalize variance
  caffe_add_scalar(variance_.count(), eps, variance_.mutable_cpu_data());
  caffe_powx(variance_.count(), variance_.cpu_data(), Dtype(0.5),
             variance_.mutable_cpu_data());

  // do mean and variance normalization
  // subtract mean
  // XXX forward should work in place, but backward doesn't, for now
  if (bottom[0] != top[0]) {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
  for (int n = 0; n < num; ++n) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
        1, -1, mean_.cpu_data(), sum_multiplier_.cpu_data(), 1.,
        top_data + top[0]->offset(n));

    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
        1, 1., variance_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
        temp_.mutable_cpu_data() + temp_.offset(n));
  }
  caffe_div(temp_.count(), top_data, temp_.cpu_data(), top_data);
}

template <typename Dtype>
void BatchNormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  CHECK(!use_global_stats_);
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int spatial_dim = bottom[0]->height() * bottom[0]->width();

  caffe_mul(temp_.count(), top_data, top_diff, bottom_diff);
  caffe_set(mean_.count(), static_cast<Dtype>(0), mean_.mutable_cpu_data());
  for (int n = 0; n < num; ++n) {
    caffe_cpu_gemv<Dtype>(CblasNoTrans, channels, spatial_dim, 1.,
        bottom_diff + bottom[0]->offset(n),
          sum_multiplier_.cpu_data(), 1., mean_.mutable_cpu_data());
  }
  for (int n = 0; n < num; ++n) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
        1, 1., mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
        bottom_diff + bottom[0]->offset(n));
  }
  caffe_mul(temp_.count(), top_data, bottom_diff, bottom_diff);

  caffe_set(mean_.count(), static_cast<Dtype>(0), mean_.mutable_cpu_data());
  for (int n = 0; n < num; ++n) {
    caffe_cpu_gemv<Dtype>(CblasNoTrans, channels, spatial_dim, 1.,
        top_diff + top[0]->offset(n), sum_multiplier_.cpu_data(), 1.,
        mean_.mutable_cpu_data());
  }
  for (int n = 0; n < num; ++n) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
        1, 1., mean_.cpu_data(), sum_multiplier_.cpu_data(), 1.,
        bottom_diff + bottom[0]->offset(n));
  }

  caffe_cpu_axpby(temp_.count(), Dtype(1), top_diff,
      Dtype(-1. / (num * spatial_dim)), bottom_diff);

  caffe_div(temp_.count(), bottom_diff, temp_.cpu_data(), bottom_diff);
}


#ifdef CPU_ONLY
STUB_GPU(BatchNormLayer);
#endif

INSTANTIATE_CLASS(BatchNormLayer);
REGISTER_LAYER_CLASS(BatchNorm);
}  // namespace caffe
