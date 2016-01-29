#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void TransformationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const TransformationParameter& param = this->layer_param_.transform_param();
  CHECK(!param.has_mean_file()) << "Mean file not supported";
  CHECK(!param.has_force_color()) << "Force color not supported";
  CHECK(!param.has_force_gray()) << "Force gray not supported";

  mirror_ = param.mirror();
  rotate_ = param.rotate();
  synchronized_ = param.synchronized();
  crop_size_ = param.crop_size();
  scale_ = param.scale();
  min_scale_ = param.min_scale();
  max_scale_ = param.max_scale();
}

template <typename Dtype>
void TransformationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const TransformationParameter& param = this->layer_param_.transform_param();
  std::vector<int> Sbot = bottom[0]->shape();
  CHECK_GE(Sbot.size(), 3) << "At least 3 dimensional image required";

  // Load the parameters
  if (mean_value_.count() != Sbot[Sbot.size()-3]) {
    mean_value_.Reshape(std::vector<int>(1, Sbot[Sbot.size()-3]));
    if (param.mean_value_size()) {
      if (param.mean_value_size() > 1)
        CHECK_GE(param.mean_value_size(), Sbot[Sbot.size()-3]) <<
          "mean value and image channels dont match";
      Dtype * pm = mean_value_.mutable_cpu_data();
      for (int i = 0; i < mean_value_.count(); i++)
        pm[i] = param.mean_value(i % param.mean_value_size());
    } else {
      Dtype * pm = mean_value_.mutable_cpu_data();
      for (int i = 0; i < mean_value_.count(); i++)
        pm[i] = 0;
    }
  }

  std::vector<int> Stop = Sbot;
  if (crop_size_ > 0) {
    Stop[Stop.size()-1] = crop_size_;
    Stop[Stop.size()-2] = crop_size_;
  }
  for (int i = 0; i < bottom.size(); i++) {
    for (int j = 0; j < Sbot.size(); j++)
      CHECK_EQ(Sbot[j], bottom[i]->shape()[j]) << "Bottoms need the same shape";
    top[i]->Reshape(Stop);
  }
}
template <typename Dtype>
std::vector<typename TransformationLayer<Dtype>::Affine2D>
TransformationLayer<Dtype>::generate(int N, int W, int H, int Wo, int Ho) {
  std::vector<Affine2D> r;
  for (int i = 0; i < N; i++) {
    if (synchronized_ && i) {
      r.push_back(r.back());
    } else {
      Dtype rot = 0, scale = 1, sx = 0, sy = 0;
      if (rotate_) caffe_rng_uniform<Dtype>(1, -M_PI, M_PI, &rot);
      Dtype sin_rot = sin(rot);
      Dtype cos_rot = cos(rot);
      Dtype rot_w = Wo*fabs(cos_rot) + Ho*fabs(sin_rot);
      Dtype rot_h = Wo*fabs(sin_rot) + Ho*fabs(cos_rot);
      Dtype max_scale = std::min(max_scale_, std::min(H / rot_h, W / rot_w));
      Dtype min_scale = std::min(min_scale_, max_scale);
      caffe_rng_uniform<Dtype>(1, min_scale, max_scale, &scale);
      Dtype scl_w = std::min(rot_w * scale, Dtype(W));
      caffe_rng_uniform<Dtype>(1, -(W-scl_w)/2, (W-scl_w)/2, &sx);
      Dtype scl_h = std::min(rot_h * scale, Dtype(H));
      caffe_rng_uniform<Dtype>(1, -(H-scl_h)/2, (H-scl_h)/2, &sy);
      Affine2D T0(1, 0, 0, 1, -0.5*Wo, -0.5*Ho);
      Affine2D RS(scale * cos_rot, -scale * sin_rot,
                  scale * sin_rot,  scale * cos_rot, 0, 0);
      if (mirror_ && caffe_rng_rand()&1) {
        RS.a00_ = -RS.a00_;
        RS.a10_ = -RS.a10_;
      }
      Affine2D T1(1, 0, 0, 1, 0.5*W + sx, 0.5*H + sy);
      r.push_back(T1*RS*T0);
    }
  }
  return r;
}
template <typename Dtype>
void TransformationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Comptute the transformation parameters
  std::vector<int> Sbot = bottom[0]->shape(), Stop = top[0]->shape();
  const int W  = Stop[Stop.size()-1], H  = Stop[Stop.size()-2];
  const int bW = Sbot[Sbot.size()-1], bH = Sbot[Sbot.size()-2];
  const int C  = Stop[Stop.size()-3];
  const int N  = top[0]->count() / (W*H*C);
  std::vector<Affine2D> affine = generate(N, bW, bH, W, H);

  // Get the mean
  const Dtype * mean = mean_value_.cpu_data();

  // Transform
  for (int i = 0; i < bottom.size(); i++) {
    const Dtype * pBot = bottom[i]->cpu_data();
    Dtype * pTop = top[i]->mutable_cpu_data();
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int n = 0; n < N; n++)
      for (int c = 0, k = n*C*W*H; c < C; c++)
        for (int y = 0; y < H; y++)
          for (int x = 0; x < W; x++, k++) {
            const float xx = affine[n].x(x, y), yy = affine[n].y(x, y);
            if (0 <= yy && yy < bH && 0 <= xx && xx < bW) {
              // Linear interpolation
              int x0 = xx, y0 = yy, x1 = xx+1, y1 = yy+1;
              if (x1 > bW-1) x1 = bW-1;
              if (y1 > bH-1) y1 = bH-1;
              const Dtype wx = x1 - xx, wy = y1 - yy;
              Dtype v = (wx)   * (wy)   * pBot[(n*C+c)*bW*bH + y0*bW + x0] +
                        (1-wx) * (wy)   * pBot[(n*C+c)*bW*bH + y0*bW + x1] +
                        (wx)   * (1-wy) * pBot[(n*C+c)*bW*bH + y1*bW + x0] +
                        (1-wx) * (1-wy) * pBot[(n*C+c)*bW*bH + y1*bW + x1];
              pTop[k] = scale_ * (v - mean[c]);
            } else {
              pTop[k] = 0;
            }
          }
  }
}

template <typename Dtype>
void TransformationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(TransformationLayer);
#endif

INSTANTIATE_CLASS(TransformationLayer);
REGISTER_LAYER_CLASS(Transformation);

}  // namespace caffe
