learn-reflectance
===========================

This implements the intrinsic image decomposition algorithm described in "Learning Data-driven Reflectance Priors for Intrinsic Image Decomposition, T. Zhou, P. Krähenbühl and A. A. Efros, ICCV 2015". If you use our code for academic purposes, please consider citing:

    @inproceedings{zhou2015learning,
	  	title={Learning data-driven reflectance priors for intrinsic image decomposition},
	  	author={Zhou, Tinghui and Kr\"ahenb\"uhl, Philipp and Efros, Alexei A},
	  	booktitle={Proceedings of the IEEE International Conference on Computer Vision},
	  	pages={3469--3477},
	  	year={2015}
	}

#### Dependencies & Compilation:
* The codebase was tested with Python 3, but it should be straightforward to adapt it to Python 2 if needed. 
* A copy of Caffe used to produce the results in our ICCV paper is included for completeness. It was based on the 'repack' branch (https://github.com/philkr/caffe/tree/repack) forked from the official BVLC Caffe repository (https://github.com/BVLC/caffe). The major reason of using this branch is its support of 'repack' layer, which allows efficient extraction of dense, per-pixel feature map despite >1 stride/pooling size of the Conv layers.
* The decomposition code is built upon the IIW codebase (https://github.com/seanbell/intrinsic), which is modified to support our learned reflectance priors. Please also cite "Intrinsic Images in the Wild, Bell et al., SIGGRAPH 2014" if you use the decomposition code.
* To compile the code from the top level directory, you need to install Caffe:
```bash
cd caffe
mkdir build
cd build
cmake ../ -Dpython_version=3
make -j8
```
and the DenseCRF package:
```bash
cd bell2014/krahenbuhl2013/
make
```
(If you encounter problems with cmake, it might be that Ubuntu 14.04 (or older version) does not recognize Python 3.4 directly. You will need to add 3.4 to the line with 'set(_PYTHON3_VERSIONS ...)' for the system's 'FindPythonInterp.cmake' and 'FindPythonLibs.cmake' files.)

#### Running the demo:
Once the above installation is complete, running
```bash
python3 demo.py
```
should output the decomposition of 'sample.png' to 'sample-r.png' and 'sample-s.png', respectively. This corresponds to the teaser example (Fig. 1) in our paper.

#### Evaluation on the IIW test split:
The original IIW dataset does not provide a training/test split. Instead, we used the split provided by "Learning Lightness from Human Judgement on Relative Reflectance, Narihira et al., CVPR 2015" for our experiments. To evaluate the network performance on relative reflectance judgment:
```base
cd iiw_experiments/
python3 eval_network.py [Path to IIW directory]
```
To evaluate the decomposition performance using our learned reflectance prior:
```base
cd iiw_experiments/
python3 eval_decomposition.py [Path to IIW directory] [Output directory]
```