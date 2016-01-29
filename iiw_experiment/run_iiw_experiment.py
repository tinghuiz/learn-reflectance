import numpy as np
import argparse
import sys
sys.path.append('../')
from util import *
from bell2014 import image_util
from bell2014 import judgements

parser = argparse.ArgumentParser(
	description="""Run decomposition and evaluation on the test split of 
	the Intrinsic Images in the Wild (IIW) dataset""")
parser.add_argument(
	'iiw_dir', type=str, help='directory of IIW data')
parser.add_argument(
	'output_dir', type=str, help='directory for storing decomposition outputs')
parser.add_argument(
	'-caffe_dir', type=str, default='../caffe/', help='Caffe directory')
parser.add_argument(
	'-gpu', type=int, default=0, help='GPU ID')
args = parser.parse_args()

sys.path.append(args.caffe_dir + '/python')
# Silence Caffe
from os import environ
environ['GLOG_minloglevel'] = '2'
from caffe import *
set_mode_gpu()
set_device(args.gpu)
feat_net = Net('../net/feat.prototxt', '../net/rref.caffemodel', 1)
rref_net = Net('../net/rref.prototxt', '../net/rref.caffemodel', 1)
test_ids = np.load('iiw_test_ids.npy').astype(int)

for t in range(len(test_ids)):
	print('Decomposing: %d/%d' % (t+1, len(test_ids)))
	id = test_ids[t]
	image_file = args.iiw_dir + str(id) + '.png'
	output_r_file = args.output_dir + str(id) + '-r.png'
	output_s_file = args.output_dir + str(id) + '-s.png'
	decompose_single_image(image_file, feat_net, rref_net, srgb=True,
		save_reflectance_file=output_r_file,
		save_shading_file=output_s_file)

# Evaluation
whdr_sum = .0
for id in test_ids:
	json_file = args.iiw_dir + str(id) + '.json'
	evaluator = judgements.HumanReflectanceJudgements.from_file(json_file)
	ref = image_util.load(args.output_dir + str(id) + '-r.png')
	whdr_sum += evaluator.compute_whdr(ref)
whdr_mean = whdr_sum/len(test_ids)
print('Mean WHDR = %.4f' % whdr_mean)