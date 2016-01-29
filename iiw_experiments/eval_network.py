import numpy as np
import argparse
import sys
sys.path.append('../')
from util import *
from bell2014 import image_util
from bell2014 import judgements
import skimage 
import json

parser = argparse.ArgumentParser(
	description="""Run decomposition and evaluation on the test split of 
	the Intrinsic Images in the Wild (IIW) dataset""")
parser.add_argument(
	'iiw_dir', type=str, help='directory of IIW data')
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
rref_net = Net('../net/rref.prototxt', '../net/rref.caffemodel', 1)
test_ids = np.load('iiw_test_ids.npy').astype(int)

# Half of the local Patch Size (63 - 1)/2. This is needed for padding, 
# so that the network can evaluate on points close to image boundaries.
hps = 31
# Size of the context image
context_size = 150
# Accumulate the weights and errors for computing WHDR
error_sum = 0.0
weight_sum = 0.0
for t in range(len(test_ids)):
	print('Evaluating: %d/%d' % (t+1, len(test_ids)))
	id = test_ids[t]
	image_file = args.iiw_dir + str(id) + '.png'
	im = skimage.io.imread(image_file)
	context_im = skimage.transform.resize(im, (context_size, context_size))
	context_im = context_im.transpose([2, 0, 1])[::-1] - channel_mean[:, None, None]
	padim = np.lib.pad(im, ((hps, hps), (hps, hps), (0,0)), 'symmetric')
	padim = padim.transpose([2,0,1])[::-1] - channel_mean[:,None,None]
	anno_file = args.iiw_dir + str(id) + '.json'
	judgements = json.load(open(anno_file))
	points = judgements['intrinsic_points']
	comparisons = judgements['intrinsic_comparisons']
	id_to_points = {p['id']: p for p in points}
	for c in comparisons:
		point1 = id_to_points[c['point1']]
		point2 = id_to_points[c['point2']]
		darker = c['darker']
		weight = c['darker_score']
		x1 = np.round(point1['x'] * im.shape[1]) + hps
		y1 = np.round(point1['y'] * im.shape[0]) + hps
		x2 = np.round(point2['x'] * im.shape[1]) + hps
		y2 = np.round(point2['y'] * im.shape[0]) + hps
		rref_net.blobs['local1'].data[0] = \
			padim[:, y1 - hps:y1 + hps + 1, x1 - hps:x1 + hps + 1]
		rref_net.blobs['local2'].data[0] = \
			padim[:, y2 - hps:y2 + hps + 1, x2 - hps:x2 + hps + 1]
		rref_net.blobs['context'].data[0] = context_im
		rref_net.blobs['coords'].data[0][0] = float(x1)/padim.shape[2]
		rref_net.blobs['coords'].data[0][1] = float(y1)/padim.shape[1]
		rref_net.blobs['coords'].data[0][2] = float(x2)/padim.shape[2]
		rref_net.blobs['coords'].data[0][3] = float(y2)/padim.shape[1]
		rref_net._forward(4, len(rref_net.layers)-1)
		pred = np.argmax(rref_net.blobs['pred'].data[0])
		if pred == 0:
			alg_darker = 'E'
		if pred == 1:
			alg_darker = '1'
		if pred == 2:
			alg_darker = '2'
		if darker != alg_darker:
			error_sum += weight
		weight_sum += weight
print('WHDR = %f' % (error_sum / weight_sum))