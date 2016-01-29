import numpy as np
import argparse
import os
from util import *
from bell2014 import image_util

parser = argparse.ArgumentParser(
	description="Demo code for the full pipeline of our ICCV'15 paper")
parser.add_argument(
	'-input', type=str, default='sample.png', help='input image')
parser.add_argument(
	'-srgb', type=bool, default=True, help='Input image is sRGB or not')
parser.add_argument(
	'-gpu', type=int, default=0, help='GPU ID')

args = parser.parse_args()

caffe_dir = 'caffe/'
sys.path.append(caffe_dir + '/python')
# Silence Caffe
from os import environ
environ['GLOG_minloglevel'] = '2'
from caffe import *
set_mode_gpu()
set_device(args.gpu)
feat_net = Net('net/feat.prototxt', 'net/rref.caffemodel', 1)
rref_net = Net('net/rref.prototxt', 'net/rref.caffemodel', 1)

r, s = decompose_single_image(args.input, feat_net, rref_net, srgb=args.srgb)
base, _ = os.path.splitext(args.input)
rfile = base + '-r.png'
sfile = base + '-s.png'
image_util.save(rfile, r, rescale=True, srgb=args.srgb)
image_util.save(sfile, s, rescale=True, srgb=args.srgb)