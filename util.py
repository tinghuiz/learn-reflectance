import numpy as np
import sys
import math
import pickle
import os
import matplotlib 
matplotlib.use('Agg') 
import skimage.io as skio
from skimage.transform import resize
from bell2014.params import IntrinsicParameters
from bell2014.solver import IntrinsicSolver
from bell2014.input import IntrinsicInput
from bell2014 import image_util
from bell2014.krahenbuhl2013.krahenbuhl2013 import DenseCRF

channel_mean = np.array([104, 117, 123])

def get_context_feat(im, net):
    """ Extract the context feature of the whole image """
    ct_size = net.blobs['context'].data.shape[2]
    ct_im = resize(im, (ct_size, ct_size))
    ct_im = ct_im.transpose([2, 0, 1])[::-1] - channel_mean[:, None, None]
    net.blobs['context'].data[0] = ct_im
    net._forward(list(net._layer_names).index('Convolution9'), 
                      list(net._layer_names).index('Convolution12'))
    context_feat = np.copy(net.blobs['Convolution12'].data[0])
    return context_feat

def get_local_feat(im, net):
    """ Extract dense conv4 features """
    h = im.shape[0]
    w = im.shape[1]
    # Half of the local Patch Size (63 - 1)/2. This is needed for padding
    # boundaries
    hps = (net.blobs['input'].data.shape[2] - 1)/2
    im = np.lib.pad(im, ((hps, hps), (hps, hps), (0,0)), 'symmetric')
    rem_y = (im.shape[0]) % 8
    rem_x = (im.shape[1]) % 8
    if rem_y != 0:
        pad_y = 8 - rem_y
    else:
        pad_y = 0
    if rem_x != 0:
        pad_x = 8 - rem_x
    else:
        pad_x = 0
    padim = np.lib.pad(im, ((0, pad_y), (0, pad_x), (0,0)), 'symmetric')
    padim = padim.transpose([2,0,1])[::-1] - channel_mean[:,None,None]
    net.blobs['input'].reshape(1, 3, padim.shape[1], padim.shape[2])
    net.blobs['input'].data[...] = np.reshape(padim, (1, 3, padim.shape[1], padim.shape[2]))
    net._forward(1, len(net.layers)-1)
    feat = np.copy(net.blobs['DenseFeat'].data)
    assert feat.shape[2] == padim.shape[1] and feat.shape[3] == padim.shape[2], \
    'ERROR! REPACK FEATURE SHAPE DOES NOT MATCH!'
    feat = feat[0, :, hps:h+hps, hps:w+hps]
    return feat

def sample_points(im, ns_sample):
    """ Divide the image into grids and sample pixels at the grid points.
    The number of returned samples might be less than what is specified by 
    ns_sample """
    h = im.shape[0]
    w = im.shape[1]
    asp_ratio = float(h)/w
    col_grids = math.floor(math.sqrt(ns_sample/asp_ratio))
    row_grids = math.floor(ns_sample/col_grids)
    col_space = w/col_grids
    row_space = h/row_grids
    sx = range(int(col_space/2), w, int(col_space))
    sy = range(int(row_space/2), h, int(row_space))
    sx, sy = np.meshgrid(sx, sy,sparse=False, indexing='xy')
    sx = sx.flatten()
    sy = sy.flatten()
    return sx, sy

def nystrom(C, si):
    """ 
        Input:
            C - sampled rows of the full comparison matrix
            si - indices of the sampled rows
        Output:
            W_ns - nystrom approximation of the full comparison matrix. 
                The full matrix W = W_ns' * W_ns
    """
    D = C[:,si]
    E,V = np.linalg.eigh(D)
    L = (V[:,E>1e-3]/(np.sqrt(E[None,E>1e-3])))
    W_ns = L.T.dot(C)
    return W_ns

def nystrom_single_image(image_file, feat_net, rref_net, ns_sample=64, sp_sigma=1e-5):
    """ 
        Input:
            image_file - file path to the input image
            feat_net - caffe net for extracting dense local features
            rref_net - caffe net for relative reflectance judgment
            ns_sample - number of rows to sample for nystrom approximation
            sp_sigma - standard deviation for distance weighting of nystrom
                approximation
        Output:
            W_ns - nystrom approximation of the pairwise comparison matrix. 
    """
    im = skio.imread(image_file)
    h = im.shape[0]
    w = im.shape[1]
    npix = h * w
    sx, sy = sample_points(im/255.0, ns_sample)
    ns_sample = len(sx)

    # Fill in the blobs with extracted features
    local_feat = get_local_feat(im, feat_net)
    rref_net.blobs['Convolution4'].reshape(w*h, local_feat.shape[0], 1, 1)
    rref_net.blobs['Convolution8'].reshape(w*h, local_feat.shape[0], 1, 1)
    rref_net.blobs['Convolution8'].data[...] = local_feat.reshape((local_feat.shape[0], -1)).T[:,:,None,None]
    gx, gy = np.meshgrid(range(w), range(h), sparse=False, indexing='xy')
    pix_coords = np.zeros((npix, 2))
    pix_coords[:, 0] = gx.flatten()
    pix_coords[:, 1] = gy.flatten()
    rref_net.blobs['coords'].reshape(w*h, 4, 1, 1)
    rref_net.blobs['coords'].data[:,2,0,0] = gx.flatten()/w
    rref_net.blobs['coords'].data[:,3,0,0] = gy.flatten()/h
    context_feat = get_context_feat(im, rref_net)
    rref_net.blobs['Convolution12'].reshape(w*h, context_feat.shape[0], 1, 1)
    rref_net.blobs['Convolution12'].data[None,:,:,:] = context_feat
    # Sampled rows of the full comparison matrix
    sample_mat = np.zeros((2*ns_sample, h*w*2))
    # Distance weighting matrix
    dist_mat = np.zeros((2*ns_sample, h*w*2))
    # For each sampled pixel, predict its relative reflectance against all other pixels
    for p in range(ns_sample):
        x = sx[p]
        y = sy[p]
        rref_net.blobs['Convolution4'].data[...] = local_feat[:,y,x][None,:,None,None]
        rref_net.blobs['coords'].data[:,0,0,0] = float(x)/w
        rref_net.blobs['coords'].data[:,1,0,0] = float(y)/h
        rref_net._forward(list(rref_net._layer_names).index('ConcatAll'), len(rref_net.layers)-1)
        scores = rref_net.blobs['pred_sm'].data[:,:].reshape((h*w, 3)).T
        # Block ordering: [w=, w>; w<, w=]
        sample_mat[2*p, ::2] = scores[0,:]
        sample_mat[2*p+1, 1::2] = scores[0,:]
        sample_mat[2*p, 1::2] = scores[2,:]
        sample_mat[2*p+1, ::2] = scores[1,:]
        xy = np.array([x, y])
        # Spatial distance weights that are later applied to the sampled comparison matrix.
        # The classifier is less reliable in judging pairs that are spatially far from each
        # other, since the distant pairwise judgment is augmented instead of being labeled 
        # when training the network. Thus, less weight is enforced on distant pairs for
        # nystrom approximation.
        dist = np.sum((xy[None,:] - pix_coords)**2, axis=1)
        dist_mat[2*p, ::2] = dist
        dist_mat[2*p+1, 1::2] = dist
        dist_mat[2*p, 1::2] = dist
        dist_mat[2*p+1, ::2] = dist
    # Do Nystrom
    sample_inds = np.zeros((2*ns_sample))
    sample_inds[::2] = 2*(sx + sy*w)
    sample_inds[1::2] = 2*(sx + sy*w)+1
    sample_inds = sample_inds.astype(int)
    # Symmetrize the sample matrix
    sample_mat[:,sample_inds] = 0.5 * sample_mat[:,sample_inds] + \
        0.5 * sample_mat[:,sample_inds].T
    # Empirically we found that multiplying the sample matrix with a 
    # small value (e.g. 0.01) makes the nystrom slightly more stable. 
    sample_mat = 0.01 * sample_mat * (np.exp(-sp_sigma*dist_mat))
    W_ns = nystrom(sample_mat, sample_inds)
    return W_ns

def decompose_single_image(image_file, feat_net, rref_net, nystrom_file=None, srgb=True,
    save_reflectance_file=None, save_shading_file=None):
    """ 
        Input:
            image_file - file path to the input image
            feat_net - caffe net for extracting dense local features
            rref_net - caffe net for relative reflectance judgment
        Output:
            Estimated reflectance and shading layers of the input image
    """
    if nystrom_file is not None:
        W_ns = pickle.load(open(nystrom_file, 'rb'))
    else:
        W_ns = nystrom_single_image(image_file, feat_net, rref_net)

    input = IntrinsicInput.from_file(image_file, image_is_srgb=srgb)
    params = IntrinsicParameters()
    solver = IntrinsicSolver(input, params, W_ns)
    reflectance, shading, decomposition = solver.solve()
    if save_reflectance_file is not None:
        image_util.save(save_reflectance_file, reflectance, rescale=True, srgb=srgb)
    if save_shading_file is not None:
        image_util.save(save_shading_file, shading, rescale=True, srgb=srgb)
    return reflectance, shading