import numpy as np
from model_ctpn.lib.fast_rcnn.config import cfg
from model_ctpn.lib.utils.cython_nms import nms; cython_nms.install()

def nms(dets, thresh):
    if dets.shape[0] == 0:
        return []
    if cfg.USE_GPU_NMS:
        return gpu_nms(dets, thresh, device_id=cfg.GPU_ID)
    else:
        return cython_nms(dets, thresh)