import os
import sys
import torch

sys.path.insert(0, os.path.abspath('../ml-cvnets'))
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from typing import Optional
from cvnets import get_model
from data import create_eval_loader
from torch import Tensor, nn
from options.opts import get_eval_arguments
from utils import logger
from utils.common_utils import device_setup, create_directories
from utils.ddp_utils import is_master, distributed_init
from torch.cuda.amp import autocast
import numpy as np
from utils.tensor_utils import to_numpy, tensor_size_from_opts
from torch.nn import functional as F
from time import time_ns

epsilon = 1.0e-2
np.random.seed(97)
eps = 1e-6

output_onnx = "mobilevit_s_det.onnx"


def predict_and_save(opts,
                     input_tensor: Tensor,
                     model: nn.Module,
                     input_arr: Optional[np.ndarray] = None,
                     device: Optional = torch.device("cpu"),
                     mixed_precision_training: Optional[bool] = False,
                     is_validation: Optional[bool] = False,
                     file_name: Optional[str] = None,
                     output_stride: Optional[int] = 32, # Default is 32 because ImageNet models have 5 downsampling stages (2^5 = 32)
                     orig_h: Optional[int] = None,
                     orig_w: Optional[int] = None
                     ):

    if input_arr is None and not is_validation:
        input_arr = (
            to_numpy(input_tensor) # convert to numpy
            .squeeze(0) # remove batch dimension
        )

    curr_height, curr_width = input_tensor.shape[2:]

    # check if dimensions are multiple of output_stride, otherwise, we get dimension mismatch errors.
    # if not, then resize them
    new_h = (curr_height // output_stride) * output_stride
    new_w = (curr_width // output_stride) * output_stride

    if new_h != curr_height or new_w != curr_width:
        # resize the input image, so that we do not get dimension mismatch errors in the forward pass
        input_tensor = F.interpolate(input=input_tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)

    # move data to device
    input_tensor = input_tensor.to(device)

    with autocast(enabled=mixed_precision_training):
        # prediction
        prediction = model.predict(input_tensor, is_scaling=False)

    # convert tensors to boxes
    boxes = prediction.boxes.cpu().numpy()
    labels = prediction.labels.cpu().numpy()
    scores = prediction.scores.cpu().numpy()

    if orig_w is None:
        assert orig_h is None
        orig_h, orig_w = input_arr.shape[:2]
    elif orig_h is None:
        assert orig_w is None
        orig_h, orig_w = input_arr.shape[:2]

    assert orig_h is not None and orig_w is not None
    boxes[..., 0::2] = boxes[..., 0::2] * orig_w
    boxes[..., 1::2] = boxes[..., 1::2] * orig_h
    boxes[..., 0::2] = np.clip(a_min=0, a_max=orig_w, a=boxes[..., 0::2])
    boxes[..., 1::2] = np.clip(a_min=0, a_max=orig_h, a=boxes[..., 1::2])

    return boxes, labels, scores

def main(opts, **kwargs):
    global output_onnx
    # CUDA_VISIBLE_DEVICES=0 cvnets-eval --common.config-file results_imagenet1k_mobilevit_s/config.yaml --common.results-loc results_imagenet1k_mobilevit_s --model.classification.pretrained results_imagenet1k_mobilevit_s/checkpoint_ema_best.pt

    # set-up the model
    device = torch.device('cpu')
    # print(help(opts))
    for k in opts.__dict__:
        # print(k)
        if k.startswith('dataset'):
            print(k, getattr(opts, k))

    model = get_model(opts)
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load("../ml-cvnets/pretrained_models/ssd_mobilevit_s_320.pt", map_location='cpu'))
    
    # Generate input tensor with random values
    input_tensor = torch.rand(1, 3, 320, 320)

    from functools import partial
    # print(model)
    ori_forward = model.forward
    model.forward = model.export_onnx_forward
    print("pytorch input shape: ", input_tensor.shape, input_tensor.dtype)

    # Export torch model to ONNX
    print("Exporting ONNX model {}".format(output_onnx))
    torch.onnx.export(model, input_tensor, output_onnx,
        opset_version = 13,
        do_constant_folding = True,
        input_names = ["input"],
        output_names = ["output"],
        dynamic_axes = {
            "input": {
                0: "b",
            },
            "output": {
                0: "b",
            },
        },
        verbose=False
    )

    model.forward = ori_forward
    print("Exporting ONNX model {} Success!".format(output_onnx))
    print('output_onnx:', output_onnx)
    
    
    # test Speed
    model.eval()
    device = torch.device('cuda')
    model.to(device)
    eval_dataloader = create_eval_loader(opts)
    # warm up
    for i, batch in enumerate(eval_dataloader):
        input_img, target_label = batch['image'], batch['label']
        model(input_img.to(device))
        if i >= 20:
            break

    start_time = time_ns()
    counter = 0
    with torch.no_grad():
        predictions = {}
        for img_idx, batch in enumerate(eval_dataloader):
            
            input_img, target_label = batch['image'], batch['label']

            batch_size = input_img.shape[0]
            assert batch_size == 1, "We recommend to run segmentation evaluation with a batch size of 1"
            counter += batch_size
            orig_w = batch["im_width"][0].item()
            orig_h = batch["im_height"][0].item()

            boxes, labels, scores = predict_and_save(
                opts=opts,
                input_tensor=input_img,
                model=model,
                device=device,
                mixed_precision_training=False,
                is_validation=True,
                orig_w=orig_w,
                orig_h=orig_h
            )

            predictions[img_idx] = (img_idx, boxes, labels, scores)
            if img_idx > 1000:
                break
        
    end_time = time_ns()
    fps = counter / ((end_time - start_time) / 1000 / 1000 / 1000)
    print('FPS:', fps)


def distributed_worker(i, main, opts, kwargs):
    setattr(opts, "dev.device_id", i)
    if torch.cuda.is_available():
        torch.cuda.set_device(i)

    ddp_rank = getattr(opts, "ddp.rank", None)
    if ddp_rank is None:  # torch.multiprocessing.spawn
        ddp_rank = kwargs.get('start_rank', 0) + i
        setattr(opts, "ddp.rank", ddp_rank)

    node_rank = distributed_init(opts)
    setattr(opts, "ddp.rank", node_rank)
    main(opts, **kwargs)

from options.opts import get_detection_eval_arguments

def main_worker(**kwargs):
    opts = get_detection_eval_arguments()
    print(opts)

    # device set-up
    opts = device_setup(opts)
    
    dataset_name = getattr(opts, "dataset.name", "imagenet")
    if dataset_name.find("coco") > -1:
        # replace model specific datasets (e.g., coco_ssd) with general COCO dataset
        setattr(opts, "dataset.name", "coco")

    num_gpus = 1
    # adjust the batch size
    train_bsize = getattr(opts, "dataset.train_batch_size0", 32) * max(1, num_gpus)
    val_bsize = getattr(opts, "dataset.val_batch_size0", 32) * max(1, num_gpus)
    setattr(opts, "dataset.train_batch_size0", train_bsize)
    setattr(opts, "dataset.val_batch_size0", val_bsize)
    setattr(opts, "dev.device_id", None)
    main(opts=opts, **kwargs)

def main_worker_detection(**kwargs):
    from engine.eval_detection import main_detection_evaluation
    main_detection_evaluation(**kwargs)

if __name__ == "__main__":
    main_worker()
