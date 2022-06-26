import argparse
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO


def eval():
    parser = argparse.ArgumentParser(description='Training arguments', add_help=True)
    parser.add_argument('--ano_filepath', type=str, default='', required=True, help="Path of annotations json path")
    parser.add_argument('--result_filepath', type=str, default='', required=True, help="Path of result json path")
    config = parser.parse_args()


    ann_file = config.ano_filepath

    coco = COCO(ann_file)

    iou_type = 'bbox'
    json_result_file = config.result_filepath


    coco_dt = coco.loadRes(json_result_file)
    coco_eval = COCOeval(coco, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == "__main__":
    eval()
