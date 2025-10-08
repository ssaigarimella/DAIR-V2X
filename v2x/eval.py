import sys
import os
import os.path as osp

sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

import argparse
import logging

logger = logging.getLogger(__name__)

from tqdm import tqdm
import numpy as np

from v2x_utils import range2box, id_to_str, Evaluator
from config import add_arguments
from dataset import SUPPROTED_DATASETS
from dataset.dataset_utils import save_pkl
from models import SUPPROTED_MODELS
from models.model_utils import Channel


def _dbg(name, d):
    import numpy as np
    b = d.get("boxes_3d", np.zeros((0,8,3)))
    s = d.get("scores_3d", np.zeros((0,)))
    l = d.get("labels_3d", np.zeros((0,)))
    print(f"[DBG] {name}: num={len(s)}  max_score={float(s.max()) if s.size else -1}  "
          f"min_score={float(s.min()) if s.size else -1}  "
          f"unique_labels={np.unique(l) if l.size else []}")

def eval_vic(args, dataset, model, evaluator):
    idx = -1
    for VICFrame, label, filt in tqdm(dataset):
        idx += 1
        # if idx % 10 != 0:
        #     continue
        if 'spd' in args.dataset:
            veh_id = VICFrame.vehicle_frame().get("frame_id")
        else:
            try:
                veh_id = dataset.data[idx][0]["vehicle_pointcloud_path"].split("/")[-1].replace(".pcd", "")
            except Exception:
                veh_id = VICFrame["vehicle_pointcloud_path"].split("/")[-1].replace(".pcd", "")

        pred = model(
            VICFrame,
            filt,
            None if not hasattr(dataset, "prev_inf_frame") else dataset.prev_inf_frame,
        )

        _dbg("pred", pred)
        
        evaluator.add_frame(pred, label)
        pipe.flush()
        pred["label"] = label["boxes_3d"]
        pred["veh_id"] = veh_id
        save_pkl(pred, osp.join(args.output, "result", pred["veh_id"] + ".pkl"))

    evaluator.print_ap("3d")
    evaluator.print_ap("bev")
    print("Average Communication Cost = %.2lf Bytes" % (pipe.average_bytes()))


def eval_single(args, dataset, model, evaluator):
    for frame, label, filt in tqdm(dataset):
        pred = model(frame, filt)
        if args.sensortype == "camera":
            evaluator.add_frame(pred, label["camera"])
        elif args.sensortype == "lidar":
            evaluator.add_frame(pred, label["lidar"])
        save_pkl({"boxes_3d": label["lidar"]["boxes_3d"]}, osp.join(args.output, "result", frame.id["camera"] + ".pkl"))

    evaluator.print_ap("3d")
    evaluator.print_ap("bev")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler="resolve")
    add_arguments(parser)
    args, _ = parser.parse_known_args()
    # add model-specific arguments
    SUPPROTED_MODELS[args.model].add_arguments(parser)
    args = parser.parse_args()

    if args.quiet:
        level = logging.ERROR
    elif args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=level,
    )

    extended_range = range2box(np.array(args.extended_range))
    logger.info("loading dataset")

    dataset = SUPPROTED_DATASETS[args.dataset](
        args.input,
        args,
        split=args.split,
        sensortype=args.sensortype,
        extended_range=extended_range,
        val_data_path=args.val_data_path
    )

    logger.info("loading evaluator")
    # before creating Evaluator
    norm = lambda s: (s or "").strip().lower()
    if args.pred_classes:
        args.pred_classes = [norm(c) for c in args.pred_classes]
    evaluator = Evaluator(args.pred_classes)

    logger.info("loading model")
    if args.eval_single:
        model = SUPPROTED_MODELS[args.model](args)
        eval_single(args, dataset, model, evaluator)
    else:
        pipe = Channel()
        model = SUPPROTED_MODELS[args.model](args, pipe)
        ### Patch for FFNet evaluation ###
        if args.model =='feature_flow':
            model.model.data_root = args.input
            model.model.test_mode = args.test_mode
        #############################
        eval_vic(args, dataset, model, evaluator)
