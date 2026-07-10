from dataclasses import dataclass
from typing import Any

import  numpy as np
import torch.nn as nn
import torch
from skimage import measure
import  numpy


@dataclass(frozen=True)
class ComponentMatchResult:
    """Connected-component matching under the repository's PD/FA rule."""

    prediction_label_map: np.ndarray
    target_label_map: np.ndarray
    prediction_regions: tuple[Any, ...]
    target_regions: tuple[Any, ...]
    matches: tuple[tuple[int, int, float], ...]
    unmatched_prediction_indices: tuple[int, ...]
    unmatched_target_indices: tuple[int, ...]


def match_connected_components(
    prediction,
    target,
    *,
    max_centroid_distance=3.0,
    connectivity=2,
):
    """Match binary components exactly as ``PD_FA.update`` historically did.

    Targets are processed in ``regionprops`` order.  Each target can consume
    its nearest still-unmatched prediction only when the centroid distance is
    strictly smaller than ``max_centroid_distance``.
    """

    prediction_array = np.asarray(prediction)
    target_array = np.asarray(target)
    if prediction_array.ndim != 2 or target_array.ndim != 2:
        raise ValueError("prediction and target must both be 2-D arrays")
    if prediction_array.shape != target_array.shape:
        raise ValueError("prediction and target shapes must match")

    prediction_label_map = measure.label(
        prediction_array.astype(bool), connectivity=connectivity
    )
    target_label_map = measure.label(
        target_array.astype(bool), connectivity=connectivity
    )
    prediction_regions = tuple(measure.regionprops(prediction_label_map))
    target_regions = tuple(measure.regionprops(target_label_map))

    unmatched_predictions = list(range(len(prediction_regions)))
    unmatched_targets = []
    matches = []
    for target_index, target_region in enumerate(target_regions):
        if not unmatched_predictions:
            unmatched_targets.extend(range(target_index, len(target_regions)))
            break
        target_centroid = np.asarray(target_region.centroid)
        distances = [
            np.linalg.norm(
                np.asarray(prediction_regions[index].centroid) - target_centroid
            )
            for index in unmatched_predictions
        ]
        nearest_position = int(np.argmin(distances))
        nearest_prediction = unmatched_predictions[nearest_position]
        nearest_distance = float(distances[nearest_position])
        if nearest_distance < float(max_centroid_distance):
            matches.append(
                (target_index, nearest_prediction, nearest_distance)
            )
            unmatched_predictions.pop(nearest_position)
        else:
            unmatched_targets.append(target_index)

    return ComponentMatchResult(
        prediction_label_map=prediction_label_map,
        target_label_map=target_label_map,
        prediction_regions=prediction_regions,
        target_regions=target_regions,
        matches=tuple(matches),
        unmatched_prediction_indices=tuple(unmatched_predictions),
        unmatched_target_indices=tuple(unmatched_targets),
    )

class ROCMetric():
    """Computes pixAcc and mIoU metric scores
    """
    def __init__(self, nclass, bins):  #bin的意义实际上是确定ROC曲线上的threshold取多少个离散值
        super(ROCMetric, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.tp_arr = np.zeros(self.bins+1)
        self.pos_arr = np.zeros(self.bins+1)
        self.fp_arr = np.zeros(self.bins+1)
        self.neg_arr = np.zeros(self.bins+1)
        self.class_pos=np.zeros(self.bins+1)
        # self.reset()

    def update(self, preds, labels):
        for iBin in range(self.bins+1):
            score_thresh = (iBin + 0.0) / self.bins
            # print(iBin, "-th, score_thresh: ", score_thresh)
            i_tp, i_pos, i_fp, i_neg,i_class_pos = cal_tp_pos_fp_neg(preds, labels, self.nclass,score_thresh)
            self.tp_arr[iBin]   += i_tp
            self.pos_arr[iBin]  += i_pos
            self.fp_arr[iBin]   += i_fp
            self.neg_arr[iBin]  += i_neg
            self.class_pos[iBin]+=i_class_pos

    def get(self):

        tp_rates    = self.tp_arr / (self.pos_arr + 0.001)
        fp_rates    = self.fp_arr / (self.neg_arr + 0.001)

        recall      = self.tp_arr / (self.pos_arr   + 0.001)
        precision   = self.tp_arr / (self.class_pos + 0.001)


        return tp_rates, fp_rates, recall, precision

    def reset(self):
        self.tp_arr   = np.zeros([self.bins+1])
        self.pos_arr  = np.zeros([self.bins+1])
        self.fp_arr   = np.zeros([self.bins+1])
        self.neg_arr  = np.zeros([self.bins+1])
        self.class_pos= np.zeros([self.bins+1])



class PD_FA():
    def __init__(self, nclass, bins, size):
        super(PD_FA, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.image_area_total = []
        self.image_area_match = []
        self.FA = np.zeros(self.bins+1)
        self.PD = np.zeros(self.bins + 1)
        self.target= np.zeros(self.bins + 1)
        self.size = size
        # Index zero remains the official operating point used by main.py.
        # The remaining entries form a probability sweep without duplicating
        # the 0.5 operating point.
        sweep = [i / self.bins for i in range(self.bins + 1)]
        self.thresholds = [0.5] + [x for x in sweep if abs(x - 0.5) > 1e-12]
        self.num_images = 0

    def update(self, preds, labels):
        probabilities = torch.sigmoid(preds.detach()).cpu().numpy()
        label_array = labels.detach().cpu().numpy()
        if probabilities.ndim == 3:
            probabilities = probabilities[:, None, :, :]
        if label_array.ndim == 3:
            label_array = label_array[:, None, :, :]

        batch_size = probabilities.shape[0]
        self.num_images += batch_size
        for iBin, score_thresh in enumerate(self.thresholds):
            for batch_index in range(batch_size):
                predits = (probabilities[batch_index, 0] > score_thresh).astype(
                    "int64"
                )
                labelss = (label_array[batch_index, 0] > 0.5).astype("int64")

                component_match = match_connected_components(predits, labelss)
                self.target[iBin] += len(component_match.target_regions)
                self.PD[iBin] += len(component_match.matches)
                self.FA[iBin] += sum(
                    component_match.prediction_regions[index].area
                    for index in component_match.unmatched_prediction_indices
                )

    def get(self,img_num=None):
        del img_num
        image_count = max(1, self.num_images)
        Final_FA =  self.FA / ((self.size*self.size) * image_count)
        Final_PD = np.divide(
            self.PD,
            self.target,
            out=np.zeros_like(self.PD),
            where=self.target != 0,
        )

        return Final_FA,Final_PD


    def reset(self):
        self.FA  = np.zeros([self.bins+1])
        self.PD  = np.zeros([self.bins+1])
        self.target = np.zeros([self.bins+1])
        self.num_images = 0

class mIoU():

    def __init__(self, nclass):
        super(mIoU, self).__init__()
        self.nclass = nclass
        self.reset()

    def update(self, preds, labels):
        # print('come_ininin')

        correct, labeled = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels, self.nclass)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union


    def get(self):

        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return pixAcc, mIoU

    def reset(self):

        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0




def cal_tp_pos_fp_neg(output, target, nclass, score_thresh):

    predict = (torch.sigmoid(output) > score_thresh).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    intersection = predict * ((predict == target).float())

    tp = intersection.sum()
    fp = (predict * ((predict != target).float())).sum()
    tn = ((1 - predict) * ((predict == target).float())).sum()
    fn = (((predict != target).float()) * (1 - predict)).sum()
    pos = tp + fn
    neg = fp + tn
    class_pos= tp+fp

    return tp, pos, fp, neg, class_pos

def batch_pix_accuracy(output, target):

    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict = (output > 0).float()
    pixel_labeled = (target > 0).float().sum()
    pixel_correct = (((predict == target).float())*((target > 0)).float()).sum()



    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):

    mini = 1
    maxi = 1
    nbins = 1
    predict = (output > 0).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * ((predict == target).float())

    area_inter, _  = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))
    area_pred,  _  = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))
    area_lab,   _  = np.histogram(target.cpu(), bins=nbins, range=(mini, maxi))
    area_union     = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union
