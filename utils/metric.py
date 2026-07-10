import  numpy as np
import torch.nn as nn
import torch
from skimage import measure
import  numpy

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

                pred_regions = list(
                    measure.regionprops(measure.label(predits, connectivity=2))
                )
                label_regions = list(
                    measure.regionprops(measure.label(labelss, connectivity=2))
                )
                self.target[iBin] += len(label_regions)

                unmatched_predictions = list(pred_regions)
                matched_targets = 0
                for label_region in label_regions:
                    if not unmatched_predictions:
                        break
                    centroid_label = np.asarray(label_region.centroid)
                    distances = [
                        np.linalg.norm(
                            np.asarray(pred_region.centroid) - centroid_label
                        )
                        for pred_region in unmatched_predictions
                    ]
                    nearest_index = int(np.argmin(distances))
                    if distances[nearest_index] < 3:
                        unmatched_predictions.pop(nearest_index)
                        matched_targets += 1

                self.PD[iBin] += matched_targets
                self.FA[iBin] += sum(
                    region.area for region in unmatched_predictions
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
