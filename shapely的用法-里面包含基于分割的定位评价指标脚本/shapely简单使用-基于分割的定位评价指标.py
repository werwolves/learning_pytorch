#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import namedtuple
import numpy as np
import os
import json
from shapely.geometry import Polygon
"""
reference from :
https://github.com/MhLiao/DB/blob/3c32b808d4412680310d3d28eeb6a2d5bf1566c5/concern/icdar2015_eval/detection/iou.py#L8
"""


class DetectionIoUEvaluator(object):
    def __init__(self, iou_constraint=0.5, area_precision_constraint=0.5):
        """
        iou_constraint: 检测box 与 真实 box 的IoU > iou_constraint 就认为是匹配上了
        area_precision_constraint: 检测box 与 要忽略的真实box 的交集 / 检测box的面积 > area_precision_constraint ,则该检测box就 被认为是 不用被关注的 检测box
        """
        self.iou_constraint = iou_constraint
        self.area_precision_constraint = area_precision_constraint

    def evaluate_image(self, gt, pred):
        def get_union(pD, pG):
            return Polygon(pD).union(Polygon(pG)).area

        def get_intersection_over_union(pD, pG):
            return get_intersection(pD, pG) / get_union(pD, pG)

        def get_intersection(pD, pG):
            return Polygon(pD).intersection(Polygon(pG)).area

        # def compute_ap(confList, matchList, numGtCare):
        #     correct = 0
        #     AP = 0
        #     if len(confList) > 0:
        #         confList = np.array(confList)
        #         matchList = np.array(matchList)
        #         sorted_ind = np.argsort(-confList)
        #         confList = confList[sorted_ind]
        #         matchList = matchList[sorted_ind]
        #         for n in range(len(confList)):
        #             match = matchList[n]
        #             if match:
        #                 correct += 1
        #                 AP += float(correct) / (n + 1)

        #         if numGtCare > 0:
        #             AP /= numGtCare

        #     return AP

        perSampleMetrics = {}

        matchedSum = 0

        Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

        numGlobalCareGt = 0
        numGlobalCareDet = 0

        arrGlobalConfidences = []
        arrGlobalMatches = []

        recall = 0
        precision = 0
        hmean = 0

        detMatched = 0

        iouMat = np.empty([1, 1])

        gtPols = []
        detPols = []

        gtPolPoints = []
        detPolPoints = []

        # Array of Ground Truth Polygons' keys marked as don't Care
        gtDontCarePolsNum = []
        # Array of Detected Polygons' matched with a don't Care GT
        detDontCarePolsNum = []

        pairs = []
        detMatchedNums = []

        arrSampleConfidences = []
        arrSampleMatch = []

        evaluationLog = ""

        for n in range(len(gt)):        # 遍历每一个真实的经过人工标注的定位框!!
            points = gt[n]['points']    # [(0, 0), (1, 0), (1, 1), (0, 1), (0, 1), (0, 1), (0, 1)]
            dontCare = gt[n]['ignore']  # 一般情况都为 False， 意思是都需要关注-->Care， 不能忽略-->ignore
            if not Polygon(points).is_valid:
                continue

            gtPol = points
            gtPols.append(gtPol)
            gtPolPoints.append(points) # 并没有什么用途
            if dontCare:
                gtDontCarePolsNum.append(len(gtPols) - 1)  # 记录下 要忽略的gt-box的index位置

        evaluationLog += "GT polygons: " + str(len(gtPols)) + (
            " (" + str(len(gtDontCarePolsNum)) + " don't care)\n"
            if len(gtDontCarePolsNum) > 0 else "\n")
        """
        1.  统计下 检测框中 需要被忽略的的检测框的 index位置 存放在 detDontCarePolsNum 中
        """
        for n in range(len(pred)):     # 遍历 每一个预测框
            points = pred[n]['points']
            if not Polygon(points).is_valid:
                continue

            detPol = points
            detPols.append(detPol)
            detPolPoints.append(points)  # 并没有什么用途
            if len(gtDontCarePolsNum) > 0:
                for dontCarePol in gtDontCarePolsNum:   # 遍历所有的 要忽略的 真实box
                    dontCarePol = gtPols[dontCarePol]
                    intersected_area = get_intersection(dontCarePol, detPol) # 计算 该预测框 与 该要忽略的真实框 的 交集面积
                    pdDimensions = Polygon(detPol).area  # 该检测框的面积
                    precision = 0 if pdDimensions == 0 else intersected_area / pdDimensions  # 交集面积/检测框的面积
                    if (precision > self.area_precision_constraint):  # 只要检测框的 > 0.5 以内的面积 in gt 区域内， 就被认为是有效的区域面积
                        detDontCarePolsNum.append(len(detPols) - 1)   # 记录下 与 要忽略的gt匹配上的  检测框的 index 位置
                        break

        evaluationLog += "DET polygons: " + str(len(detPols)) + (
            " (" + str(len(detDontCarePolsNum)) + " don't care)\n"
            if len(detDontCarePolsNum) > 0 else "\n")
        # ---------------------------------------------------- 以下才是关键的核心代码 --------------------------------------------------------- # 

        if len(gtPols) > 0 and len(detPols) > 0:
            # Calculate IoU and precision matrixs
            outputShape = [len(gtPols), len(detPols)]  # [2, 1] ----> 表示 gt-box 的数量是2， det-box的数量为1
            iouMat = np.empty(outputShape)
            gtRectMat = np.zeros(len(gtPols), np.int8)   # array([0, 0], dtype=int8)
            detRectMat = np.zeros(len(detPols), np.int8) # array([0], dtype=int8)
            """
            1. 计算 gt-box 与 det-box 两两之间 IoU 
            """
            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    pG = gtPols[gtNum]
                    pD = detPols[detNum]
                    iouMat[gtNum, detNum] = get_intersection_over_union(pD, pG)  # 计算所有的 gt-box 与 所有的 det-box 之间  IoU
            """
            1. 统计出所有的匹配上的 gt-box 与 det-box 的 index位置
            """
            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    if gtRectMat[gtNum] == 0 and detRectMat[
                            detNum] == 0 and gtNum not in gtDontCarePolsNum and detNum not in detDontCarePolsNum:
                        if iouMat[gtNum, detNum] > self.iou_constraint:
                            gtRectMat[gtNum] = 1    # 表示 该 gt已经被 匹配上了
                            detRectMat[detNum] = 1  # 表示 该 det已经被 匹配上了
                            detMatched += 1
                            pairs.append({'gt': gtNum, 'det': detNum})
                            detMatchedNums.append(detNum)
                            evaluationLog += "Match GT #" + \
                                             str(gtNum) + " with Det #" + str(detNum) + "\n"

        numGtCare = (len(gtPols) - len(gtDontCarePolsNum))
        numDetCare = (len(detPols) - len(detDontCarePolsNum))
        # if numGtCare == 0:
        #     recall = float(1)
        #     precision = float(0) if numDetCare > 0 else float(1)
        # else:
        #     recall = float(detMatched) / numGtCare  # 计算召回率
        #     precision = 0 if numDetCare == 0 else float(detMatched) / numDetCare  # 计算准确率

        # hmean = 0 if (precision + recall) == 0 else 2.0 * \
        #                                             precision * recall / (precision + recall)

        matchedSum += detMatched
        numGlobalCareGt += numGtCare
        numGlobalCareDet += numDetCare

        perSampleMetrics = {
            'gtCare': numGtCare,
            'detCare': numDetCare,
            'detMatched': detMatched,
        }
        return perSampleMetrics

    def combine_results(self, results):
        numGlobalCareGt = 0
        numGlobalCareDet = 0
        matchedSum = 0
        for result in results:
            numGlobalCareGt += result['gtCare']
            numGlobalCareDet += result['detCare']
            matchedSum += result['detMatched']

        methodRecall = 0 if numGlobalCareGt == 0 else float(
            matchedSum) / numGlobalCareGt
        methodPrecision = 0 if numGlobalCareDet == 0 else float(
            matchedSum) / numGlobalCareDet
        methodHmean = 0 if methodRecall + methodPrecision == 0 else 2 * \
                                                                    methodRecall * methodPrecision / (
                                                                            methodRecall + methodPrecision)
        methodMetrics = {
            'precision': methodPrecision,
            'recall': methodRecall,
            'hmean': methodHmean
        }

        return methodMetrics


if __name__ == '__main__':
    evaluator = DetectionIoUEvaluator()
    gt_dir = r'E:\RFID\datasets\book_loc\sh_east_lib\test\test-gt'
    pred_dir = r'E:\RFID\datasets\book_loc\sh_east_lib\test\test-pred-2025-3-4'
    
    gts,preds = [],[]
    for i in os.listdir(gt_dir):
        if i.endswith('.json'):
            gt_path = os.path.join(gt_dir,i)
            pred_path = os.path.join(pred_dir,i)
            with open(gt_path,'r', encoding='utf-8') as f:
                gt_data = json.load(f)
            with open(pred_path,'r', encoding='utf-8') as f:
                pred_data = json.load(f)
            
            new_sample_gt = []
            for gt_info in gt_data['shapes']:
                box_dict = {}
                box_dict["points"] = gt_info["points"]
                box_dict["text"] = gt_info["label"]
                box_dict["ignore"] = False
                new_sample_gt.append(box_dict)
            gts.append(new_sample_gt)
            new_sample_pred = []
            for pred_info in pred_data['shapes']:
                box_dict = {}
                box_dict["points"] = pred_info["points"]
                box_dict["text"] = pred_info["label"]
                box_dict["ignore"] = False
                new_sample_pred.append(box_dict)
            preds.append(new_sample_pred)   
    
    

    results = []
    for gt, pred in zip(gts, preds):
        results.append(evaluator.evaluate_image(gt, pred))
    metrics = evaluator.combine_results(results)
    print(metrics)
