import numpy as np
from tqdm import tqdm

def getImgDetMetrics(detections,gt, th_dist=5):

    tp_img, fp_img, fn_img = 0,0,0

    if len(gt) > 0:
        tp, fp, fn = np.zeros(4),np.zeros(4),np.zeros(4)
        for j in range(4):
            np_gt = len(gt[j])
            np_dt = len(detections[j])
            # print('len',np_gt,np_dt)
            for gt_point in gt[j]:
                # print('gt_point',gt_point)
                tp_flag = False
                for det_point in detections[j]:
                    # print('det_point',det_point)
                    vector = np.array([[det_point[0]] - gt_point[0],[det_point[1]] - gt_point[1]])
                    dist = np.linalg.norm(vector)
                    if dist < th_dist:
                            tp_flag = True
                if not tp_flag:
                    fn[j] += 1
                        
            for det_point in detections[j]:
                for gt_point in gt[j]:
                    vector = np.array([[det_point[0]] - gt_point[0],[det_point[1]] - gt_point[1]])
                    dist = np.linalg.norm(vector)
                    if dist < th_dist:
                            tp_flag = True
                            tp[j] += 1
                if not tp_flag:
                    fp[j] += 1
            # tp[j] = np_dt - fp[j] - fn[j]
            # if tp[j] < 0:
            #     tp[j] = 0
        
        fp_img = np.sum(fp)
        fn_img = np.sum(fn)
        tp_img = np.sum(tp)

        prec_img = tp_img / (tp_img + fp_img)
        rec_img = tp_img / (tp_img + fn_img)

    else:
        print('gt = 0')
        if len(detections) > 0:
            fp_img += len(detections)
            prec_img = 0
            rec_img = 0

    if (tp_img + fp_img + fn_img) > 4:
        print('Más de una puerta')

    # print(tp_img, fp_img, fn_img)
    # print(prec_img, rec_img)

    return tp_img, fp_img, fn_img, prec_img, rec_img

def getDetectionMetrics(detections, gt, th_dist = 5):

    tp_sum, fp_sum, fn_sum = 0,0,0
    precision_list, recall_list = [],[]

    for i in tqdm(range(len(detections))):

        tp_img, fp_img, fn_img, precision_img, recall_img = getImgDetMetrics(detections[i], gt[i], th_dist)

        tp_sum += tp_img
        fp_sum += fp_img
        fn_sum += fn_img

        precision_list.append(precision_img)
        recall_list.append(recall_img)

    prec_tot = tp_sum / (tp_sum + fp_sum)
    rec_tot = tp_sum / (tp_sum + fn_sum)

    precision_mean = np.mean(precision_list)
    recall_mean = np.mean(recall_list)

    print('m_x_sum',tp_sum, fp_sum, fn_sum)
    print('TOTAL',prec_tot, rec_tot)
    print('MEDIA',precision_mean, recall_mean)
        
    return

def getEstimationMetrics(detections, gt):


    return

