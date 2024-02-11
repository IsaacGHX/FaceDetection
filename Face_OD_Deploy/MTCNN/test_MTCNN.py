import cv2, random
import numpy as np
from tqdm import tqdm
from read_annots import get_data
from infer_path import infer_image


def calculate_iou(box1, box2):
    """
    计算两个边界框的IoU（交并比）。
    box1和box2的格式应该为：[x1, y1, x2, y2]，其中(x1, y1)是左上角，(x2, y2)是右下角。
    """
    # print(box1, box2)
    x1, y1, x2, y2, _ = box1
    x1_, y1_, x2_, y2_ = box2

    # 计算交集的左上角和右下角坐标
    intersection_x1 = max(x1, x1_)
    intersection_y1 = max(y1, y1_)
    intersection_x2 = min(x2, x2_)
    intersection_y2 = min(y2, y2_)

    # 计算交集的面积
    intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)

    # 计算并集的面积
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)
    union_area = box1_area + box2_area - intersection_area

    # 计算IoU
    iou = intersection_area / union_area
    return iou


def calculate_iou_for_all_boxes(boxes1, boxes2):
    """
    计算两个边界框列表中所有边界框之间的IoU。
    boxes1和boxes2应该是形如[[x1, y1, x2, y2], ...]的列表。
    返回一个IoU矩阵，其中每个元素[i, j]表示boxes1中第i个框与boxes2中第j个框的IoU。
    """
    dim = max(len(boxes1), len(boxes2))
    iou_matrix = np.zeros((dim, dim))

    for i, box1 in enumerate(boxes1):
        for j, box2 in enumerate(boxes2):
            iou_matrix[i, j] = calculate_iou(box1, box2)

    return iou_matrix


def calculate_ap(gt_boxes, predictions, iou_matrix, iou_threshold=0.5):
    # print(gt_boxes)
    # print(predictions)
    # print(iou_matrix)
    # 将预测结果按照置信度降序排序
    # predictions = sorted(predictions, key=lambda x: x[-1], reverse=True)

    # 初始化变量
    true_positives = np.zeros(len(gt_boxes))
    false_positives = np.zeros(len(gt_boxes))
    false_negatives = len(gt_boxes) - 1

    precision = []
    recall = []

    # 遍历预测结果
    for i, prediction in enumerate(gt_boxes):
        if len(gt_boxes) == 0:
            break

        confidence = prediction[-1]  # 置信度

        # 查找与当前预测框具有最大IoU的真实边界框
        max_iou_index = np.argmax(iou_matrix[i])
        max_iou = np.max(iou_matrix[i])

        if max_iou >= iou_threshold:  # 设置IoU阈值
            true_positives[i] = 1
        else:
            false_positives[i] = 1

        # false_negatives -= 1  # 减少未匹配的真实边界框数

    # 计算精确度和召回率
    precision = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_positives))
    recall = np.sum(true_positives) / (np.sum(true_positives) + false_negatives)
    return precision, recall


def compute_ap(precision, recall):
    """
    计算平均精确度 (Average Precision, AP)
    precision: 一个包含精确度值的列表
    recall: 一个包含召回率值的列表
    """
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


if __name__ == '__main__':
    data_path = "D:/datasets/WIDER_val/WIDER_val/images"
    label_path = "D:/datasets/wider_face_split/wider_face_val_bbx_gt.txt"

    img_paths, gt_boxes, gt_labels = get_data(data_path, label_path)

    iou = 0
    precision = 0
    num_img = len(img_paths)
    print("Total face num: ", num_img)

    num_images = len(img_paths)
    AVG_IOUS = 0
    AVG_PRECISION = 0
    P = []
    R = []

    for i in tqdm(range(10)):
        # 从数据集中随机选择一张图片
        random_index = random.randint(0, num_images - 1)
        selected_img_path = img_paths[random_index]
        selected_gt_boxes = gt_boxes[random_index]
        # print(gt_boxes[0][0], gt_labels[0][0])

        # selected_img_path = img_paths[i]
        # selected_gt_boxes = gt_boxes[i]
        converted_gt_boxes = []

        for box in selected_gt_boxes:
            xlt, ylt, w, h = box
            x1 = xlt
            y1 = ylt
            x2 = xlt + w
            y2 = ylt + h
            converted_gt_boxes.append([x1, y1, x2, y2])

        # 读取选定的图片
        image = cv2.imread(selected_img_path)

        # 在这里运行你的目标检测模型以获取检测框
        onet_boxes, _ = infer_image(selected_img_path)
        # print(onet_boxes)
        if onet_boxes is None:
            print("No face detected by onet.")
            continue

        print("Detected faces: ", len(onet_boxes), "GT faces: ", len(converted_gt_boxes))

        # for box in onet_boxes:
        #     x1, y1, x2, y2, confidence = box
        #     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.putText(image, f"{confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #
        # for box in converted_gt_boxes:
        #     x1, y1, x2, y2 = box
        #     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        #     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #     cv2.putText(image, f"{1.00}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 计算IoU并显示
        ious = calculate_iou_for_all_boxes(onet_boxes, converted_gt_boxes)
        average_iou = np.sum(ious) / ious.shape[0]
        print("Single Img IOU: ", average_iou)
        # cv2.putText(image, f"Average IoU: {average_iou:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 计算AP并显示
        # print(selected_gt_boxes, onet_boxes)
        # cv2.putText(image, f"P: {precision:.2f}, R: {recall:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
        #             2)
        precision, recall = calculate_ap(selected_gt_boxes, onet_boxes, ious)
        P.append(precision)
        R.append(recall)
        print("Single Img P, R: ", precision, recall)

        ## 显示图片和信息
        # cv2.imshow("Test Result", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        AVG_IOUS += average_iou

    AVG_IOUS /= num_img
    ap = compute_ap(P, R)
    print(f"AP:{ap}")
    print(AVG_IOUS)
