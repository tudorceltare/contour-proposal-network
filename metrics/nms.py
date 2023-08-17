import torch

from metrics.iou import intersection_over_union


def nms(
        predictions,
        iou_threshold,
        prob_threshold,
        box_format="corners"
):

    # predictions = [ class, confidence, x1, y1, x2, y2 ]
    assert type(predictions) == list

    bboxes = [box for box in predictions if box[1] > prob_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0] or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format
            )
            < iou_threshold
        ]
        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms
