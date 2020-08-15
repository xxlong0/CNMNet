import numpy as np


# https://github.com/davisvideochallenge/davis/blob/master/python/lib/davis/measures/jaccard.py
def eval_iou(annotation,segmentation):
    """ Compute region similarity as the Jaccard Index.

    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.

    Return:
        jaccard (float): region similarity

    """

    annotation   = annotation.astype(np.bool)
    segmentation = segmentation.astype(np.bool)

    if np.isclose(np.sum(annotation),0) and np.isclose(np.sum(segmentation),0):
        return 1
    else:
        return np.sum((annotation & segmentation)) / \
                np.sum((annotation | segmentation),dtype=np.float32)


# https://github.com/art-programmer/PlaneNet/blob/master/utils.py#L2115
def eval_plane_prediction(predSegmentations, gtSegmentations, predDepths, gtDepths, threshold=0.5):
    predNumPlanes = len(np.unique(predSegmentations)) - 1
    gtNumPlanes = len(np.unique(gtSegmentations)) - 1

    if len(gtSegmentations.shape) == 2:
        gtSegmentations = (np.expand_dims(gtSegmentations, -1) == np.arange(gtNumPlanes)).astype(np.float32)
    if len(predSegmentations.shape) == 2:
        predSegmentations = (np.expand_dims(predSegmentations, -1) == np.arange(predNumPlanes)).astype(np.float32)

    planeAreas = gtSegmentations.sum(axis=(0, 1))
    intersectionMask = np.expand_dims(gtSegmentations, -1) * np.expand_dims(predSegmentations, 2) > 0.5

    # depthDiffs = np.expand_dims(gtDepths, -1) - np.expand_dims(predDepths, 2)
    depthDiffs = gtDepths - predDepths
    depthDiffs = depthDiffs[:, :, np.newaxis, np.newaxis]

    intersection = np.sum((intersectionMask).astype(np.float32), axis=(0, 1))

    planeDiffs = np.abs(depthDiffs * intersectionMask).sum(axis=(0, 1)) / np.maximum(intersection, 1e-4)

    planeDiffs[intersection < 1e-4] = 1

    union = np.sum(((np.expand_dims(gtSegmentations, -1) + np.expand_dims(predSegmentations, 2)) > 0.5).astype(np.float32), axis=(0, 1))
    planeIOUs = intersection / np.maximum(union, 1e-4)

    numPredictions = int(predSegmentations.max(axis=(0, 1)).sum())

    numPixels = planeAreas.sum()

    IOUMask = (planeIOUs > threshold).astype(np.float32)
    minDiff = np.min(planeDiffs * IOUMask + 1000000 * (1 - IOUMask), axis=1)
    stride = 0.05
    pixelRecalls = []
    planeStatistics = []
    for step in range(int(0.61 / stride + 1)):
        diff = step * stride
        pixelRecalls.append(np.minimum((intersection * (planeDiffs <= diff).astype(np.float32) * IOUMask).sum(1),
                                       planeAreas).sum() / numPixels)
        planeStatistics.append(((minDiff <= diff).sum(), gtNumPlanes, numPredictions))

    return pixelRecalls, planeStatistics


#https://github.com/art-programmer/PlaneNet
def evaluateDepths(predDepths, gtDepths, validMasks, planeMasks=True, printInfo=True):
    masks = np.logical_and(np.logical_and(validMasks, planeMasks), gtDepths > 1e-4)

    numPixels = float(masks.sum())

    rmse = np.sqrt((pow(predDepths - gtDepths, 2) * masks).sum() / numPixels)
    rmse_log = np.sqrt((pow(np.log(predDepths) - np.log(gtDepths), 2) * masks).sum() / numPixels)
    log10 = (np.abs(
        np.log10(np.maximum(predDepths, 1e-4)) - np.log10(np.maximum(gtDepths, 1e-4))) * masks).sum() / numPixels
    rel = (np.abs(predDepths - gtDepths) / np.maximum(gtDepths, 1e-4) * masks).sum() / numPixels
    rel_sqr = (pow(predDepths - gtDepths, 2) / np.maximum(gtDepths, 1e-4) * masks).sum() / numPixels
    deltas = np.maximum(predDepths / np.maximum(gtDepths, 1e-4), gtDepths / np.maximum(predDepths, 1e-4)) + (
            1 - masks.astype(np.float32)) * 10000
    accuracy_1 = (deltas < 1.25).sum() / numPixels
    accuracy_2 = (deltas < pow(1.25, 2)).sum() / numPixels
    accuracy_3 = (deltas < pow(1.25, 3)).sum() / numPixels
    recall = float(masks.sum()) / validMasks.sum()
    if printInfo:
        print(('evaluate', rel, rel_sqr, log10, rmse, rmse_log, accuracy_1, accuracy_2, accuracy_3, recall))
        pass
    return rel, rel_sqr, log10, rmse, rmse_log, accuracy_1, accuracy_2, accuracy_3, recall


def eval_plane_and_pixel_recall_normal(segmentation, gt_segmentation, param, gt_param, threshold=0.5):
    """
    :param segmentation: label map for plane segmentation [h, w] where 20 indicate non-planar
    :param gt_segmentation: ground truth label for plane segmentation where 20 indicate non-planar
    :param threshold: value for iou
    :return: percentage of correctly predicted ground truth planes correct plane
    """
    depth_threshold_list = np.linspace(0.0, 30, 13)

    # both prediction and ground truth segmentation contains non-planar region which indicated by label 20
    # so we minus one
    plane_num = len(np.unique(segmentation)) - 1
    gt_plane_num = len(np.unique(gt_segmentation)) - 1

    # 13: 0:0.05:0.6
    plane_recall = np.zeros((gt_plane_num, len(depth_threshold_list)))
    pixel_recall = np.zeros((gt_plane_num, len(depth_threshold_list)))

    plane_area = 0.0

    gt_param = gt_param.reshape(20, 3)

    # check if plane is correctly predict
    for i in range(gt_plane_num):
        gt_plane = gt_segmentation == i
        plane_area += np.sum(gt_plane)

        for j in range(plane_num):
            pred_plane = segmentation == j
            iou = eval_iou(gt_plane, pred_plane)

            if iou > threshold:
                # mean degree difference over overlap region:
                gt_p = gt_param[i]
                pred_p = param[j]

                n_gt_p = gt_p / np.linalg.norm(gt_p)
                n_pred_p = pred_p / np.linalg.norm(pred_p)

                angle = np.arccos(np.clip(np.dot(n_gt_p, n_pred_p), -1.0, 1.0))
                degree = np.degrees(angle)
                depth_diff = degree

                # compare with threshold difference
                plane_recall[i] = (depth_diff < depth_threshold_list).astype(np.float32)
                pixel_recall[i] = (depth_diff < depth_threshold_list).astype(np.float32) * \
                      (np.sum(gt_plane * pred_plane))
                break

    pixel_recall = np.sum(pixel_recall, axis=0).reshape(1, -1) / plane_area

    return plane_recall, pixel_recall


def compute_valid_depth_mask(d1, d2=None, min_thred=0.3, max_thred=8.0):
    """Computes the mask of valid values for one or two depth maps

    Returns a valid mask that only selects values that are valid depth value
    in both depth maps (if d2 is given).
    Valid depth values are >0 and finite.
    """
    if d2 is None:
        valid_mask = (d1 < max_thred) & (d1 > min_thred) & (np.isfinite(d1))

    else:
        valid_mask = (d1 < max_thred) & (d2 < max_thred)
        valid_mask[valid_mask] = (d1[valid_mask] > min_thred) & (d2[valid_mask] > min_thred)
    return valid_mask


def l1(depth1, depth2):
    """
    Computes the l1 errors between the two depth maps.
    Takes preprocessed depths (no nans, infs and non-positive values)

    depth1:  one depth map
    depth2:  another depth map

    Returns:
        L1(log)

    """
    assert (np.all(np.isfinite(depth1) & np.isfinite(depth2) & (depth1 > 0) & (depth2 > 0)))
    diff = depth1 - depth2
    num_pixels = float(diff.size)

    if num_pixels == 0:
        return np.nan
    else:
        return np.sum(np.absolute(diff)) / num_pixels


def l1_inverse(depth1, depth2):
    """
    Computes the l1 errors between inverses of two depth maps.
    Takes preprocessed depths (no nans, infs and non-positive values)

    depth1:  one depth map
    depth2:  another depth map

    Returns:
        L1(log)

    """
    assert (np.all(np.isfinite(depth1) & np.isfinite(depth2) & (depth1 > 0) & (depth2 > 0)))
    diff = np.reciprocal(depth1) - np.reciprocal(depth2)
    num_pixels = float(diff.size)

    if num_pixels == 0:
        return np.nan
    else:
        return np.sum(np.absolute(diff)) / num_pixels


def rmse_log(depth1, depth2):
    """
    Computes the root min square errors between the logs of two depth maps.
    Takes preprocessed depths (no nans, infs and non-positive values)

    depth1:  one depth map
    depth2:  another depth map

    Returns:
        RMSE(log)

    """
    assert (np.all(np.isfinite(depth1) & np.isfinite(depth2) & (depth1 > 0) & (depth2 > 0)))
    log_diff = np.log(depth1) - np.log(depth2)
    num_pixels = float(log_diff.size)

    if num_pixels == 0:
        return np.nan
    else:
        return np.sqrt(np.sum(np.square(log_diff)) / num_pixels)


def rmse(depth1, depth2):
    """
    Computes the root min square errors between the two depth maps.
    Takes preprocessed depths (no nans, infs and non-positive values)

    depth1:  one depth map
    depth2:  another depth map

    Returns:
        RMSE(log)

    """
    assert (np.all(np.isfinite(depth1) & np.isfinite(depth2) & (depth1 > 0) & (depth2 > 0)))
    diff = depth1 - depth2
    num_pixels = float(diff.size)

    if num_pixels == 0:
        return np.nan
    else:
        return np.sqrt(np.sum(np.square(diff)) / num_pixels)


def scale_invariant(depth1, depth2):
    """
    Computes the scale invariant loss based on differences of logs of depth maps.
    Takes preprocessed depths (no nans, infs and non-positive values)

    depth1:  one depth map
    depth2:  another depth map

    Returns:
        scale_invariant_distance

    """
    # sqrt(Eq. 3)
    assert (np.all(np.isfinite(depth1) & np.isfinite(depth2) & (depth1 > 0) & (depth2 > 0)))
    log_diff = np.log(depth1) - np.log(depth2)
    num_pixels = float(log_diff.size)

    if num_pixels == 0:
        return np.nan
    else:
        return np.sqrt(np.sum(np.square(log_diff)) / num_pixels - np.square(np.sum(log_diff)) / np.square(num_pixels))


def abs_relative(depth_pred, depth_gt):
    """
    Computes relative absolute distance.
    Takes preprocessed depths (no nans, infs and non-positive values)

    depth_pred:  depth map prediction
    depth_gt:    depth map ground truth

    Returns:
        abs_relative_distance

    """
    assert (np.all(np.isfinite(depth_pred) & np.isfinite(depth_gt) & (depth_pred > 0) & (depth_gt > 0)))
    diff = depth_pred - depth_gt
    num_pixels = float(diff.size)

    if num_pixels == 0:
        return np.nan
    else:
        return np.sum(np.absolute(diff) / depth_gt) / num_pixels


def avg_log10(depth1, depth2):
    """
    Computes average log_10 error (Liu, Neural Fields, 2015).
    Takes preprocessed depths (no nans, infs and non-positive values)

    depth1:  one depth map
    depth2:  another depth map

    Returns:
        abs_relative_distance

    """
    assert (np.all(np.isfinite(depth1) & np.isfinite(depth2) & (depth1 > 0) & (depth2 > 0)))
    log_diff = np.log10(depth1) - np.log10(depth2)
    num_pixels = float(log_diff.size)

    if num_pixels == 0:
        return np.nan
    else:
        return np.sum(np.absolute(log_diff)) / num_pixels


def sq_relative(depth_pred, depth_gt):
    """
    Computes relative squared distance.
    Takes preprocessed depths (no nans, infs and non-positive values)

    depth_pred:  depth map prediction
    depth_gt:    depth map ground truth

    Returns:
        squared_relative_distance

    """
    assert (np.all(np.isfinite(depth_pred) & np.isfinite(depth_gt) & (depth_pred > 0) & (depth_gt > 0)))
    diff = depth_pred - depth_gt
    num_pixels = float(diff.size)

    if num_pixels == 0:
        return np.nan
    else:
        return np.sum(np.square(diff) / depth_gt) / num_pixels


def ratio_threshold(depth1, depth2, threshold):
    """
    Computes the percentage of pixels for which the ratio of the two depth maps is less than a given threshold.
    Takes preprocessed depths (no nans, infs and non-positive values)

    depth1:  one depth map
    depth2:  another depth map

    Returns:
        percentage of pixels with ratio less than the threshold

    """
    assert (threshold > 0.)
    assert (np.all(np.isfinite(depth1) & np.isfinite(depth2) & (depth1 > 0) & (depth2 > 0)))
    log_diff = np.log(depth1) - np.log(depth2)
    num_pixels = float(log_diff.size)

    if num_pixels == 0:
        return np.nan
    else:
        return float(np.sum(np.absolute(log_diff) < np.log(threshold))) / num_pixels


def compute_errors(depth_pred, depth_gt, distances_to_compute=None):
    """
    Computes different distance measures between two depth maps.

    depth_pred:           depth map prediction
    depth_gt:             depth map ground truth
    distances_to_compute: which distances to compute

    Returns:
        a dictionary with computed distances, and the number of valid pixels

    """

    valid_mask = compute_valid_depth_mask(depth_pred, depth_gt)
    depth_pred = depth_pred[valid_mask]
    depth_gt = depth_gt[valid_mask]
    num_valid = np.sum(valid_mask)

    if distances_to_compute is None:
        distances_to_compute = ['l1',
                                'l1_inverse',
                                'scale_invariant',
                                'abs_relative',
                                'sq_relative',
                                'avg_log10',
                                'rmse_log',
                                'rmse',
                                'ratio_threshold_1.25',
                                'ratio_threshold_1.5625',
                                'ratio_threshold_1.953125']

    results = {'num_valid': num_valid}
    for dist in distances_to_compute:
        if dist.startswith('ratio_threshold'):
            threshold = float(dist.split('_')[-1])
            results[dist] = ratio_threshold(depth_pred, depth_gt, threshold)
        else:
            results[dist] = globals()[dist](depth_pred, depth_gt)

    return results


def compute_depth_scale_factor(depth1, depth2, depth_scaling='abs'):
    """
    Computes the scale factor for depth1 to minimize the least squares error to depth2
    """

    assert (np.all(np.isfinite(depth1) & np.isfinite(depth2) & (depth1 > 0) & (depth2 > 0)))

    if depth_scaling == 'abs':
        # minimize MSE on depth
        d1d1 = np.multiply(depth1, depth1)
        d1d2 = np.multiply(depth1, depth2)
        mask = compute_valid_depth_mask(d1d2)
        sum_d1d1 = np.sum(d1d1[mask])
        sum_d1d2 = np.sum(d1d2[mask])
        if sum_d1d1 > 0.:
            scale = sum_d1d2 / sum_d1d1
        else:
            print('compute_depth_scale_factor: Norm=0 during scaling')
            scale = 1.
    elif depth_scaling == 'log':
        # minimize MSE on log depth
        log_diff = np.log(depth2) - np.log(depth1)
        scale = np.exp(np.mean(log_diff))
    elif depth_scaling == 'inv':
        # minimize MSE on inverse depth
        d1d1 = np.multiply(np.reciprocal(depth1), np.reciprocal(depth1))
        d1d2 = np.multiply(np.reciprocal(depth1), np.reciprocal(depth2))
        mask = compute_valid_depth_mask(d1d2)
        sum_d1d1 = np.sum(d1d1[mask])
        sum_d1d2 = np.sum(d1d2[mask])
        if sum_d1d1 > 0.:
            scale = np.reciprocal(sum_d1d2 / sum_d1d1)
        else:
            print('compute_depth_scale_factor: Norm=0 during scaling')
            scale = 1.
    else:
        raise Exception('Unknown depth scaling method')

    return scale


def evaluate_depth(translation_gt, depth_gt_in, depth_pred_in,
                   distances_to_compute=None, inverse_gt=True, inverse_pred=True,
                   depth_scaling='abs', depth_pred_max=np.inf):
    """
    Computes different error measures for the inverse depth map without scaling and with scaling.

    translation_gt: 1d numpy array with [tx,ty,tz]
        The translation that corresponds to the ground truth depth

    depth_gt: 2d numpy array
        This is the ground truth depth

    depth_pred: depth prediction being evaluated

    distances_to_compute: which distances to compute

    returns (err, err_after_scaling)
        errs is the dictionary of errors without optimally scaling the prediction

        errs_pred_scaled is the dictionary of errors after minimizing
        the least squares error by scaling the prediction
    """

    valid_mask = compute_valid_depth_mask(depth_pred_in, depth_gt_in)
    depth_pred = depth_pred_in[valid_mask]
    depth_gt = depth_gt_in[valid_mask]
    if inverse_gt:
        depth_gt = np.reciprocal(depth_gt)
    if inverse_pred:
        depth_pred = np.reciprocal(depth_pred)

    # if depth_pred_max < np.inf:
    # depth_pred[depth_pred>depth_pred_max] = depth_pred_max

    # we need to scale the ground truth depth if the translation is not normalized
    translation_norm = np.sqrt(translation_gt.dot(translation_gt))
    scale_gt_depth = not np.isclose(1.0, translation_norm)
    if scale_gt_depth:
        depth_gt_scaled = depth_gt / translation_norm
    else:
        depth_gt_scaled = depth_gt

    errs = compute_errors(depth_pred, depth_gt_scaled, distances_to_compute)

    # minimize the least squares error and compute the errors again
    scale = compute_depth_scale_factor(depth_pred, depth_gt_scaled, depth_scaling=depth_scaling)
    depth_pred_scaled = depth_pred * scale

    errs_pred_scaled = compute_errors(depth_pred_scaled, depth_gt_scaled, distances_to_compute)

    return errs, errs_pred_scaled