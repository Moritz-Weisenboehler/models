import tensorflow as tf

from object_detection.core import standard_fields
from object_detection.utils import object_detection_evaluation

from sig.metrics.pick_evaluation import PickEvaluation


class PickEvaluator(object_detection_evaluation.DetectionEvaluator):
  """Class to evaluate pick metrics."""

  def __init__(self,
         categories: list,
         category_weights: dict = {},
         max_picks: int = 100,
         score_threshold: float = 0,
         iou_threshold: float = 0):
    """Constructor.

    Args:
    categories: A list of dicts, each of which has the following keys -
      'id': (required) an integer id uniquely identifying this category.
      'name': (required) string representing category name e.g., 'cat', 'dog'.
    category_weights: A dictionary mapping category ids to weights e.g., {1: 3, 2: 1}
      Positive weights are considered as objects to accept
      Negative weights are considered as objects to reject
    max_picks: Maximum picks (detections per class) per image to consider
    score_threshold: Min score for detections to pass clean step.
    iou_threshold: Min IoU for detections to pass clean step.
    """
    super(PickEvaluator, self).__init__(categories)

    classes = {}

    for category in categories:
      class_id = category['id']

      classes[class_id] = {
        'name': category['name'],
        'weight': (category_weights[class_id]
               if class_id in category_weights else 1)}

    self._evaluation = PickEvaluation(classes,
                      max_picks,
                      score_threshold,
                      iou_threshold)

    self.clear()

  def add_single_ground_truth_image_info(self, image_id, groundtruth_dict):
    """Adds groundtruth for a single image to be used for evaluation.

    Args:
    image_id: A unique string/integer identifier for the image.
    groundtruth_dict: A dictionary of groundtruth numpy arrays required for
      evaluations.
    """
    input_fields = standard_fields.InputDataFields

    self._groundtruth_nums.append(
      len(groundtruth_dict[input_fields.groundtruth_classes][0].numpy()))
    self._groundtruth_classes.append(
      groundtruth_dict[input_fields.groundtruth_classes][0].numpy())
    self._groundtruth_boxes.append(
      groundtruth_dict[input_fields.groundtruth_boxes][0].numpy())

  def add_single_detected_image_info(self, image_id, detections_dict):
    """Adds detections for a single image to be used for evaluation.

    Args:
    image_id: A unique string/integer identifier for the image.
    detections_dict: A dictionary of detection numpy arrays required for
      evaluation.
    """
    detection_fields = standard_fields.DetectionResultFields

    self._detection_nums.append(
      detections_dict[detection_fields.num_detections][0].numpy())
    self._detection_classes.append(
      detections_dict[detection_fields.detection_classes][0].numpy())
    self._detection_boxes.append(
      detections_dict[detection_fields.detection_boxes][0].numpy())
    self._detection_scores.append(
      detections_dict[detection_fields.detection_scores][0].numpy())

  def _pack_evaluation_data(self):

    evaluation_data = []

    for image_id in self._image_ids:
      image_index = self._image_ids[image_id]

      groundtruth = []

      for label_index in range(self._groundtruth_nums[image_index]):
        class_id = self._groundtruth_classes[image_index][label_index]

        groundtruth.append({
          "class_id": class_id,
          "bounding_box": self._groundtruth_boxes[image_index][label_index]})

      detections = []

      for label_index in range(self._detection_nums[image_index]):
        class_id = self._detection_classes[image_index][label_index]

        detections.append({
          "class_id": class_id,
          "bounding_box": self._detection_boxes[image_index][label_index],
          "score": self._detection_scores[image_index][label_index]})

      evaluation_data.append({"labels":  groundtruth,
                  "detections": detections})

    return evaluation_data

  def evaluate(self):
    """Evaluates detections and returns a dictionary of metrics."""
    tf.python.logging.info('Performing evaluation on %d images.',
                 len(self._image_ids))

    evaluation_data = self._pack_evaluation_data()
    evaluation_results = self._evaluation.calculate(evaluation_data)[0]

    metric_results = {}
    
    for class_id in evaluation_results:
        class_name = f'C{class_id}'

        for category in self._categories:
            if category['id'] == class_id:
                class_name = category['name']

        for key, value in evaluation_results[class_id].items():
            metric_results[f'PickMetric/{class_name}/{key}'] = value

    return metric_results

  def clear(self):
    """Clears the state to prepare for a fresh evaluation."""

    self._image_ids = {}

    self._groundtruth_nums = []
    self._groundtruth_classes = []
    self._groundtruth_boxes = []

    self._detection_nums = []
    self._detection_classes = []
    self._detection_boxes = []
    self._detection_scores = []

  def add_eval_dict(self, eval_dict: dict):

    input_fields = standard_fields.InputDataFields

    image_id = eval_dict[input_fields.key][0].numpy()

    if image_id in self._image_ids:
      tf.python.logging.warning(f'Ignoring ground truth with image id {image_id} '
                    'since it was previously added')
      return

    self._image_ids[image_id] = len(self._image_ids.keys())

    self.add_single_ground_truth_image_info(image_id, eval_dict)
    self.add_single_detected_image_info(image_id, eval_dict)
