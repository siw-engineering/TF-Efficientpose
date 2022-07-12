import argparse
import os
import sys
import tensorflow as tf
from model import build_EfficientPose
from eval.common import evaluate
from PIL import Image
import numpy as np
import cv2
from utils.visualization import draw_detections, draw_annotations
import math


args = '--phi 0 --weights ./Weights/Linemod/object_9/phi_0_linemod_best_ADD.h5 --validation-image-save-path ' \
        './predicted_images linemod /home/pc/EfficientPose/Datasets/Linemod_preprocessed --object-id 9'.split(' ')

obj = '/home/pc/EfficientPose/Datasets/Linemod_preprocessed/models/obj_09.ply'


def parse_args(args):
    """
    Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Simple EfficientPose evaluation script.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    linemod_parser = subparsers.add_parser('linemod')
    linemod_parser.add_argument('linemod_path', help='Path to dataset directory (ie. /Datasets/Linemod_preprocessed).')
    linemod_parser.add_argument('--object-id', help='ID of the Linemod Object to train on', type=int, default=8)

    occlusion_parser = subparsers.add_parser('occlusion')
    occlusion_parser.add_argument('occlusion_path',
                                  help='Path to dataset directory (ie. /Datasets/Linemod_preprocessed).')

    parser.add_argument('--rotation-representation',
                        help='Which representation of the rotation should be used. Choose from "axis_angle", "rotation_matrix" and "quaternion"',
                        default='axis_angle')

    parser.add_argument('--weights', help='File containing weights to init the model parameter')

    parser.add_argument('--batch-size', help='Size of the batches.', default=1, type=int)
    parser.add_argument('--phi', help='Hyper parameter phi', default=0, type=int, choices=(0, 1, 2, 3, 4, 5, 6))
    parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--score-threshold', help='score threshold for non max suppresion', type=float, default=0.5)
    parser.add_argument('--validation-image-save-path',
                        help='path where to save the predicted validation images after each epoch', default=None)

    print(vars(parser.parse_args(args)))
    return parser.parse_args(args)


def allow_gpu_growth_memory():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    _ = tf.Session(config = config)

allow_gpu_growth_memory()


args = parse_args(args)
if args.validation_image_save_path:
    os.makedirs(args.validation_image_save_path, exist_ok = True)


def create_generators(args):
    common_args = {
        'batch_size': args.batch_size,
        'phi': args.phi,
    }

    from generators.linemod import LineModGenerator
    generator = LineModGenerator(
        args.linemod_path,
        args.object_id,
        train=False,
        shuffle_dataset=False,
        shuffle_groups=False,
        rotation_representation=args.rotation_representation,
        use_colorspace_augmentation=False,
        use_6DoF_augmentation=False,
        **common_args
    )
    return generator


print("\nCreating the Generators...")
generator = create_generators(args)
print("Done!")
num_rotation_parameters = generator.get_num_rotation_parameters()
num_classes = generator.num_classes()
num_anchors = generator.num_anchors


print("Building the Model...")
_, prediction_model, _ = build_EfficientPose(args.phi,
                                             num_classes = num_classes,
                                             num_anchors = num_anchors,
                                             freeze_bn = True,
                                             score_threshold = args.score_threshold,
                                             num_rotation_parameters = num_rotation_parameters,
                                             print_architecture = False)
print("Done!")
print('Loading model, this may take a second...')
prediction_model.load_weights(args.weights, by_name = True)
print("Done!")


model = prediction_model
generator = generator
save_path = args.validation_image_save_path
score_threshold = args.score_threshold
iou_threshold = 0.5
max_detections = 100
diameter_threshold = 0.1
i = 0
raw_image = generator.load_image(i)

capture = cv2.VideoCapture(0)
ret, frame = capture.read()
image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

image, scale = generator.preprocess_image(image.copy())
camera_matrix = generator.load_camera_matrix(i)
camera_matrix = np.array([
         [721.14, 0.00, 414.75],
         [0.00, 728.66, 280.26],
         [0.00, 0.00, 1.00],
    ])
dist_coeff = np.array([0.05679849, -0.30853488, -0.00217107,  0.00847797,  0.56349182])


camera_input = generator.get_camera_parameter_input(camera_matrix, scale, generator.translation_scale_norm)
class_to_bbox_3D = generator.get_bbox_3d_dict()
label_to_name = generator.label_to_name

image_size = 512


def preprocess_image(image):
    image_height, image_width = image.shape[:2]
    if image_height > image_width:
        scale = image_size / image_height
        resized_height = image_size
        resized_width = int(image_width * scale)
    else:
        scale = image_size / image_width
        resized_height = int(image_height * scale)
        resized_width = image_size

    image = cv2.resize(image, (resized_width, resized_height))
    image = image.astype(np.float32)
    image /= 255.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image -= mean
    image /= std
    pad_h = image_size - resized_height
    pad_w = image_size - resized_width
    image = np.pad(image, [(0, pad_h), (0, pad_w), (0, 0)], mode='constant')

    return image, scale


def resizes(image):
    image_height, image_width = image.shape[:2]
    if image_height > image_width:
        scale = image_size / image_height
        resized_height = image_size
        resized_width = int(image_width * scale)
    else:
        scale = image_size / image_width
        resized_height = int(image_height * scale)
        resized_width = image_size

    image = cv2.resize(image, (resized_width, resized_height))
    return image

#capture = cv2.VideoCapture("WIN_20220606_16_07_45_Pro.mp4")
# capture = cv2.VideoCapture(0)
#cv2.namedWindow("Video",cv2.WINDOW_NORMAL)
cv2.namedWindow("Video_2",cv2.WINDOW_NORMAL)
while True:
    ret, frame = capture.read()
    if not ret:
        break
    # frame = resizes(frame)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image, scale = preprocess_image(image)

    boxes, scores, labels, rotations, translations = model.predict_on_batch(
        [np.expand_dims(image, axis=0), np.expand_dims(camera_input, axis=0)])[:5]

    # correct boxes for image scale
    boxes /= scale

    # rescale rotations and translations
    rotations *= math.pi
    # height, width, _ = raw_image.shape

    # select indices which have a score above the threshold
    indices = np.where(scores[0, :] > score_threshold)[0]

    # select those scores
    scores = scores[0][indices]

    # find the order with which to sort the scores
    scores_sort = np.argsort(-scores)[:max_detections]

    # select detections
    image_boxes = boxes[0, indices[scores_sort], :]
    image_rotations = rotations[0, indices[scores_sort], :]
    image_translations = translations[0, indices[scores_sort], :]
    image_scores = scores[scores_sort]
    image_labels = labels[0, indices[scores_sort]]
    image_detections = np.concatenate(
        [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

    print(image_translations, scale)
    raw_image = frame.copy()
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
    # raw_image = draw_detections(raw_image, image_boxes, image_scores, image_labels, image_rotations, image_translations,
    #                             class_to_bbox_3D=generator.get_bbox_3d_dict(),
    #                             camera_matrix=generator.load_camera_matrix(0), label_to_name=generator.label_to_name)

    raw_image = draw_detections(raw_image, image_boxes, image_scores, image_labels, image_rotations, image_translations,
                                class_to_bbox_3D=class_to_bbox_3D,
                                camera_matrix=camera_matrix, label_to_name=label_to_name, obj=obj, distCoeffs=dist_coeff)

    cv2.imshow("Video", frame)
    # if raw_image is None:
    cv2.imshow("Video_2", np.uint8(raw_image[..., ::-1]))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
