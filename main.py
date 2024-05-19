import cv2
import cvlib
# import numpy as np
from imutils.video import FPS
from cvlib.object_detection import draw_bbox

video = cv2.VideoCapture('sample_videos/classroom.mp4')  # for sample videos
# video = cv2.VideoCapture(0) # for webcam

# available models: yolov4 and yolov4-tiny
# use model='yolov4-tiny' for faster but less accurate version
model = 'yolov4'


def count_objects(labels):
    label_dict = dict()
    for _label in set(labels):
        label_dict[_label.title()] = 0
    for _label in labels:
        label_dict[_label.title()] += 1

    return label_dict


while True:
    fps = FPS().start()
    ret, frame = video.read()
    if not ret:
        break

    bbox, label, conf = cvlib.detect_common_objects(frame, model=model)
    output_image = draw_bbox(frame, bbox, label, conf, write_conf=False)

    for i, (key, value) in enumerate(count_objects(label).items()):
        cv2.putText(output_image, f'{key}={value}', (0, 25 + i * 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 2)

    fps.update()
    fps.stop()

    cv2.putText(output_image, f'FPS={fps.fps():0.2f}', (frame.shape[1] - 150, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 0), 2)

    cv2.putText(output_image, model, (0, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 0, 0), 2)

    # for name, coord in zip(label, bbox):
    #     if name != 'person':
    #         continue
    #
    #     (startX, startY) = coord[0], coord[1]
    #     (endX, endY) = coord[2], coord[3]
    #
    #     person_image = np.copy(frame[startY:endY, startX:endX])
    #     face, _ = cvlib.detect_face(person_image)
    #
    #     (faceStartX, faceStartY) = coord[0], coord[1]
    #     (faceEndX, faceEndY) = coord[2], coord[3]
    #
    #     face_image = np.copy(frame[faceStartY:faceEndY, faceStartX:faceEndX])
    #
    #     gender_raw = cvlib.detect_gender(face_image)
    #     preds = gender_raw[1]
    #     gender, gender_conf = ('indeterminate', 0)
    #     if preds[0] > 0.5 and preds[1] > 0.5:
    #         gender, gender_conf = ('male', preds[0]) if preds[0] > preds[1] else ('female', preds[1])
    #     elif preds[0] > 0.5:
    #         gender, gender_conf = ('male', preds[0])
    #     elif preds[1] > 0.5:
    #         gender, gender_conf = ('female', preds[1])
    #
    #     # print(gender)
    #
    #     cv2.putText(output_image, f'{gender} ({gender_conf * 100:.02f}%)', (coord[0], coord[1] - 10),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
    #                 2)

    cv2.imshow("Object Detection", output_image)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
