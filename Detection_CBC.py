import torch
import cv2
import numpy as np
import time
import os

from datetime import datetime
from deep_sort_realtime.deepsort_tracker import DeepSort


class ObjectDetection:

    def __init__(self):

        self.path_crop = '/home/mark/vision/scripts/crop/'
        self.source = './vids/test28.avi'
        self.model = self.load_model()
        self.classes = self.model.names
        self.line_cascade = cv2.CascadeClassifier('cascade.xml')
        self.device = 'cuda'
        print("\n\nDevice Used:", self.device)

        self.id = 0
        self.prev_id = 0
        self.cas_x = 0
        self.cas_y = 0
        self.counter = 0

        self.det()

    def load_model(self):

        model = torch.hub.load('../yolov5', 'custom', path='../yolov5/runs/train/exp17/weights/best.pt', source='local',
                               )
        return model

    def score_frame(self, frame):

        self.model.to(self.device)
        w = int(frame.shape[1] / 2)
        h = int(frame.shape[0] / 2)

        results = self.model(frame, (w, h))

        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        return self.classes[int(x)]

    def plot_boxes(self, frame, results, h, w, conf):

        labels, cord = results
        detections = []
        n = len(labels)
        x_shape, y_shape = w, h

        for i in range(n):
            row = cord[i]
            if row[4] >= conf:
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), \
                    int(row[3] * y_shape)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 200, 0), 2)
                cv2.putText(frame, self.class_to_label(labels[i]) + " ID: " + str(self.id), (x1 + 50, y1 + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
                # x_center = x1 + (x2 - x1)
                # y_center = y1 + ((y2 - y1) / 2)
                # tlwh = np.asarray([x1, y1, int(x2 - x1), int(y2 - y1)])
                # conf = float(row[4].item())

                detections.append(([int(x1), int(y1), int(x2 - x1), int(y2 - y1)], row[4].item(), 'Pack'))

        return detections

    def cascade(self, frame):

        line_cas = self.line_cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        line_rect = line_cas.detectMultiScale(gray,
                                              scaleFactor=4,
                                              minNeighbors=6)
        if line_rect is ():
            flag = False
        else:
            flag = True

        for (x, y, w, h) in line_rect:
            cv2.rectangle(frame, (x, y),
                          (x + w, y + h), (255, 255, 255), 2)
            x_center = x + w / 2
            y_center = y + (h / 2)
            self.cas_x = x_center
            self.cas_y = y_center

        return frame, flag

    def z_detection(self, bbox):

        cas_x, cas_y = self.cas_x, self.cas_y
        date = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

        if cas_x > 0 and cas_y > 0:
            if (bbox[0] < cas_x < bbox[0] + bbox[2]) and (bbox[1] < cas_y < bbox[1] + bbox[3]):
                if int(self.id) != int(self.prev_id):
                    self.prev_id = self.id
                    return self.prev_id, date

    def det(self):

        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        obj_tracker = DeepSort(max_age=5,
                               n_init=2,
                               nms_max_overlap=1.0,
                               max_cosine_distance=0.1,
                               nn_budget=None,
                               override_track_class=None,
                               embedder='mobilenet',
                               half=True,
                               bgr=True,
                               embedder_gpu=True,
                               embedder_model_name=None,
                               embedder_wts=None,
                               polygon=False,
                               today=None)

        cap = cv2.VideoCapture('./vids/test28.avi')
        while True:
            resolution = [1024, 768]
            start_time = time.perf_counter()
            ret, frame_orig = cap.read()
            frame_res = cv2.resize(frame_orig, resolution)
            frame = frame_res.copy()

            results = self.score_frame(frame)
            detections = self.plot_boxes(frame, results, h=frame.shape[0], w=frame.shape[1], conf=0.5)

            tracks = obj_tracker.update_tracks(detections, frame=frame)
            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                self.id = track_id

            end_time = time.perf_counter()
            fps = 1 / np.round(end_time - start_time, 3)
            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            frame1, flag_cas = self.cascade(frame)
            crop = frame1.copy()

            dt = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

            if flag_cas:
                for i in detections:
                    if 660 < self.cas_x < 750:
                        match = self.z_detection(i[0])
                        os.chdir(self.path_crop)
                        if match is not None:
                            name = f'{str(dt)}-ID:{match}'
                            filename = '%s.png' % name
                            cv2.imwrite(filename, crop)

            key = cv2.waitKey(30) & 0xff
            if key == ord('p'):

                while True:

                    key2 = cv2.waitKey(30) or 0xff
                    cv2.imshow('Main_Frame', frame1)
                    if key2 == ord('p'):
                        break

            cv2.imshow('Main_Frame', frame1)

            if key == 27:
                break
        cap.release()


if __name__ == "__main__":
    # path_crop = '/home/mark/vision/scripts/crop/'
    detection = ObjectDetection()
    cv2.destroyAllWindows()
