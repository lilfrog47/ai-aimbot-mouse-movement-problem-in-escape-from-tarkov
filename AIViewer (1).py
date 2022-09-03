import keyboard
import numpy as np
import cv2
import time
import random
import mss
import torch
from threading import Thread
import \
    win32api  # This is not safe -- Fortnite(tournaments), COD MW(all), Valorant(all), The Cycle(safe), Apex Legends(safe)
import win32con

running = False
tracking_bool = False
rel_x_target = 0  # False
rel_y_target = 0


class Targeting(Thread):
    def __init__(self, toggle_key='x'):
        Thread.__init__(self)
        self.toggle_key = toggle_key

    def run(self):
        print('Beginning Target loop!')
        self.target_loop()

    def target_loop(self):

        global tracking_bool, rel_x_target, rel_y_target

        print('In target loop!')
        prev_x = 0
        prev_y = 0
        scale = 8
        while True:
            time.sleep(random.uniform(0.001, 0.002))

            if running:

                if keyboard.is_pressed(self.toggle_key):
                    tracking_bool = not tracking_bool
                    print('Tracking:', tracking_bool)
                    time.sleep(0.3)

                if not tracking_bool:
                    time.sleep(0.1)

                if tracking_bool:
                    if rel_x_target and rel_y_target:
                        prev_x = rel_x_target
                        prev_y = rel_y_target
                        win32api.SetCursorPos((rel_x_target, rel_y_target))
                        rel_y_target = 0
                        rel_x_target = 0


class AIView(Thread):
    def __init__(self, weight, exit_key='p'):
        """Initializes the viewing window class for the model
        1. Initializes an mss object - mss is used for taking multiple screenshots(record)
        2. Select the monitor you wish to capture from
        3. Loads your custom model | https://github.com/ultralytics/yolov5/issues/36
        4. Stores the exit key
        5. Checks if CUDA compatible
        """
        Thread.__init__(self)
        self.screen = mss.mss()  # 1.
        self.monitor = self.screen.monitors[1]  # 2.
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weight, force_reload=True)  # 3.
        self.exit_key = exit_key  # 4.
        print(self.cuda_info())  # 5.

    def cuda_info(self):
        """Outputs CUDA compatibility
        Returns:
            : str :
        """
        if torch.cuda.is_available():
            return 'Cuda acceleration ON!'
        else:
            return 'Cuda acceleration OFF!'

    def resize_aspect_ratio(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        """Resize while maintaining aspect ratio"""
        dim = None
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        return cv2.resize(image, dim, interpolation=inter)

    def run(self):
        global rel_x_target, rel_y_target, running

        print('Initializing loop!')
        running = True
        while True:
            frame = np.array(self.screen.grab(self.monitor))
            results = self.model(frame)

            if len(results.xyxy[0]) != 0:
                detect_info = {}
                for *box, confidence, cls in results.xyxy[0]:
                    x1y1 = [int(x.item()) for x in box[:2]]  # X1 and Y1 = top left
                    x2y2 = [int(x.item()) for x in box[2:]]  # X2 and Y2 = bottom right
                    x1, y1, x2, y2, conf = *x1y1, *x2y2, confidence.item()
                    # x1, y1 = top left
                    # x2, y2 = bottom right
                    d_height = y2 - y1
                    d_width = x2 - x1
                    if conf > 0.4:
                        target_x_pos = int((x1 + x2) / 2)
                        target_y_pos = int((y1 + y2) / 2) - int(d_height / 3.0)
                        detect_info = {
                            "x1y1": x1y1,
                            "x2y2": x2y2,
                            "head_x_pos": target_x_pos,
                            "head_y_pos": target_y_pos,
                            "conf": conf
                        }

                    cv2.putText(frame, f'Conf: {conf:.2f}', (x1y1[0], x1y1[1] - 10),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
                    cv2.rectangle(frame, x1y1, x2y2, (133, 122, 255), 3)

                if detect_info:
                    rel_x_target = detect_info['head_x_pos'] - self.monitor['left']
                    rel_y_target = detect_info['head_y_pos'] - self.monitor['top']

            cv2.putText(frame, f"Mouse Tracking: {'On' if tracking_bool else 'Off'}",
                        (self.monitor['left'], self.monitor['top'] + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 255), 1)

            # Resize image for one monitor users
            if one_monitor:
                resized_img = self.resize_aspect_ratio(frame, 500, 500)
                cv2.imshow('Viewer', resized_img)
            else:
                cv2.imshow("Viewer", frame)

            cv2.waitKey(10)  # Paints
            time.sleep(0.01)  # Blocks

            if keyboard.is_pressed(self.exit_key):
                print('Closed viewer!')
                break

        cv2.destroyAllWindows()
        running = False
        print('Loop ended, cv2 window closed!')


def user_input():
    """Waits for user input for model name and exit key

    Returns:
        weight_name : str : name of the model used
        exit_key : str : key used to exit from app
    """

    model_name = input('Name of weight( without .pt): ')
    exit_key = input('Exit key: ')
    aim_assist_key = input('Aim assist key: ')
    one_monitor = input("Only have one monitor? (y) or (n)")
    if one_monitor.lower().strip() == 'y':
        one_monitor = True
    else:
        one_monitor = False
    return model_name, exit_key, aim_assist_key, one_monitor


if __name__ == "__main__":
    print('Initialized')
    weight_name, exit_key, aim_assist_key, one_monitor = user_input()
    view = AIView(weight_name, exit_key)
    print('Success! Running...')
    target_track = Targeting(toggle_key=aim_assist_key)
    target_track.start()
    view.start()
    target_track.join()
    view.join()
