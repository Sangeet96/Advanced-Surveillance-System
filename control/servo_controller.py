# import numpy as np

# class ServoController:
#     def __init__(self, sensitivity=0.004):
#         self.sensitivity = sensitivity
#         self.servo_x = 0.0
#         self.servo_y = 0.0

#     def update(self, target_x, target_y, center_x, center_y):
#         dx = target_x - center_x
#         dy = target_y - center_y

#         delta_x = -dx * self.sensitivity
#         delta_y = dy * self.sensitivity

#         self.servo_x = np.clip(self.servo_x + delta_x, -1.0, 1.0)
#         self.servo_y = np.clip(self.servo_y + delta_y, -1.0, 1.0)

#         move_x = "Right" if delta_x > 0.5 else "Left"
#         move_y = "Up" if delta_y > 0.5 else "Down"
#         print(f"ServoX → {move_x} ({delta_x:+.3f}) | ServoY → {move_y} ({delta_y:+.3f})")

#         return self.servo_x, self.servo_y


import time

class ServoController:
    def __init__(self):
        self.x_angle = 90
        self.y_angle = 90

    def update(self, target):
        target_x, target_y, tid = target
        frame_center_x, frame_center_y = 320, 240

        dx = target_x - frame_center_x
        dy = target_y - frame_center_y

        if abs(dx) > 15:
            self.x_angle -= dx * 0.01
        if abs(dy) > 15:
            self.y_angle += dy * 0.01

        # clamp angles
        self.x_angle = max(0, min(180, self.x_angle))
        self.y_angle = max(0, min(180, self.y_angle))

        print(f"[Servo] Target ID {tid} | X: {self.x_angle:.2f}°, Y: {self.y_angle:.2f}°")
        time.sleep(0.01)
