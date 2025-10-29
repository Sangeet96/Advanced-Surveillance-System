import numpy as np

class ServoController:
    def __init__(self, sensitivity=0.002):
        self.sensitivity = sensitivity
        self.servo_x = 0.0
        self.servo_y = 0.0

    def update(self, target_x, target_y, center_x, center_y):
        dx = target_x - center_x
        dy = target_y - center_y

        delta_x = -dx * self.sensitivity
        delta_y = dy * self.sensitivity

        self.servo_x = np.clip(self.servo_x + delta_x, -1.0, 1.0)
        self.servo_y = np.clip(self.servo_y + delta_y, -1.0, 1.0)

        move_x = "Right" if delta_x > 0 else "Left"
        move_y = "Up" if delta_y > 0 else "Down"
        print(f"ServoX → {move_x} ({delta_x:+.3f}) | ServoY → {move_y} ({delta_y:+.3f})")

        return self.servo_x, self.servo_y
