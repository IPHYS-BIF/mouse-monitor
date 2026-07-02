import time
from PySide6.QtCore import QThread, Signal

class PIDController:
    def __init__(self, kp, ki, kd, setpoint):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.setpoint = setpoint
        self.integral, self.prev_error = 0.0, 0.0

    def compute(self, measurement, dt):
        error = self.setpoint - measurement
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        return max(0.0, min(100.0, output)) # Constrain PWM between 0 and 100%

class HardwareWorker(QThread):
    temps_updated = Signal(float, float, float) # mouse_temp, bed_temp, pwm
    
    def __init__(self, test_mode=False):
        super().__init__()
        self.test_mode = test_mode
        self.running = True
        self.target_temp = 37.5
        self.pid = PIDController(kp=5.0, ki=0.1, kd=1.0, setpoint=self.target_temp)
        
        # Sim physics state
        self.sim_mouse_temp = 34.0
        self.sim_bed_temp = 34.0

    def set_target(self, temp):
        self.target_temp = temp
        self.pid.setpoint = temp

    def run(self):
        last_time = time.time()
        while self.running:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            if self.test_mode:
                # 1. Run PID based on Mouse Temp
                pwm = self.pid.compute(self.sim_mouse_temp, dt)
                
                # 2. Simulate thermal physics (Bed heats up based on PWM, loses heat to ambient)
                bed_heat_rate = (pwm / 100.0) * 0.5 
                ambient_loss = (self.sim_bed_temp - 22.0) * 0.01
                self.sim_bed_temp += (bed_heat_rate - ambient_loss) * dt
                
                # 3. Simulate Mouse Temp (Absorbs heat from bed, loses heat to ambient)
                mouse_heat_gain = (self.sim_bed_temp - self.sim_mouse_temp) * 0.05
                mouse_heat_loss = (self.sim_mouse_temp - 22.0) * 0.005
                self.sim_mouse_temp += (mouse_heat_gain - mouse_heat_loss) * dt
                
                self.temps_updated.emit(self.sim_mouse_temp, self.sim_bed_temp, pwm)
            else:
                # REAL HARDWARE LOGIC GOES HERE (I2C read, GPIO PWM write)
                pass 
            
            self.msleep(100) # Run loop at 10Hz

    def stop(self):
        self.running = False
        self.wait()
