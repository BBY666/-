from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.go2.sport.sport_client import (
    SportClient,
    PathPoint,
    SPORT_PATH_POINT_SIZE,
)
import threading
import time



def test_file():
    print("测试unitree_dog_control")


def HighStateHandler(msg: SportModeState_):
    global robot_state
    robot_state = msg


class SportModeTest:
    def __init__(self) -> None:
        # Time count
        self.t = 0
        self.dt = 0.01

        # Initial poition and yaw
        self.px0 = 0
        self.py0 = 0
        self.yaw0 = 0

        self.client = SportClient()  # Create a sport client
        self.client.SetTimeout(10.0)
        self.client.Init()

        # 全局事件，用于控制函数执行
        self.execution_event = threading.Event()

        # 当前执行的函数
        self.current_function = None

    def GetInitState(self, robot_state: SportModeState_):
        self.px0 = robot_state.position[0]
        self.py0 = robot_state.position[1]
        self.yaw0 = robot_state.imu_state.rpy[2]

    def StandUpDown(self,measurement,velocity=1):
        self.client.StandDown()
        print("Stand down !!!")
        time.sleep(1)

        self.client.StandUp()
        print("Stand up !!!")
        time.sleep(1)

        self.client.StandDown()
        print("Stand down !!!")
        time.sleep(1)

        self.client.Damp()

    def VelocityMove(self,measurement,velocity=1):
        elapsed_time = 1
        for i in range(int(elapsed_time / self.dt)):
            self.client.Move(0.3, 0, 0)  # vx, vy vyaw
            time.sleep(self.dt)
        self.client.StopMove()

    def Move_forward(self,measurement,velocity=1):
        elapsed_time = measurement*4/velocity*1
        v=velocity/1*0.24
        while True:
            for i in range(int(elapsed_time / self.dt)):
                self.client.Move(v, 0, 0)  # vx, vy vyaw
                if self.execution_event.is_set():
                    break
                time.sleep(self.dt)
            self.client.StopMove()
            break

    def StopMove(self,measurement,velocity=1):
    	self.client.StopMove()

    def Move_backward(self,measurement,velocity=1):
        elapsed_time = measurement*5.5/velocity*1
        v = velocity / 1 * 0.24
        while True:
            for i in range(int(elapsed_time / self.dt)):
                self.client.Move(-v, 0, 0)  # vx, vy vyaw
                if self.execution_event.is_set():
                    break
                time.sleep(self.dt)
            self.client.StopMove()
            break

    def Move_left(self,measurement,velocity=1):
        elapsed_time = measurement*5.5/velocity*1
        v = velocity / 1 * 0.24
        while True:
            for i in range(int(elapsed_time / self.dt)):
                self.client.Move(0, v, 0)  # vx, vy vyaw
                if self.execution_event.is_set():
                    break
                time.sleep(self.dt)
            self.client.StopMove()
            break
        
    def Move_right(self,measurement,velocity=1):
        elapsed_time = measurement*5.5/velocity*1
        v = velocity / 1 * 0.24
        while True:
            for i in range(int(elapsed_time / self.dt)):
                self.client.Move(0, -v, 0)  # vx, vy vyaw
                if self.execution_event.is_set():
                    break
                time.sleep(self.dt)
            self.client.StopMove()
            break
    
    def Move_cycle_right(self,measurement,velocity=1):
        elapsed_time = (float(measurement)/90.0)*2
        # v = velocity / 5 * 1
        while True:
            for i in range(int(elapsed_time / self.dt)):
                self.client.Move(0.1, 0, -v)  # vx, vy vyaw
                if self.execution_event.is_set():
                    break
                time.sleep(self.dt)
            self.client.StopMove()
            break

    def Move_cycle_left(self,measurement,velocity=1):
        elapsed_time = (float(measurement)/90.0)*2
        # v = velocity / 5 * 1
        while True:
            for i in range(int(elapsed_time / self.dt)):
                self.client.Move(0.1, 0, v)  # vx, vy vyaw
                if self.execution_event.is_set():
                    break
                time.sleep(self.dt)
            self.client.StopMove()
            break

    def BalanceAttitude(self,measurement,velocity=1):
        self.client.Euler(0.1, 0.2, 0.3)  # roll, pitch, yaw
        self.client.BalanceStand()

    def TrajectoryFollow(self,measurement=1,velocity=1):
        time_seg = 0.2
        time_temp = self.t - time_seg
        path = []
        for i in range(SPORT_PATH_POINT_SIZE):
            time_temp += time_seg

            px_local = 0.5 * math.sin(0.5 * time_temp)
            py_local = 0
            yaw_local = 0
            vx_local = 0.25 * math.cos(0.5 * time_temp)
            vy_local = 0
            vyaw_local = 0

            path_point_tmp = PathPoint(0, 0, 0, 0, 0, 0, 0)

            path_point_tmp.timeFromStart = i * time_seg
            path_point_tmp.x = (
                px_local * math.cos(self.yaw0)
                - py_local * math.sin(self.yaw0)
                + self.px0
            )
            path_point_tmp.y = (
                px_local * math.sin(self.yaw0)
                + py_local * math.cos(self.yaw0)
                + self.py0
            )
            path_point_tmp.yaw = yaw_local + self.yaw0
            path_point_tmp.vx = vx_local * math.cos(self.yaw0) - vy_local * math.sin(
                self.yaw0
            )
            path_point_tmp.vy = vx_local * math.sin(self.yaw0) + vy_local * math.cos(
                self.yaw0
            )
            path_point_tmp.vyaw = vyaw_local

            path.append(path_point_tmp)

            self.client.TrajectoryFollow(path)
    
    def Stretch(self,measurement=1,velocity=1):
        
        self.client.Stretch()
        print("Stretch !!!")
        time.sleep(1)  
        
    def Stand(self,measurement=1,velocity=1):
        self.client.RecoveryStand()
        print("RecoveryStand !!!")
        time.sleep(1)
       
    def Sit(self,measurement=1,velocity=1):   
        self.client.StandDown()
        print("Stand down !!!")
        time.sleep(1)
       
            
    def SpecialMotions(self,measurement=1,velocity=1):
        self.client.RecoveryStand()
        print("RecoveryStand !!!")
        time.sleep(1)
        
        self.client.Stretch()
        print("Sit !!!")
        time.sleep(1)  
        
        self.client.RecoveryStand()
        print("RecoveryStand !!!")
        time.sleep(1)
 
    def execute_function(self,func,measurement=1,velocity=1):
        self.current_function = func.__name__
        func(measurement)

    def start_execution(self,func,measurement,velocity=1):
        self.execution_thread = threading.Thread(target=self.execute_function, args=(func,measurement))
        self.execution_thread.start()

    def stop_execution(self):
        if self.current_function:
            self.execution_event.set()
            self.execution_thread.join()
            self.execution_event.clear()