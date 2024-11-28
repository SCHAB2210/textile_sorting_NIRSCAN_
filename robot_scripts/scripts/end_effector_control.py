from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from scripts.pick_and_place import cotton, polyester, wool, pick, hover
from scripts.server import start_server, get_sensor_data
import threading
import time


host, port = ['192.168.1.20', 12345]

bot = InterbotixManipulatorXS(
    robot_model='vx300',
    group_name='arm',
    gripper_name='gripper',
)

def main():
    robot_startup()

    server_thread = threading.Thread(target=start_server, args=(host, port), daemon=True)
    server_thread.start()

    bot.arm.set_trajectory_time(2, 1)

    hover(bot)
    bot.gripper.release(2.0)

    while True:
        label = get_sensor_data()
        if label != '':
            time.sleep(4)
            pick(bot)
            if label == '0':
                cotton(bot)
            elif label == '1':
                wool(bot)
            elif label == '2':
                polyester(bot)
            hover(bot)

        time.sleep(1)

    robot_shutdown()

if __name__ == "__main__":
    main()

