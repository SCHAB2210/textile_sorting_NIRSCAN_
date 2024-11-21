from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from pick_and_place import cotton, polyester, wool
from server import start_server, get_sensor_data
import threading
import time

host, port = ['192.168.1.20', 12345]

bot = InterbotixManipulatorXS(
robot_model='vx300',
group_name='arm',
gripper_name='gripper',
)

robot_startup()
bot.arm.go_to_home_pose()

server_thread = threading.Thread(target=start_server, args=(host, port), daemon=True)
server_thread.start()

while True:
	label = get_sensor_data()
	if label == '0':
	    cotton(bot)

	elif label == '1':
	    polyester(bot)

	elif label == '2':
	    wool(bot)

	time.sleep(1)

robot_shutdown()

