from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import time

def home(bot):
    bot.arm.set_ee_pose_components(x=0.1190, y=-0.0098, z=0.3029, pitch=0.9)

def hover(bot):
    home(bot)
    bot.arm.set_ee_pose_components(x=0.22, y=0.6, z=0.18, pitch=0.4, roll=0.85)

def pick(bot):
    bot.gripper.grasp(5.0)
    home(bot)

def cotton(bot):
    bot.arm.set_ee_pose_components(x=0.3, y=-0.49, z=0.35, pitch=1)
    bot.arm.set_ee_pose_components(x=0.3, y=-0.49, z=0.2, pitch=0.4)
    bot.gripper.release(1.0)
    bot.arm.set_ee_pose_components(x=0.3, y=-0.49, z=0.35, pitch=1)

def wool(bot):
    bot.arm.set_ee_pose_components(x=0.13, y=-0.4207, z=0.2846, pitch=1)
    time.sleep(1)
    bot.gripper.release(1.0)

def polyester(bot):
    bot.arm.set_ee_pose_components(x=-0.23, y=-0.4207, z=0.2546, pitch=1)
    bot.arm.set_ee_pose_components(x=-0.23, y=-0.4207, z=0, pitch=1)
    bot.gripper.release(1.0)
    bot.arm.set_ee_pose_components(x=-0.23, y=-0.4207, z=0.2546, pitch=1)

def main():
    print("Pick and Place module initialized.")

if __name__ == "__main__":
    main()

