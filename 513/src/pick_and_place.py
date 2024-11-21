from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

def cotton(bot):
    bot.arm.set_ee_pose_components(x=0.4, y=0.4, z=0.3)
    bot.gripper.grasp(2.0)
    bot.arm.go_to_home_pose()

    bot.arm.set_ee_pose_components(x=0.7, y=0.0, z=0.2)
    bot.gripper.release(2.0)
    bot.arm.go_to_home_pose()

def polyester(bot):
    bot.arm.set_ee_pose_components(x=0.3, y=0.4, z=0.3)
    bot.gripper.grasp(2.0)
    bot.arm.go_to_home_pose()

    bot.arm.set_ee_pose_components(x=0.7, y=0.0, z=0.2)
    bot.gripper.release(2.0)
    bot.arm.go_to_home_pose()

def wool(bot):
    bot.arm.set_ee_pose_components(x=0.5, y=0.4, z=0.3)
    bot.gripper.grasp(2.0)
    bot.arm.go_to_home_pose()

    bot.arm.set_ee_pose_components(x=0.7, y=0.0, z=0.2)
    bot.gripper.release(2.0)
    bot.arm.go_to_home_pose()
