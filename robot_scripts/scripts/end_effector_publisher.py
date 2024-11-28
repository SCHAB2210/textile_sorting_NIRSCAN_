import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from tf2_ros import Buffer, TransformListener
from tf_transformations import quaternion_matrix


class EndEffectorPublisher(Node):
    def __init__(self):
        super().__init__('end_effector_publisher')

        # Publisher for the end-effector position
        self.publisher_ = self.create_publisher(Point, 'end_effector_position', 10)

        # TF2 buffer and listener to get transforms
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Timer to publish pose at regular intervals (10 Hz)
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        try:
            # Look up the transform from vx300/base_link to vx300/ee_arm_link
            transform = self.tf_buffer.lookup_transform('vx300/base_link', 'vx300/ee_arm_link', rclpy.time.Time())

            # Extract translation (x, y, z) from the transform
            translation = transform.transform.translation
            x, y, z = translation.x, translation.y, translation.z

            # Create and publish the Point message
            point_msg = Point()
            point_msg.x = x
            point_msg.y = y
            point_msg.z = z
            self.publisher_.publish(point_msg)

            self.get_logger().info(f'Publishing: x={x}, y={y}, z={z}')

        except Exception as e:
            self.get_logger().error(f'Failed to get end-effector transform: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = EndEffectorPublisher()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

