#!/usr/bin/env python3
"""
Hover controller - handles ONLY Z (altitude) control.
X/Y/Yaw handled by planar_move plugin (smooth physics).
TF broadcasting for RTAB-Map.
"""
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import SetEntityState, GetEntityState
from gazebo_msgs.msg import EntityState
from tf2_ros import TransformBroadcaster
import math


class DroneHoverController(Node):
    def __init__(self):
        super().__init__('drone_hover_controller')
        self.set_parameters([Parameter('use_sim_time',
            Parameter.Type.BOOL, True)])

        self.x = 10.47; self.y = -20.56
        self.z = 0.1;   self.yaw = math.pi
        self.initialised = False

        # Only track Z velocity
        self.cmd_vz = 0.0
        self.vz = 0.0
        self.alpha = 0.3
        self.dt = 0.02
        self.entity_name = 'x3_lidar_drone'

        self.tf_broadcaster = TransformBroadcaster(self)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)

        # Subscribe to cmd_vel only for Z
        self.cmd_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_cb, 10)

        self.set_client = self.create_client(
            SetEntityState, '/set_entity_state')
        self.get_client = self.create_client(
            GetEntityState, '/get_entity_state')

        self.get_logger().info('Waiting for Gazebo services...')
        self.set_client.wait_for_service(timeout_sec=10.0)
        self.get_client.wait_for_service(timeout_sec=10.0)
        self.get_logger().info('Services ready. Hover controller active.')

        self.create_timer(0.1, self.update_pose)
        self.create_timer(self.dt, self.control_loop)

    def cmd_cb(self, msg):
        self.cmd_vz = msg.linear.z

    def update_pose(self):
        """Continuously sync pose from Gazebo."""
        req = GetEntityState.Request()
        req.name = self.entity_name
        req.reference_frame = 'world'
        self.get_client.call_async(req).add_done_callback(self._got_state)

    def _got_state(self, future):
        try:
            res = future.result()
            if res.success:
                self.x = res.state.pose.position.x
                self.y = res.state.pose.position.y
                self.z = res.state.pose.position.z
                q = res.state.pose.orientation
                self.yaw = math.atan2(
                    2*(q.w*q.z+q.x*q.y), 1-2*(q.y**2+q.z**2))
                if not self.initialised:
                    self.initialised = True
                    self.get_logger().info(
                        f'Initial pose: x={self.x:.2f} '
                        f'y={self.y:.2f} z={self.z:.2f} '
                        f'yaw={math.degrees(self.yaw):.1f}°')
        except Exception:
            pass

    def control_loop(self):
        now = self.get_clock().now()
        stamp = now.to_msg()

        # Smooth Z velocity
        self.vz += self.alpha * (self.cmd_vz - self.vz)
        if abs(self.vz) < 0.01:
            self.vz = 0.0

        if self.initialised and abs(self.vz) > 0.01:
            # Only update Z — let planar_move handle X/Y/yaw
            new_z = max(0.1, self.z + self.vz * self.dt)
            req = SetEntityState.Request()
            req.state = EntityState()
            req.state.name = self.entity_name
            req.state.reference_frame = 'world'
            req.state.pose.position.x = self.x
            req.state.pose.position.y = self.y
            req.state.pose.position.z = new_z
            qz = math.sin(self.yaw/2.0)
            qw = math.cos(self.yaw/2.0)
            req.state.pose.orientation.z = qz
            req.state.pose.orientation.w = qw
            self.set_client.call_async(req)

        # Always broadcast TF for RTAB-Map
        qz = math.sin(self.yaw/2.0)
        qw = math.cos(self.yaw/2.0)
        self._broadcast_tf(stamp, qz, qw)

        # Publish odometry
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = self.z
        qz2 = math.sin(self.yaw/2.0)
        qw2 = math.cos(self.yaw/2.0)
        odom.pose.pose.orientation.z = qz2
        odom.pose.pose.orientation.w = qw2
        odom.twist.twist.linear.z = self.vz
        self.odom_pub.publish(odom)

    def _broadcast_tf(self, stamp, qz=0.0, qw=1.0):
        t1 = TransformStamped()
        t1.header.stamp = stamp
        t1.header.frame_id = 'map'
        t1.child_frame_id = 'odom'
        t1.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t1)

        t2 = TransformStamped()
        t2.header.stamp = stamp
        t2.header.frame_id = 'odom'
        t2.child_frame_id = 'base_link'
        t2.transform.translation.x = self.x
        t2.transform.translation.y = self.y
        t2.transform.translation.z = self.z
        t2.transform.rotation.z = qz
        t2.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(t2)


def main(args=None):
    rclpy.init(args=args)
    node = DroneHoverController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
