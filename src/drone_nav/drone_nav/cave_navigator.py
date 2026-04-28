#!/usr/bin/env python3
"""
Cave Navigator - Follows recorded waypoints exactly
Loads waypoints.json and flies through them in order
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from gazebo_msgs.srv import GetEntityState
import json
import math


class CaveNavigator(Node):
    def __init__(self):
        super().__init__('cave_navigator')
        self.set_parameters([
            rclpy.parameter.Parameter('use_sim_time',
                rclpy.parameter.Parameter.Type.BOOL, True)])

        # Load waypoints from file
        with open('/home/jayakar/RSN_Proj/maps/waypoints.json') as f:
            data = json.load(f)
        self.waypoints = [(w['x'], w['y'], w['z'], w['label'])
                         for w in data]

        self.get_logger().info(
            f'Loaded {len(self.waypoints)} waypoints')

        # State
        self.x = 0.0; self.y = 0.0
        self.z = 0.0; self.yaw = 0.0
        self.initialized = False
        self.wp_idx = 0
        self.stuck_timer = 0
        self.stuck_limit = 20
        self.done = False

        # Publishers
        self.cmd_pub = self.create_publisher(
            Twist, '/cmd_vel', 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/cave_path', 10)

        # Gazebo service
        self.get_client = self.create_client(
            GetEntityState, '/get_entity_state')
        self.get_client.wait_for_service(timeout_sec=10.0)

        self.get_logger().info('='*50)
        self.get_logger().info('CAVE NAVIGATOR READY')
        self.get_logger().info(
            f'Mission: {len(self.waypoints)} waypoints')
        self.get_logger().info(
            f'Start: ({self.waypoints[0][0]}, {self.waypoints[0][1]})')
        self.get_logger().info(
            f'End: ({self.waypoints[-1][0]}, {self.waypoints[-1][1]})')
        self.get_logger().info('='*50)

        self.create_timer(0.1,  self.update_pose)
        self.create_timer(0.5,  self.nav_loop)
        self.create_timer(0.05, self.move_loop)
        self.create_timer(3.0,  self.publish_markers)

    def update_pose(self):
        req = GetEntityState.Request()
        req.name = 'x3_lidar_drone'
        req.reference_frame = 'world'
        self.get_client.call_async(req).add_done_callback(self._cb)

    def _cb(self, future):
        try:
            res = future.result()
            if res.success:
                p = res.state.pose.position
                self.x, self.y, self.z = p.x, p.y, p.z
                q = res.state.pose.orientation
                self.yaw = math.atan2(
                    2*(q.w*q.z+q.x*q.y),
                    1-2*(q.y**2+q.z**2))
                self.initialized = True
        except Exception:
            pass

    def dist_to_wp(self):
        tx, ty, tz, _ = self.waypoints[self.wp_idx]
        return math.sqrt(
            (tx-self.x)**2+(ty-self.y)**2+(tz-self.z)**2)

    def nav_loop(self):
        if not self.initialized or self.done:
            return

        if self.wp_idx >= len(self.waypoints):
            self.get_logger().info('='*50)
            self.get_logger().info('MISSION COMPLETE!')
            self.get_logger().info('All waypoints visited.')
            self.get_logger().info('='*50)
            self.cmd_pub.publish(Twist())
            self.done = True
            return

        d = self.dist_to_wp()
        tx, ty, tz, label = self.waypoints[self.wp_idx]

        if d < 1.2:
            self.get_logger().info(
                f'✓ [{self.wp_idx+1}/{len(self.waypoints)}] '
                f'{label} reached')
            self.wp_idx += 1
            self.stuck_timer = 0
            return

        self.stuck_timer += 1
        if self.stuck_timer > self.stuck_limit:
            self.get_logger().warn(
                f'Stuck at {label} dist={d:.1f}m — skipping')
            self.wp_idx += 1
            self.stuck_timer = 0
            return

        # Log every 5th update
        if self.stuck_timer % 5 == 0:
            self.get_logger().info(
                f'→ [{self.wp_idx+1}/{len(self.waypoints)}] '
                f'{label} ({tx:.1f},{ty:.1f}) dist={d:.1f}m')

    def move_loop(self):
        if not self.initialized or self.done:
            return
        if self.wp_idx >= len(self.waypoints):
            return

        tx, ty, tz, _ = self.waypoints[self.wp_idx]
        dx = tx-self.x; dy = ty-self.y; dz = tz-self.z
        dxy = math.sqrt(dx*dx+dy*dy)

        twist = Twist()
        tyaw = math.atan2(dy, dx) if dxy > 0.3 else self.yaw
        yerr = tyaw - self.yaw
        while yerr >  math.pi: yerr -= 2*math.pi
        while yerr < -math.pi: yerr += 2*math.pi

        twist.angular.z = max(-1.5, min(1.5, yerr*2.5))
        if abs(yerr) < 0.35:
            twist.linear.x = min(0.5, dxy*0.4)
        twist.linear.z = max(-0.5, min(0.5, dz*1.5))
        self.cmd_pub.publish(twist)

    def publish_markers(self):
        markers = MarkerArray()

        # Draw all waypoints as small dots
        for i, (wx, wy, wz, label) in enumerate(self.waypoints):
            mk = Marker()
            mk.header.frame_id = 'map'
            mk.header.stamp = self.get_clock().now().to_msg()
            mk.ns = 'waypoints'
            mk.id = i
            mk.type = Marker.SPHERE
            mk.action = Marker.ADD
            mk.pose.position.x = wx
            mk.pose.position.y = wy
            mk.pose.position.z = wz
            mk.pose.orientation.w = 1.0
            mk.scale.x = mk.scale.y = mk.scale.z = 0.4

            if i < self.wp_idx:
                # Visited - green
                mk.color = ColorRGBA(
                    r=0.0, g=1.0, b=0.0, a=1.0)
            elif i == self.wp_idx:
                # Current target - yellow
                mk.color = ColorRGBA(
                    r=1.0, g=1.0, b=0.0, a=1.0)
            else:
                # Not yet visited - blue
                mk.color = ColorRGBA(
                    r=0.2, g=0.2, b=1.0, a=0.4)
            markers.markers.append(mk)

        # Current target label
        if self.wp_idx < len(self.waypoints):
            tx, ty, tz, label = self.waypoints[self.wp_idx]
            txt = Marker()
            txt.header.frame_id = 'map'
            txt.header.stamp = self.get_clock().now().to_msg()
            txt.ns = 'current_target'
            txt.id = 9999
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position.x = tx
            txt.pose.position.y = ty
            txt.pose.position.z = tz + 1.5
            txt.pose.orientation.w = 1.0
            txt.scale.z = 1.2
            txt.text = (f'TARGET: {label}\n'
                       f'{self.wp_idx+1}/{len(self.waypoints)}')
            txt.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
            markers.markers.append(txt)

        self.marker_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = CaveNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.cmd_pub.publish(Twist())
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
