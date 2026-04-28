#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid
from gazebo_msgs.srv import GetEntityState
import numpy as np
import math
import random

class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')
        self.set_parameters([
            rclpy.parameter.Parameter('use_sim_time',
                rclpy.parameter.Parameter.Type.BOOL, True)])
        self.x = 10.47; self.y = -20.56
        self.z = 0.1;   self.yaw = math.pi
        self.initialized = False
        self.map_data = None
        self.map_info = None
        self.current_goal = None
        self.visited = []
        self.stuck_timer = 0
        self.target_x = 10.47
        self.target_y = -20.56
        self.target_z = 1.5
        self.entry_waypoints = [
            (10.47, -20.56, 1.5),
            (7.0,   -20.56, 1.5),
            (3.0,   -20.56, 1.5),
            (-2.0,  -20.56, 1.5),
            (-7.0,  -20.56, 1.5),
            (-12.0, -20.56, 1.5),
        ]
        self.entry_idx = 0
        self.phase = 'entry'
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_cb, 1)
        self.get_client = self.create_client(
            GetEntityState, '/get_entity_state')
        self.get_client.wait_for_service(timeout_sec=10.0)
        self.get_logger().info('Cave Explorer ready! Phase 1: Tunnel entry')
        self.create_timer(0.1,  self.update_pose)
        self.create_timer(1.0,  self.navigation_loop)
        self.create_timer(0.05, self.movement_loop)

    def map_cb(self, msg):
        self.map_info = msg.info
        self.map_data = np.array(msg.data, dtype=np.int8).reshape(
            msg.info.height, msg.info.width)

    def update_pose(self):
        req = GetEntityState.Request()
        req.name = 'x3_lidar_drone'
        req.reference_frame = 'world'
        self.get_client.call_async(req).add_done_callback(self._pose_cb)

    def _pose_cb(self, future):
        try:
            res = future.result()
            if res.success:
                p = res.state.pose.position
                self.x, self.y, self.z = p.x, p.y, p.z
                q = res.state.pose.orientation
                self.yaw = math.atan2(
                    2*(q.w*q.z+q.x*q.y), 1-2*(q.y**2+q.z**2))
                self.initialized = True
        except Exception:
            pass

    def dist_to(self, tx, ty, tz=None):
        if tz is None:
            return math.sqrt((tx-self.x)**2+(ty-self.y)**2)
        return math.sqrt((tx-self.x)**2+(ty-self.y)**2+(tz-self.z)**2)

    def map_to_world(self, mx, my):
        wx = mx * self.map_info.resolution + self.map_info.origin.position.x
        wy = my * self.map_info.resolution + self.map_info.origin.position.y
        return wx, wy

    def find_frontiers(self):
        if self.map_data is None or self.map_info is None:
            return []
        h, w = self.map_data.shape
        frontiers = []
        for my in range(1, h-1, 2):
            for mx in range(1, w-1, 2):
                if self.map_data[my, mx] != -1:
                    continue
                neighbors = [
                    self.map_data[my-1, mx],
                    self.map_data[my+1, mx],
                    self.map_data[my, mx-1],
                    self.map_data[my, mx+1],
                ]
                if 0 in neighbors:
                    wx, wy = self.map_to_world(mx, my)
                    if wx < 9.0:
                        frontiers.append((wx, wy))
        return frontiers

    def navigation_loop(self):
        if not self.initialized:
            return
        if self.phase == 'entry':
            if self.entry_idx >= len(self.entry_waypoints):
                self.get_logger().info(
                    'Tunnel entry complete! Switching to frontier exploration.')
                self.phase = 'frontier'
                self.current_goal = None
                return
            wp = self.entry_waypoints[self.entry_idx]
            d = self.dist_to(*wp)
            if d < 1.2:
                self.get_logger().info(
                    f'Entry WP {self.entry_idx+1}/'
                    f'{len(self.entry_waypoints)} reached!')
                self.entry_idx += 1
                if self.entry_idx < len(self.entry_waypoints):
                    wp = self.entry_waypoints[self.entry_idx]
            self.target_x = wp[0]
            self.target_y = wp[1]
            self.target_z = wp[2]
            self.get_logger().info(
                f'[Entry {self.entry_idx+1}/{len(self.entry_waypoints)}] '
                f'→ ({wp[0]:.1f},{wp[1]:.1f}) dist={d:.1f}m')

        elif self.phase == 'frontier':
            if self.current_goal is not None:
                d = self.dist_to(*self.current_goal)
                if d < 1.5:
                    self.get_logger().info(
                        f'Frontier reached! Total: {len(self.visited)+1}')
                    self.visited.append(self.current_goal)
                    self.current_goal = None
                    self.stuck_timer = 0
            self.stuck_timer += 1
            if self.current_goal is not None and self.stuck_timer > 20:
                self.get_logger().warn('Stuck! Skipping.')
                self.visited.append(self.current_goal)
                self.current_goal = None
                self.stuck_timer = 0
            if self.current_goal is None:
                frontiers = self.find_frontiers()
                self.get_logger().info(f'Found {len(frontiers)} frontiers')
                best = None
                best_dist = float('inf')
                random.shuffle(frontiers)
                for fx, fy in frontiers[:150]:
                    already = any(
                        math.sqrt((fx-vx)**2+(fy-vy)**2) < 2.0
                        for vx, vy in self.visited)
                    if already:
                        continue
                    d = self.dist_to(fx, fy)
                    if 1.0 < d < 25.0 and d < best_dist:
                        best_dist = d
                        best = (fx, fy)
                if best is not None:
                    self.current_goal = best
                    self.target_x = best[0]
                    self.target_y = best[1]
                    self.target_z = 1.5
                    self.stuck_timer = 0
                    self.get_logger().info(
                        f'New frontier: ({best[0]:.1f},{best[1]:.1f}) '
                        f'dist={best_dist:.1f}m')
                else:
                    self.get_logger().info('All frontiers visited! Cave mapped!')

    def movement_loop(self):
        if not self.initialized:
            return
        twist = Twist()
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dz = self.target_z - self.z
        dist_xy = math.sqrt(dx*dx + dy*dy)
        target_yaw = math.atan2(dy, dx) if dist_xy > 0.3 else self.yaw
        yaw_err = target_yaw - self.yaw
        while yaw_err >  math.pi: yaw_err -= 2*math.pi
        while yaw_err < -math.pi: yaw_err += 2*math.pi
        twist.angular.z = max(-1.0, min(1.0, yaw_err * 2.0))
        if abs(yaw_err) < 0.4:
            twist.linear.x = min(0.6, dist_xy * 0.5)
        twist.linear.z = max(-0.5, min(0.5, dz * 1.2))
        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.cmd_pub.publish(Twist())
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
