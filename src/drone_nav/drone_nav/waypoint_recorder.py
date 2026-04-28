#!/usr/bin/env python3
"""
Auto Waypoint Recorder
Records drone position every time it moves more than 1 meter.
Just fly normally — waypoints are saved automatically.
Press Ctrl+C to stop and save.
"""
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import GetEntityState
import json
import math

class WaypointRecorder(Node):
    def __init__(self):
        super().__init__('waypoint_recorder')
        self.set_parameters([
            rclpy.parameter.Parameter('use_sim_time',
                rclpy.parameter.Parameter.Type.BOOL, True)])

        self.waypoints = []
        self.x = 0.0; self.y = 0.0; self.z = 0.0
        self.last_x = None; self.last_y = None; self.last_z = None

        # Record a waypoint every time drone moves this far
        self.min_distance = 1.0  # meters

        self.get_client = self.create_client(
            GetEntityState, '/get_entity_state')
        self.get_client.wait_for_service(timeout_sec=10.0)

        # Check position every 0.5 seconds
        self.create_timer(0.5, self.update_and_record)

        print('\n' + '='*50)
        print('AUTO WAYPOINT RECORDER')
        print('='*50)
        print(f'Recording waypoint every {self.min_distance}m moved')
        print('Just fly the drone normally!')
        print('Press Ctrl+C to stop and save waypoints.json')
        print('='*50 + '\n')

    def update_and_record(self):
        req = GetEntityState.Request()
        req.name = 'x3_lidar_drone'
        req.reference_frame = 'world'
        self.get_client.call_async(req).add_done_callback(self._cb)

    def _cb(self, future):
        try:
            res = future.result()
            if not res.success:
                return

            p = res.state.pose.position
            self.x, self.y, self.z = p.x, p.y, p.z

            # First waypoint
            if self.last_x is None:
                self.record_waypoint()
                return

            # Record if moved enough
            dist = math.sqrt(
                (self.x - self.last_x)**2 +
                (self.y - self.last_y)**2 +
                (self.z - self.last_z)**2)

            if dist >= self.min_distance:
                self.record_waypoint()

        except Exception:
            pass

    def record_waypoint(self):
        wp = {
            'label': f'wp_{len(self.waypoints)+1}',
            'x': round(self.x, 2),
            'y': round(self.y, 2),
            'z': round(self.z, 2)
        }
        self.waypoints.append(wp)
        self.last_x = self.x
        self.last_y = self.y
        self.last_z = self.z
        print(f'✓ wp_{len(self.waypoints):03d}: '
              f'({wp["x"]:.2f}, {wp["y"]:.2f}, {wp["z"]:.2f})')

    def save(self):
        if not self.waypoints:
            print('No waypoints recorded!')
            return
        path = '/home/jayakar/RSN_Proj/maps/waypoints.json'
        with open(path, 'w') as f:
            json.dump(self.waypoints, f, indent=2)
        print('\n' + '='*50)
        print(f'Saved {len(self.waypoints)} waypoints to:')
        print(path)
        print('='*50)


def main(args=None):
    rclpy.init(args=args)
    node = WaypointRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
