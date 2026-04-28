#!/usr/bin/env python3
"""
Simulated NIR Spectroscopy Node
- Subscribes to /front_camera/image_raw
- Extracts dominant color from center of image
- Matches against USGS spectral signatures
- Publishes mineral detection with confidence
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String, ColorRGBA
from gazebo_msgs.srv import GetEntityState
import numpy as np
import math

MINERAL_SIGNATURES = {
    'Quartz': {
        'rgb': (230, 220, 210),
        'color_rviz': (0.9, 0.9, 0.9),
        'description': 'Silicon dioxide — common in granite veins',
        'confidence_threshold': 0.65,
    },
    'Hematite': {
        'rgb': (180, 40, 30),
        'color_rviz': (0.8, 0.1, 0.1),
        'description': 'Iron oxide ore — primary iron source',
        'confidence_threshold': 0.65,
    },
    'Malachite': {
        'rgb': (30, 160, 60),
        'color_rviz': (0.1, 0.7, 0.2),
        'description': 'Copper carbonate — indicator of copper deposits',
        'confidence_threshold': 0.65,
    },
    'Chalcopyrite': {
        'rgb': (200, 160, 20),
        'color_rviz': (0.8, 0.6, 0.0),
        'description': 'Copper iron sulfide — primary copper ore',
        'confidence_threshold': 0.65,
    },
    'Limestone': {
        'rgb': (60, 100, 200),
        'color_rviz': (0.2, 0.4, 0.8),
        'description': 'Calcium carbonate — sedimentary rock',
        'confidence_threshold': 0.65,
    },
}


class SpectroscopyNode(Node):
    def __init__(self):
        super().__init__('spectroscopy_node')
        self.set_parameters([
            rclpy.parameter.Parameter('use_sim_time',
                rclpy.parameter.Parameter.Type.BOOL, True)])

        self.detections = {}
        self.detection_cooldown = 0
        self.drone_x = 0.0
        self.drone_y = 0.0
        self.drone_z = 0.0

        self.result_pub = self.create_publisher(
            String, '/spectroscopy/result', 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/spectroscopy/markers', 10)

        self.image_sub = self.create_subscription(
            Image, '/front_camera/image_raw', self.image_cb, 1)

        self.get_client = self.create_client(
            GetEntityState, '/get_entity_state')
        self.get_client.wait_for_service(timeout_sec=10.0)

        self.create_timer(0.1, self.update_pose)
        self.create_timer(1.0, self.publish_markers)

        self.get_logger().info('Spectroscopy node ready!')
        self.get_logger().info(
            'Watching /front_camera/image_raw for mineral signatures')

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
                self.drone_x = p.x
                self.drone_y = p.y
                self.drone_z = p.z
        except Exception:
            pass

    def image_cb(self, msg):
        if self.detection_cooldown > 0:
            self.detection_cooldown -= 1
            return
        try:
            img = np.frombuffer(msg.data, dtype=np.uint8)
            img = img.reshape(msg.height, msg.width, 3)

            h, w = msg.height, msg.width
            cy, cx = h//2, w//2
            rh, rw = h//5, w//5
            center = img[cy-rh:cy+rh, cx-rw:cx+rw]

            avg_r = float(np.mean(center[:,:,0]))
            avg_g = float(np.mean(center[:,:,1]))
            avg_b = float(np.mean(center[:,:,2]))

            best_mineral = None
            best_confidence = 0.0
            max_dist = math.sqrt(3 * 255**2)

            for mineral, sig in MINERAL_SIGNATURES.items():
                ref_r, ref_g, ref_b = sig['rgb']
                dist = math.sqrt(
                    (avg_r-ref_r)**2 +
                    (avg_g-ref_g)**2 +
                    (avg_b-ref_b)**2)
                confidence = 1.0 - (dist / max_dist)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_mineral = mineral

            threshold = MINERAL_SIGNATURES[best_mineral][
                'confidence_threshold']

            if best_confidence > threshold:
                if best_mineral not in self.detections:
                    self.detections[best_mineral] = {
                        'x': self.drone_x,
                        'y': self.drone_y,
                        'z': self.drone_z,
                        'confidence': best_confidence,
                    }
                    self.get_logger().info('='*50)
                    self.get_logger().info(
                        f'MINERAL DETECTED: {best_mineral}')
                    self.get_logger().info(
                        f'Confidence: {best_confidence*100:.1f}%')
                    self.get_logger().info(
                        f'RGB: ({avg_r:.0f},{avg_g:.0f},{avg_b:.0f})')
                    self.get_logger().info(
                        f'Location: ({self.drone_x:.2f},'
                        f'{self.drone_y:.2f},{self.drone_z:.2f})')
                    self.get_logger().info(
                        f'Info: {MINERAL_SIGNATURES[best_mineral]["description"]}')
                    self.get_logger().info(
                        f'Total: {len(self.detections)}/5')
                    self.get_logger().info('='*50)

                    msg_out = String()
                    msg_out.data = (
                        f'{best_mineral}|'
                        f'{best_confidence*100:.1f}%|'
                        f'{self.drone_x:.2f},'
                        f'{self.drone_y:.2f},'
                        f'{self.drone_z:.2f}')
                    self.result_pub.publish(msg_out)
                    self.publish_markers()
                    self.detection_cooldown = 20

        except Exception as e:
            self.get_logger().warn(f'Image error: {e}')

    def publish_markers(self):
        markers = MarkerArray()
        for i, (mineral, info) in enumerate(self.detections.items()):
            sig = MINERAL_SIGNATURES[mineral]
            r, g, b = sig['color_rviz']

            mk = Marker()
            mk.header.frame_id = 'map'
            mk.header.stamp = self.get_clock().now().to_msg()
            mk.ns = 'spectroscopy'
            mk.id = i
            mk.type = Marker.SPHERE
            mk.action = Marker.ADD
            mk.pose.position.x = info['x']
            mk.pose.position.y = info['y']
            mk.pose.position.z = info['z']
            mk.pose.orientation.w = 1.0
            mk.scale.x = mk.scale.y = mk.scale.z = 1.0
            mk.color = ColorRGBA(r=r, g=g, b=b, a=1.0)
            markers.markers.append(mk)

            txt = Marker()
            txt.header.frame_id = 'map'
            txt.header.stamp = self.get_clock().now().to_msg()
            txt.ns = 'spectroscopy_labels'
            txt.id = i + 100
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position.x = info['x']
            txt.pose.position.y = info['y']
            txt.pose.position.z = info['z'] + 1.5
            txt.pose.orientation.w = 1.0
            txt.scale.z = 1.0
            txt.text = (f'{mineral}\n'
                       f'{info["confidence"]*100:.1f}% confidence')
            txt.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            markers.markers.append(txt)

        self.marker_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = SpectroscopyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
