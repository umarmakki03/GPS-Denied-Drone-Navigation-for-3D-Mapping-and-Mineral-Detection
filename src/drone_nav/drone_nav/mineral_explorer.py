#!/usr/bin/env python3
"""
Mineral Explorer
Pipeline:
  1. Camera identifies mineral type from color
  2. LiDAR point cloud clustered in drone's forward region
  3. DBSCAN finds mineral cluster
  4. 3D bounding box + colored cluster visualized in RViz
  5. Drone follows waypoints, detects along the way
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import Image, PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from gazebo_msgs.srv import GetEntityState
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import math
import json

# Mineral color signatures adjusted for tunnel lighting
MINERAL_SIGNATURES = {
    'Quartz':       {'rgb': (178,178,178), 'rviz': (0.9,0.9,0.9)},
    'Hematite':     {'rgb': (178,0,0),     'rviz': (0.8,0.1,0.1)},
    'Malachite':    {'rgb': (0,153,25),    'rviz': (0.1,0.7,0.2)},
    'Chalcopyrite': {'rgb': (178,127,0),   'rviz': (0.8,0.6,0.0)},
    'Limestone':    {'rgb': (25,76,178),   'rviz': (0.2,0.4,0.8)},
}


def dbscan(pts, eps=0.5, min_samples=4):
    """Pure numpy DBSCAN."""
    n = len(pts)
    if n == 0:
        return np.array([], dtype=int)
    labels = -np.ones(n, dtype=int)
    cid = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        d = np.sqrt(((pts - pts[i])**2).sum(axis=1))
        nb = list(np.where(d <= eps)[0])
        if len(nb) < min_samples:
            continue
        labels[i] = cid
        seen = set(nb)
        q = nb[:]
        while q:
            j = q.pop(0)
            labels[j] = cid
            d2 = np.sqrt(((pts - pts[j])**2).sum(axis=1))
            nb2 = list(np.where(d2 <= eps)[0])
            if len(nb2) >= min_samples:
                for k in nb2:
                    if k not in seen:
                        seen.add(k); q.append(k)
        cid += 1
    return labels


class MineralExplorer(Node):
    def __init__(self):
        super().__init__('mineral_explorer')
        self.set_parameters([
            rclpy.parameter.Parameter('use_sim_time',
                rclpy.parameter.Parameter.Type.BOOL, True)])

        # Load waypoints
        with open('/home/jayakar/RSN_Proj/maps/waypoints.json') as f:
            data = json.load(f)
        self.waypoints = [(w['x'],w['y'],w['z'],w['label'])
                         for w in data]

        # Drone state
        self.x=0.0; self.y=0.0; self.z=0.0; self.yaw=0.0
        self.initialized = False

        # Navigation
        self.wp_idx = 0
        self.stuck_timer = 0
        self.done = False

        # Sensor data
        self.latest_image = None
        self.latest_cloud = None

        # Detections: name → detection dict
        self.detections = {}
        self.cooldown = {}

        # Publishers
        self.cmd_pub = self.create_publisher(Twist,'/cmd_vel',10)
        self.marker_pub = self.create_publisher(
            MarkerArray,'/mineral_detections',10)

        # Subscribers
        self.create_subscription(
            Image,'/front_camera/image_raw',self.img_cb,1)
        self.create_subscription(
            PointCloud2,'/lidar/points',self.cloud_cb,1)

        self.get_client = self.create_client(
            GetEntityState,'/get_entity_state')
        self.get_client.wait_for_service(timeout_sec=10.0)

        self.get_logger().info('='*50)
        self.get_logger().info('MINERAL EXPLORER READY')
        self.get_logger().info(
            f'Waypoints: {len(self.waypoints)} | '
            'Camera+LiDAR detection pipeline active')
        self.get_logger().info('='*50)

        self.create_timer(0.1,  self.update_pose)
        self.create_timer(0.5,  self.nav_loop)
        self.create_timer(0.05, self.move_loop)
        self.create_timer(0.25, self.detect_minerals)
        self.create_timer(1.0,  self.publish_markers)

    def img_cb(self, msg):
        self.latest_image = msg

    def cloud_cb(self, msg):
        self.latest_cloud = msg

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
                self.x,self.y,self.z = p.x,p.y,p.z
                q = res.state.pose.orientation
                self.yaw = math.atan2(
                    2*(q.w*q.z+q.x*q.y),
                    1-2*(q.y**2+q.z**2))
                self.initialized = True
        except Exception:
            pass

    # ── Step 1: Camera color classification ───────────────────
    def classify_camera(self):
        """Returns (mineral_name, confidence) or (None, 0)."""
        if self.latest_image is None:
            return None, 0.0
        try:
            msg = self.latest_image
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3)
            h,w = msg.height, msg.width
            cy,cx = h//2, w//2
            rh,rw = h//4, w//4
            crop = img[cy-rh:cy+rh, cx-rw:cx+rw]
            avg = crop.mean(axis=(0,1))  # [R,G,B]

            # Check color saturation — walls are grey/brown (low sat)
            # minerals are vivid (high saturation)
            r_val = float(avg[0])
            g_val = float(avg[1])
            b_val = float(avg[2])

            max_c = max(r_val, g_val, b_val)
            min_c = min(r_val, g_val, b_val)
            saturation = (max_c - min_c) / (max_c + 1e-6)
            brightness = max_c

            self.get_logger().info(
                f'RGB:({r_val:.0f},{g_val:.0f},{b_val:.0f}) '
                f'sat={saturation:.2f} bright={brightness:.0f}')

            # Reject dull grey/brown walls
            # minerals must have visible color saturation
            if saturation < 0.20 or brightness < 10:
                return None, 0.0

            best=None; best_conf=0.0
            max_d = math.sqrt(3*255**2)
            for name,sig in MINERAL_SIGNATURES.items():
                ref = np.array(sig['rgb'], dtype=float)
                d = float(np.linalg.norm(avg - ref))
                conf = 1.0 - d/max_d
                if conf > best_conf:
                    best_conf=conf; best=name

            self.get_logger().info(
                f'Camera: {best} {best_conf*100:.1f}% ')
            return best, best_conf
        except Exception as e:
            self.get_logger().warn(f'Camera: {e}')
            return None, 0.0

    # ── Step 2+3: LiDAR clustering ─────────────────────────────
    def cluster_lidar(self):
        """
        Get LiDAR points in front of drone,
        run DBSCAN, return best cluster.
        Returns (cluster_pts, center, min_pt, max_pt) or None.
        """
        if self.latest_cloud is None:
            return None
        try:
            # Read point cloud
            raw = list(pc2.read_points(
                self.latest_cloud,
                field_names=('x','y','z'),
                skip_nans=True))
            if len(raw) < 10:
                return None

            # Build Nx3 array safely
            pts = np.array(
                [[float(p[0]),float(p[1]),float(p[2])]
                 for p in raw], dtype=np.float64)

            # Transform base_link → world frame
            cy = math.cos(self.yaw); sy = math.sin(self.yaw)
            world = np.zeros_like(pts)
            world[:,0] = pts[:,0]*cy - pts[:,1]*sy + self.x
            world[:,1] = pts[:,0]*sy + pts[:,1]*cy + self.y
            world[:,2] = pts[:,2] + self.z

            # Keep points in FORWARD cone, 1–8m away
            fwd_x = math.cos(self.yaw)
            fwd_y = math.sin(self.yaw)
            rel = world[:,:2] - np.array([self.x,self.y])
            fwd_dot = rel[:,0]*fwd_x + rel[:,1]*fwd_y
            dist2d  = np.linalg.norm(rel, axis=1)

            mask = (
                (fwd_dot > 0.5) &
                (dist2d > 0.5)  &
                (dist2d < 12.0)  &
                (world[:,2] > 0.05) &
                (world[:,2] < 4.0))
            front = world[mask]

            if len(front) < 8:
                return None

            # DBSCAN cluster
            labels = dbscan(front, eps=0.6, min_samples=4)
            unique = set(labels[labels >= 0])
            if not unique:
                return None

            # Pick closest cluster
            best_cl = None; best_d = float('inf')
            for lbl in unique:
                cl = front[labels==lbl]
                if len(cl) < 4:
                    continue
                ctr = cl.mean(axis=0)
                d = math.sqrt((ctr[0]-self.x)**2+(ctr[1]-self.y)**2)
                if d < best_d:
                    best_d=d; best_cl=cl

            if best_cl is None:
                return None

            mn  = best_cl.min(axis=0)
            mx  = best_cl.max(axis=0)
            ctr = (mn+mx)/2
            return best_cl, ctr, mn, mx

        except Exception as e:
            self.get_logger().warn(f'LiDAR cluster: {e}')
            return None

    # ── Combined detection ─────────────────────────────────────
    def detect_minerals(self):
        if not self.initialized:
            return

        # Camera classification
        mineral, conf = self.classify_camera()
        if mineral is None or conf < 0.55:
            return
        if mineral in self.detections:
            return
        if self.cooldown.get(mineral,0) > 0:
            self.cooldown[mineral] -= 1
            return

        self.get_logger().info(
            f'Camera: {mineral} {conf*100:.1f}% — clustering LiDAR...')

        # LiDAR clustering
        result = self.cluster_lidar()
        if result is None:
            self.get_logger().warn(
                f'No LiDAR cluster found for {mineral}')
            self.cooldown[mineral] = 10
            return

        cluster, center, mn, mx = result

        # Store detection
        self.detections[mineral] = {
            'confidence': conf,
            'center':  tuple(center),
            'min':     tuple(mn),
            'max':     tuple(mx),
            'cluster': cluster,
            'n_pts':   len(cluster),
        }
        self.cooldown[mineral] = 40

        self.get_logger().info('='*50)
        self.get_logger().info(f'MINERAL DETECTED: {mineral}')
        self.get_logger().info(
            f'Confidence:  {conf*100:.1f}%')
        self.get_logger().info(
            f'Location:    ({center[0]:.2f},'
            f'{center[1]:.2f},{center[2]:.2f})')
        self.get_logger().info(
            f'Cluster pts: {len(cluster)}')
        self.get_logger().info(
            f'BBox:        ({mn[0]:.1f},{mn[1]:.1f},'
            f'{mn[2]:.1f}) → ({mx[0]:.1f},'
            f'{mx[1]:.1f},{mx[2]:.1f})')
        self.get_logger().info(
            f'Total found: {len(self.detections)}/5')
        self.get_logger().info('='*50)
        self.publish_markers()

    # ── Navigation ─────────────────────────────────────────────
    def nav_loop(self):
        if not self.initialized or self.done:
            return
        if self.wp_idx >= len(self.waypoints):
            found = len(self.detections)
            self.get_logger().info('='*50)
            self.get_logger().info('MISSION COMPLETE!')
            self.get_logger().info(
                f'Minerals found: {found}/5')
            for n,d in self.detections.items():
                cx,cy,cz = d['center']
                self.get_logger().info(
                    f'  ✓ {n} @ ({cx:.1f},{cy:.1f},'
                    f'{cz:.1f}) {d["confidence"]*100:.1f}%')
            self.get_logger().info('='*50)
            self.cmd_pub.publish(Twist())
            self.done = True
            return

        tx,ty,tz,label = self.waypoints[self.wp_idx]
        d = math.sqrt(
            (tx-self.x)**2+(ty-self.y)**2+(tz-self.z)**2)

        if d < 1.2:
            self.wp_idx += 1; self.stuck_timer=0
            return

        self.stuck_timer += 1
        if self.stuck_timer > 20:
            self.wp_idx += 1; self.stuck_timer=0
            return

        if self.stuck_timer % 10 == 0:
            self.get_logger().info(
                f'WP {self.wp_idx+1}/{len(self.waypoints)} '
                f'{label} dist={d:.1f}m | '
                f'Found:{len(self.detections)}/5')

    def move_loop(self):
        if not self.initialized or self.done:
            return
        if self.wp_idx >= len(self.waypoints):
            return
        tx,ty,tz,_ = self.waypoints[self.wp_idx]
        dx=tx-self.x; dy=ty-self.y; dz=tz-self.z
        dxy=math.sqrt(dx*dx+dy*dy)
        twist=Twist()
        tyaw=math.atan2(dy,dx) if dxy>0.3 else self.yaw
        yerr=tyaw-self.yaw
        while yerr> math.pi: yerr-=2*math.pi
        while yerr<-math.pi: yerr+=2*math.pi
        twist.angular.z=max(-1.5,min(1.5,yerr*2.5))
        if abs(yerr)<0.35:
            twist.linear.x=min(0.5,dxy*0.4)
        twist.linear.z=max(-0.5,min(0.5,dz*1.5))
        self.cmd_pub.publish(twist)

    # ── RViz visualization ─────────────────────────────────────
    def publish_markers(self):
        markers = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        # Waypoint trail
        for i,(wx,wy,wz,_) in enumerate(self.waypoints):
            mk=Marker()
            mk.header.frame_id='map'; mk.header.stamp=stamp
            mk.ns='wps'; mk.id=i
            mk.type=Marker.SPHERE; mk.action=Marker.ADD
            mk.pose.position.x=wx
            mk.pose.position.y=wy
            mk.pose.position.z=wz
            mk.pose.orientation.w=1.0
            mk.scale.x=mk.scale.y=mk.scale.z=0.25
            if i<self.wp_idx:
                mk.color=ColorRGBA(r=0.0,g=1.0,b=0.0,a=0.7)
            elif i==self.wp_idx:
                mk.color=ColorRGBA(r=1.0,g=1.0,b=0.0,a=1.0)
            else:
                mk.color=ColorRGBA(r=0.2,g=0.2,b=1.0,a=0.25)
            markers.markers.append(mk)

        # Mineral detections
        for i,(name,det) in enumerate(self.detections.items()):
            r,g,b = MINERAL_SIGNATURES[name]['rviz']
            cx,cy,cz = det['center']
            mn = det['min']; mx = det['max']

            # ── Colored LiDAR cluster points ──────────────────
            cl = det['cluster']
            for j,pt in enumerate(cl[::2]):  # every 2nd point
                dot=Marker()
                dot.header.frame_id='map'
                dot.header.stamp=stamp
                dot.ns=f'cl_{name}'; dot.id=i*50000+j
                dot.type=Marker.SPHERE; dot.action=Marker.ADD
                dot.pose.position.x=float(pt[0])
                dot.pose.position.y=float(pt[1])
                dot.pose.position.z=float(pt[2])
                dot.pose.orientation.w=1.0
                dot.scale.x=dot.scale.y=dot.scale.z=0.18
                dot.color=ColorRGBA(
                    r=float(r),g=float(g),b=float(b),a=1.0)
                markers.markers.append(dot)

            # ── 3D Bounding box (solid transparent) ───────────
            bbox=Marker()
            bbox.header.frame_id='map'; bbox.header.stamp=stamp
            bbox.ns='bbox'; bbox.id=i+500
            bbox.type=Marker.CUBE; bbox.action=Marker.ADD
            bbox.pose.position.x=cx
            bbox.pose.position.y=cy
            bbox.pose.position.z=cz
            bbox.pose.orientation.w=1.0
            bbox.scale.x=max(0.4,abs(mx[0]-mn[0]))
            bbox.scale.y=max(0.4,abs(mx[1]-mn[1]))
            bbox.scale.z=max(0.4,abs(mx[2]-mn[2]))
            bbox.color=ColorRGBA(
                r=float(r),g=float(g),b=float(b),a=0.2)
            markers.markers.append(bbox)

            # ── Wireframe edges ────────────────────────────────
            wire=Marker()
            wire.header.frame_id='map'; wire.header.stamp=stamp
            wire.ns='wire'; wire.id=i+600
            wire.type=Marker.LINE_LIST; wire.action=Marker.ADD
            wire.pose.orientation.w=1.0
            wire.scale.x=0.04
            wire.color=ColorRGBA(
                r=float(r),g=float(g),b=float(b),a=1.0)
            x0,y0,z0 = mn[0],mn[1],mn[2]
            x1,y1,z1 = mx[0],mx[1],mx[2]
            corners=[
                Point(x=x0,y=y0,z=z0),
                Point(x=x1,y=y0,z=z0),
                Point(x=x0,y=y1,z=z0),
                Point(x=x1,y=y1,z=z0),
                Point(x=x0,y=y0,z=z1),
                Point(x=x1,y=y0,z=z1),
                Point(x=x0,y=y1,z=z1),
                Point(x=x1,y=y1,z=z1),
            ]
            for a,b_i in [(0,1),(2,3),(4,5),(6,7),
                          (0,2),(1,3),(4,6),(5,7),
                          (0,4),(1,5),(2,6),(3,7)]:
                wire.points.append(corners[a])
                wire.points.append(corners[b_i])
            markers.markers.append(wire)

            # ── Text label ─────────────────────────────────────
            txt=Marker()
            txt.header.frame_id='map'; txt.header.stamp=stamp
            txt.ns='lbl'; txt.id=i+700
            txt.type=Marker.TEXT_VIEW_FACING
            txt.action=Marker.ADD
            txt.pose.position.x=cx
            txt.pose.position.y=cy
            txt.pose.position.z=cz+1.5
            txt.pose.orientation.w=1.0
            txt.scale.z=1.2
            txt.text=(f'✓ {name}\n'
                     f'{det["confidence"]*100:.1f}%\n'
                     f'{det["n_pts"]} pts')
            txt.color=ColorRGBA(r=1.0,g=1.0,b=1.0,a=1.0)
            markers.markers.append(txt)

        self.marker_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = MineralExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.cmd_pub.publish(Twist())
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
