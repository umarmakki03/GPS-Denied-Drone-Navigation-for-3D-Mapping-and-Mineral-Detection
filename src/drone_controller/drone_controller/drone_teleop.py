#!/usr/bin/env python3
"""
Drone Teleop
- HOLD key to move, RELEASE to stop
- Speed adjustment keys
- Works with planar_move (X/Y/Yaw) + hover controller (Z)
"""
import sys
import tty
import termios
import threading
import time
import select
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

KEY_BINDINGS = {
    'w': ( 1,  0,  0,  0),
    's': (-1,  0,  0,  0),
    'a': ( 0,  1,  0,  0),
    'd': ( 0, -1,  0,  0),
    'r': ( 0,  0,  1,  0),
    'f': ( 0,  0, -1,  0),
    'q': ( 0,  0,  0,  1),
    'e': ( 0,  0,  0, -1),
}

SPEED_UP_KEY    = '='
SPEED_DOWN_KEY  = '-'
VERT_UP_KEY     = ']'
VERT_DOWN_KEY   = '['
STOP_KEY        = ' '

DEFAULT_LINEAR   = 0.6
DEFAULT_VERTICAL = 0.6
DEFAULT_ANGULAR  = 1.0
SPEED_STEP       = 0.1
MIN_SPEED        = 0.05
MAX_SPEED        = 1.5

BANNER = """
Drone Teleop — HOLD key to move, RELEASE to stop
=================================================
Movement:
  w / s      : forward / backward
  a / d      : strafe left / right
  r / f      : up / down
  q / e      : yaw left / right
  SPACE      : emergency stop

Speed control:
  = / -      : increase / decrease linear speed
  ] / [      : increase / decrease vertical speed

Ctrl+C to quit.
=================================================
"""


class DroneTeleop(Node):
    def __init__(self):
        super().__init__('drone_teleop')
        self.set_parameters([
            rclpy.parameter.Parameter('use_sim_time',
                rclpy.parameter.Parameter.Type.BOOL, True)])
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.linear_speed   = DEFAULT_LINEAR
        self.vertical_speed = DEFAULT_VERTICAL
        self.angular_speed  = DEFAULT_ANGULAR
        self.held_keys = {}
        self.lock = threading.Lock()
        self.e_stop = False
        self.create_timer(0.02, self.publish_velocity)

    def publish_velocity(self):
        twist = Twist()
        now = time.time()
        with self.lock:
            if self.e_stop:
                self.pub.publish(twist)
                return
            expired = [k for k, t in self.held_keys.items()
                      if now - t > 0.25]
            for k in expired:
                del self.held_keys[k]
            active = set(self.held_keys.keys())
        for key in active:
            if key in KEY_BINDINGS:
                vx, vy, vz, wz = KEY_BINDINGS[key]
                twist.linear.x  += vx * self.linear_speed
                twist.linear.y  += vy * self.linear_speed
                twist.linear.z  += vz * self.vertical_speed
                twist.angular.z += wz * self.angular_speed
        twist.linear.x  = max(-MAX_SPEED, min(MAX_SPEED, twist.linear.x))
        twist.linear.y  = max(-MAX_SPEED, min(MAX_SPEED, twist.linear.y))
        twist.linear.z  = max(-MAX_SPEED, min(MAX_SPEED, twist.linear.z))
        twist.angular.z = max(-MAX_SPEED, min(MAX_SPEED, twist.angular.z))
        self.pub.publish(twist)

    def key_seen(self, key):
        with self.lock:
            if key == STOP_KEY:
                self.e_stop = not self.e_stop
                print(f'\r  Emergency stop: {"ON" if self.e_stop else "OFF"}   ')
                return
            if key == SPEED_UP_KEY:
                self.linear_speed = min(MAX_SPEED, round(self.linear_speed + SPEED_STEP, 2))
                self._print_speeds(); return
            if key == SPEED_DOWN_KEY:
                self.linear_speed = max(MIN_SPEED, round(self.linear_speed - SPEED_STEP, 2))
                self._print_speeds(); return
            if key == VERT_UP_KEY:
                self.vertical_speed = min(MAX_SPEED, round(self.vertical_speed + SPEED_STEP, 2))
                self._print_speeds(); return
            if key == VERT_DOWN_KEY:
                self.vertical_speed = max(MIN_SPEED, round(self.vertical_speed - SPEED_STEP, 2))
                self._print_speeds(); return
            if key in KEY_BINDINGS:
                self.e_stop = False
                self.held_keys[key] = time.time()

    def _print_speeds(self):
        print(f'\r  Linear: {self.linear_speed:.2f} m/s  |  '
              f'Vertical: {self.vertical_speed:.2f} m/s  |  '
              f'Angular: {self.angular_speed:.2f} rad/s    ',
              end='', flush=True)


def read_keys(node, stop_event):
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while not stop_event.is_set() and rclpy.ok():
            ready, _, _ = select.select([sys.stdin], [], [], 0.02)
            if ready:
                key = sys.stdin.read(1)
                if key == '\x03':
                    stop_event.set(); break
                if key == '\x1b':
                    try:
                        rest = sys.stdin.read(2)
                        arrow_map = {
                            '\x1b[A': 'w',
                            '\x1b[B': 's',
                            '\x1b[C': 'd',
                            '\x1b[D': 'a',
                        }
                        mapped = arrow_map.get(key+rest)
                        if mapped:
                            node.key_seen(mapped)
                    except Exception:
                        pass
                    continue
                node.key_seen(key)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        node.pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = DroneTeleop()
    print(BANNER)
    node._print_speeds()
    print()
    stop_event = threading.Event()
    key_thread = threading.Thread(
        target=read_keys, args=(node, stop_event), daemon=True)
    key_thread.start()
    try:
        while rclpy.ok() and not stop_event.is_set():
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub.publish(Twist())
        print('\nDrone stopped.')
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
