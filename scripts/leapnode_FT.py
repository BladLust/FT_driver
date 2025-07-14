#!/usr/bin/env python3

import numpy as np
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import String

# from FT_client import *
from leap_hardware.FT_client import *
import leap_hardware.leap_hand_utils as lhu
from leap_hardware.srv import *

# LEAP hand conventions:
# 180 is flat out home pose for the index, middle, ring, finger MCPs.
# Applying a positive angle closes the joints more and more to curl closed.
# The MCP is centered at 180 and can move positive or negative to that.
# This convention aligns well with FT motors using a [0, 360] degree range.

# The joint numbering goes from Index (0-3), Middle(4-7), Ring(8-11) to Thumb(12-15) and from MCP Side, MCP Forward, PIP, DIP for each finger.
# For instance, the MCP Side of Index is ID 0, the MCP Forward of Ring is 9, the DIP of Ring is 11

# I recommend you only query when necessary and below 90 samples a second.  Used the combined commands if you can to save time.  Also don't forget about the USB latency settings in the readme.
# The services allow you to always have the latest data when you want it, and not spam the communication lines with unused data.


class LeapNode:
    def __init__(self):
        # Some parameters to control the hand
        self.kP = float(rospy.get_param("/leaphand_node/kP", 30.0))
        self.kI = float(rospy.get_param("/leaphand_node/kI", 0.0))
        self.kD = float(rospy.get_param("/leaphand_node/kD", 70.0))
        self.curr_lim = float(rospy.get_param("/leaphand_node/curr_lim", 350.0))  # Current limit
        self.free_move = rospy.get_param("/leaphand_node/free_move", True)  # Free movement mode
        self.ema_amount = 0.2
        # Position is assumed to be in degrees, matching the LEAP hand convention.
        self.curr_pos = np.ones(16) * np.pi
        self.curr_pos[13] += 1.61
        self.prev_pos = self.pos = self.curr_pos

        # Subscribes to a variety of sources that can command the hand
        rospy.Subscriber("/leaphand_node/cmd_leap", JointState, self._receive_pose)
        rospy.Subscriber("/leaphand_node/cmd_allegro", JointState, self._receive_allegro)
        rospy.Subscriber("/leaphand_node/cmd_ones", JointState, self._receive_ones)

        # Creates services that can give information about the hand out
        rospy.Service("leap_position", leap_position, self.pos_srv)
        rospy.Service("leap_velocity", leap_velocity, self.vel_srv)
        rospy.Service("leap_effort", leap_effort, self.eff_srv)
        # rospy.Service("leap_pos_vel", leap_posveleff, self.pos_vel_srv)
        # You can put the correct port here or have the node auto-search for a hand at the first 3 ports.
        # For example ls /dev/serial/by-id/* to find your LEAP Hand. Then use the result.
        # For example: /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7W91VW-if00-port0
        # You can put the correct port here or have the node auto-search for a hand at the first 3 ports.
        self.motors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

        # --- FTClient Implementation ---
        # Search for the hand on the first 3 USB ports
        for i in range(3):
            port = f"/dev/ttyUSB{i}"
            try:
                rospy.loginfo(f"Attempting to connect to LEAP Hand on port {port}...")
                # Instantiate FTClient, using degrees to match hand conventions
                self.ft_client = FTClient(
                    self.motors, port, baudrate=115200, use_degrees=False
                )
                self.ft_client.connect()
                rospy.loginfo(f"Successfully connected to LEAP Hand on port {port}.")
                break
            except Exception as e:
                rospy.logwarn(f"Failed to connect on {port}: {e}")
                if i == 2:
                    rospy.logerr("Could not connect to LEAP Hand on any port.")
                    raise

        # Enable torque if free_move is False
        if not self.free_move:
            self.ft_client.set_torque_enabled(self.motors, True)

        # Set PID gains
        self.ft_client.set_pid_gains(p=int(self.kP), i=int(self.kI), d=int(self.kD))

        # Set initial position
        self.ft_client.write_desired_pos_simple(self.motors, self.curr_pos)

    # Receive LEAP pose and directly control the robot.
    def _receive_pose(self, msg):
        pose = msg.position
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.ft_client.write_desired_pos_simple(self.motors, self.curr_pos)

    # Allegro compatibility, first read the allegro publisher and then convert to leap
    def _receive_allegro(self, msg):
        pose = lhu.allegro_to_LEAPhand(msg.position, zeros=False)
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.ft_client.write_desired_pos_simple(self.motors, self.curr_pos)

    # Sim compatibility, first read the sim publisher and then convert to leap
    def _receive_ones(self, msg):
        pose = lhu.sim_ones_to_LEAPhand(np.array(msg.position))
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.ft_client.write_desired_pos_simple(self.motors, self.curr_pos)

    # Service that reads and returns the pos of the robot in degrees.
    def pos_srv(self, req):
        return {"position": self.ft_client.read_pos().tolist()}

    # Service that reads and returns the vel of the robot in deg/s.
    def vel_srv(self, req):
        return {"velocity": self.ft_client.read_vel().tolist()}

    # Service that reads and returns the effort/current of the robot.
    # NOTE: FTClient does not support current reading, returns zeros.
    def eff_srv(self, req):
        return {"effort": np.zeros(len(self.motors)).tolist()}

    # Use these combined services to save latency if you need multiple datapoints
    # NOTE: FTClient does not support current reading, returns zeros.
    # def pos_vel_srv(self, req):
    #     pos, vel = self.ft_client.read_pos_vel()
    #     return {
    #         "position": pos.tolist(),
    #         "velocity": vel.tolist(),
    #         "effort": np.zeros_like(pos).tolist(),
    #     }


def main():
    rospy.init_node("leaphand_node")
    leaphand_node = LeapNode()
    rospy.spin()


if __name__ == "__main__":
    main()
