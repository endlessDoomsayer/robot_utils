#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

class WaitForTopics:
    def __init__(self, node: Node, topics: list, on_ready_callback, check_period=1.0):
        """
        :param node: The parent ROS2 node instance (self)
        :param topics: List of topic strings to wait for
        :param on_ready_callback: Function to call when all topics have publishers
        :param check_period: How often to check the ROS graph (seconds)
        """
        self.node = node
        self.topics = [topics] if isinstance(topics, str) else topics
        self.callback = on_ready_callback
        
        self.node.get_logger().info(f"WaitForTopics: Waiting for publishers on: {self.topics}")
        
        # Start the internal monitoring timer
        self.timer = self.node.create_timer(check_period, self._check_topics)

    def _check_topics(self):
        if all(self.node.count_publishers(t) > 0 for t in self.topics):
            self.node.get_logger().info("All required topics detected. Initializing Node...")
            self.timer.destroy()
            self.callback()