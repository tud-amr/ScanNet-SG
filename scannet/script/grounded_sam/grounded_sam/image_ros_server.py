#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from image_service.srv import image_service, image_serviceResponse
from message_filters import Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class ImageFetcher:
    def __init__(self):
        self.bridge = CvBridge()
        self.color_image = None
        self.depth_image = None
        
        # Subscribe to the color and depth image topics
        self.color_sub = Subscriber('/dingo1/dinova/camera/rgb/image', Image)
        self.depth_sub = Subscriber('/dingo1/dinova/camera/depth/image', Image)

        rospy.loginfo("Subscribed to color and depth image topics.")

        ats = ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], queue_size=1, slop=0.1)
        ats.registerCallback(self.sync_callback)

    def sync_callback(self, color, depth):
        rospy.loginfo("Received RGB and Depth images")
        try:
            color = self.bridge.imgmsg_to_cv2(color, 'bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth, "32FC1")*1000
            depth = np.clip(depth, 0, 65535)
            depth = depth.astype('uint16')
            print("Depth range:", depth.min(), depth.max())
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return
        
        # Store the images
        self.color_image = color
        self.depth_image = depth

    def handle_service(self, req):
        if self.color_image is None or self.depth_image is None:
            rospy.logwarn("Color or depth image is not yet available.")
            return image_serviceResponse()
        rospy.loginfo("Sending images to the client.")
        # Convert the OpenCV images back to ROS Image messages
        color_ros_image = self.bridge.cv2_to_imgmsg(self.color_image, "bgr8")
        depth_ros_image = self.bridge.cv2_to_imgmsg(self.depth_image, "16UC1")  

        return image_serviceResponse(color_image=color_ros_image, depth_image=depth_ros_image)

def image_fetcher_service():
    rospy.init_node('image_ros_server')

    image_fetcher = ImageFetcher()
    
    # Advertise the service
    service = rospy.Service('fetch_images', image_service, image_fetcher.handle_service)
    rospy.loginfo("Image fetcher service is ready.")

    rospy.spin()

if __name__ == "__main__":
    image_fetcher_service()
