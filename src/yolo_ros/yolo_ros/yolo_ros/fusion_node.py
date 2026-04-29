import numpy as np

import rclpy
from rclpy.node import Node

from ament_index_python.packages import get_package_share_directory
import yaml

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray
from yolo_msgs.msg import Detection, DetectionArray


class FusionNode(Node):
    def __init__(self) -> None:
        super().__init__("fusion_node")

        topics_path = get_package_share_directory("perception_common") + "/topics.yaml"
        with open(topics_path, "r", encoding="utf-8") as f:
            topic_config = yaml.safe_load(f)

        self.lidar_proj_topic = topic_config["topics"]["transform"]["lidar_2d_projection"]
        self.fused_bbox_topic = topic_config["topics"]["yolo"]["fused_bbox"]
        self._latest_detections_msg = None

        self._pub = self.create_publisher(DetectionArray, self.fused_bbox_topic, 10)
        self._markers_pub = self.create_publisher(MarkerArray, "fused_bbox_markers", 10)

        # Avoid strict timestamp sync because tracking and lidar projection may come
        # from different clock domains (bag time vs wall clock).
        self.create_subscription(DetectionArray, "tracking", self._detections_cb, 10)
        self.create_subscription(PointCloud2, self.lidar_proj_topic, self._lidar_cb, 10)

        self.get_logger().info(
            f"Fusion node ready: tracking + {self.lidar_proj_topic} -> {self.fused_bbox_topic}"
        )

    def _detections_cb(self, detections_msg: DetectionArray) -> None:
        self._latest_detections_msg = detections_msg

    def _lidar_cb(self, lidar_msg: PointCloud2) -> None:
        detections_msg = self._latest_detections_msg
        if detections_msg is None:
            return

        pts = point_cloud2.read_points_numpy(
            lidar_msg, field_names=["x", "y", "z", "u", "v"], skip_nans=True
        )

        out = DetectionArray()
        out.header = lidar_msg.header

        if pts.size == 0:
            self._pub.publish(out)
            self._publish_markers(out)
            return

        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        u = pts[:, 3]
        v = pts[:, 4]

        for det in detections_msg.detections:
            cx = float(det.bbox.center.position.x)
            cy = float(det.bbox.center.position.y)
            w = float(det.bbox.size.x)
            h = float(det.bbox.size.y)

            x_min = cx - w / 2.0
            x_max = cx + w / 2.0
            y_min = cy - h / 2.0
            y_max = cy + h / 2.0

            mask = (u >= x_min) & (u <= x_max) & (v >= y_min) & (v <= y_max)
            if not np.any(mask):
                continue

            px = x[mask]
            py = y[mask]
            pz = z[mask]

            fused_det = Detection()
            fused_det.class_id = det.class_id
            fused_det.class_name = det.class_name
            fused_det.score = det.score
            fused_det.id = det.id
            fused_det.extra_info = det.extra_info
            fused_det.bbox = det.bbox
            fused_det.mask = det.mask
            fused_det.keypoints = det.keypoints
            fused_det.keypoints3d = det.keypoints3d

            fused_det.bbox3d.center.position.x = float(np.median(px))
            fused_det.bbox3d.center.position.y = float(np.median(py))
            fused_det.bbox3d.center.position.z = float(np.median(pz))
            fused_det.bbox3d.center.orientation.w = 1.0
            fused_det.bbox3d.size.x = 1.5
            fused_det.bbox3d.size.y = 1.5
            fused_det.bbox3d.size.z = 1.5
            fused_det.bbox3d.frame_id = lidar_msg.header.frame_id

            out.detections.append(fused_det)

        self._pub.publish(out)
        self._publish_markers(out)

    def _publish_markers(self, fused_msg: DetectionArray) -> None:
        marker_array = MarkerArray()

        delete_all = Marker()
        delete_all.header = fused_msg.header
        delete_all.ns = "fused_bbox"
        delete_all.id = 0
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        for i, det in enumerate(fused_msg.detections):
            m = Marker()
            m.header = fused_msg.header
            m.ns = "fused_bbox"
            m.id = i
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose = det.bbox3d.center
            m.scale.x = max(float(det.bbox3d.size.x), 0.1)
            m.scale.y = max(float(det.bbox3d.size.y), 0.1)
            m.scale.z = max(float(det.bbox3d.size.z), 0.1)
            m.color.r = 0.15
            m.color.g = 0.85
            m.color.b = 0.25
            m.color.a = 0.45
            marker_array.markers.append(m)

        self._markers_pub.publish(marker_array)


def main() -> None:
    rclpy.init()
    node = FusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
