"""Fuses tracked 2D detections with the LiDAR projection to give each object a 3D position.

A detection describes an image captured several hundred milliseconds before the message
arrives (camera transport plus YOLO inference plus tracking), while the projection arrives a
few tens of milliseconds behind its own capture. Pairing each detection with whatever
projection is current at arrival time therefore lifts stale pixels onto fresh geometry, and
the object lands wherever the vehicle was when the camera saw it.

So the projections are buffered instead, and each detection array is fused against the cloud
captured closest to its own header stamp. yolo_node and tracking_node both copy the source
image header through untouched, so that stamp is the camera capture time, in the same host
clock domain as the LiDAR. Skew drops to at most half a LiDAR period, independent of how far
behind the camera pipeline runs. Stamps are only ever compared to each other and never to
the node clock, which keeps this valid under bag replay without any use_sim_time
configuration.

Detections drive the output rather than projections: with a 10Hz cloud stream and an 80ms
pairing bound, two consecutive clouds can match the same detection array, so a
projection-driven node would publish the same objects twice at two different geometries from
one detector frame.
"""

import time as _time

import numpy as np

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult

from ament_index_python.packages import get_package_share_directory
import yaml

from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from yolo_msgs.msg import Detection, DetectionArray

from perception_common.stamp_sync import StampMatchedBuffer, apply_bounded_parameters


class FusionNode(Node):
    def __init__(self) -> None:
        super().__init__("fusion_node")

        topics_path = get_package_share_directory("perception_common") + "/topics.yaml"
        with open(topics_path, "r", encoding="utf-8") as f:
            topic_config = yaml.safe_load(f)

        self.lidar_proj_topic = topic_config["topics"]["transform"]["lidar_2d_projection"]
        self.fused_bbox_topic = topic_config["topics"]["yolo"]["fused_bbox"]

        # Half a 10Hz LiDAR period is the worst case for a dense buffer; the default
        # leaves headroom for jitter and the occasional dropped scan without ever
        # admitting a cloud from a neighbouring frame. Err tight here: a misplaced
        # obstacle is worse than a dropped one, and the next detection is ~100ms away.
        max_pairing_skew = float(
            self.declare_parameter("max_pairing_skew", 0.08)
            .get_parameter_value()
            .double_value
        )
        # Must exceed the camera-to-detection latency, or the matching cloud will have
        # been evicted before its detection arrives.
        buffer_duration = float(
            self.declare_parameter("projection_buffer_duration", 2.0)
            .get_parameter_value()
            .double_value
        )
        # How long the fused output may go unrefreshed before the node publishes an empty
        # array. Without this a stalled detector leaves its last 3D boxes standing in RViz
        # and in planning, which reads as valid, current geometry.
        self.fusion_timeout = float(
            self.declare_parameter("fusion_timeout", 0.5)
            .get_parameter_value()
            .double_value
        )
        # The LiDAR stamp is end-of-sweep, so a point's true capture time is uniform over
        # the preceding sweep. -0.05 centres pairing on the sweep instead of biasing it
        # half a period late. Left at 0.0 until measured against a bag.
        stamp_offset = float(
            self.declare_parameter("projection_stamp_offset", 0.0)
            .get_parameter_value()
            .double_value
        )

        self._projections = StampMatchedBuffer(
            "projection",
            buffer_duration=max(0.0, buffer_duration),
            max_skew=max(0.0, max_pairing_skew),
            stamp_offset=stamp_offset,
        )
        self.fusion_timeout = max(0.0, self.fusion_timeout)
        self._last_publish = 0.0
        self._last_unmatched_log = 0.0

        self._pub = self.create_publisher(DetectionArray, self.fused_bbox_topic, 10)
        self._markers_pub = self.create_publisher(MarkerArray, "fused_bbox_markers", 10)

        self.create_subscription(DetectionArray, "tracking", self._detections_cb, 10)
        self.create_subscription(PointCloud2, self.lidar_proj_topic, self._lidar_cb, 10)

        # Fixed period, so fusion_timeout is enforced to within 0.5s of granularity.
        self._watchdog_timer = self.create_timer(0.5, self._watchdog)

        # The pairing bound and the buffer depth both depend on measured pipeline latency,
        # so keep them settable at runtime for calibration against a replaying bag without
        # restarting the node.
        self.add_on_set_parameters_callback(self._on_set_parameters)

        self.get_logger().info(
            f"Fusion node ready: tracking + {self.lidar_proj_topic} -> {self.fused_bbox_topic}, "
            f"max_pairing_skew={self._projections.max_skew:.3f}s "
            f"projection_buffer_duration={self._projections.buffer_duration:.3f}s "
            f"fusion_timeout={self.fusion_timeout:.3f}s (stamp-matched pairing)"
        )

    def _on_set_parameters(self, params) -> SetParametersResult:
        targets = {
            "max_pairing_skew": (self._projections, "max_skew"),
            "projection_buffer_duration": (self._projections, "buffer_duration"),
            "fusion_timeout": (self, "fusion_timeout"),
        }
        ok, reason, applied = apply_bounded_parameters(params, targets)
        if not ok:
            return SetParametersResult(successful=False, reason=reason)
        for name, value in applied:
            self.get_logger().info(f"{name} set to {value:.3f}s")

        # Unbounded: a negative offset is the expected direction for an end-of-sweep stamp.
        for p in params:
            if p.name == "projection_stamp_offset":
                self._projections.stamp_offset = float(p.value)
                self.get_logger().info(
                    f"projection_stamp_offset set to {float(p.value):.3f}s"
                )
        return SetParametersResult(successful=True)

    @staticmethod
    def _foreground_points(px, py, pz):
        """Keep only the nearest depth cluster inside a bounding box.

        Two objects that overlap in image space drop two separated groups of returns into
        one bbox, and a median over both lands between them where nothing is. Sorting by
        depth and cutting at the first significant gap keeps the foreground object alone.

        Depth is ``px``: transform.py publishes x,y,z in the LiDAR frame, where x is
        forward range (cropped to [0, 100]) and z is height (cropped to [-3.5, 1]).
        Clustering on z instead splits by height, which on flat ground finds no gap at all
        and lets the very blending this guards against through.
        """
        if len(px) < 2:
            return px, py, pz

        order = np.argsort(px)
        sx = px[order]

        diffs = np.diff(sx)
        med = np.median(diffs)
        mad = np.median(np.abs(diffs - med))
        gap_thresh = max(med + 3.0 * mad, 0.05)

        gaps = np.where(diffs > gap_thresh)[0]
        if len(gaps) == 0:
            return px, py, pz

        fg = order[: gaps[0] + 1]
        return px[fg], py[fg], pz[fg]

    def _lidar_cb(self, lidar_msg: PointCloud2) -> None:
        """Buffer the projection so a later detection array can be paired against it."""
        self._projections.add(lidar_msg)

    def _detections_cb(self, detections_msg: DetectionArray) -> None:
        entry, skew = self._projections.match(detections_msg.header)
        if entry is None:
            self._log_unmatched(skew)
            return
        self._fuse(detections_msg, entry)

    def _log_unmatched(self, skew: float) -> None:
        """Rate-limited warning so a starved or misaligned pipeline stays visible."""
        now = _time.monotonic()
        if now - self._last_unmatched_log < 1.0:
            return
        self._last_unmatched_log = now
        self.get_logger().warning(
            f"Unmatched detections: "
            f"{self._projections.describe_unmatched(skew, 'detection')}; "
            f"{self._projections.status()}"
        )

    def _fuse(self, detections_msg: DetectionArray, entry) -> None:
        xyz, u, v = entry.arrays()

        out = DetectionArray()
        # The fused points came from this cloud at this time, so the output carries its
        # stamp and frame rather than the detection's.
        out.header = entry.header

        if xyz.shape[0] == 0:
            self._publish(out)
            return

        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

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

            px, py, pz = self._foreground_points(x[mask], y[mask], z[mask])

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
            fused_det.bbox3d.frame_id = entry.header.frame_id

            out.detections.append(fused_det)

        self._publish(out)

    def _publish(self, fused_msg: DetectionArray) -> None:
        self._pub.publish(fused_msg)
        self._publish_markers(fused_msg)
        self._last_publish = _time.monotonic()

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

    def _watchdog(self) -> None:
        """Publish an empty result while detections are not being refreshed.

        Without this, a dead detector is indistinguishable from a scene with no objects
        only by absence — the last boxes simply stay on screen and in planning.
        """
        if self.fusion_timeout <= 0.0:
            return
        newest = self._projections.newest()
        if newest is None:
            return
        if _time.monotonic() - self._last_publish > self.fusion_timeout:
            empty = DetectionArray()
            empty.header = newest.header
            self._publish(empty)


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
