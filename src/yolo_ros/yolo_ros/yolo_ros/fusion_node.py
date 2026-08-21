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
behind the camera pipeline runs.

Matching is two-sided: a detection with no cloud yet captured at or after it is deferred for
up to wait_for_newer rather than forced backwards onto an older cloud. Deferred detections are
resolved by _pump, which runs after every buffered cloud and on a short timer.

Message stamps are only ever compared to each other, so the pairing itself is valid under bag
replay regardless of use_sim_time. Deferral expiry and the watchdog do read the node clock, so
run with use_sim_time:=true against a bag if you want them to track playback.

Detections drive the output rather than projections: with a 10Hz cloud stream and an 80ms
pairing bound, two consecutive clouds can match the same detection array, so a
projection-driven node would publish the same objects twice at two different geometries from
one detector frame.
"""

import numpy as np

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult

from ament_index_python.packages import get_package_share_directory
import yaml

from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from yolo_msgs.msg import Detection, DetectionArray

from perception_common.stamp_sync import (
    DEFERRED,
    StampMatchedBuffer,
    apply_bounded_parameters,
)


class FusionNode(Node):
    def __init__(self) -> None:
        super().__init__("fusion_node")

        topics_path = get_package_share_directory("perception_common") + "/topics.yaml"
        with open(topics_path, "r", encoding="utf-8") as f:
            topic_config = yaml.safe_load(f)

        self.lidar_proj_topic = topic_config["topics"]["transform"]["lidar_2d_projection"]
        self.fused_bbox_topic = topic_config["topics"]["yolo"]["fused_bbox"]

        # Measured on the vehicle: tracking arrives 0.014s after its capture stamp while
        # the projection arrives 0.031s after its own, so a detection is processed ~17ms
        # BEFORE the cloud it should pair with has been buffered. That is what makes
        # wait_for_newer necessary: without it, matching can only reach backwards into
        # clouds already held, and the worst case is a full 10Hz LiDAR period rather than
        # the half period a two-sided search gives. The camera and LiDAR free-run on
        # separate oscillators, so their phase sweeps that whole range with a ~60-100s
        # beat; a one-sided node at 0.08 matched in long stretches and starved in others,
        # and 0.12 was the width needed to cover the one-sided range.
        #
        # With deferral restoring two-sided matching, 0.06 covers half a period plus
        # jitter, so a pair is at most ~0.3m apart at 5m/s. If this ever starves, set
        # wait_for_newer to 0.0 and max_pairing_skew back to 0.12 at runtime -- that is
        # exactly the old behaviour -- and check the unmatched/expired counters.
        max_pairing_skew = float(
            self.declare_parameter("max_pairing_skew", 0.06)
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
        # How long a detection may wait for a cloud captured at or after it. This is what
        # makes matching two-sided; see the max_pairing_skew comment above. Set to 0.0 to
        # restore the old one-sided behaviour at runtime.
        wait_for_newer = float(
            self.declare_parameter("wait_for_newer", 0.06)
            .get_parameter_value()
            .double_value
        )
        # Deferrals are almost always resolved by the _pump that follows each buffered cloud;
        # this timer only bounds the wait when the projection stream stalls. Read-only because
        # a timer period cannot be changed after construction, and silently ignoring a
        # ros2 param set would be worse than rejecting it.
        pump_period = float(
            self.declare_parameter(
                "deferral_pump_period", 0.02, ParameterDescriptor(read_only=True)
            )
            .get_parameter_value()
            .double_value
        )
        stats_log_period = float(
            self.declare_parameter(
                "stats_log_period", 5.0, ParameterDescriptor(read_only=True)
            )
            .get_parameter_value()
            .double_value
        )

        # A 2D box around a distant vehicle also contains the road in front of it, caught
        # by the box's bottom rows. Those returns are nearer than the vehicle, so the depth
        # clustering below adopts them and the object is published several metres early.
        # Measured on the 2026-08-20 bag: the published range sat 14.3m median (22.2m worst)
        # in front of the vehicle's own returns, and the adopted points were at road height,
        # pinned to the bottom 8% of the box. See _reject_ground for why each bound is here.
        #
        # Validated on that same replay by fitting the ground as a sloped line through each
        # box's lowest quartile (these roads fall ~1.4cm per metre, so a flat threshold
        # would have flattered the result) and measuring how far above it the points that
        # decide the output sit. Before: median +0.00m, i.e. the road itself, with 1 of 32
        # detections drawn from elevated returns. After: median +0.78m, vehicle-body
        # height, with 30 of 32 from elevated returns. Median published range moved
        # 76.05m -> 90.98m.
        self._ground_min_range = float(
            self.declare_parameter("ground_rejection_min_range", 25.0)
            .get_parameter_value()
            .double_value
        )
        self._ground_margin = float(
            self.declare_parameter("ground_margin", 0.4)
            .get_parameter_value()
            .double_value
        )
        self._ground_min_points = int(
            self.declare_parameter("ground_min_points", 2)
            .get_parameter_value()
            .integer_value
        )

        self._projections = StampMatchedBuffer(
            "projection",
            buffer_duration=max(0.0, buffer_duration),
            max_skew=max(0.0, max_pairing_skew),
            stamp_offset=stamp_offset,
            wait_for_newer=max(0.0, wait_for_newer),
        )
        self.fusion_timeout = max(0.0, self.fusion_timeout)
        # None until the first publish: under sim time the node clock reads 0 until /clock
        # arrives, and 0.0 here would make the first watchdog tick see a ~1.7e9s gap.
        self._last_publish = None
        self._last_unmatched_log = None

        self._pub = self.create_publisher(DetectionArray, self.fused_bbox_topic, 10)
        self._markers_pub = self.create_publisher(MarkerArray, "fused_bbox_markers", 10)

        self.create_subscription(DetectionArray, "tracking", self._detections_cb, 10)
        self.create_subscription(PointCloud2, self.lidar_proj_topic, self._lidar_cb, 10)

        # Fixed period, so fusion_timeout is enforced to within 0.5s of granularity.
        self._watchdog_timer = self.create_timer(0.5, self._watchdog)
        if pump_period > 0.0:
            self._pump_timer = self.create_timer(pump_period, self._pump)
        if stats_log_period > 0.0:
            self._stats_timer = self.create_timer(stats_log_period, self._log_stats)

        # The pairing bound and the buffer depth both depend on measured pipeline latency,
        # so keep them settable at runtime for calibration against a replaying bag without
        # restarting the node.
        self.add_on_set_parameters_callback(self._on_set_parameters)

        self.get_logger().info(
            f"Fusion node ready: tracking + {self.lidar_proj_topic} -> {self.fused_bbox_topic}, "
            f"max_pairing_skew={self._projections.max_skew:.3f}s "
            f"projection_buffer_duration={self._projections.buffer_duration:.3f}s "
            f"wait_for_newer={self._projections.wait_for_newer:.3f}s "
            f"fusion_timeout={self.fusion_timeout:.3f}s "
            f"ground_rejection>={self._ground_min_range:.1f}m "
            f"ground_margin={self._ground_margin:.2f}m "
            f"ground_min_points={self._ground_min_points} "
            f"use_sim_time={self.get_parameter('use_sim_time').value} "
            f"(stamp-matched pairing)"
        )

    def _now(self) -> float:
        """Seconds on the node clock -- sim time when use_sim_time is set, else wall time."""
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_set_parameters(self, params) -> SetParametersResult:
        # Checked before anything is applied, so an out-of-range count cannot leave the
        # node half-updated -- the same guarantee apply_bounded_parameters gives the rest.
        for p in params:
            if p.name == "ground_min_points" and int(p.value) < 0:
                return SetParametersResult(
                    successful=False, reason="ground_min_points must be >= 0"
                )

        targets = {
            "max_pairing_skew": (self._projections, "max_skew"),
            "projection_buffer_duration": (self._projections, "buffer_duration"),
            "fusion_timeout": (self, "fusion_timeout"),
            "wait_for_newer": (self._projections, "wait_for_newer"),
            "ground_rejection_min_range": (self, "_ground_min_range"),
            "ground_margin": (self, "_ground_margin"),
        }
        # These two are metres; everything else here is seconds.
        metres = {"ground_rejection_min_range", "ground_margin"}
        ok, reason, applied = apply_bounded_parameters(params, targets)
        if not ok:
            return SetParametersResult(successful=False, reason=reason)
        for name, value in applied:
            self.get_logger().info(
                f"{name} set to {value:.3f}{'m' if name in metres else 's'}"
            )

        for p in params:
            if p.name == "ground_min_points":
                self._ground_min_points = int(p.value)
                self.get_logger().info(
                    f"ground_min_points set to {self._ground_min_points}"
                )

        # Unbounded: a negative offset is the expected direction for an end-of-sweep stamp.
        for p in params:
            if p.name == "projection_stamp_offset":
                self._projections.stamp_offset = float(p.value)
                self.get_logger().info(
                    f"projection_stamp_offset set to {float(p.value):.3f}s"
                )
        return SetParametersResult(successful=True)

    def _reject_ground(self, px, py, pz):
        """Drop road returns from a box before the depth clustering runs.

        The road in front of a distant vehicle falls inside its 2D box and is nearer than
        the vehicle, so _foreground_points adopts it. The size of that error is set by how
        far a pixel of box-edge error moves the ground intercept, which grows as
        r^2/(f*h): 0.01m per pixel at 10m, 0.15m at 40m, 0.60m at 80m. That quadratic is
        why near objects are placed correctly and far ones are not.

        Hence the range gate rather than a height test alone. Below
        ground_rejection_min_range the error is under a decimetre, while a short object --
        a cone stands ~0.5m -- lies almost entirely within ground_margin of the road, so
        filtering there would strip it for no benefit. Beyond the gate, a box left with
        fewer than ground_min_points falls back to the unfiltered set, so a short or
        sparsely-sampled object is never placed worse than it would have been.

        Sensor geometry bounds what this can recover, and explains a transient that looks
        like a bug but is not. The 32-ring LiDAR has a 1.29deg ring pitch, so ring spacing
        at range r is r*tan(1.29deg): 0.45m at 20m, 0.90m at 40m, 1.80m at 80m. A 1.5m car
        subtends less than one ring spacing beyond ~67m, so past that range whether any
        ring lands on the vehicle at all is luck, frame to frame. On a frame where none
        does, the box holds road and nothing else -- measured z-spread 0.03m across all 16
        such detections on the 2026-08-20 replay -- ground_min_points declines to filter,
        and the published range is the road ring in front of the vehicle, exactly as it was
        before this function existed. That is why a distant object can sit on a ground ring
        for several frames and then snap onto the vehicle as it closes to ~65m and the
        rings begin to strike it. The split is not purely by range: on that replay the
        filtered detections sat at 91.9m median and the fallback ones at 87.6m.

        Known limitation: those no-return frames still publish a bbox3d, at the road
        position, carrying no field that distinguishes them from a genuine fix. A near-zero
        z-spread inside the box identifies them cheaply if a consumer ever needs to.
        """
        if len(px) < 2 or self._ground_margin <= 0.0:
            return px, py, pz
        # The nearest return decides, not the median: it is the one the clustering would
        # adopt, so it is what determines whether this box is close enough to leave alone.
        if float(np.min(px)) < self._ground_min_range:
            return px, py, pz

        # A low percentile rather than the minimum, so a single stray low return cannot
        # drag the estimate down and quietly disable the margin.
        ground_z = float(np.percentile(pz, 10.0))
        keep = pz > ground_z + self._ground_margin
        if int(np.count_nonzero(keep)) < self._ground_min_points:
            return px, py, pz
        return px[keep], py[keep], pz[keep]

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

        The road in front of a distant vehicle is a separate problem that this gap cut
        cannot solve at any axis: the road is genuinely nearer, so the nearest cluster is
        the correct answer to the question asked here, just not the wanted one. It is
        handled upstream by _reject_ground, which strips near-ground returns before this
        runs. Do not try to fix it by clustering on height again -- that reintroduces the
        blending described above without addressing the road.
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
        """Buffer the projection, then release any detection that was waiting for it."""
        self._projections.add(lidar_msg)
        # This is what resolves nearly every deferral, in the same tick the awaited cloud
        # lands; the pump timer only matters when this stream stalls.
        self._pump()

    def _detections_cb(self, detections_msg: DetectionArray) -> None:
        pairing = self._projections.match(
            detections_msg.header, now=self._now(), payload=detections_msg
        )
        if pairing.outcome is not DEFERRED:
            self._complete(pairing)

    def _pump(self) -> None:
        for pairing in self._projections.drain(self._now()):
            self._complete(pairing)

    def _complete(self, pairing) -> None:
        if pairing.value is None:
            self._log_unmatched(pairing.skew, pairing.reason)
            return
        self._fuse(pairing.payload, pairing.value)

    def _log_unmatched(self, skew: float, reason=None) -> None:
        """Rate-limited warning so a starved or misaligned pipeline stays visible."""
        now = self._now()
        if self._last_unmatched_log is not None and now - self._last_unmatched_log < 1.0:
            return
        self._last_unmatched_log = now
        self.get_logger().warning(
            f"Unmatched detections: "
            f"{self._projections.describe_unmatched(skew, 'detection', reason=reason)}; "
            f"{self._projections.status()}"
        )

    def _log_stats(self) -> None:
        """Periodic pairing health, so a skew regression is visible without instrumentation."""
        self.get_logger().info(f"pairing: {self._projections.status()}")

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

            bx, by, bz = self._reject_ground(x[mask], y[mask], z[mask])
            px, py, pz = self._foreground_points(bx, by, bz)

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
        self._last_publish = self._now()

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
        if newest is None or self._last_publish is None:
            return
        if self._now() - self._last_publish > self.fusion_timeout:
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
