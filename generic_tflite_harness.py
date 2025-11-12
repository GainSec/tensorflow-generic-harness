#!/usr/bin/env python3
\"\"\"Run object-recognition TFLite models against live or recorded footage.\"\"\"
import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import cv2
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:  # fallback to full TF installation
    from tensorflow.lite.python.interpreter import Interpreter  # type: ignore


@dataclass
class InputConfig:
    name: str
    height: int
    width: int
    channels: int
    scale: str
    fill_color: int = 0


@dataclass
class ModelConfig:
    name: str
    file: Path
    model_type: str
    api: str
    label_map: Path
    confidence_threshold: float
    cluster_threshold: float
    inputs: InputConfig
    output_name: str
    classes: int
    anchor_file: Optional[Path] = None
    anchor_scales: Optional[Dict[str, float]] = None
    bbox_output_name: Optional[str] = None


@dataclass
class FramePacket:
    frame: np.ndarray
    source: str
    index: int
    timestamp_ms: float
    media_path: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class DetectionStats:
    frames: int = 0
    detections: int = 0
    label_histogram: Counter = field(default_factory=Counter)
    inference_ms_total: float = 0.0


BANNER = r\"\"\"
 ____  _         _   _   _             _   _
| __ )(_)_ __ __| | | | | |_   _ _ __ | |_(_)_ __   __ _
|  _ \| | '__/ _` | | |_| | | | | '_ \| __| | '_ \ / _` |
| |_) | | | | (_| | |  _  | |_| | | | | |_| | | | | (_| |
|____/|_|_|  \__,_| |_| |_|\__,_|_| |_|\__|_|_| |_|\__, |
                                                   |___/
 ____                                 ____  _         _ _____
/ ___|  ___  __ _ ___  ___  _ __  _  | __ )(_)_ __ __| | ____|   _  ___
\___ \ / _ \/ _` / __|/ _ \| '_ \(_) |  _ \| | '__/ _` |  _|| | | |/ _ \
 ___) |  __/ (_| \__ \ (_) | | | |_  | |_) | | | | (_| | |__| |_| |  __/
|____/ \___|\__,_|___/\___/|_| |_(_) |____/|_|_|  \__,_|_____\__, |\___/
                                                             |___/
         ____           ____       _       ____
        | __ ) _   _   / ___| __ _(_)_ __ / ___|  ___  ___
 _____  |  _ \| | | | | |  _ / _` | | '_ \\___ \ / _ \/ __|
|_____| | |_) | |_| | | |_| | (_| | | | | |___) |  __/ (__
        |____/ \__, |  \____|\__,_|_|_| |_|____/ \___|\___|
               |___/
\"\"\"

IMAGE_EXTENSIONS = {\".jpg\", \".jpeg\", \".png\", \".bmp\", \".webp\"}
VIDEO_EXTENSIONS = {\".mp4\", \".mov\", \".mkv\", \".avi\", \".mpg\", \".mpeg\"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_metadata(models_json: Path, assets_root: Path, model_name: str) -> 'ModelConfig':
    data = json.loads(models_json.read_text())
    for entry in data.get(\"models\", []):
        if entry[\"name\"] != model_name:
            continue
        inputs = entry[\"inputs\"][0]
        label_map_path = assets_root / entry[\"classes\"]
        anchor_file = None
        anchor_scales = None
        outputs = entry.get(\"outputs\", [])
        if not outputs:
            raise ValueError(f\"Model '{model_name}' is missing outputs in metadata\")
        output_name = outputs[0][\"name\"]
        bbox_output_name: Optional[str] = None
        if entry[\"type\"].lower() == \"ssd\":
            anchor_file = assets_root / entry[\"anchors\"][\"locations\"]
            anchor_scales = {k: entry[\"anchors\"][k] for k in (\"x_scale\", \"y_scale\", \"h_scale\", \"w_scale\")}
            bbox_entry = next((o for o in outputs if o.get(\"type\", \"\").lower() in {\"bbox\", \"box\", \"bbox_enc\"}), None)
            confidence_entry = next((o for o in outputs if o is not bbox_entry), None)
            if bbox_entry:
                bbox_output_name = bbox_entry[\"name\"]
            if confidence_entry:
                output_name = confidence_entry[\"name\"]
        primary_dims = next((o for o in outputs if o[\"name\"] == output_name), outputs[0])[\"dims\"][\"size\"]
        classes = primary_dims[2] - 5 if entry[\"type\"].lower().startswith(\"yolo\") else primary_dims[2]
        return ModelConfig(
            name=entry[\"name\"],
            file=assets_root / entry[\"file\"],
            model_type=entry[\"type\"],
            api=entry[\"api\"],
            label_map=label_map_path,
            confidence_threshold=float(entry.get(\"confidenceThreshold\", 0.4)),
            cluster_threshold=float(entry.get(\"clusterThreshold\", 0.45)),
            inputs=InputConfig(
                name=inputs[\"name\"],
                height=int(inputs[\"dims\"][0]),
                width=int(inputs[\"dims\"][1]),
                channels=int(inputs[\"dims\"][2]),
                scale=inputs.get(\"scale\", \"squish\"),
                fill_color=int(inputs.get(\"fillColor\", 0)),
            ),
            output_name=output_name,
            classes=classes,
            anchor_file=anchor_file,
            anchor_scales=anchor_scales,
            bbox_output_name=bbox_output_name,
        )
    raise ValueError(f\"Model '{model_name}' not found in {models_json}\")


def load_label_map(path: Path) -> List[str]:
    data = json.loads(path.read_text())
    return [label[\"name\"] for label in data.get(\"labels\", [])]


# ---------------------------------------------------------------------------
# Pre/post-processing
# ---------------------------------------------------------------------------

def letterbox(image: np.ndarray, cfg: InputConfig) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
    target = (cfg.width, cfg.height)
    if cfg.scale == \"squish\":
        resized = cv2.resize(image, target)
        return resized, (cfg.width / image.shape[1], cfg.height / image.shape[0]), (0, 0)
    # maintain aspect ratio
    h, w = image.shape[:2]
    scale = min(cfg.width / w, cfg.height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    canvas = np.full((cfg.height, cfg.width, 3), cfg.fill_color, dtype=np.uint8)
    pad_x = (cfg.width - new_w) // 2
    pad_y = (cfg.height - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, (scale, scale), (pad_x, pad_y)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def nms(boxes, scores, threshold):
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes)
    scores = np.asarray(scores)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / ((boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]) + (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1]) - inter + 1e-9)
        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]
    return keep


def decode_yolo(raw: np.ndarray, conf_thresh: float, cluster_thresh: float, input_dims: Tuple[int, int], scale: Tuple[float, float], pad: Tuple[int, int], labels: List[str]) -> List[Dict]:
    data = raw.reshape(-1, raw.shape[-1])
    obj = sigmoid(data[:, 4])
    cls_logits = data[:, 5:]
    cls_scores = sigmoid(cls_logits)
    class_ids = np.argmax(cls_scores, axis=1)
    class_scores = cls_scores[np.arange(len(class_ids)), class_ids]
    conf = obj * class_scores
    mask = conf >= conf_thresh
    data = data[mask]
    conf = conf[mask]
    class_ids = class_ids[mask]
    if data.size == 0:
        return []
    boxes_xywh = data[:, :4]
    # Convert to xyxy relative to letterboxed input
    x_center = boxes_xywh[:, 0]
    y_center = boxes_xywh[:, 1]
    width = boxes_xywh[:, 2]
    height = boxes_xywh[:, 3]
    x1 = (x_center - width / 2) * input_dims[1]
    y1 = (y_center - height / 2) * input_dims[0]
    x2 = (x_center + width / 2) * input_dims[1]
    y2 = (y_center + height / 2) * input_dims[0]
    # map back to original frame coords
    scale_x = scale[0] if len(scale) > 1 else scale[0]
    scale_y = scale[1] if len(scale) > 1 else scale[0]
    pad_x, pad_y = pad
    x1 = (x1 - pad_x) / scale_x
    x2 = (x2 - pad_x) / scale_x
    y1 = (y1 - pad_y) / scale_y
    y2 = (y2 - pad_y) / scale_y
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    keep = nms(boxes, conf, cluster_thresh)
    detections = []
    for idx in keep:
        label = labels[class_ids[idx]] if class_ids[idx] < len(labels) else f\"class_{class_ids[idx]}\"
        detections.append({
            \"bbox\": boxes[idx].tolist(),
            \"confidence\": float(conf[idx]),
            \"label\": label,
        })
    return detections


def load_anchors(path: Path) -> np.ndarray:
    data = json.loads(path.read_text())
    anchors = [(a[\"x0\"], a[\"y0\"], a[\"x1\"], a[\"y1\"]) for a in data.get(\"anchors\", [])]
    return np.asarray(anchors, dtype=np.float32)


def decode_ssd(raw_boxes: np.ndarray, raw_scores: np.ndarray, anchors: np.ndarray, scales: Dict[str, float], conf_thresh: float, cluster_thresh: float, input_dims: Tuple[int, int], scale: Tuple[float, float], pad: Tuple[int, int], labels: List[str]) -> List[Dict]:
    # raw_boxes shape: (1, num_boxes, 4)
    boxes = raw_boxes[0]
    scores = raw_scores[0]
    num_boxes = boxes.shape[0]
    ycenter = boxes[:, 0] / scales[\"y_scale\"] * (anchors[:, 3] - anchors[:, 1]) + (anchors[:, 1] + anchors[:, 3]) / 2.0
    xcenter = boxes[:, 1] / scales[\"x_scale\"] * (anchors[:, 2] - anchors[:, 0]) + (anchors[:, 0] + anchors[:, 2]) / 2.0
    half_h = np.exp(boxes[:, 2] / scales[\"h_scale\"]) * (anchors[:, 3] - anchors[:, 1]) / 2.0
    half_w = np.exp(boxes[:, 3] / scales[\"w_scale\"]) * (anchors[:, 2] - anchors[:, 0]) / 2.0
    y1 = ycenter - half_h
    x1 = xcenter - half_w
    y2 = ycenter + half_h
    x2 = xcenter + half_w
    # convert to pixel coordinates in input space
    x1 = x1 * input_dims[1]
    x2 = x2 * input_dims[1]
    y1 = y1 * input_dims[0]
    y2 = y2 * input_dims[0]
    class_probs = sigmoid(scores)
    class_ids = np.argmax(class_probs[:, 1:], axis=1) + 1  # ignore background
    confidences = class_probs[np.arange(num_boxes), class_ids]
    mask = confidences >= conf_thresh
    boxes = np.stack([x1, y1, x2, y2], axis=1)[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]
    if boxes.size == 0:
        return []
    scale_x = scale[0] if len(scale) > 1 else scale[0]
    scale_y = scale[1] if len(scale) > 1 else scale[0]
    pad_x, pad_y = pad
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale_x
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale_y
    keep = nms(boxes, confidences, cluster_thresh)
    detections = []
    for idx in keep:
        label_index = class_ids[idx]
        label_name = labels[label_index] if label_index < len(labels) else f\"class_{label_index}\"
        detections.append({
            \"bbox\": boxes[idx].tolist(),
            \"confidence\": float(confidences[idx]),
            \"label\": label_name,
        })
    return detections


# ---------------------------------------------------------------------------
# Runner, recorders, and streams
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    return datetime.utcnow().isoformat(timespec=\"seconds\") + \"Z\"


def serialize_source(source: Dict[str, Any]) -> Dict[str, Any]:
    serialized: Dict[str, Any] = {}
    for key, value in source.items():
        if isinstance(value, Path):
            serialized[key] = str(value)
        else:
            serialized[key] = value
    return serialized


def draw_detections(frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
    canvas = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det[\"bbox\"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f\"{det['label']} {det['confidence']:.2f}\"
        cv2.putText(canvas, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return canvas


def build_interpreter(model_path: Path) -> Interpreter:
    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter


class DetectionRecorder:
    def __init__(self, model_name: str, source: Dict[str, Any], run_id: Optional[str] = None):
        self.model_name = model_name
        self.source = source
        self.run_id = run_id
        self.started_at = utc_now_iso()
        self.completed_at: Optional[str] = None
        self.frames: List[Dict[str, Any]] = []
        self.stats = DetectionStats()

    def add_record(self, packet: 'FramePacket', inference_ms: float, detections: List[Dict[str, Any]]) -> None:
        record = {
            \"frame_index\": packet.index,
            \"timestamp_ms\": packet.timestamp_ms,
            \"media_path\": packet.media_path,
            \"session_id\": packet.session_id,
            \"detections\": detections,
            \"inference_ms\": inference_ms,
        }
        self.frames.append(record)
        self.stats.frames += 1
        self.stats.detections += len(detections)
        self.stats.inference_ms_total += inference_ms
        for det in detections:
            label = det.get(\"label\", \"unknown\")
            self.stats.label_histogram[label] += 1

    def build_report(self) -> Dict[str, Any]:
        if self.completed_at is None:
            self.completed_at = utc_now_iso()
        avg_inf = self.stats.inference_ms_total / max(1, self.stats.frames)
        return {
            \"model\": self.model_name,
            \"run_id\": self.run_id,
            \"source\": serialize_source(self.source),
            \"started_at\": self.started_at,
            \"completed_at\": self.completed_at,
            \"frames\": self.frames,
            \"stats\": {
                \"frames\": self.stats.frames,
                \"detections\": self.stats.detections,
                \"avg_inference_ms\": avg_inf,
                \"labels\": dict(self.stats.label_histogram),
            },
        }

    def write_json(self, path: Path) -> Dict[str, Any]:
        report = self.build_report()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
        return report


class ModelRunner:
    def __init__(self, cfg: 'ModelConfig'):
        self.cfg = cfg
        self.labels = load_label_map(cfg.label_map)
        self.interpreter = build_interpreter(cfg.file)
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()
        self.anchors = load_anchors(cfg.anchor_file) if cfg.anchor_file else None

    def _prepare_input(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
        preprocessed, scale, pad = letterbox(frame, self.cfg.inputs)
        rgb = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
        if self.input_details[\"dtype\"] == np.uint8:
            input_data = rgb.astype(np.uint8)
        else:
            input_data = rgb.astype(np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)
        return input_data, scale, pad

    def _collect_outputs(self) -> Dict[str, np.ndarray]:
        return {d[\"name\"]: self.interpreter.get_tensor(d[\"index\"]) for d in self.output_details}

    def _resolve_output(self, outputs: Dict[str, np.ndarray], name: Optional[str]) -> np.ndarray:
        if name and name in outputs:
            return outputs[name]
        if name:
            for key, value in outputs.items():
                if key.endswith(name) or name.endswith(key):
                    return value
        # fallback to first
        return next(iter(outputs.values()))

    def infer(self, frame: np.ndarray) -> Tuple[List[Dict], float]:
        input_data, scale, pad = self._prepare_input(frame)
        self.interpreter.set_tensor(self.input_details[\"index\"], input_data)
        start = time.time()
        self.interpreter.invoke()
        inference_ms = (time.time() - start) * 1000.0
        outputs = self._collect_outputs()
        detections: List[Dict]
        if self.cfg.model_type.lower().startswith(\"yolo\"):
            main_output = self._resolve_output(outputs, self.cfg.output_name)
            detections = decode_yolo(main_output, self.cfg.confidence_threshold, self.cfg.cluster_threshold, (self.cfg.inputs.height, self.cfg.inputs.width), scale, pad, self.labels)
        elif self.cfg.model_type.lower() == \"ssd\":
            if not self.anchors or not self.cfg.anchor_scales:
                raise RuntimeError(\"SSD model missing anchors or scale metadata\")
            box_name = self.cfg.bbox_output_name
            box_tensor = self._resolve_output(outputs, box_name)
            score_tensor = self._resolve_output(outputs, self.cfg.output_name)
            detections = decode_ssd(box_tensor, score_tensor, self.anchors, self.cfg.anchor_scales, self.cfg.confidence_threshold, self.cfg.cluster_threshold, (self.cfg.inputs.height, self.cfg.inputs.width), scale, pad, self.labels)
        else:
            raise NotImplementedError(f\"Model type {self.cfg.model_type} not supported\")
        return detections, inference_ms


def close_frame_stream(stream: Any) -> None:
    close_func = getattr(stream, \"close\", None)
    if callable(close_func):
        close_func()


def camera_frame_stream(camera_index: int) -> Iterator['FramePacket']:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f\"Could not open camera index {camera_index}\")
    idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
            if timestamp <= 0:
                timestamp = time.time() * 1000.0
            yield FramePacket(frame=frame, source=\"camera\", index=idx, timestamp_ms=timestamp)
            idx += 1
    finally:
        cap.release()


def video_frame_stream(video_path: Path) -> Iterator['FramePacket']:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f\"Could not open video file {video_path}\")
    idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
            if timestamp <= 0:
                timestamp = time.time() * 1000.0
            yield FramePacket(frame=frame, source=\"video\", index=idx, timestamp_ms=timestamp, media_path=str(video_path))
            idx += 1
    finally:
        cap.release()


def frames_dir_stream(frames_dir: Path) -> Iterator['FramePacket']:
    files = sorted([p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])
    idx = 0
    for path in files:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        timestamp = path.stat().st_mtime * 1000.0
        yield FramePacket(frame=frame, source=\"frames\", index=idx, timestamp_ms=timestamp, media_path=str(path))
        idx += 1


def list_session_media(session_dir: Path) -> List[Path]:
    media: List[Path] = []
    for path in sorted(session_dir.rglob(\"*\")):
        if not path.is_file():
            continue
        if path.suffix.lower() in IMAGE_EXTENSIONS.union(VIDEO_EXTENSIONS):
            media.append(path)
    return media


def session_frame_stream(session_dir: Path) -> Iterator['FramePacket']:
    media_files = list_session_media(session_dir)
    global_index = 0
    for media_path in media_files:
        suffix = media_path.suffix.lower()
        rel = str(media_path.relative_to(session_dir))
        if suffix in IMAGE_EXTENSIONS:
            frame = cv2.imread(str(media_path))
            if frame is None:
                continue
            timestamp = media_path.stat().st_mtime * 1000.0
            yield FramePacket(frame=frame, source=\"session\", index=global_index, timestamp_ms=timestamp, media_path=rel, session_id=session_dir.name)
            global_index += 1
        elif suffix in VIDEO_EXTENSIONS:
            cap = cv2.VideoCapture(str(media_path))
            if not cap.isOpened():
                continue
            frame_in_clip = 0
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
                    if timestamp <= 0:
                        timestamp = media_path.stat().st_mtime * 1000.0
                    media_label = f\"{rel}#frame{frame_in_clip}\"
                    yield FramePacket(frame=frame, source=\"session\", index=global_index, timestamp_ms=timestamp, media_path=media_label, session_id=session_dir.name)
                    global_index += 1
                    frame_in_clip += 1
            finally:
                cap.release()


WINDOW_TITLE = \"Generic Visual Inference Harness\"


def run_stream(runner: 'ModelRunner', frame_stream: Iterator['FramePacket'], recorder: 'DetectionRecorder', display: bool, max_frames: Optional[int]) -> 'DetectionRecorder':
    processed = 0
    try:
        for packet in frame_stream:
            detections, inference_ms = runner.infer(packet.frame)
            recorder.add_record(packet, inference_ms, detections)
            if display:
                overlay = draw_detections(packet.frame, detections)
                cv2.imshow(WINDOW_TITLE, overlay)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord(\"q\")):
                    break
            processed += 1
            if max_frames and processed >= max_frames:
                break
    finally:
        close_frame_stream(frame_stream)
        if display:
            cv2.destroyAllWindows()
    return recorder


def append_summary_markdown(summary: Dict[str, Any], md_path: Path) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not md_path.exists()
    lines: List[str] = []
    if needs_header:
        lines.append(\"# Visual Model Replay Summary\\n\\n\")
    source = summary.get(\"source\", {})
    descriptor_parts = [source.get(\"type\", \"unknown\")]
    if source.get(\"run_id\"):
        descriptor_parts.append(str(source[\"run_id\"]))
    if source.get(\"session_id\"):
        descriptor_parts.append(str(source[\"session_id\"]))
    descriptor = \" / \".join(part for part in descriptor_parts if part)
    started = summary.get(\"started_at\", utc_now_iso())
    lines.append(f\"## {started} – {descriptor}\\n\")
    lines.append(f\"- Model: `{summary.get('model', 'unknown')}`\\n\")
    stats = summary.get(\"stats\", {})
    avg_inf = stats.get(\"avg_inference_ms\", 0.0)
    lines.append(f\"- Frames: {stats.get('frames', 0)} | Total detections: {stats.get('detections', 0)} | Avg inference: {avg_inf:.2f} ms\\n\")
    labels = stats.get(\"labels\", {})
    label_line = \", \".join(f\"{label} ({count})\" for label, count in labels.items()) if labels else \"none\"
    lines.append(f\"- Labels: {label_line}\\n\\n\")
    with md_path.open(\"a\", encoding=\"utf-8\") as fh:
        fh.writelines(lines)


class SessionWatcher:
    def __init__(self, runner: 'ModelRunner', session_root: Path, stage: str, output_root: Path, summary_md: Path, max_frames: Optional[int], poll_interval: float):
        self.runner = runner
        self.stage_dir = session_root / stage
        self.output_root = output_root
        self.summary_md = summary_md
        self.max_frames = max_frames
        self.poll_interval = poll_interval
        self.index_path = output_root / \"session_index.json\"
        self.report_dir = output_root / \"detections\"
        self.detection_processed = output_root / \"detectionProcessed\"
        self.discarded = output_root / \"discarded\"
        self._index = self._load_index()

    def _load_index(self) -> Dict[str, Any]:
        if self.index_path.exists():
            return json.loads(self.index_path.read_text())
        return {}

    def _save_index(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps(self._index, indent=2))

    def process_available_sessions(self) -> bool:
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        processed_any = False
        for session_dir in sorted(self.stage_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            if session_dir.name in self._index:
                continue
            report = self._process_session(session_dir)
            if report is None:
                continue
            processed_any = True
            self._index[session_dir.name] = {
                \"report\": str(report[\"report_path\"]),
                \"detections\": report[\"summary\"][\"stats\"][\"detections\"],
                \"frames\": report[\"summary\"][\"stats\"][\"frames\"],
                \"stage\": self.stage_dir.name,
                \"completed_at\": report[\"summary\"][\"completed_at\"],
            }
            self._save_index()
        return processed_any

    def watch(self) -> None:
        while True:
            self.process_available_sessions()
            time.sleep(self.poll_interval)

    def _process_session(self, session_dir: Path) -> Optional[Dict[str, Any]]:
        media_files = list_session_media(session_dir)
        if not media_files:
            return None
        source_desc = {
            \"type\": \"session\",
            \"session_id\": session_dir.name,
            \"stage\": self.stage_dir.name,
            \"path\": str(session_dir),
        }
        recorder = DetectionRecorder(self.runner.cfg.name, source_desc, run_id=session_dir.name)
        stream = session_frame_stream(session_dir)
        recorder = run_stream(self.runner, stream, recorder, display=False, max_frames=self.max_frames)
        report_path = self.report_dir / f\"{session_dir.name}.json\"
        summary = recorder.write_json(report_path)
        append_summary_markdown(summary, self.summary_md)
        target_root = self.detection_processed if summary[\"stats\"][\"detections\"] > 0 else self.discarded
        target_dir = target_root / session_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)
        marker = target_dir / \"replay.txt\"
        marker.write_text(f\"source={session_dir}\\nreport={report_path}\\n\")
        return {\"summary\": summary, \"report_path\": report_path}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    print(BANNER)
    parser = argparse.ArgumentParser(description=\"Replay object-recognition models against live or recorded footage.\")
    parser.add_argument(\"--models-json\", type=Path, required=True, help=\"Path to models.json describing model files and metadata\")
    parser.add_argument(\"--assets-root\", type=Path, required=True, help=\"Root directory containing the assets (label maps, anchors, models)\")
    parser.add_argument(\"--model-name\", required=True, help=\"Model entry name from models.json\")
    parser.add_argument(\"--camera\", type=int, default=0, help=\"Webcam index (default 0) when no other source is selected\")
    parser.add_argument(\"--video\", type=Path, help=\"Path to a recorded video file to replay\")
    parser.add_argument(\"--frames-dir\", type=Path, help=\"Directory full of still frames to replay\")
    parser.add_argument(\"--session-root\", type=Path, help=\"Root of the media tree\")
    parser.add_argument(\"--session-stage\", default=\"processed\", help=\"Stage directory under --session-root to monitor\")
    parser.add_argument(\"--session-output-root\", type=Path, default=Path(\"detections_pipeline\"), help=\"Destination for processed and discarded mirrors and reports\")
    parser.add_argument(\"--session-watch\", action=\"store_true\", help=\"Continuously poll for new sessions under --session-root/--session-stage\")
    parser.add_argument(\"--session-poll-interval\", type=float, default=5.0, help=\"Seconds between polls when --session-watch is set\")
    parser.add_argument(\"--run-id\", help=\"Identifier for this replay; defaults to source-derived value\")
    parser.add_argument(\"--detections-dir\", type=Path, default=Path(\"detections\"), help=\"Directory to store detection reports for non-session runs\")
    parser.add_argument(\"--output-json\", type=Path, help=\"Explicit path for the detection JSON (overrides --detections-dir)\")
    parser.add_argument(\"--summary-md\", type=Path, default=Path(\"OUTPUT/visualmodel-summary.md\"), help=\"Markdown summary ledger to append results into\")
    parser.add_argument(\"--no-display\", action=\"store_true\", help=\"Disable OpenCV visualization\")
    parser.add_argument(\"--max-frames\", type=int, help=\"Stop after N frames for non-session runs or per-session replays\")
    args = parser.parse_args()

    if sum(bool(x) for x in (args.video, args.frames_dir, args.session_root)) > 1:
        parser.error(\"Select only one input source: --video, --frames-dir, or --session-root.\")

    cfg = load_metadata(args.models_json, args.assets_root, args.model_name)
    runner = ModelRunner(cfg)

    if args.session_root:
        watcher = SessionWatcher(
            runner=runner,
            session_root=args.session_root,
            stage=args.session_stage,
            output_root=args.session_output_root,
            summary_md=args.summary_md,
            max_frames=args.max_frames,
            poll_interval=args.session_poll_interval,
        )
        processed = watcher.process_available_sessions()
        if args.session_watch:
            try:
                watcher.watch()
            except KeyboardInterrupt:
                pass
        elif not processed:
            print(f\"No unprocessed sessions found under {watcher.stage_dir}\", file=sys.stderr)
        return

    if args.video and not args.video.exists():
        parser.error(f\"Video file {args.video} does not exist\")
    if args.frames_dir and not args.frames_dir.exists():
        parser.error(f\"Frames directory {args.frames_dir} does not exist\")

    source_desc: Dict[str, Any]
    if args.video:
        frame_stream = video_frame_stream(args.video)
        run_id = args.run_id or args.video.stem
        source_desc = {\"type\": \"video\", \"path\": args.video.resolve()}
    elif args.frames_dir:
        frame_stream = frames_dir_stream(args.frames_dir)
        run_id = args.run_id or args.frames_dir.name
        source_desc = {\"type\": \"frames\", \"path\": args.frames_dir.resolve()}
    else:
        frame_stream = camera_frame_stream(args.camera)
        run_id = args.run_id or f\"camera{args.camera}_{int(time.time())}\"
        source_desc = {\"type\": \"camera\", \"camera_index\": args.camera}

    source_desc[\"run_id\"] = run_id
    recorder = DetectionRecorder(cfg.name, source_desc, run_id=run_id)
    recorder = run_stream(runner, frame_stream, recorder, display=not args.no_display, max_frames=args.max_frames)

    detections_dir = args.detections_dir
    detections_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_json or detections_dir / f\"{run_id}.json\"
    summary = recorder.write_json(output_json)
    append_summary_markdown(summary, args.summary_md)
    print(json.dumps({\"run_id\": run_id, \"stats\": summary[\"stats\"]}))

if __name__ == \"__main__\":
    main()
