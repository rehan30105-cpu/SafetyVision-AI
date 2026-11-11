import os
import time
import torch
from ultralytics import YOLO

class ModelManager:
    _instance = None

    @classmethod
    def get_instance(cls, model_path=None):
        if cls._instance is None:
            if model_path is None:
                model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'best.pt')
            cls._instance = ModelManager(model_path)
        return cls._instance

    def __init__(self, model_path):
        self.model_path = model_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        
        # Load model centrally
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
            
        self.model = YOLO(self.model_path)
        self.classes = self.model.names # Dict of {id: name}
        
        # Class color mapping for high-contrast safety visualization
        self.class_colors = {
            'Hardhat': '#10B981',       # Emerald Green (Safe)
            'Mask': '#06B6D4',          # Cyan (Safe)
            'Safety Vest': '#3B82F6',   # Blue (Safe)
            'NO-Hardhat': '#EF4444',    # Red (Violation)
            'NO-Mask': '#F97316',       # Orange (Violation)
            'NO-Safety Vest': '#DC2626',# Dark Red (Violation)
            'Person': '#8B5CF6',        # Violet
            'Safety Cone': '#F59E0B',   # Amber Warning
            'machinery': '#EC4899',     # Pink/Magenta Hazard
            'vehicle': '#6366F1'        # Indigo Hazard
        }

        # Real performance tracking metrics
        self.total_inferences = 0
        self.total_latency_ms = 0.0
        self.last_latency_ms = 0.0
        self.last_fps = 0.0
        self.total_confidence_sum = 0.0
        self.total_detections_count = 0
        self.model_loaded_time = time.time()

    def predict(self, frame_or_path, conf=0.35, iou=0.45, imgsz=640):
        """
        Runs centralized YOLO inference on a numpy frame or image path.
        Returns list of detection dictionaries and raw results.
        """
        start_t = time.perf_counter()
        results = self.model.predict(
            source=frame_or_path,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=self.device,
            verbose=False
        )
        latency_ms = (time.perf_counter() - start_t) * 1000.0
        
        self.last_latency_ms = round(latency_ms, 1)
        self.last_fps = round(1000.0 / max(latency_ms, 1.0), 1)
        self.total_inferences += 1
        self.total_latency_ms += latency_ms

        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                xyxy = [float(v) for v in box.xyxy[0].tolist()]
                cls_name = self.classes.get(cls_id, f"Class {cls_id}")

                self.total_confidence_sum += confidence
                self.total_detections_count += 1

                detections.append({
                    'class_id': cls_id,
                    'class_name': cls_name,
                    'confidence': round(confidence, 3),
                    'bbox': [round(x, 1) for x in xyxy], # [x1, y1, x2, y2]
                    'color': self.class_colors.get(cls_name, '#94A3B8')
                })

        return detections, results[0]

    def get_stats(self):
        avg_latency = round(self.total_latency_ms / max(self.total_inferences, 1), 1)
        avg_conf = round(self.total_confidence_sum / max(self.total_detections_count, 1), 3) if self.total_detections_count > 0 else 0.0
        return {
            'model_name': 'YOLOv8n PPE & Hazard Detector',
            'model_path': self.model_path,
            'device': self.device.upper(),
            'device_name': self.device_name,
            'classes': self.classes,
            'total_inferences': self.total_inferences,
            'total_detections': self.total_detections_count,
            'last_latency_ms': self.last_latency_ms,
            'average_latency_ms': avg_latency,
            'current_fps': self.last_fps,
            'average_confidence': avg_conf,
            'uptime_seconds': round(time.time() - self.model_loaded_time, 1)
        }
