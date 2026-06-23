"""
Bird Tracker v2
===============
Rastrea pájaros en vídeo con segmentación por sustracción de fondo.

Uso:
    python bird_tracker.py <video> [--config config.json] [--gen-config]

Si no se especifica --config, busca 'config.json' en el directorio actual.
Con --gen-config genera una plantilla de configuración y sale.

Modos de visualización (visualization.render_mode):
  - "overlay"    : trayectorias sobre el vídeo original
  - "greenscreen": fondo verde, solo los pájaros recortados
  - "blackscreen": fondo negro, solo los pájaros recortados
  - "mask"       : muestra la máscara binaria de segmentación

Modos de color (visualization.color_mode):
  - "rainbow"    : cada pájaro con su propio color del arcoíris
  - "single"     : todos del mismo color (visualization.single_color)
  - "heatmap"    : la intensidad del color aumenta con la velocidad

Modos de trayectoria (visualization.trail_mode):
  - "line"       : línea continua
  - "dots"       : puntos equiespaciados
  - "fade"       : línea que se desvanece hacia el pasado
"""

import cv2
import numpy as np
import argparse
import sys
import os
import json
import copy
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Dict, Any
from scipy.optimize import linear_sum_assignment


# ---Paleta arcoíris (BGR) ---
RAINBOW_COLORS: List[Tuple[int, int, int]] = [
    (0,   0,   255),  # rojo
    (0,  69,   255),  # naranja
    (0,  255,  255),  # amarillo
    (0,  255,    0),  # verde
    (255,  0,    0),  # azul
    (130,  0,   75),  # índigo
    (255,  0,  139),  # violeta
    (0,  128,  255),  # naranja oscuro
    (255, 255,   0),  # cian
    (128,  0,  255),  # rosa
]

NAMED_COLORS: Dict[str, Tuple[int, int, int]] = {
    "red":     (0,   0,   255),
    "green":   (0,   255,   0),
    "blue":    (255,  0,    0),
    "white":   (255, 255,  255),
    "black":   (0,    0,    0),
    "yellow":  (0,   255,  255),
    "cyan":    (255, 255,   0),
    "magenta": (255,  0,   255),
    "orange":  (0,   165,  255),
    "purple":  (128,  0,   128),
    "lime":    (0,   255,  128),
}


# ---Config dataclass ---
@dataclass
class DetectionConfig:
    threshold: int = 30            # Threshold de diferencia (0-255)
    blur_kernel: int = 5           # Tamaño del kernel de desenfoque (impar)
    min_area: int = 20             # Área mínima de contorno (px²)
    max_area: int = 5000           # Área máxima de contorno (px²)
    morph_iterations: int = 2      # Iteraciones de morfología de cierre
    bg_frames: int = 50            # Frames para calcular el fondo mediana
    adaptive_threshold: bool = False  # Usar threshold adaptativo (Otsu)

@dataclass
class TrackingConfig:
    max_distance: int = 80         # Distancia máxima de asociación (px)
    max_lost_frames: int = 8       # Frames máximos sin detección antes de borrar

@dataclass
class VisualizationConfig:
    render_mode: str = "overlay"     # overlay | greenscreen | blackscreen | mask
    color_mode: str = "rainbow"      # rainbow | single | heatmap
    trail_mode: str = "fade"         # line | dots | fade
    single_color: str = "red"        # Color en modo single
    background_color: str = "#00B140" # Color de fondo en modos greenscreen/blackscreen
    trail_length: int = 80           # Longitud de trayectoria (0 = sin límite)
    trail_thickness: int = 2         # Grosor de la línea de trayectoria
    dot_radius: int = 5              # Radio del punto actual del pájaro
    trail_alpha: float = 0.7         # Opacidad de la capa de trayectorias (overlay)
    bird_contour: bool = True        # Dibujar contorno blanco en el pájaro
    show_id: bool = False            # Mostrar ID numérico sobre cada pájaro
    show_hud: bool = True            # Mostrar info (frame, nº pájaros) en esquina
    hud_color: str = "white"         # Color del texto HUD
    dots_spacing: int = 8            # Separación entre puntos (trail_mode=dots)
    bird_mask_padding: int = 8       # Padding (px) alrededor del pájaro en modos recorte

@dataclass
class OutputConfig:
    output_file: Optional[str] = None  # None = auto (output_<video>.mp4)
    show_preview: bool = False          # Mostrar ventana en tiempo real
    preview_only: bool = False          # No guardar vídeo al usar vista previa interactiva
    loop_video: bool = False            # Repetir el vídeo en modo preview
    save_mask_video: bool = False       # Guardar también el vídeo de máscara
    export_csv: bool = False            # Exportar trayectorias a CSV

@dataclass
class Config:
    video: str = ""
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


# ---Config helpers ---
def config_to_dict(cfg: Config) -> dict:
    return asdict(cfg)

def config_from_dict(d: dict) -> Config:
    cfg = Config()
    if "video" in d:
        cfg.video = d["video"]
    if "detection" in d:
        det = d["detection"]
        cfg.detection = DetectionConfig(**{k: v for k, v in det.items()
                                           if k in DetectionConfig.__dataclass_fields__})
    if "tracking" in d:
        trk = d["tracking"]
        cfg.tracking = TrackingConfig(**{k: v for k, v in trk.items()
                                         if k in TrackingConfig.__dataclass_fields__})
    if "visualization" in d:
        viz = d["visualization"]
        cfg.visualization = VisualizationConfig(**{k: v for k, v in viz.items()
                                                    if k in VisualizationConfig.__dataclass_fields__})
    if "output" in d:
        out = d["output"]
        cfg.output = OutputConfig(**{k: v for k, v in out.items()
                                     if k in OutputConfig.__dataclass_fields__})
    return cfg

def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return config_from_dict(d)

def save_config(cfg: Config, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config_to_dict(cfg), f, indent=2, ensure_ascii=False)

def generate_default_config(path: str):
    """Genera un config.json de ejemplo con comentarios inline."""
    cfg = Config()
    cfg.video = "birds.mp4"
    d = config_to_dict(cfg)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
    print(f"[OK] Config de ejemplo generado en: {path}")


# ---Utilidades de color ---
def parse_color(s: str) -> Tuple[int, int, int]:
    """Convierte nombre o #RRGGBB a BGR."""
    s = s.strip().lower()
    if s in NAMED_COLORS:
        return NAMED_COLORS[s]
    if s.startswith("#") and len(s) == 7:
        r = int(s[1:3], 16)
        g = int(s[3:5], 16)
        b = int(s[5:7], 16)
        return (b, g, r)
    raise ValueError(f"Color no reconocido: '{s}'. Usa nombre (red, green…) o #RRGGBB")

def velocity_to_color(velocity: float, max_vel: float = 30.0) -> Tuple[int, int, int]:
    """Mapea velocidad a color: azul (lento) → verde → rojo (rápido)."""
    t = min(velocity / max(max_vel, 1.0), 1.0)
    # Interpolación: azul (0) -> cian (0.5) -> rojo (1)
    if t < 0.5:
        r = int(0)
        g = int(255 * t * 2)
        b = int(255 * (1 - t * 2))
    else:
        r = int(255 * (t - 0.5) * 2)
        g = int(255 * (1 - (t - 0.5) * 2))
        b = 0
    return (b, g, r)


# Modos interactivos
RENDER_MODES = ["overlay", "greenscreen", "blackscreen", "mask"]
COLOR_MODES = ["rainbow", "single", "heatmap"]
TRAIL_MODES = ["line", "dots", "fade"]
PRESET_COLORS = ["red", "green", "blue", "yellow", "cyan", "magenta", "orange", "purple", "white", "black"]


def clamp(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(max_value, value))


def ensure_odd(value: int) -> int:
    value = clamp(value, 1, 101)
    return value if value % 2 == 1 else value + 1


def build_background(video_path: str, n_frames: int) -> np.ndarray:
    """Fondo como mediana de n_frames muestras distribuidas uniformemente."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se puede abrir el vídeo para calcular el fondo: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample = min(max(1, n_frames), total)
    indices = np.linspace(0, total - 1, sample, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.release()

    if not frames:
        raise RuntimeError("No se pudieron leer frames para calcular el fondo.")
    return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)


def setup_preview_window(cfg: Config):
    control_window = "Bird Tracker Controls"
    preview_window = "Bird Tracker Preview"

    cv2.namedWindow(control_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(control_window, 420, 700)
    cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(preview_window, 1200, 600)

    def create_trackbar(name: str, value: int, max_value: int, callback):
        cv2.createTrackbar(name, control_window, value, max_value, callback)

    def set_visualization(field: str, value: Any):
        setattr(cfg.visualization, field, value)

    create_trackbar("render_mode", RENDER_MODES.index(cfg.visualization.render_mode), len(RENDER_MODES) - 1,
                    lambda v: set_visualization("render_mode", RENDER_MODES[v]))
    create_trackbar("color_mode", COLOR_MODES.index(cfg.visualization.color_mode), len(COLOR_MODES) - 1,
                    lambda v: set_visualization("color_mode", COLOR_MODES[v]))
    create_trackbar("trail_mode", TRAIL_MODES.index(cfg.visualization.trail_mode), len(TRAIL_MODES) - 1,
                    lambda v: set_visualization("trail_mode", TRAIL_MODES[v]))
    create_trackbar("single_color", PRESET_COLORS.index(cfg.visualization.single_color)
                    if cfg.visualization.single_color in PRESET_COLORS else 0,
                    len(PRESET_COLORS) - 1,
                    lambda v: set_visualization("single_color", PRESET_COLORS[v]))
    create_trackbar("threshold", clamp(cfg.detection.threshold, 0, 255), 255,
                    lambda v: setattr(cfg.detection, "threshold", v))
    create_trackbar("blur_kernel", cfg.detection.blur_kernel, 31,
                    lambda v: setattr(cfg.detection, "blur_kernel", ensure_odd(max(1, v))))
    create_trackbar("min_area", clamp(cfg.detection.min_area, 0, 5000), 5000,
                    lambda v: setattr(cfg.detection, "min_area", max(1, v)))
    create_trackbar("max_area", clamp(cfg.detection.max_area, 0, 30000), 30000,
                    lambda v: setattr(cfg.detection, "max_area", max(v, cfg.detection.min_area)))
    create_trackbar("morph_iter", clamp(cfg.detection.morph_iterations, 0, 10), 10,
                    lambda v: setattr(cfg.detection, "morph_iterations", v))
    create_trackbar("max_dist", clamp(cfg.tracking.max_distance, 1, 200), 200,
                    lambda v: setattr(cfg.tracking, "max_distance", max(1, v)))
    create_trackbar("max_lost", clamp(cfg.tracking.max_lost_frames, 0, 30), 30,
                    lambda v: setattr(cfg.tracking, "max_lost_frames", v))
    create_trackbar("trail_len", clamp(cfg.visualization.trail_length, 0, 200), 200,
                    lambda v: setattr(cfg.visualization, "trail_length", v))
    create_trackbar("trail_thk", clamp(cfg.visualization.trail_thickness, 1, 10), 10,
                    lambda v: setattr(cfg.visualization, "trail_thickness", max(1, v)))
    create_trackbar("dot_radius", clamp(cfg.visualization.dot_radius, 1, 20), 20,
                    lambda v: setattr(cfg.visualization, "dot_radius", max(1, v)))
    create_trackbar("dots_space", clamp(cfg.visualization.dots_spacing, 1, 50), 50,
                    lambda v: setattr(cfg.visualization, "dots_spacing", max(1, v)))
    create_trackbar("trail_alpha", int(clamp(cfg.visualization.trail_alpha * 100, 0, 100)), 100,
                    lambda v: setattr(cfg.visualization, "trail_alpha", v / 100.0))
    create_trackbar("show_id", int(cfg.visualization.show_id), 1,
                    lambda v: setattr(cfg.visualization, "show_id", bool(v)))
    create_trackbar("bird_contour", int(cfg.visualization.bird_contour), 1,
                    lambda v: setattr(cfg.visualization, "bird_contour", bool(v)))
    create_trackbar("show_hud", int(cfg.visualization.show_hud), 1,
                    lambda v: setattr(cfg.visualization, "show_hud", bool(v)))
    create_trackbar("adaptive", int(cfg.detection.adaptive_threshold), 1,
                    lambda v: setattr(cfg.detection, "adaptive_threshold", bool(v)))

    print("[INFO] Interfaz interactiva activada. Usa 'q' para salir, 'r' para recalcular el fondo.")
    print("       Cambia modos, parámetros y observa el vídeo original junto al modificado.")
    print("       El vídeo se repite en bucle mientras ajustes parámetros.")


# ---Detección ---
def detect_birds(frame_gray: np.ndarray,
                 background: np.ndarray,
                 cfg: DetectionConfig):
    """
    Devuelve:
      centroids  : lista de (cx, cy)
      contours   : lista de contornos (para recorte)
      mask       : máscara binaria uint8
    """
    diff = cv2.absdiff(frame_gray, background)

    bk = cfg.blur_kernel
    if bk > 1:
        bk = bk if bk % 2 == 1 else bk + 1
        diff = cv2.GaussianBlur(diff, (bk, bk), 0)

    if cfg.adaptive_threshold:
        _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(diff, cfg.threshold, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,
                            iterations=cfg.morph_iterations)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if cfg.min_area <= area <= cfg.max_area:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))
                valid_contours.append(cnt)

    return centroids, valid_contours, mask


# ---Tracker ---
class BirdTracker:
    def __init__(self, cfg: TrackingConfig):
        self.cfg = cfg
        self.next_id = 0
        self.tracks: Dict[int, dict] = {}
        self._color_cache: Dict[int, int] = {}

    def update(self, detections: List[Tuple[int, int]]) -> Dict[int, Tuple[int, int]]:
        if not self.tracks:
            for det in detections:
                self._new_track(det)
            return {tid: t["pos"] for tid, t in self.tracks.items() if t["lost"] == 0}

        track_ids = list(self.tracks.keys())
        track_pos = np.array([self.tracks[tid]["pos"] for tid in track_ids], dtype=float)

        if not detections:
            for tid in track_ids:
                self.tracks[tid]["lost"] += 1
            self._remove_lost()
            return {}

        det_arr = np.array(detections, dtype=float)
        diff = track_pos[:, None, :] - det_arr[None, :, :]
        dist_matrix = np.sqrt((diff ** 2).sum(axis=2))

        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        assigned_tracks, assigned_dets = set(), set()
        for r, c in zip(row_ind, col_ind):
            if dist_matrix[r, c] < self.cfg.max_distance:
                tid = track_ids[r]
                prev_pos = self.tracks[tid]["pos"]
                self.tracks[tid]["prev_pos"] = prev_pos
                self.tracks[tid]["pos"] = detections[c]
                dx = detections[c][0] - prev_pos[0]
                dy = detections[c][1] - prev_pos[1]
                self.tracks[tid]["velocity"] = float(np.sqrt(dx*dx + dy*dy))
                self.tracks[tid]["lost"] = 0
                assigned_tracks.add(tid)
                assigned_dets.add(c)

        for tid in track_ids:
            if tid not in assigned_tracks:
                self.tracks[tid]["lost"] += 1

        for c, det in enumerate(detections):
            if c not in assigned_dets:
                self._new_track(det)

        self._remove_lost()
        return {tid: t["pos"] for tid, t in self.tracks.items() if t["lost"] == 0}

    def _new_track(self, pos):
        color_idx = self.next_id % len(RAINBOW_COLORS)
        self.tracks[self.next_id] = {
            "pos": pos,
            "prev_pos": pos,
            "lost": 0,
            "color_idx": color_idx,
            "velocity": 0.0,
        }
        self._color_cache[self.next_id] = color_idx
        self.next_id += 1

    def _remove_lost(self):
        to_del = [tid for tid, t in self.tracks.items()
                  if t["lost"] > self.cfg.max_lost_frames]
        for tid in to_del:
            del self.tracks[tid]

    def get_color(self, tid: int, color_mode: str) -> Tuple[int, int, int]:
        track = self.tracks.get(tid)
        if track is None:
            # Track ya eliminado: usar color guardado en cache o fallback
            color_idx = self._color_cache.get(tid, 0)
            if color_mode == "rainbow":
                return RAINBOW_COLORS[color_idx % len(RAINBOW_COLORS)]
            return (0, 0, 255)
        if color_mode == "rainbow":
            return RAINBOW_COLORS[track["color_idx"] % len(RAINBOW_COLORS)]
        if color_mode == "heatmap":
            vel = track.get("velocity", 0.0)
            return velocity_to_color(vel)
        return (0, 0, 255)  # fallback rojo


# ---Renderizado ---
def render_frame(frame: np.ndarray,
                 mask: np.ndarray,
                 valid_contours,
                 active: Dict[int, Tuple[int, int]],
                 trails: Dict[int, List[Tuple[int, int]]],
                 tracker: BirdTracker,
                 cfg: VisualizationConfig,
                 single_color: Tuple[int, int, int],
                 bg_color: Tuple[int, int, int],
                 frame_idx: int,
                 total_frames: int) -> np.ndarray:

    render_mode = cfg.render_mode
    h, w = frame.shape[:2]

    # ---Construir fondo del frame de salida ---
    if render_mode == "mask":
        frame_out = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    elif render_mode in ("greenscreen", "blackscreen"):
        # Fondo liso del color especificado
        background_frame = np.full((h, w, 3), bg_color, dtype=np.uint8)

        # Dilatamos la máscara para incluir un poco de contexto alrededor del pájaro
        pad = cfg.bird_mask_padding
        if pad > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad*2+1, pad*2+1))
            dilated_mask = cv2.dilate(mask, k, iterations=1)
        else:
            dilated_mask = mask

        # Copiar los píxeles del frame original donde hay pájaro
        bird_pixels = cv2.bitwise_and(frame, frame, mask=dilated_mask)
        inv_mask = cv2.bitwise_not(dilated_mask)
        bg_pixels = cv2.bitwise_and(background_frame, background_frame, mask=inv_mask)
        frame_out = cv2.add(bird_pixels, bg_pixels)

    else:  # overlay
        frame_out = frame.copy()

    # ---Dibujar trayectorias ---
    trail_layer = frame_out.copy()

    for tid, trail in trails.items():
        if len(trail) < 1:
            continue

        if cfg.color_mode == "single":
            color = single_color
        else:
            color = tracker.get_color(tid, cfg.color_mode)

        n = len(trail)

        if cfg.trail_mode == "line" and n >= 2:
            pts = np.array(trail, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(trail_layer, [pts], False, color, cfg.trail_thickness)

        elif cfg.trail_mode == "dots" and n >= 1:
            spacing = max(1, cfg.dots_spacing)
            for i, pt in enumerate(trail):
                if i % spacing == 0:
                    cv2.circle(trail_layer, pt, max(1, cfg.trail_thickness), color, -1)

        elif cfg.trail_mode == "fade" and n >= 2:
            # Dibuja segmento a segmento con opacidad creciente hacia el presente
            for i in range(1, n):
                alpha_seg = i / n  # 0 (pasado) → 1 (presente)
                seg_color = tuple(int(c * alpha_seg) for c in color)
                thickness = max(1, int(cfg.trail_thickness * alpha_seg + 0.5))
                cv2.line(trail_layer, trail[i-1], trail[i], seg_color, thickness)

    # Mezclar capa de trayectorias
    frame_out = cv2.addWeighted(trail_layer, cfg.trail_alpha,
                                frame_out, 1 - cfg.trail_alpha, 0)

    # ---Dibujar puntos actuales ---
    for tid, pos in active.items():
        if cfg.color_mode == "single":
            color = single_color
        else:
            color = tracker.get_color(tid, cfg.color_mode)

        cv2.circle(frame_out, pos, cfg.dot_radius, color, -1)
        if cfg.bird_contour:
            cv2.circle(frame_out, pos, cfg.dot_radius + 2, (255, 255, 255), 1)
        if cfg.show_id:
            cv2.putText(frame_out, str(tid),
                        (pos[0] + cfg.dot_radius + 2, pos[1] - cfg.dot_radius),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # ---HUD ---
    if cfg.show_hud:
        try:
            hud_color = parse_color(cfg.hud_color)
        except ValueError:
            hud_color = (255, 255, 255)

        mode_str = f"{cfg.render_mode} | {cfg.color_mode} | trail:{cfg.trail_mode}"
        cv2.putText(frame_out,
                    f"Frame {frame_idx+1}/{total_frames}  Pajaros activos: {len(active)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_color, 2, cv2.LINE_AA)
        cv2.putText(frame_out, mode_str,
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, hud_color, 1, cv2.LINE_AA)

    return frame_out


# ---Main ---
def main():
    parser = argparse.ArgumentParser(
        description="Bird Tracker v2 — seguimiento de pájaros con config JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("video", nargs="?", default=None,
                        help="Ruta al vídeo (sobreescribe config.video).")
    parser.add_argument("--config", default="config.json",
                        help="Fichero de configuración JSON. Default: config.json")
    parser.add_argument("--preview", action="store_true",
                        help="Muestra ventana interactiva con controles y vídeo original + modificado.")
    parser.add_argument("--gen-config", action="store_true",
                        help="Genera un config.json de ejemplo y sale.")
    args = parser.parse_args()

    # ---Generar config de ejemplo ---
    if args.gen_config:
        generate_default_config(args.config)
        sys.exit(0)

    # ---Cargar config ---
    if os.path.isfile(args.config):
        print(f"[INFO] Cargando configuración desde: {args.config}")
        cfg = load_config(args.config)
    else:
        print(f"[WARN] No se encontró '{args.config}'. Usando configuración por defecto.")
        cfg = Config()

    # El argumento de vídeo en CLI sobreescribe el del config
    if args.video:
        cfg.video = args.video

    if args.preview:
        cfg.output.show_preview = True
        cfg.output.preview_only = True
        cfg.output.loop_video = True

    if not cfg.video:
        print("[ERROR] No se especificó vídeo. Ponlo en config.json o como argumento.", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(cfg.video):
        print(f"[ERROR] Vídeo no encontrado: {cfg.video}", file=sys.stderr)
        sys.exit(1)

    # ---Abrir vídeo ---
    cap = cv2.VideoCapture(cfg.video)
    if not cap.isOpened():
        print(f"[ERROR] No se puede abrir: {cfg.video}", file=sys.stderr)
        sys.exit(1)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] {cfg.video}  {width}x{height} @ {fps:.1f}fps  {total} frames")

    if args.preview:
        cfg.output.show_preview = True

    # ---Validar colores ---
    try:
        single_color = parse_color(cfg.visualization.single_color)
        bg_color = parse_color(cfg.visualization.background_color)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    # ---Fichero de salida ---
    out = None
    mask_writer = None
    csv_file = None
    if not cfg.output.preview_only:
        if not cfg.output.output_file:
            base, _ = os.path.splitext(os.path.basename(cfg.video))
            cfg.output.output_file = f"output_{base}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(cfg.output.output_file, fourcc, fps, (width, height))
        if not out.isOpened():
            print(f"[ERROR] No se puede escribir en: {cfg.output.output_file}", file=sys.stderr)
            sys.exit(1)

        if cfg.output.save_mask_video:
            mask_path = cfg.output.output_file.replace(".mp4", "_mask.mp4")
            mask_writer = cv2.VideoWriter(mask_path, fourcc, fps, (width, height))

        if cfg.output.export_csv:
            csv_path = cfg.output.output_file.replace(".mp4", "_trajectories.csv")
            csv_file = open(csv_path, "w")
            csv_file.write("frame,bird_id,x,y\n")

    elif cfg.output.show_preview:
        print("[INFO] Preview solo: no se guardará vídeo de salida ni CSV mientras pruebas parámetros.")
    if cfg.output.export_csv:
        csv_path = cfg.output.output_file.replace(".mp4", "_trajectories.csv")
        csv_file = open(csv_path, "w")
        csv_file.write("frame,bird_id,x,y\n")

    # ---Calcular fondo ---
    print(f"[INFO] Calculando fondo con {cfg.detection.bg_frames} frames…")
    background = build_background(cfg.video, cfg.detection.bg_frames)
    print("[INFO] Fondo calculado.")

    # ---Tracker y trayectorias ---
    tracker = BirdTracker(cfg.tracking)
    trails: Dict[int, List[Tuple[int, int]]] = defaultdict(list)

    if cfg.output.show_preview:
        setup_preview_window(cfg)

    frame_idx = 0
    print("[INFO] Procesando…")

    while True:
        ret, frame = cap.read()
        if not ret:
            if cfg.output.show_preview and cfg.output.loop_video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                tracker = BirdTracker(cfg.tracking)
                trails = defaultdict(list)
                continue
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        centroids, valid_contours, mask = detect_birds(gray, background, cfg.detection)
        active = tracker.update(centroids)

        # Actualizar trayectorias
        tl = cfg.visualization.trail_length
        for tid, pos in active.items():
            trails[tid].append(pos)
            if tl > 0 and len(trails[tid]) > tl:
                trails[tid] = trails[tid][-tl:]

        # CSV export
        if csv_file:
            for tid, pos in active.items():
                csv_file.write(f"{frame_idx},{tid},{pos[0]},{pos[1]}\n")

        try:
            single_color = parse_color(cfg.visualization.single_color)
        except ValueError:
            single_color = NAMED_COLORS["red"]
        try:
            bg_color = parse_color(cfg.visualization.background_color)
        except ValueError:
            bg_color = NAMED_COLORS["green"]

        # Renderizar frame
        frame_out = render_frame(
            frame, mask, valid_contours, active, trails,
            tracker, cfg.visualization,
            single_color, bg_color,
            frame_idx, total,
        )

        if out is not None:
            out.write(frame_out)

        if mask_writer is not None:
            mask_writer.write(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))

        if cfg.output.show_preview:
            preview = np.hstack([frame, frame_out]) if frame.shape == frame_out.shape else np.hstack([frame, cv2.resize(frame_out, (frame.shape[1], frame.shape[0]))])
            cv2.putText(preview, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(preview, "Modificado", (frame.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Bird Tracker Preview", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[INFO] Interrumpido.")
                break
            elif key == ord("r"):
                print("[INFO] Recalculando fondo con los parámetros actuales...")
                background = build_background(cfg.video, cfg.detection.bg_frames)
                print("[INFO] Fondo recalculado.")
            elif key == ord("a"):
                cfg.detection.adaptive_threshold = not cfg.detection.adaptive_threshold
                print(f"[INFO] Adaptive threshold {'activado' if cfg.detection.adaptive_threshold else 'desactivado'}.")

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  {frame_idx}/{total} ({frame_idx/max(total,1)*100:.1f}%)")

    # ---Cerrar ---
    cap.release()
    out.release()
    if mask_writer:
        mask_writer.release()
    if csv_file:
        csv_file.close()
    if cfg.output.show_preview:
        cv2.destroyAllWindows()

    print(f"\n[OK] Guardado en : {cfg.output.output_file}")
    print(f"     Frames       : {frame_idx}")
    print(f"     Pájaros total: {tracker.next_id}")


if __name__ == "__main__":
    main()
