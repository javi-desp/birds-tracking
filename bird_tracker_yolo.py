"""
YOLO26 bird trajectory tracker con interfaz interactiva.

Este script ejecuta segmentación y seguimiento de pájaros usando YOLO26
con persistencia de trackers y visualización en tiempo real.
Solo usa YOLO26 como motor de detección y permite cambiar modos
al vuelo mientras el vídeo se reproduce en bucle.

Modelos YOLO26 disponibles:
  - yolo26n-seg.pt (nano, más rápido)
  - yolo26s-seg.pt (small)
  - yolo26m-seg.pt (medium)
  - yolo26l-seg.pt (large)
  - yolo26x-seg.pt (xlarge, más preciso)

Requisitos:
  pip install ultralytics torch torchvision opencv-python numpy

Uso:
  python bird_tracker_yolo.py --video birds.mp4 --model yolo26x-seg.pt --preview --device 0
"""

import argparse
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: no se encuentra el paquete 'ultralytics'. Instala con:")
    print("  pip install ultralytics torch torchvision opencv-python numpy")
    sys.exit(1)

RENDER_MODES = ["overlay", "greenscreen", "blackscreen", "mask"]
COLOR_MODES = ["rainbow", "single", "heatmap", "emotion", "palette"]
TRAIL_MODES = ["line", "dots", "fade", "particles"]
ART_MODES = ["none", "neon", "sunrise", "ocean"]
PRESET_COLORS = ["red", "green", "blue", "yellow", "cyan", "magenta", "orange", "purple", "white", "black"]
ART_PALETTES = {
    "neon": [
        (200, 0, 255),
        (255, 128, 0),
        (0, 255, 255),
        (255, 0, 255),
        (0, 255, 128),
    ],
    "sunrise": [
        (20, 120, 255),
        (80, 190, 255),
        (30, 180, 240),
        (45, 160, 255),
        (70, 200, 255),
    ],
    "ocean": [
        (220, 130, 0),
        (50, 170, 255),
        (100, 210, 210),
        (165, 190, 255),
        (255, 230, 100),
    ],
}

BIRD_CLASS = 14
SKY_COLOR = (205, 180, 95)  # cielo warm minimalista en BGR


@dataclass
class GuiConfig:
    render_mode: str = "overlay"
    color_mode: str = "rainbow"
    trail_mode: str = "fade"
    art_mode: str = "none"
    single_color: str = "red"
    trail_length: int = 80
    trail_thickness: int = 3
    dot_radius: int = 6
    trail_alpha: float = 0.8
    show_id: bool = True
    show_hud: bool = True
    hud_color: str = "white"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO bird tracker con interfaz interactiva y modo bucle."
    )
    parser.add_argument("--video", required=True, help="Ruta al archivo de vídeo.")
    parser.add_argument("--model", default="yolo26x-seg.pt",
                        help="Modelo YOLO26 de segmentación. Ej: yolo26n-seg.pt, yolo26x-seg.pt")
    parser.add_argument("--device", default="0",
                        help="Dispositivo CUDA. Usa 0 para GPU. Ej: 0 o cuda:0")
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="Tamaño de imagen de inferencia. Recomendado >= 1280.")
    parser.add_argument("--tracker", default="bytetrack.yaml",
                        help="Tracker persistente. Ej: bytetrack.yaml o botsort.yaml")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confianza mínima de detección.")
    parser.add_argument("--output", default="yolo_bird_trajectory.mp4",
                        help="Ruta de salida para el vídeo renderizado.")
    parser.add_argument("--preview", action="store_true",
                        help="Mostrar ventana con vídeo original + resultado.")
    parser.add_argument("--no-save", action="store_true",
                        help="No guardar vídeo de salida.")
    parser.add_argument("--loop", action="store_true",
                        help="Repetir el vídeo en bucle en modo preview.")
    return parser.parse_args()


def parse_color(value: str) -> Tuple[int, int, int]:
    value = value.strip().lower()
    named = {
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "yellow": (0, 255, 255),
        "cyan": (255, 255, 0),
        "magenta": (255, 0, 255),
        "orange": (0, 165, 255),
        "purple": (128, 0, 128),
    }
    if value in named:
        return named[value]
    if value.startswith("#") and len(value) == 7:
        r = int(value[1:3], 16)
        g = int(value[3:5], 16)
        b = int(value[5:7], 16)
        return (b, g, r)
    return named["white"]


def velocity_to_color(velocity: float, max_vel: float = 30.0) -> Tuple[int, int, int]:
    t = min(velocity / max(max_vel, 1.0), 1.0)
    if t < 0.5:
        r = 0
        g = int(255 * t * 2)
        b = int(255 * (1 - t * 2))
    else:
        r = int(255 * (t - 0.5) * 2)
        g = int(255 * (1 - (t - 0.5) * 2))
        b = 0
    return (b, g, r)


def get_track_color(track_id: int, cfg: GuiConfig, velocity: float = 0.0) -> Tuple[int, int, int]:
    if cfg.color_mode == "single":
        return parse_color(cfg.single_color)
    if cfg.color_mode == "rainbow":
        return COLOR_PALETTE[track_id % len(COLOR_PALETTE)]
    if cfg.color_mode == "heatmap":
        return velocity_to_color(velocity)
    if cfg.color_mode == "emotion":
        return velocity_to_color(velocity, max_vel=40.0)
    if cfg.color_mode == "palette" and cfg.art_mode in ART_PALETTES:
        palette = ART_PALETTES[cfg.art_mode]
        return palette[track_id % len(palette)]
    return COLOR_PALETTE[track_id % len(COLOR_PALETTE)]


def safe_centroid_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    if mask is None or mask.size == 0:
        return None
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.mean()), int(ys.mean())


def create_sky_background(shape: Tuple[int, int, int]) -> np.ndarray:
    canvas = np.full(shape, SKY_COLOR, dtype=np.uint8)
    return cv2.GaussianBlur(canvas, (31, 31), 0)


def draw_trails(canvas: np.ndarray,
                trails: Dict[int, List[Tuple[int, int]]],
                tracker_velocities: Dict[int, float],
                cfg: GuiConfig) -> np.ndarray:
    for track_id, points in trails.items():
        if len(points) < 1:
            continue
        velocity = tracker_velocities.get(track_id, 0.0)
        color = get_track_color(track_id, cfg, velocity)
        n = len(points)
        if cfg.trail_mode == "line" and n >= 2:
            pts = np.array(points[-cfg.trail_length:], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], False, color, cfg.trail_thickness, cv2.LINE_AA)
        elif cfg.trail_mode == "dots":
            spacing = max(1, cfg.trail_length // 10)
            for i, pt in enumerate(points[-cfg.trail_length:]):
                if i % spacing == 0:
                    cv2.circle(canvas, pt, max(1, cfg.trail_thickness), color, -1)
        elif cfg.trail_mode == "fade" and n >= 2:
            pts = points[-cfg.trail_length:]
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                seg_color = tuple(int(c * alpha) for c in color)
                thickness = max(1, int(cfg.trail_thickness * alpha + 0.5))
                cv2.line(canvas, pts[i - 1], pts[i], seg_color, thickness, cv2.LINE_AA)
        elif cfg.trail_mode == "particles":
            pts = points[-cfg.trail_length:]
            for i, pt in enumerate(pts):
                alpha = (i + 1) / len(pts)
                radius = max(1, int(cfg.dot_radius * (0.4 + 0.6 * alpha)))
                part_color = tuple(int(c * (0.2 + 0.8 * alpha)) for c in color)
                cv2.circle(canvas, pt, radius, part_color, -1)
    return canvas


def render_frame(frame: np.ndarray,
                 boxes: np.ndarray,
                 ids: List[int],
                 masks: Optional[List[np.ndarray]],
                 trails: Dict[int, List[Tuple[int, int]]],
                 tracker_velocities: Dict[int, float],
                 cfg: GuiConfig) -> np.ndarray:
    h, w = frame.shape[:2]
    art_bg = None
    if cfg.art_mode != "none":
        art_bg = create_sky_background(frame.shape)

    if cfg.render_mode == "mask":
        mask_canvas = np.zeros_like(frame)
        if masks is not None:
            for mask, track_id in zip(masks, ids):
                if mask is None:
                    continue
                color = get_track_color(track_id, cfg, tracker_velocities.get(track_id, 0.0))
                colored = np.zeros_like(frame)
                colored[mask > 0] = color
                alpha = 0.6
                cv2.addWeighted(colored, alpha, mask_canvas, 1 - alpha, 0, mask_canvas)
        frame_out = mask_canvas
    elif cfg.render_mode in ("greenscreen", "blackscreen"):
        bg_color = SKY_COLOR if cfg.render_mode == "greenscreen" else (0, 0, 0)
        if art_bg is not None:
            background = art_bg
        else:
            background = np.full((h, w, 3), bg_color, dtype=np.uint8)
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        if masks is not None:
            for mask in masks:
                if mask is None:
                    continue
                combined_mask = cv2.bitwise_or(combined_mask, mask.astype(np.uint8) * 255)
        inv_mask = cv2.bitwise_not(combined_mask)
        birds = cv2.bitwise_and(frame, frame, mask=combined_mask)
        bg = cv2.bitwise_and(background, background, mask=inv_mask)
        frame_out = cv2.add(birds, bg)
    else:
        frame_out = frame.copy()
        if art_bg is not None:
            frame_out = cv2.addWeighted(frame_out, 0.7, art_bg, 0.3, 0)

    trail_canvas = frame_out.copy()
    trail_canvas = draw_trails(trail_canvas, trails, tracker_velocities, cfg)
    frame_out = cv2.addWeighted(trail_canvas, cfg.trail_alpha, frame_out, 1 - cfg.trail_alpha, 0)

    for idx, track_id in enumerate(ids):
        color = get_track_color(track_id, cfg, tracker_velocities.get(track_id, 0.0))
        if idx < len(boxes):
            x0, y0, x1, y1 = map(int, boxes[idx])
            if cfg.render_mode == "overlay":
                cv2.rectangle(frame_out, (x0, y0), (x1, y1), color, 2)
            centroid = (int((x0 + x1) / 2), int((y0 + y1) / 2))
            cv2.circle(frame_out, centroid, cfg.dot_radius, color, -1)
            if cfg.show_id:
                cv2.putText(frame_out, str(track_id), (x0, max(0, y0 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    if cfg.show_hud:
        hud_color = parse_color(cfg.hud_color)
        mode_str = f"{cfg.render_mode} | art:{cfg.art_mode} | {cfg.color_mode} | trail:{cfg.trail_mode}"
        cv2.putText(frame_out, mode_str, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_color, 2, cv2.LINE_AA)
        cv2.putText(frame_out, "Presiona 'q' para salir", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hud_color, 1, cv2.LINE_AA)
    return frame_out


def create_gui(cfg: GuiConfig) -> None:
    window = "YOLO Bird Controls"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 420, 750)

    def put_trackbar(name: str, value: int, max_val: int, callback):
        cv2.createTrackbar(name, window, value, max_val, callback)

    put_trackbar("render_mode", RENDER_MODES.index(cfg.render_mode), len(RENDER_MODES) - 1,
                 lambda v: setattr(cfg, "render_mode", RENDER_MODES[v]))
    put_trackbar("color_mode", COLOR_MODES.index(cfg.color_mode), len(COLOR_MODES) - 1,
                 lambda v: setattr(cfg, "color_mode", COLOR_MODES[v]))
    put_trackbar("art_mode", ART_MODES.index(cfg.art_mode), len(ART_MODES) - 1,
                 lambda v: setattr(cfg, "art_mode", ART_MODES[v]))
    put_trackbar("trail_mode", TRAIL_MODES.index(cfg.trail_mode), len(TRAIL_MODES) - 1,
                 lambda v: setattr(cfg, "trail_mode", TRAIL_MODES[v]))
    put_trackbar("single_color", PRESET_COLORS.index(cfg.single_color) if cfg.single_color in PRESET_COLORS else 0,
                 len(PRESET_COLORS) - 1,
                 lambda v: setattr(cfg, "single_color", PRESET_COLORS[v]))
    put_trackbar("trail_len", cfg.trail_length, 200,
                 lambda v: setattr(cfg, "trail_length", max(1, v)))
    put_trackbar("trail_thk", cfg.trail_thickness, 12,
                 lambda v: setattr(cfg, "trail_thickness", max(1, v)))
    put_trackbar("dot_rad", cfg.dot_radius, 20,
                 lambda v: setattr(cfg, "dot_radius", max(1, v)))
    put_trackbar("trail_a", int(cfg.trail_alpha * 100), 100,
                 lambda v: setattr(cfg, "trail_alpha", max(0.05, v / 100.0)))
    put_trackbar("show_id", int(cfg.show_id), 1,
                 lambda v: setattr(cfg, "show_id", bool(v)))
    put_trackbar("show_hud", int(cfg.show_hud), 1,
                 lambda v: setattr(cfg, "show_hud", bool(v)))

    print("[INFO] Control UI lista. Cambia modos y parámetros al vuelo.")


def load_yolo_model(model_path: str) -> YOLO:
    try:
        return YOLO(model_path)
    except Exception as exc:
        print(f"Error cargando el modelo YOLO: {exc}")
        sys.exit(1)


def safe_iter_results(results):
    try:
        for item in results:
            yield item
    except Exception as exc:
        print(f"Error durante la inferencia: {exc}")
        return


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.video):
        print(f"Error: vídeo no encontrado: {args.video}")
        sys.exit(1)
    if not os.path.isfile(args.model):
        print(f"Error: modelo no encontrado: {args.model}")
        sys.exit(1)

    model = load_yolo_model(args.model)
    device_str = f"cuda:{args.device}" if args.device.isdigit() else args.device

    print("[INFO] Usando modelo:", args.model)
    print("[INFO] Dispositivo:", device_str)
    print("[INFO] Tracker:", args.tracker)
    print("[INFO] Resolución de inferencia:", args.imgsz)
    print("[INFO] Modelos YOLO26 recomendados: yolo26n, yolo26s, yolo26m, yolo26l, yolo26x")

    cfg = GuiConfig()
    if args.preview:
        create_gui(cfg)

    save_output = not args.no_save and not args.preview
    writer = None
    width = height = fps = None
    if save_output:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"Error abriendo el vídeo: {args.video}")
            sys.exit(1)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width * 2, height))
        if not writer.isOpened():
            print(f"Error: no se puede crear el vídeo de salida en {args.output}")
            sys.exit(1)
        cap.release()

    bird_tracks: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    tracker_velocities: Dict[int, float] = defaultdict(float)
    total_frames = 0

    def process_once() -> int:
        nonlocal total_frames
        frame_count = 0
        results = model.track(
            source=args.video,
            device=device_str,
            imgsz=args.imgsz,
            tracker=args.tracker,
            persist=True,
            classes=[BIRD_CLASS],
            conf=args.conf,
            verbose=False,
        )

        for result in safe_iter_results(results):
            frame = getattr(result, 'orig_img', None)
            if frame is None:
                continue

            boxes = np.array(result.boxes.xyxy.cpu()) if hasattr(result.boxes, 'xyxy') else np.array([])
            ids = []
            if hasattr(result.boxes, 'id') and result.boxes.id is not None:
                ids = [int(x) for x in result.boxes.id.cpu().numpy()]
            else:
                ids = list(range(len(boxes)))

            masks = None
            if hasattr(result, 'masks') and result.masks is not None:
                try:
                    masks = [mask.cpu().numpy().astype(bool) for mask in result.masks.data]
                except Exception:
                    masks = None

            prev_positions = {track_id: bird_tracks[track_id][-1] for track_id in ids if bird_tracks[track_id]}
            for idx, track_id in enumerate(ids):
                if idx >= len(boxes):
                    break
                box = boxes[idx]
                centroid = centroid_from_box(box)
                if masks is not None and idx < len(masks):
                    mask_centroid = safe_centroid_from_mask(masks[idx])
                    if mask_centroid is not None:
                        centroid = mask_centroid
                bird_tracks[track_id].append(centroid)
                if track_id in prev_positions:
                    px, py = prev_positions[track_id]
                    dx = centroid[0] - px
                    dy = centroid[1] - py
                    tracker_velocities[track_id] = np.sqrt(dx * dx + dy * dy)
                else:
                    tracker_velocities[track_id] = 0.0

            frame_out = render_frame(frame, boxes, ids, masks, bird_tracks, tracker_velocities, cfg)
            combined = np.hstack([frame, frame_out])

            if args.preview:
                cv2.imshow("YOLO Bird Preview", combined)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    return -1
                if key == ord('r'):
                    print("[INFO] Reproducción en bucle reiniciada.")
                    return frame_count

            if writer is not None:
                writer.write(combined)

            frame_count += 1
            total_frames += 1
        return frame_count

    while True:
        count = process_once()
        if count < 0:
            break
        if not args.loop or not args.preview:
            break
        print("[INFO] Bucle completo, reiniciando reproducción.")
        bird_tracks.clear()
        tracker_velocities.clear()

    if writer is not None:
        writer.release()
    if args.preview:
        cv2.destroyAllWindows()

    print(f"[OK] Procesados {total_frames} frames.")
    if writer is not None:
        print(f"[OK] Vídeo guardado en: {args.output}")


if __name__ == "__main__":
    main()
