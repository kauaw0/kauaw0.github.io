"""
vehicle_counter.py
==================
Classe reutilizável para detectar e contar veículos em vídeos de estacionamento.
Detecta e conta CARROS e MOTOS separadamente usando YOLOv10 + ByteTrack.

Dependências:
    pip install ultralytics supervision opencv-python

Uso:
    from vehicle_counter import VehicleCounter

    counter = VehicleCounter(
        video_path="entrada.mp4",
        line_y_start=0.5,
        line_y_end=0.5,
        total_vagas_carros=90,
        total_vagas_motos=40,
    )
    counter.process()
"""

import os as _os
import sys
from pathlib import Path

import cv2
import numpy as np

# ── verificação de GPU ────────────────────────────────────────────────────────
def _detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"[INFO] GPU CUDA disponível: {name}")
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps and mps.is_available() and mps.is_built():
            print("[INFO] GPU MPS disponível (Apple Silicon) ✓")
            return "mps"
        print("[INFO] Nenhuma GPU detectada – usando CPU.")
        return "cpu"
    except Exception as e:
        print(f"[WARN] Erro ao detectar GPU: {e} – usando CPU.")
        return "cpu"

GPU_DEVICE = _detect_device()


# ── imports principais ────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("[ERRO] ultralytics não instalado. Execute: pip install ultralytics")

try:
    import supervision as sv
except ImportError:
    sys.exit("[ERRO] supervision não instalada. Execute: pip install supervision")

# ── classes COCO ──────────────────────────────────────────────────────────────
# Apenas carros, ônibus e caminhões. Motos e pessoas não serão detectados.
CAR_CLASS_IDS  = {2: "car", 5: "bus", 7: "truck"}

class VehicleCounter:
    """
    Detecta e conta VEÍCULOS (carros, ônibus, caminhões) cruzando uma linha virtual em um vídeo.

    Parâmetros
    ----------
    video_path : str
        Caminho para o vídeo de entrada (.mp4).
    output_path : str | None
        Caminho do vídeo de saída anotado. None = gera nome automático.
    line_y_start : float
        Posição Y relativa do ponto ESQUERDO da linha (0.0=topo, 1.0=base).
    line_y_end : float
        Posição Y relativa do ponto DIREITO da linha.
        left < right → diagonal esquerda-alta / direita-baixa.
    model_name : str
        Modelo YOLO a carregar (ex.: "yolov10n.pt").
    confidence : float
        Limiar de confiança geral para veículos (0–1).
    total_vagas : int
        Capacidade total de vagas. Padrão: 90.
    show_preview : bool
        Se True, exibe janela em tempo real.
    save_video : bool
        Se True, grava o vídeo anotado em output_path.
    imgsz : int
        Resolução de inferência YOLO (menor = mais rápido).
    display_scale : float
        Escala da janela de preview (0.6 = 60% do tamanho original).
    display_skip : int
        Exibe 1 a cada N frames (inferência ocorre em todos).
    """

    def __init__(
        self,
        video_path: str,
        output_path: str = None,
        line_y_start: float = 0.5,
        line_y_end: float = 0.5,
        model_name: str = "yolov10n.pt",
        confidence: float = 0.35,
        total_vagas: int = 90,
        show_preview: bool = True,
        save_video: bool = False,
        imgsz: int = 640,
        display_scale: float = 0.6,
        display_skip: int = 2,
    ):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")

        if output_path is None:
            self.output_path = self.video_path.parent / f"{self.video_path.stem}_resultado.mp4"
        else:
            self.output_path = Path(output_path)

        self.line_y_start       = line_y_start
        self.line_y_end         = line_y_end
        self.model_name         = model_name
        self.confidence         = confidence
        self.total_vagas        = total_vagas
        self.show_preview       = show_preview
        self.save_video         = save_video
        self.imgsz              = imgsz
        self.display_scale      = display_scale
        self.display_skip       = display_skip

        # Contadores
        self.count_in  = 0
        self.count_out = 0
        self.extra_out = 0 # Usado para diminuir de outras câmeras no cálculo de vagas

        # Componentes (inicializados em _setup)
        self.model           = None
        self.tracker         = None
        self.line_zone       = None
        self.line_annotator  = None
        self.box_annotator   = None
        self.label_annotator = None

    # ── configuração ─────────────────────────────────────────────────────────

    def _setup(self, frame_width: int, frame_height: int):
        print(f"[INFO] Carregando modelo '{self.model_name}' no dispositivo '{GPU_DEVICE}'...")
        self.model = YOLO(self.model_name)
        self.model.to(GPU_DEVICE)

        self.tracker = sv.ByteTrack(lost_track_buffer=120)

        # Linha de contagem
        y_left  = int(frame_height * self.line_y_start)
        y_right = int(frame_height * self.line_y_end)
        start = sv.Point(0,           y_left)
        end   = sv.Point(frame_width, y_right)

        self.line_zone = sv.LineZone(start=start, end=end)

        self.line_annotator  = sv.LineZoneAnnotator(thickness=3, text_thickness=2, text_scale=0.8)
        self.box_annotator   = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.55, text_thickness=1)

        print(f"[INFO] Linha: esquerda Y={y_left}px → direita Y={y_right}px")
        print(f"[INFO] Confiança veículos ≥ {self.confidence}")

    # ── processamento de frame ────────────────────────────────────────────────

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self.model(
            frame,
            conf=self.confidence,
            classes=list(CAR_CLASS_IDS.keys()),
            imgsz=self.imgsz,
            device=GPU_DEVICE,
            verbose=False,
        )[0]

        all_det = sv.Detections.from_ultralytics(results)

        # Atualiza rastreamento
        tracked_det = self.tracker.update_with_detections(all_det)

        # ── Contagem por cruzamento ───────────────────────────────────────
        if len(tracked_det) > 0:
            in_c, out_c = self.line_zone.trigger(tracked_det)
            self.count_in  += int(in_c.sum())
            self.count_out += int(out_c.sum())

        # ── Anotações ────────────────────────────────────────────────────
        labels = []
        for i in range(len(tracked_det)):
            tid  = tracked_det.tracker_id[i] if tracked_det.tracker_id is not None else "?"
            cid  = int(tracked_det.class_id[i])
            name = CAR_CLASS_IDS.get(cid, "vehicle")
            conf = tracked_det.confidence[i] if tracked_det.confidence is not None else 0.0
            labels.append(f"#{tid} {name} {conf:.2f}")

        annotated = frame.copy()
        annotated = self.box_annotator.annotate(scene=annotated, detections=tracked_det)
        annotated = self.label_annotator.annotate(scene=annotated, detections=tracked_det, labels=labels)
        annotated = self.line_annotator.annotate(frame=annotated, line_counter=self.line_zone)
        annotated = self._draw_counter_panel(annotated, is_secondary=getattr(self, "is_secondary", False))

        return annotated

    # ── painel de contadores ──────────────────────────────────────────────────

    def _draw_counter_panel(self, frame: np.ndarray, is_secondary: bool = False) -> np.ndarray:
        """Painel único: Apenas Veículos."""
        no_carros = max(0, self.count_in  - self.count_out - self.extra_out)
        vagas_c   = max(0, self.total_vagas - no_carros)

        overlay = frame.copy()
        panel_h = 100
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], panel_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

        cv2.line(frame, (0, panel_h), (frame.shape[1], panel_h), (60, 60, 60), 1)

        font  = cv2.FONT_HERSHEY_DUPLEX
        s     = 1.0
        thick = 2
        W     = frame.shape[1]
        c1    = 20
        c2    = W // 4 + 10
        c3    = W // 2 + 10
        c4    = 3 * W // 4 - 10

        cv2.putText(frame, "VEICULOS", (c1, 35), font, 1.0, (200, 200, 200), 2)
        if is_secondary:
            cv2.putText(frame, "CAMERA 2: APENAS SAIDAS", (c2, 35), font, 1.0, (0, 140, 255), 2)
        
        cv2.putText(frame, f"Entradas: {self.count_in}",
                    (c1, 80), font, s, (0, 255, 120), thick)
        cv2.putText(frame, f"Saidas: {self.count_out}",
                    (c2, 80), font, s, (0, 140, 255), thick)

        if not is_secondary:
            cv2.putText(frame, f"No estac.: {no_carros}",
                        (c3, 80), font, s, (255, 220, 0), thick)
            cor_c = (0, 220, 50) if vagas_c > 10 else (0, 60, 255)
            cv2.putText(frame, f"Vagas: {vagas_c}/{self.total_vagas}",
                        (c4, 80), font, s, cor_c, thick)

        return frame

    # ── pipeline principal ────────────────────────────────────────────────────

    def _run_video(self, video_path: Path, frame_width: int, frame_height: int,
                   global_frame: list) -> bool:
        """Loop de frames de um arquivo. Retorna False se o user encerrou com 'q'."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Nao foi possivel abrir: {video_path}")

        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"\n[INFO] Reproducao: {video_path.name}  ({total_frames} frames @ {fps:.0f} fps)")

        writer = None
        if self.save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_path = video_path.parent / f"{video_path.stem}_resultado.mp4"
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (frame_width, frame_height))

        interrupted = False
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                annotated = self._process_frame(frame)

                if writer is not None:
                    writer.write(annotated)

                if self.show_preview and global_frame[0] % self.display_skip == 0:
                    dw = int(frame_width  * self.display_scale)
                    dh = int(frame_height * self.display_scale)
                    preview = cv2.resize(annotated, (dw, dh))
                    cv2.imshow("Estacionamento - Monitoramento", preview)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("[INFO] Interrompido pelo usuario.")
                        interrupted = True
                        break

                global_frame[0] += 1
                if global_frame[0] % 30 == 0:
                    print(f"[INFO] frame {global_frame[0]:>4} | "
                          f"Veículos E={self.count_in} S={self.count_out}")
        finally:
            cap.release()
            if writer is not None:
                writer.release()

        return not interrupted

    def _build_result(self) -> dict:
        no_carros = max(0, self.count_in  - self.count_out)
        return {
            "entradas":  self.count_in,
            "saidas":    self.count_out,
            "no_estac":  no_carros,
            "vagas_liv": max(0, self.total_vagas - no_carros),
        }

    def _print_result(self, result: dict):
        print("\n" + "=" * 56)
        print(f"  [VEICULOS]  Entradas: {result['entradas']}  |  Saidas: {result['saidas']}")
        print(f"              No estac.: {result['no_estac']}  |  Vagas livres: {result['vagas_liv']}/{self.total_vagas}")
        print("=" * 56)

    # ── API publica ───────────────────────────────────────────────────────────

    def process(self) -> dict:
        """Processa o video unico indicado no construtor."""
        cap_probe = cv2.VideoCapture(str(self.video_path))
        w = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_probe.release()
        self._setup(w, h)
        print("[INFO] Pressione 'q' para encerrar.")
        self._run_video(self.video_path, w, h, [0])
        cv2.destroyAllWindows()
        result = self._build_result()
        self._print_result(result)
        return result

    def process_many(self, video_paths: list) -> dict:
        """
        Processa multiplos videos em sequencia como se fossem UM so.
        Contadores e rastreadores sao mantidos entre os videos.
        """
        paths = [Path(p) for p in video_paths]

        cap_probe = cv2.VideoCapture(str(paths[0]))
        w = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_probe.release()

        self._setup(w, h)
        print(f"[INFO] Sessao unificada: {len(paths)} video(s) em sequencia.")
        print("[INFO] Pressione 'q' para encerrar.")

        global_frame = [0]
        for vpath in paths:
            ok = self._run_video(vpath, w, h, global_frame)
            if not ok:
                break

        cv2.destroyAllWindows()

        result = self._build_result()
        print("\n" + "=" * 56)
        print("  RESULTADO FINAL (sessao unificada)")
        print("=" * 56)
        self._print_result(result)
        return result
