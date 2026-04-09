"""
main.py
=======
Processa DUAS câmeras simultaneamente. A Câmera 1 registra Entradas principais e
Saídas principais. A Câmera 2 registra APENAS SAÍDAS (ex: de uma rótula), e os 
veículos que saem por ela liberam vagas no painel geral da Câmera 1.

Mostra ambos os vídeos lado a lado em tempo real.

Uso:
    python3 main.py
"""

import cv2
import numpy as np
from vehicle_counter import VehicleCounter

def main():
    print("\n" + "━" * 56)
    print("  CONFIGURAÇÃO DE CÂMERAS SIMULTÂNEAS")
    print("━" * 56)
    resposta = input("Deseja salvar o vídeo final (Lado a Lado)? (S/N): ").strip().upper()
    salvar_video = resposta == 'S'

    # 1. Instanciamos a CÂMERA PRINCIPAL (Entrada/Saída geral)
    cam_principal = VehicleCounter(
        video_path          = "entrada.mp4",
        line_y_start        = 0.55,      # Calibração da linha (esquerda)
        line_y_end          = 0.35,      # Calibração da linha (direita)
        model_name          = "yolov10n.pt", # Modelo Nano (super rápido / fluido)
        confidence          = 0.30,      
        imgsz               = 640,       # Resolução padrão para máxima velocidade
        total_vagas         = 90,
    )
    
    # 2. Instanciamos a CÂMERA SECUNDÁRIA (Apenas registro de Saída extra)
    cam_secundaria = VehicleCounter(
        video_path          = "saida.mp4",
        line_y_start        = 0.55,      
        line_y_end          = 0.35,      
        model_name          = "yolov10n.pt", # Modelo Nano para CPU/GPU livre
        confidence          = 0.30,      
        imgsz               = 640,       
        total_vagas         = 90,
    )
    cam_secundaria.is_secondary = True

    cap1 = cv2.VideoCapture(str(cam_principal.video_path))
    cap2 = cv2.VideoCapture(str(cam_secundaria.video_path))
    
    if not cap1.isOpened() or not cap2.isOpened():
        raise IOError("Não foi possível carregar os dois vídeos. Verifique se entrada.mp4 e saida.mp4 existem.")

    # Pega fps base do primeiro para regular a velocidade de exibição
    fps = cap1.get(cv2.CAP_PROP_FPS) or 30.0

    # Configuração interna (passa dimensões reais do vídeo para instanciar as zonas de cruzamento)
    w1, h1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w2, h2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_principal._setup(w1, h1)
    cam_secundaria._setup(w2, h2)

    writer = None
    if salvar_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # Largura dobrada para armazenar um vídeo ao lado do outro
        writer = cv2.VideoWriter("monitoramento_duplo_resultado.mp4", fourcc, fps, (w1 + w2, max(h1, h2)))

    print("\n" + "━" * 56)
    print("  INICIANDO MONITORAMENTO MULTI-CÂMERA")
    print("━" * 56)
    print("[INFO] Pressione 'q' na janela para encerrar.")

    frame_idx = 0
    try:
        while True:
            # Ler simultaneamente
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or not ret2:
                # Se um dos vídeos acabar, encerra (ou ajuste se quiser usar loop)
                print("[INFO] Fim da transmissão de um dos vídeos.")
                break

            # ── Lógica de Sincronia entre Câmeras ───────────────────────
            # A base de vagas atualizadas da Cam1 pega o total de saídas detectadas na Cam2
            cam_principal.extra_out = cam_secundaria.count_out

            # ── Processamento (YOLO + ByteTrack) ───────────────────────
            ann1 = cam_principal._process_frame(frame1)
            ann2 = cam_secundaria._process_frame(frame2)

            # Acoplar as duas imagens lado a lado (horizontal)
            # Para hconcat as alturas devem ser iguais. Força o tamanho da cam2 virar cam1:
            if ann1.shape[0] != ann2.shape[0]:
                ann2 = cv2.resize(ann2, (ann1.shape[1], ann1.shape[0]))
                
            combined = cv2.hconcat([ann1, ann2])

            if writer is not None:
                writer.write(combined)

            # Reduzir escala para desenhar a imagem mais rápido
            display_scale = 0.5
            comb_w, comb_h = int(combined.shape[1] * display_scale), int(combined.shape[0] * display_scale)
            preview = cv2.resize(combined, (comb_w, comb_h))
            
            # Remove o delay artificial, roda o mais rápido que a placa aguentar
            cv2.imshow("Monitoramento Duplo - Estacionamento", preview)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Interrompido pelo usuário.")
                break

            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"[INFO] Frame {frame_idx:04d} | "
                      f"CAM PRINCIPAL: Entraram={cam_principal.count_in} Saíram={cam_principal.count_out} "
                      f"| ROTULA: Saíram={cam_secundaria.count_out}")
    finally:
        cap1.release()
        cap2.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    # ── Resumo consolidado ──────────────────────────────────────────
    no_estac = max(0, cam_principal.count_in - cam_principal.count_out - cam_secundaria.count_out)
    total_saidas = cam_principal.count_out + cam_secundaria.count_out
    
    print("\n" + "═" * 56)
    print("  RESUMO CONSOLIDADO FINAL")
    print("═" * 56)
    print(f"  Total ENTRADAS              : {cam_principal.count_in}")
    print(f"  Total SAÍDAS (Cam 1 + Rótula) : {total_saidas}")
    print(f"  Veículos no estacionamento  : {no_estac}")
    print(f"  Vagas livres                : {max(0, cam_principal.total_vagas - no_estac)}/{cam_principal.total_vagas}")
    print("═" * 56)

if __name__ == "__main__":
    main()
