import cv2
import sys
import torch
import numpy as np
import argparse
sys.path.append('.')
import config as cfg
from Trainer import Model
from benchmark.utils.padder import InputPadder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='ours_t', type=str)
    parser.add_argument('--n', default=3, type=int, help='Interpolation factor')
    parser.add_argument('--video', default='path_to_your_video.mp4', type=str, help='Input video path')
    parser.add_argument('--output', default='output_video.mp4', type=str, help='Output video path')
    args = parser.parse_args()

    model = Model(-1)
    model.load_model()
    model.eval()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps * args.n, (width, height))

    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to read video")
        return

    while ret:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert frames to tensors
        I0 = torch.tensor(prev_frame.transpose(2, 0, 1)).float().cuda() / 255.
        I1 = torch.tensor(frame.transpose(2, 0, 1)).float().cuda() / 255.
        I0, I1 = I0.unsqueeze(0), I1.unsqueeze(0)
        
        padder = InputPadder(I0.shape, divisor=32)
        I0, I1 = padder.pad(I0, I1)
        
        preds = model.multi_inference(I0, I1, TTA=True, time_list=[(i + 1) * (1. / args.n) for i in range(args.n - 1)])
        
        # Remove batch dimension and transpose
        frame_to_write = padder.unpad(I0).squeeze(0).cpu().numpy() * 255
        frame_to_write = frame_to_write.transpose(1, 2, 0).astype(np.uint8)
        out.write(cv2.cvtColor(frame_to_write, cv2.COLOR_RGB2BGR))
        
        for pred in preds:
            pred_frame = padder.unpad(pred).squeeze(0).cpu().numpy() * 255
            pred_frame = pred_frame.transpose(1, 2, 0).astype(np.uint8)
            out.write(cv2.cvtColor(pred_frame, cv2.COLOR_RGB2BGR))
        
        prev_frame = frame

    cap.release()
    out.release()

if __name__ == "__main__":
    main()
