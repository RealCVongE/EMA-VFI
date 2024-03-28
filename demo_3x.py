import cv2
import sys
import torch
import numpy as np
import argparse
from imageio import mimsave

'''==========import from our code=========='''
sys.path.append('.')
import config as cfg
from Trainer import Model
from benchmark.utils.padder import InputPadder

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ours_t', type=str)
parser.add_argument('--n', default=3, type=int)  # 보간할 배율 설정
parser.add_argument('--video', default='path_to_your_video.mp4', type=str, help='Input video path')
parser.add_argument('--output', default='output_video.mp4', type=str, help='Output video path')
args = parser.parse_args()

assert args.model in ['ours_t', 'ours_small_t'], 'Model not exists!'

'''==========Model setting=========='''
TTA = True
if args.model == 'ours_small_t':
    TTA = False
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours_small_t'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F=16,
        depth=[2, 2, 2, 2, 2]
    )
else:
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours_t'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F=32,
        depth=[2, 2, 2, 4, 4]
    )

model = Model(-1)
model.load_model()  # 체크포인트 로드, 실제 모델 로드 방식에 맞게 수정 필요
model.eval()
# model = model.cuda()  # 모델을 CUDA 디바이스로 옮김

print(f'=========================Start Generating=========================')

# 비디오 로드
cap = cv2.VideoCapture(args.video)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 결과 비디오 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(args.output, fourcc, fps * args.n, (width, height))  # 원래 fps의 n배로 설정

ret, prev_frame = cap.read()
if ret:
    prev_frame_rgb = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)  # BGR에서 RGB로 변환
    # 첫 번째 프레임을 BGR로 다시 변환하여 비디오에 쓰기
    out.write(cv2.cvtColor(prev_frame_rgb, cv2.COLOR_RGB2BGR))
while ret:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR에서 RGB로 변환
    I0 = torch.tensor(prev_frame_rgb.transpose(2, 0, 1)).float().cuda() / 255.
    I1 = torch.tensor(frame_rgb.transpose(2, 0, 1)).float().cuda() / 255.
    I0 = I0.unsqueeze(0)
    I1 = I1.unsqueeze(0)
    
    padder = InputPadder(I0.shape, divisor=32)
    I0, I1 = padder.pad(I0, I1)
    
    preds = model.multi_inference(I0, I1, TTA=TTA, time_list=[(i + 1) * (1. / args.n) for i in range(args.n - 1)], fast_TTA=TTA)
    
    for pred in preds:
        pred_frame = (padder.unpad(pred).detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
        pred_frame_bgr = cv2.cvtColor(pred_frame, cv2.COLOR_RGB2BGR)  # RGB에서 BGR로 변환
        out.write(pred_frame_bgr)  # BGR 프레임 저장
    
    prev_frame_rgb = frame_rgb  # 다음 루프를 위해 현재 프레임 업데이트

if prev_frame is not None:
    # 마지막 프레임을 BGR로 변환하여 저장
    last_frame_bgr = cv2.cvtColor(prev_frame_rgb, cv2.COLOR_RGB2BGR)
    out.write(last_frame_bgr)

# 비디오 처리 완료 후 자원 해제
cap.release()
out.release()

print(f'=========================Done=========================')
