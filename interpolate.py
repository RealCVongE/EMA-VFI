import os
import subprocess
from pathlib import Path

input_dir = "/home/bigdeal/mnt2/238-1.실내(편의점,_매장)_사람_구매행동_데이터/PreProcessToOneMinute"
output_dir = "/home/bigdeal/mnt2/238-1.실내(편의점,_매장)_사람_구매행동_데이터/PreProcessToOneMinute9FPS"
model_name = 'ours_t'
n = 3

# 입력 디렉토리에서 모든 mp4 파일 목록을 가져옵니다.
video_files = list(Path(input_dir).glob('**/*.mp4'))

# 비디오 파일들을 반복 처리합니다.
for video_file in video_files:
    input_path = str(video_file)
    output_path = Path(output_dir) / video_file.name  # 같은 파일 이름을 사용합니다.
    output_path=str(output_path)
    if(os.path.exists(output_path)):
        print("skipping")
        continue
    # Python 스크립트를 실행하는 명령어를 구성합니다.
    command = [
        'python', 'demo_3x_gpt.py',
        '--model', model_name,
        '--n', str(n),
        '--video', input_path,
        '--output', output_path
    ]
    
    # subprocess.run을 사용하여 명령어를 실행합니다.
    with subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=1, text=True) as proc:
        for line in proc.stdout:
            print(line, end='')
    proc.wait()