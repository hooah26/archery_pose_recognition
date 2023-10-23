import os
import cv2
import torch
from transformers import ViTForImageClassification
from transformers import ViTFeatureExtractor

model_name_or_path = "./vit-base-beans"
video_dir = "../raw_videos"

labels =  ["rest","shoot"]
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
).to(device)

class_info = {0:"rest",1:"shoot"}

for video_file in os.listdir(video_dir):

    if not ".mp4" in video_file:
        continue

    # 웹캠, 영상 파일의 경우 이것을 사용하세요.:
    cap = cv2.VideoCapture(os.path.join(video_dir,video_file))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('{}_prediction_whole.mp4'.format(video_file.split('.')[0]), fourcc, 30.0,
                          (int(width), int(height)))
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("카메라를 찾을 수 없습니다.")
            # 동영상을 불러올 경우는 'continue' 대신 'break'를 사용합니다.
            break

        batch =  feature_extractor([image])
        cv2.putText(image, class_info[model(torch.tensor(batch["pixel_values"][0]).unsqueeze(0).cuda()).logits.argmax().detach().cpu().numpy().max()], (200, 200), cv2.FONT_HERSHEY_PLAIN, 3, (150, 222, 209), 3)
        out.write(image)
    out.release()

