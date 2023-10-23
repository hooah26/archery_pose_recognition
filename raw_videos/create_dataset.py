import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
from PIL import Image
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)

# pose_config = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py'  # hrnet_w48_coco_256x192.py'
# pose_checkpoint = 'demo/models/vitpose-b.pth'  # 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
# det_config = 'mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
# det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
# try:
#     from mmdet.apis import inference_detector, init_detector
#     has_mmdet = True
# except (ImportError, ModuleNotFoundError):
#     has_mmdet = False
# det_model = init_detector(
#     det_config, det_checkpoint, device="cuda:0")

frame_dict = {"version":"0.1",
              "data": []}

video_lists = ["raw_video2.mp4","raw_video3.mp4"]
images = []
labels = []
count = 0
for video_file in video_lists:
    print(video_file)
    # 웹캠, 영상 파일의 경우 이것을 사용하세요.:
    cap = cv2.VideoCapture(video_file)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('output1.mp4', fourcc, 30.0,
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))


    with mp_pose.Pose(
            # static_image_mode=True,
            enable_segmentation=True,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=.7) as pose:
        angle_buffer = []
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("카메라를 찾을 수 없습니다.")
                # 동영상을 불러올 경우는 'continue' 대신 'break'를 사용합니다.
                break
            # images.append(image)

            # 필요에 따라 성능 향상을 위해 이미지 작성을 불가능함으로 기본 설정합니다.
            # image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            coord_x1 , coord_y1 = results.pose_landmarks.landmark[11].x, results.pose_landmarks.landmark[11].y
            coord_x2, coord_y2 = results.pose_landmarks.landmark[13].x, results.pose_landmarks.landmark[13].y

            offset_x, offset_y = coord_x2-coord_x1 , coord_y2-coord_y1

            a= np.rad2deg(np.arccos(np.dot([offset_x, offset_y] / np.linalg.norm([offset_x, offset_y]),[0, 1])))
            angle_buffer.append(a)
            angle_buffer = angle_buffer[-20:]

            # mmdet_results = inference_detector(det_model, image)

            # # keep the person class bounding boxes.
            # person_results = process_mmdet_results(mmdet_results, 1)
            # left,top,right,bottom, _ = person_results[0]["bbox"]
            # image = image[int(top):int(bottom),int(left):int(right),:]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 포즈 주석을 이미지 위에 그립니다.
            image.flags.writeable = True
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # mp_drawing.draw_landmarks(
            #     image,
            #     results.pose_landmarks,
            #     mp_pose.POSE_CONNECTIONS,
            #     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            # 보기 편하게 이미지를 좌우 반전합니다.
            #cv2.putText(image, str(round(a,2)), (100, 700), cv2.FONT_HERSHEY_PLAIN, 3, (150, 222, 209), 3)
            angle_buffer = angle_buffer[-20:]
            if round(np.mean(angle_buffer),2)>60:
                # labels.append("shoot")
                #
                # print(np.fromstring(image.tostring()))
                # print(np.array(image.tolist()))
                cv2.imwrite("./images/video_file_{}.jpg".format(count),image)
                frame_dict["data"].append({"label": "shoot", "image":"./images/video_file_{}.jpg".format(count)})
                # cv2.putText(image, "SHOOT", (200, 700), cv2.FONT_HERSHEY_PLAIN, 3, (150, 222, 209), 3)
            else:
                # labels.append("rest")
                # image.astype('uint8').tolist()
                # print(np.array(image.tolist()))
                cv2.imwrite("./images/video_file_{}.jpg".format(count),image)
                frame_dict["data"].append({"label": "rest", "image":"./images/video_file_{}.jpg".format(count)})
                # cv2.putText(image, "REST", (200, 700), cv2.FONT_HERSHEY_PLAIN, 3, (150, 222, 209), 3)
            count += 1
            print(image.shape)

            # out.write(image)
            # cv2.imwrite("{}.jpg".format(count),image)
            #cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        out.release()

# images = np.array(images)
# images = images.tolist()
# from json import JSONEncoder
# class NumpyArrayEncoder(JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return JSONEncoder.default(self, obj)
# from tqdm import tqdm
# for label, image in tqdm(zip(labels,images),total=len(labels)):
#     frame_dict["data"].append({"images":image.tolist(), "labels":label})

import json
with open("pose_dataset.json","w") as json_file:
    json.dump(frame_dict,json_file)


