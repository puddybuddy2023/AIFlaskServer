
import os
import cv2
import tensorflow.compat.v1 as tf
import sys
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from utils import label_map_util
from utils import visualization_utils as vis_util
import math
from PIL import Image
import numpy as np
from PIL import Image, ImageFilter
import requests
import json
import base64

def process_image_from_url(image_url):
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            return image
        else:
            return None
    except Exception as e:
        print("Error occurred:", str(e))
        return None


def petsnal_process_with_img(image):
    # This is needed since the notebook is stored in the object_detection folder.
    sys.path.append("...")

    # Name of the directory containing the object detection module we're using
    MODEL_NAME = 'inference_graph'
    
    # Grab path to current working directory
    CWD_PATH = os.getcwd()

    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = os.path.join(CWD_PATH,'frozen_inference_graph.pb')

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH,'labelmap.pbtxt')

        
    # Number of classes the object detector can identify
    NUM_CLASSES = 4

    # Load the label map.
    # Label maps map indices to category names, so that when our convolution
    # network predicts `5`, we know that this corresponds to `king`.
    # Here we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    # Define input and output tensors (i.e. data) for the object detection classifier

        
    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 이미지의 높이와 너비 가져오기
    height, width = image.shape[:2]

    # 자를 크기 결정 (가로와 세로 중 더 작은 값)
    crop_size = min(height, width)

    # 가운데 좌표 계산
    start_x = (width - crop_size) // 2
    start_y = (height - crop_size) // 2

    # 이미지의 중앙을 기준으로 정사각형 모양으로 자르기
    cropped_image = image[start_y:start_y+crop_size, start_x:start_x+crop_size]

    # 이미지 크기 조정
    image = cv2.resize(cropped_image, (400, 400), interpolation=cv2.INTER_AREA)


    imageNP = np.copy(image)


    image_expanded = np.expand_dims(image, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})
    
    
    # Draw the results of the detection (aka 'visulaize the results')

    array, boxes, tags = vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=2,
        min_score_thresh=0.60)
    
    data = {}

    for tag, box in zip(tags, boxes):
        tag_name = tag[0].split(': ')[0]  # 태그에서 이름 추출 (예: 'leg')
        data[tag_name] = box

    angle = calculate_angle(data['face'], data['leg'])
    midpoint = calculate_midpoint(data['leg'], data['face'])

    _width, _height = calculate_size(data['face'], data['leg'])

    point = (int(midpoint[0]*400-_width/2), int((midpoint[1]*400-_height/2)*1.1))

    # 이미지 크기는 항상 400 x 400

    _image = np.copy(imageNP)

    # 이미지 파일 불러오기 수정!!
    image_path = "NY.png"  # 삽입할 이미지 파일 경로
    insert_image = Image.open(image_path)

    # 이미지 회전 및 변환
    insert_image = insert_image.rotate(angle, resample=Image.BICUBIC)  # 회전
    insert_image = insert_image.resize((int(_width), int(_height)), resample=Image.BICUBIC)  # 크기 조절

    # 흐릿하게 만들 부분의 너비
    blur_width = int(width * 0.2)  # 예시: 전체 너비의 20%를 흐릿하게 만듦

    # 이미지 흐릿하게 처리
    blurred_image = insert_image.copy()
    blurred_image[:, -blur_width:] = cv2.blur(blurred_image[:, -blur_width:], (50, 50))  # 블러 처리


    # Pillow(PIL) 이미지로 변환
    pillow_image = Image.fromarray(_image)


    # 이미지 합성
    pillow_image.paste(insert_image, point, mask=insert_image)


    # 다시 numpy 배열로 변환
    _image = np.array(pillow_image)

    url = upload_to_s3(Image.fromarray(_image))
    return url


    # 합성된 이미지 표시 혹은 저장
    # Image.fromarray(_image).show()  # 이미지 표시



def calculate_angle(box1, box2):
    # 각 박스의 중심 지점 계산
    center_box1 = ((box1[1] + box1[3]) / 2, (box1[0] + box1[2]) / 2)
    center_box2 = ((box2[1] + box2[3]) / 2, (box2[0] + box2[2]) / 2)
    
    # 두 중심을 연결하는 선의 기울기 계산
    slope = (center_box2[1] - center_box1[1]) / (center_box2[0] - center_box1[0])
    
    # y축과의 각도 계산 (라디안 값을 돌려주므로 이를 degree로 변환)
    angle_radians = math.atan(slope)
    angle_degrees = math.degrees(angle_radians)
    
    return 90- angle_degrees 

def calculate_midpoint(box1, box2):
    # 각 박스의 중심 지점 계산
    center_box1 = ((box1[1] + box1[3]) / 2, (box1[0] + box1[2]) / 2)
    center_box2 = ((box2[1] + box2[3]) / 2, (box2[0] + box2[2]) / 2)

    
    # 두 중심을 연결하는 선의 중점 계산
    midpoint = ((center_box1[0] + center_box2[0]) / 2, (center_box1[1] + center_box2[1]) / 2)
    
    return midpoint

def calculate_size(box1, box2):
    height = box2[0] - box1[2]
    width = box1[3] - box1[1]
    return width*400*0.9, height*400*1.2

def upload_to_s3(pil_image):
    # 이미지를 BytesIO로 변환하여 파일 형태로 생성
    image_file = io.BytesIO()
    pil_image.save(image_file, format='JPEG')  # 원하는 포맷으로 저장할 수 있습니다.

    # 파일 객체의 포인터를 처음으로 되돌림
    image_file.seek(0)

    # 멀티파트 요청을 위한 파일 형태로 변환
    files = {'file': ('image.jpg', image_file, 'image/jpeg')}  # 파일 이름 및 MIME 타입 지정

    # Flask 서버의 엔드포인트 URL
    url = 'http://ec2-13-124-164-167.ap-northeast-2.compute.amazonaws.com/uploadNewImg'

    # 멀티파트 요청으로 이미지 파일을 서버에 전송
    response = requests.post(url, files=files)

    print(response.text)  # 서버로부터의 응답 출력