
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
from urllib.parse import urlparse
import posixpath
import io
from requests_toolbelt.multipart.encoder import MultipartEncoder


def get_file_extension(url):
    """URL에서 파일 확장자를 추출하는 함수"""
    parsed = urlparse(url)
    root, ext = posixpath.splitext(parsed.path)
    print(ext)
    return ext

def process_image_from_url(image_url):
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))

            # 이미지가 RGBA 모드인 경우 RGB로 변환
            if image.mode == 'RGBA':
                image = image.convert('RGB')

            # URL에서 파일 확장자 추출
            file_extension = get_file_extension(image_url)

            # 이미지를 저장할 디렉토리 경로 설정
            directory = 'assets/temp'
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            # 이미지를 특정 경로에 저장 (파일명은 URL에서 추출한 확장자로 설정)
            image.save(os.path.join(directory, f'image{file_extension}'))
            print("Image saved successfully!")
            return image
        else:
            return None
    except Exception as e:
        print("Error occurred:", str(e))
        return None


def virtual_fit_process_with_img(image, insert_image):
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

    image = np.array(image)
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

    # 이미지 회전 및 변환
    insert_image = insert_image.rotate(angle, resample=Image.BICUBIC)  # 회전
    insert_image = insert_image.resize((int(_width), int(_height)), resample=Image.BICUBIC)  # 크기 조절

    # Pillow(PIL) 이미지로 변환
    pillow_image = Image.fromarray(_image)


    # 이미지 합성
    pillow_image.paste(insert_image, point, mask=insert_image)


    # 다시 numpy 배열로 변환
    _image = np.array(pillow_image)

    code, json_data = upload_to_s3(Image.fromarray(_image))
    return json_data['uploadImg']


    # 합성된 이미지 표시 혹은 저장
    # Image.fromarray(_image).show()  # 이미지 표시

def petsnal_color(image, preferId):
    # 폴더 내의 모든 파일에 대해 반복하여 작업 수행
    folder_path = './assets/petsnals'  # 작업할 폴더 경로
    img_urls = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.png'):
            file_path = os.path.join(folder_path, filename)  # 파일 경로 생성

            # 이미지 파일 열기
            insert_image = Image.open(file_path)

            # 가상 fit 처리 함수 실행
            img_urls.append(virtual_fit_process_with_img(image, insert_image))

    save_petsnal_test(preferId, img_urls)
    return True


def save_petsnal_test(preferId, img_urls):
    for url in img_urls:
        pass
    # post 요청
    return True



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
    image_bytes = io.BytesIO()
    pil_image.save(image_bytes, format='JPEG')  # 필요한 포맷으로 저장 (JPEG, PNG 등)

    # BytesIO의 포인터를 처음으로 되돌림
    image_bytes.seek(0)

    # 업로드할 이미지 데이터 설정
    field= {'file': ('image.jpg', image_bytes, 'image/jpeg')}

    return post('http://ec2-13-124-164-167.ap-northeast-2.compute.amazonaws.com/uploadNewImg', field)
   

def post(url, field_data) :
    m = MultipartEncoder(fields=field_data)
    headers = {'Content-Type' : m.content_type}
    res = requests.post(url, headers=headers, data=m)
    return res.status_code, res.json()