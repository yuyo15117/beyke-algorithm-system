import requests
import base64
import urllib
import pycocotools.mask as mask_util
import requests
import urllib.request
import cv2
import numpy as np
import time
import mediapipe as mp
import os
def read_image_from_url(url):
    try:
        # 下载图片
        response = urllib.request.urlopen(url)
        image_data = response.read()

        # 将图像数据解码为OpenCV可读取的格式
        np_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        return image
    except Exception as e:
        print("Error occurred while reading image from URL:", str(e))
        return None


def url_to_base64(url):
    # 发送HTTP请求获取图片内容
    response = requests.get(url)

    # 将图片内容转换为Base64编码的字符串
    image_content = response.content
    base64_data = base64.b64encode(image_content)
    base64_str = base64_data.decode('utf-8')

    return base64_str


def change_lightness(img, value=30):
    # Convert the image to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(hls)

    # Increase the lightness channel by the given value
    l = cv2.add(l, value)
    l[l > 255] = 255
    l[l < 0] = 0

    # Merge the modified channels back to HLS image
    final_hls = cv2.merge((h, l, s))

    # Convert the image back to BGR color space
    img = cv2.cvtColor(final_hls, cv2.COLOR_HLS2BGR)

    return img


def change_saturation(img, value=30):
    # Convert the image to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(hls)

    # Increase the saturation channel by the given value
    s = cv2.add(s, value)
    s[s > 255] = 255
    s[s < 0] = 0

    # Merge the modified channels back to HLS image
    final_hls = cv2.merge((h, l, s))

    # Convert the image back to BGR color space
    img = cv2.cvtColor(final_hls, cv2.COLOR_HLS2BGR)

    return img


def make_teeth_yellow(image):
    # Convert image to HLS color space
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    # Split HLS image into channels
    h, l, s = cv2.split(hls)

    # Increase yellow color in teeth region (adjust the value as needed)
    h_teeth = h                              #很黄  较黄
    s_teeth = s + 30  # Increase saturation  #30   10
    l_teeth = l - 15  # Decrease brightness  #-15  -10

    # Merge modified channels back into HLS image
    hls_teeth = cv2.merge((h_teeth, l_teeth, s_teeth))

    # Convert HLS image back to BGR color space
    result = cv2.cvtColor(hls_teeth, cv2.COLOR_HLS2BGR)

    return result

def image_to_base64(image):
    # 将图像转换为base64编码的字符串
    retval, buffer = cv2.imencode('.jpg', image)
    image_str = base64.b64encode(buffer).decode('utf-8')
    return image_str


def detect_mouth(img):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,  # False处理视频，True处理单张图片
                                      max_num_faces=1,
                                      refine_landmarks=True,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)  # 静态图片不用设置

    # 读取一帧图像
    height, width, channels = np.shape(img)
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 特征点提取
    results = face_mesh.process(img_RGB)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            # 绘制人脸关键点
            face_landmarks = results.multi_face_landmarks[0]
            list_lms = []
            if face_landmarks:
                for i in range(478):
                    pos_x = int(face_landmarks.landmark[i].x * width)
                    pos_y = int(face_landmarks.landmark[i].y * height)
                    list_lms.append([pos_x, pos_y])
                    # cv2.circle(img, (pos_x, pos_y), 3, (0, 255, 0), -1)

        list_lms = np.array(list_lms, dtype=np.int32)
        index_lip_up = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 311, 312, 13, 82, 80, 191, 78,
                        61]
        index_lip_down = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 291, 375, 321, 405, 314, 17, 84, 181, 91,
                          146, 61, 78]

        # 计算嘴巴的边界框
        x_min = np.min(list_lms[index_lip_up][:, 0])
        x_max = np.max(list_lms[index_lip_up][:, 0])
        y_min = np.min(list_lms[index_lip_up][:, 1])
        y_max = np.max(list_lms[index_lip_down][:, 1])

        # 绘制嘴巴边界框
        # cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        # mouth_left_top_x,mouth_left_top_y,mouth_right_down_x,mouth_right_down_y = detect_mouth(image)

        return x_min, y_min, x_max, y_max


class ToothProcessor:
    def __init__(self, token_request_url, img_path, service_url, access_token):
        self.token_request_url = token_request_url
        self.img_path = img_path
        self.service_url = service_url
        self.access_token = access_token



    def get_image(self):
        st = time.time()
        base64_image = self.img_path

        # response = requests.get(self.token_request_url)
        try:
            service_url = self.service_url + self.access_token

            body = {
                "image": f'{base64_image}'
            }

            resp = requests.post(url=service_url, json=body)

            # params = "{\"url\":\"" + self.img_path + "\"}"

            result = resp.json()
            # print(f'result{result}')
            results = result['results']
        except Exception as e:
            print(f"Error: {e}")
            return 2001

        et = time.time()

        execution_time = et - st
        print(f'牙齿分割接口响应时间:{execution_time:.2f}')
        # 将base64图片字符串解码为图像数据
        image_data = base64.b64decode(base64_image)

        # # 将图像数据转换为NumPy数组
        nparr = np.frombuffer(image_data, np.uint8)
        #
        # # 解码图像数组
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        #进行嘴巴检测，获取嘴巴右上角y值，并返回
        mouth_left_top_x,mouth_left_top_y,mouth_right_down_x,mouth_right_down_y = detect_mouth(image)
        # image = read_image_from_url(self.img_path).astype(np.float32)

        source_image = image.astype(np.uint8)
        height, width = image.shape[:2]
        # alpha = 0.5
        mask_img = np.zeros_like(image)
        non_mask_img = np.zeros_like(image).astype(np.uint8)

        for item in results:
            rightdx = item['location']['left'] + item['location']['width']
            rightdy = item['location']['top'] + item['location']['height']
            # print(f"rightdx{rightdx}")
            # print(f"rightdy{rightdy}")

            # if item['location']['top'] < mouth_left_top_y or item['location']['left'] < mouth_left_top_x or item['location']['left'] + item['location']['width']<mouth_right_down_x or item['location']['top']+item['location']['height']<mouth_right_down_y:
            if item['location']['top'] < mouth_left_top_y:
                continue
            elif item['location']['left'] < mouth_left_top_x:
                continue
            elif rightdx>mouth_right_down_x:
                continue
            elif rightdy>mouth_right_down_y:
                continue

            # print(f"item{item['location']['top']}")
            rle_obj = {"counts": item['mask'],
                       "size": [height, width]}
            mask = mask_util.decode(rle_obj)

            random_color = np.array([np.random.random() * 255.0,
                                     np.random.random() * 255.0,
                                     np.random.random() * 255.0])

            idx = np.nonzero(mask)

            # 保留mask部分，其余部分设置为0像素值
            # mask_img = np.zeros_like(image)
            mask_img[idx[0], idx[1], :] = image[idx[0], idx[1], :]

            # 将除mask部分以外的像素值设置为0
            # non_mask_img = np.zeros_like(image).astype(np.uint8)
            non_mask_img[idx[0], idx[1], :] = 255
            # print(f"non_mask_img{non_mask_img}")

        inverse_mask = cv2.bitwise_not(non_mask_img)
        mask01 = non_mask_img


        # cv2.imshow('inverse_mask',inverse_mask)
        masked_image = cv2.bitwise_and(source_image, mask01)
        inverse_masked_image = cv2.bitwise_and(source_image, inverse_mask)
        # cv2.imshow('masked_image',masked_image)

        processed_masked_image = change_lightness(masked_image, value=22)

        processed_masked_image = change_saturation(processed_masked_image, value=2)
        # processed_masked_image = make_teeth_yellow(masked_image)
        #去白

        processed_masked_image = cv2.bitwise_and(processed_masked_image, mask01)
        output_image = cv2.bitwise_or(processed_masked_image, inverse_masked_image)
        # cv2.imshow('inverse_masked_image',inverse_masked_image)
        # cv2.imshow('processed_masked_image',processed_masked_image)
        #
        # cv2.waitKey(0)
        return 2003,output_image, inverse_mask, mask01
