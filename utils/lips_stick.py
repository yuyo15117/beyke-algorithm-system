import json
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.fmu.v20191213 import fmu_client, models
import re
from flask import Flask, request, jsonify,Blueprint

import cv2
import urllib.request
import numpy as np
import base64
def detect_face(image_url):

    # 解码Base64字符串为二进制数据
    decoded_data = base64.b64decode(image_url)

    # 将二进制数据转换为NumPy数组
    np_data = np.frombuffer(decoded_data, np.uint8)

    # 使用OpenCV加载图像
    image = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

    # 加载人脸检测器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 将图像转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 人脸检测
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 如果检测到人脸
    if len(faces) > 0:
        # 获取第一个人脸的坐标和尺寸
        (x, y, w, h) = faces[0]

        # # 在图像上绘制人脸框
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #
        # # 显示带有人脸框的图像
        # cv2.imshow('Face Detection', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # 返回人脸框的位置和尺寸
        return (x, y, w, h)
    else:
        print("未检测到人脸。")
        return None

class LipsColor():
    def __init__(self, image_url, SecretId, SecretKey, endpoint_):
        self.image_url = image_url
        self.SecretId = SecretId
        self.SecretKey = SecretKey
        self.endpoint_ = endpoint_

    def convet_lips_color(self):
        # 调用人脸检测函数
        face_coordinates = detect_face(self.image_url)
        # flag = 0
        if face_coordinates == None:
            flag = 0
            extracted_url = 0
            return extracted_url,flag
        else:
            flag = 1
            # response = {
            #     'code': '2000'
            # }
            # return jsonify(response)

            # print("人脸框的左上角横坐标:", face_coordinates[0])
            # print("人脸框的左上角纵坐标:", face_coordinates[1])
            # print("人脸框的宽度:", face_coordinates[2])
            # print("人脸框的高度:", face_coordinates[3])

            # 将face_coordinates中的值转换为int类型
            x = int(face_coordinates[0].item())
            y = int(face_coordinates[1].item())
            width = int(face_coordinates[2].item())
            height = int(face_coordinates[3].item())
            extracted_url_list = []
            try:
                # 实例化一个认证对象，入参需要传入腾讯云账户 SecretId 和 SecretKey，此处还需注意密钥对的保密
                # 代码泄露可能会导致 SecretId 和 SecretKey 泄露，并威胁账号下所有资源的安全性。以下代码示例仅供参考，建议采用更安全的方式来使用密钥，请参见：https://cloud.tencent.com/document/product/1278/85305
                # 密钥可前往官网控制台 https://console.cloud.tencent.com/cam/capi 进行获取
                # SecretId:AKIDeDakvyrccVNXW1Sp250rY3VsGUu9nGWa
                # SecretKey:zpTpw3SflnfLp8FAsW0JQwJTQCsPn1zo
                cred = credential.Credential("AKIDeDakvyrccVNXW1Sp250rY3VsGUu9nGWa", "zpTpw3SflnfLp8FAsW0JQwJTQCsPn1zo")
                # 实例化一个http选项，可选的，没有特殊需求可以跳过
                httpProfile = HttpProfile()
                httpProfile.endpoint = "fmu.tencentcloudapi.com"

                # 实例化一个client选项，可选的，没有特殊需求可以跳过
                clientProfile = ClientProfile()
                clientProfile.httpProfile = httpProfile
                # 实例化要请求产品的client对象,clientProfile是可选的
                client = fmu_client.FmuClient(cred, "ap-shanghai", clientProfile)
                # 实例化一个请求对象,每个接口都会对应一个request对象
                req = models.TryLipstickPicRequest()
                params = {
                    "Image": f"{self.image_url}",
                    "LipColorInfos": [
                        {
                            "RGBA": {
                                "R": 168,
                                "G": 38,
                                "B": 40,
                                "A": 100
                            },
                            "FaceRect": {
                                "X": x,
                                "Y": y,
                                "Width": width,
                                "Height": height
                            },
                            "ModelAlpha": 100
                        }
                    ],
                    "RspImgType": "url"
                }
                req.from_json_string(json.dumps(params))#迪奥999 168,38,40,100   #147,71,65 兰蔻274  180 73 91 雅诗兰黛420

                # 返回的resp是一个TryLipstickPicResponse的实例，与请求对象对应
                resp = client.TryLipstickPic(req)
                # print(f' resp{resp}')

                # 输出json格式的字符串回包
                # print(f'resp.to_json_string(){resp.to_json_string()}')
                resp_str = resp.to_json_string()

                # 使用正则表达式提取URL
                url_pattern = r'"ResultUrl": "(.*?)"'
                match = re.search(url_pattern, resp_str)
                if match:
                    extracted_url = match.group(1)
                    # print(f'extracted_url{extracted_url}')

                    extracted_url_list.append(extracted_url)
                    # print(f'extracted_url{extracted_url}')

            except TencentCloudSDKException as err:
                print(f'extracted_url_list{extracted_url_list}')
            return extracted_url,flag