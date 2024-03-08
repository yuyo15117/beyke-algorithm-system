
import urllib
import pycocotools.mask as mask_util
import requests
import urllib.request
import cv2
import numpy as np
import time
from PIL import Image
from nn_models.deeplab import DeeplabV3
from yolov9_main import detect_yolov9
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


def calculate_intersection_over_union(mask, leftdx,leftdy,rightdx,rightdy,rect_width,rect_height):
    mask_y = mask[0]  # 不规则图形的横坐标集合
    mask_x = mask[1]  # 不规则图形的纵坐标集合

    # rect_width = rightdx - leftdx
    # rect_height = rightdy - leftdy

    intersection_count = 0
    # union_count = len(mask_x)

    for x, y in zip(mask_x, mask_y):
        if leftdx <= x <= rightdx and leftdy <= y <= rightdy:
            intersection_count += 1
    # cv2.imwrite('source_imagecopy02.jpg',source_imagecopy)
    # union_count += rect_width * rect_height - intersection_count
    union_count = rect_width * rect_height
    # print(f"intersection_count{intersection_count},{union_count}")
    iou = (intersection_count / union_count) * 100

    return iou
class ToothProcessor_seg:
    def __init__(self, token_request_url, img_url, service_url, access_token, image_coordinate, image_class):
        self.token_request_url = token_request_url
        self.img_url = img_url
        self.service_url = service_url
        self.access_token = access_token
        self.image_coordinate = image_coordinate
        self.image_class = image_class
        # self.image_position = image_position

    def get_image(self):
        try:
            service_url = self.service_url + self.access_token
            body = {
                "url": f'{self.img_url}'
            }
            resp = requests.post(url=service_url, json=body)

            result = resp.json()
            results = result['results']
            if not results:
                return False, 2002
        except Exception as e:
            print(f"Error: {e}")
            return False, 2001

        source_image = read_image_from_url(self.img_url)
        source_image_copy = source_image.copy()

        source_image = source_image.astype(np.uint8)
        height, width = source_image.shape[:2]
        idx_all = []
        if self.image_class == 2:
            for item in results:
                if item['name'] == 'gingiva':
                    rle_obj = {"counts": item['mask'],
                               "size": [height, width]}
                    mask = mask_util.decode(rle_obj)
                    idx = np.nonzero(mask)
                    # 将元素一一对应地合并并添加到 idx_all 中
                    if not idx_all:
                        idx_all = list(idx)
                    else:
                        idx_all = [np.concatenate((idx_all[i], idx[i])) for i in range(len(idx))]

                    # 将 idx_all 转换为元组
                    idx_all = tuple(idx_all)
                    source_image_copy[idx_all[0], idx_all[1], :] = 127

        elif self.image_class == 1:
            for item in results:
                rle_obj = {"counts": item['mask'],
                           "size": [height, width]}
                mask = mask_util.decode(rle_obj)
                idx = np.nonzero(mask)
                # 将元素一一对应地合并并添加到 idx_all 中
                if not idx_all:
                    idx_all = list(idx)
                else:
                    idx_all = [np.concatenate((idx_all[i], idx[i])) for i in range(len(idx))]

                # 将 idx_all 转换为元组
                idx_all = tuple(idx_all)
                source_image_copy[idx_all[0], idx_all[1], :] = 127

        IOUs = []

        for gingiva_list in self.image_coordinate:
            """
            # 0: height 1: left 2: top 3: width
            """
            IOU_dic =[]
            #     {
            #     # "topX": None,
            #     # "topY": None,
            #     # "bottomX": None,
            #     # "bottomY": None,
            #     "bool": None
            # }
            leftdx = gingiva_list["topX"]
            leftdy = gingiva_list["topY"]
            rightdx = gingiva_list["bottomX"]
            rightdy = gingiva_list["bottomY"]
            # test = cv2.rectangle(source_image_copy, (leftdx, leftdy), (rightdx, rightdy), (0, 255, 0), 2)
            rect_width = rightdx-leftdx
            rect_height = rightdy-leftdy
            IOU = calculate_intersection_over_union(idx_all, leftdx,leftdy,rightdx,rightdy,rect_width,rect_height)
            # print(IOU)
            if IOU > 5:
                # IOU_dic["topX"] = leftdx
                # IOU_dic["topY"] = leftdy
                # IOU_dic["bottomX"] = rightdx
                # IOU_dic["bottomY"] = rightdy
                # IOU_dic["bool"] = True
                IOU_dic = True
                IOUs.append(IOU_dic)
            else:
                # IOU_dic["topX"] = leftdx
                # IOU_dic["topY"] = leftdy
                # IOU_dic["bottomX"] = rightdx
                # IOU_dic["bottomY"] = rightdy
                # IOU_dic["bool"] = False
                IOU_dic = False
                IOUs.append(IOU_dic)
        return IOUs,2003

        # result = all(num > 5 for num in IOUs)

        # if result:
        #     # True代表这个坐标框符合条件可以通过
        #     #2003表示正常
        #     return True,2003
        # else:
        #     # False代表这个坐标框不符合条件不可以通过
        #     return False,2003

class ToothProcessor_seg_left_right:
    def __init__(self, token_request_url, img_url, service_url, access_token, image_coordinate, image_class, image_position):
        self.token_request_url = token_request_url
        self.img_url = img_url
        self.service_url = service_url
        self.access_token = access_token
        self.image_coordinate = image_coordinate
        self.image_class = image_class
        self.image_position = image_position

    def get_image(self):
        try:
            service_url = self.service_url + self.access_token
            body = {
                "url": f'{self.img_url}'
            }
            resp = requests.post(url=service_url, json=body)

            result = resp.json()
            results = result['results']
        except Exception as e:
            print(f"Error: {e}")
            return False, 2001

        source_image = read_image_from_url(self.img_url)
        source_image_copy = source_image.copy()

        source_image = source_image.astype(np.uint8)
        height, width = source_image.shape[:2]
        idx_all = []

        for item in results:
            if item['name'] == 'tooth':
                rle_obj = {"counts": item['mask'],
                           "size": [height, width]}
                mask = mask_util.decode(rle_obj)
                idx = np.nonzero(mask)
                # 将元素一一对应地合并并添加到 idx_all 中
                if not idx_all:
                    idx_all = list(idx)
                else:
                    idx_all = [np.concatenate((idx_all[i], idx[i])) for i in range(len(idx))]

                # 将 idx_all 转换为元组
                idx_all = tuple(idx_all)
                source_image_copy[idx_all[0], idx_all[1], :] = 127

        IOUs = []

        for gingiva_list in self.image_coordinate:
            """
            # 0: height 1: left 2: top 3: width
            """
            IOU_dic =[]
            #     {
            #     # "topX": None,
            #     # "topY": None,
            #     # "bottomX": None,
            #     # "bottomY": None,
            #     "bool": None
            # }
            leftdx = gingiva_list["topX"]
            leftdy = gingiva_list["topY"]
            rightdx = gingiva_list["bottomX"]
            rightdy = gingiva_list["bottomY"]
            # test = cv2.rectangle(source_image_copy, (leftdx, leftdy), (rightdx, rightdy), (0, 255, 0), 2)
            rect_width = rightdx-leftdx
            rect_height = rightdy-leftdy
            IOU = calculate_intersection_over_union(idx_all, leftdx,leftdy,rightdx,rightdy,rect_width,rect_height)
            # print(IOU)
            if IOU > 20:
                # IOU_dic["topX"] = leftdx
                # IOU_dic["topY"] = leftdy
                # IOU_dic["bottomX"] = rightdx
                # IOU_dic["bottomY"] = rightdy
                # IOU_dic["bool"] = True
                IOU_dic = True
                IOUs.append(IOU_dic)
            else:
                # IOU_dic["topX"] = leftdx
                # IOU_dic["topY"] = leftdy
                # IOU_dic["bottomX"] = rightdx
                # IOU_dic["bottomY"] = rightdy
                # IOU_dic["bool"] = False
                IOU_dic = False
                IOUs.append(IOU_dic)
        return IOUs,2003

        # result = all(num > 5 for num in IOUs)

        # if result:
        #     # True代表这个坐标框符合条件可以通过
        #     #2003表示正常
        #     return True,2003
        # else:
        #     # False代表这个坐标框不符合条件不可以通过
        #     return False,2003

#牙垢牙结石——楔缺筛选
class ToothProcessor_tartar_defect_seg:
    def __init__(self, token_request_url, img_url, service_url, access_token):
        self.token_request_url = token_request_url
        self.img_url = img_url
        self.service_url = service_url
        self.access_token = access_token
        # self.image_coordinate = image_coordinate
        # self.image_class = image_class
        # self.image_position = image_position

    def get_image(self):
        try:
            service_url = self.service_url + self.access_token
            body = {
                "url": f'{self.img_url}',
                "threshold": 0.8
            }
            resp = requests.post(url=service_url, json=body)

            result = resp.json()
            results = result['results']
        except Exception as e:
            print(f"Error: {e}")
            return False, 2001

        source_image = read_image_from_url(self.img_url)
        # 将OpenCV图像对象转换为RGB格式
        image_rgb = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)

        # 将图像数据转换为PIL Image对象
        pil_image = Image.fromarray(image_rgb)
        #加载DeeplabV3
        deeplab = DeeplabV3()
        #定义类别
        name_classes = ["background", "tartar"]
        pr = deeplab.detect_image(image=pil_image, count=False, name_classes=name_classes)
        pr[pr == 1] = 255
        pr = pr.astype(np.uint8)

        # 进行连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pr, connectivity=8)

        for item in results:
            # print(f'item{item}')
            leftdx = item['location']['left']
            leftdy = item['location']['top']

            rightdx = item['location']['left'] + item['location']['width']
            rightdy = item['location']['top'] + item['location']['height']
            for label in range(1, num_labels):
                x, y, w, h, area = stats[label]
                intersection_x1 = max(x, leftdx)
                intersection_y1 = max(y, leftdy)
                intersection_x2 = min(x + w, rightdx)
                intersection_y2 = min(y + h, rightdy)
                intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)

                if intersection_area > 0:
                    pr[labels == label] = 0

        # 进行连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pr, connectivity=8)
        resul_list = []


        # 返回色块的边界框
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < 200:
                pr[labels == i] = 0
            else:
                # cv2.rectangle(source_image_copy02, (x, y), (x + w, y + h), (0, 255, 0), 2)

                location_dic = {
                    'height': f'{h}',
                    'left': f'{x}',
                    'top': f'{y}',
                    'width': f'{w}'
                }
                response_dic = {
                    'location': location_dic,
                    'name': 'tartar'
                }

            resul_list.append(response_dic)
        return resul_list,2000

#龋齿识别
class ToothProcessor_decay_detection:
    def __init__(self, img_url):
        # self.token_request_url = token_request_url
        self.img_url = img_url
        # self.service_url = service_url
        # self.access_token = access_token
        # self.image_coordinate = image_coordinate
        # self.image_class = image_class
        # self.image_position = image_position

    def get_image(self):
        source_image = read_image_from_url(self.img_url)
        # # 将OpenCV图像对象转换为RGB格式
        # image_rgb = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)

        # # 将图像数据转换为PIL Image对象
        # pil_image = Image.fromarray(image_rgb)

        #加载 yolo-v9
        prediction = detect_yolov9.YOLOV9_detect(source_image)

        results = prediction.detect_image()
        return results,2000









        # # 进行连通域分析
        # # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pr, connectivity=8)
        # resul_list = []
        #
        #
        # # 返回色块的边界框
        # for i in range(1, num_labels):
        #     x, y, w, h, area = stats[i]
        #     if area < 200:
        #         pr[labels == i] = 0
        #     else:
        #         # cv2.rectangle(source_image_copy02, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #
        #         location_dic = {
        #             'height': f'{h}',
        #             'left': f'{x}',
        #             'top': f'{y}',
        #             'width': f'{w}'
        #         }
        #         response_dic = {
        #             'location': location_dic,
        #             'name': 'tartar'
        #         }
        #
        #     resul_list.append(response_dic)
        # return resul_list,2000
