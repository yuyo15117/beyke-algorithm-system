import cv2
import requests
import numpy as np
from flask import Flask, request, jsonify,Blueprint
import base64
import urllib
app = Flask(__name__)
import urllib.parse
from utils.tooth_whitening import ToothProcessor
from utils.tooth_seg import ToothProcessor_seg,ToothProcessor_seg_left_right,ToothProcessor_tartar_defect_seg,ToothProcessor_decay_detection
from utils.lips_stick import LipsColor
import concurrent.futures
# 创建蓝图
blueprint = Blueprint('my_blueprint', __name__, url_prefix='/api/diagnosis')


def download_image(image_url):
    # 下载图片并返回图像对象
    with urllib.request.urlopen(image_url) as url:
        data = url.read()
    nparr = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image

def convert_to_grayscale(image):
    # 将图像转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def image_to_base64(image):
    # 将图像转换为base64编码的字符串
    retval, buffer = cv2.imencode('.jpg', image)
    image_str = base64.b64encode(buffer).decode('utf-8')
    return image_str

def get_access_token(url):
    response = requests.post(url)
    data = response.json()
    access_token = data['access_token']
    return access_token

# 定义函数1：获取白牙图片以及掩膜
def get_whitetooth_image(image_url):
    token_url = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=SB6ljXvwQhWv8936PdC3g8Gg&client_secret=HeqKaz7AaK3VmTNj7u76ZeRxRIN2HKdX'
    access_token = get_access_token(token_url)
    processor = ToothProcessor(
        token_request_url="https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=SB6ljXvwQhWv8936PdC3g8Gg&client_secret=HeqKaz7AaK3VmTNj7u76ZeRxRIN2HKdX",
        img_path=f"{image_url}",
        service_url="https://aip.baidubce.com/rpc/2.0/ai_custom/v1/segmentation/tooth_seg?url&access_token=",
        access_token=f'{access_token}'
    )
    # whitetooth_image, inverse_mask, mask01 = processor.get_image()
    error_code_3return = processor.get_image()

    # print(f'whitetooth_image, inverse_mask, mask01{whitetooth_image, inverse_mask, mask01}')
    # return whitetooth_image, inverse_mask, mask01
    return error_code_3return

# 定义函数2：获取涂口红后的图片URL
def get_lips_image_url(image_url):
    lipcolor = LipsColor(
        image_url=f"{image_url}",
        SecretId="AKIDeDakvyrccVNXW1Sp250rY3VsGUu9nGWa",
        SecretKey="zpTpw3SflnfLp8FAsW0JQwJTQCsPn1zo",
        endpoint_="fmu.tencentcloudapi.com"
    )
    lips_image_url, flag = lipcolor.convet_lips_color()
    return lips_image_url, flag

#口红白牙算法
def lips_white(source_image_lips,source_image_tooth,inverse_mask,mask):
    #用inverse_mask将source_image_lips原有的黄牙去除
    inverse_masked_image = cv2.bitwise_and(source_image_lips, inverse_mask)

    #用mask将source_image_tooth的白牙保留
    masked_image = cv2.bitwise_and(source_image_tooth, mask)

    #将masked_image与inverse_masked_image合并
    output_image = cv2.bitwise_or(masked_image, inverse_masked_image)
    return output_image




# 在蓝图上定义路由和视图函数
@blueprint.route('/lips_color_and_whitening_tooth', methods=['GET', 'POST'])
def white_tooth_lipstick():
    if request.method == 'POST':
        # 获取前端发送的JSON数据
        data = request.json
        image_url = data['image_url']
        # print(f'image_url{image_url}')
        # 下载图片
        # image = download_image(image_url)

        # 创建线程池
        with concurrent.futures.ThreadPoolExecutor() as executor:

            # # 提交函数get_whitetooth_image任务到线程池
            whitetooth_futures = executor.submit(get_whitetooth_image, image_url)

            # # 提交函数get_lips_image_url任务到线程池
            lips_image_url_futures = executor.submit(get_lips_image_url, image_url)

            # 等待函数2的任务完成
            lips_image_url, flag = lips_image_url_futures.result()
            if lips_image_url == 0:
                response = {

                    'code': '2000',
                    'msg': '未检测到人脸'
                }
            else:
                # 提交函数3的任务到线程池
                image_future = executor.submit(download_image, lips_image_url)

                # 等待函数3的任务完成
                lips_image = image_future.result()

                # 处理涂口红后的逻辑
                # 等待函数1的任务完成
                error_code_3return = whitetooth_futures.result()
                error_code = error_code_3return
                if error_code == 2001:
                    response = {

                        'code': '2001',
                        'msg': '百度接口未正常返回'
                    }
                    return jsonify(response)
                else:
                    whitetooth_image = error_code_3return[1]
                    inverse_mask = error_code_3return[2]
                    mask01 = error_code_3return[3]

                    # whitetooth_image, inverse_mask, mask01 = whitetooth_futures.result()



                    lips_whitetooth_image = lips_white(lips_image, whitetooth_image, inverse_mask, mask01)
                    # 对processed_image进行后续操作
                    # 将原图转换为base64编码的字符串
                    # source_image_str = image_to_base64(image)

                    # 将无口红牙齿变白图像转换为base64编码的字符串
                    nolips_white_image_str = image_to_base64(whitetooth_image)

                    # 将有口红牙齿没变白图像转换为base64编码的字符串
                    lips_nowhitetooth_image_str = image_to_base64(lips_image)

                    # 将有口红牙齿变白图像转换为base64编码的字符串
                    lips_whitetooth_image_str = image_to_base64(lips_whitetooth_image)

                    response = {
                                'nolips_white_image_url': nolips_white_image_str,
                                'lips_nowhitetooth_image_url': lips_nowhitetooth_image_str,
                                'lips_whitetooth_image_url': lips_whitetooth_image_str,
                                'code':'200'
                    }


            return jsonify(response)
    elif request.method == 'GET':
        # 在这里处理GET请求

        return 'Hello World'

def get_source_image(data):
    image_url = data['image_url']
    image_box = data['boxes']
    image_class = data['healthType']
    # image_position = data['maxillofacial']

    #牙龈炎
    if image_class == 2:
        token_url = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=SB6ljXvwQhWv8936PdC3g8Gg&client_secret=HeqKaz7AaK3VmTNj7u76ZeRxRIN2HKdX'
        access_token = get_access_token(token_url)
        processor = ToothProcessor_seg(
            token_request_url="https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=SB6ljXvwQhWv8936PdC3g8Gg&client_secret=HeqKaz7AaK3VmTNj7u76ZeRxRIN2HKdX",
            img_url=f"{image_url}",
            service_url="https://aip.baidubce.com/rpc/2.0/ai_custom/v1/segmentation/tooth_gingiva?input_type=url&access_token=",
            access_token=f'{access_token}',
            image_coordinate=image_box,
            image_class=image_class
        )

        # whitetooth_image, inverse_mask, mask01 = processor.get_image()
        flag,error_code = processor.get_image()

        # return flag, whitetooth_image, inverse_mask, mask01
        return flag,error_code
    #龋齿
    elif image_class == 1:
        token_url = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=SB6ljXvwQhWv8936PdC3g8Gg&client_secret=HeqKaz7AaK3VmTNj7u76ZeRxRIN2HKdX'
        access_token = get_access_token(token_url)
        processor = ToothProcessor_seg(
            token_request_url="https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=SB6ljXvwQhWv8936PdC3g8Gg&client_secret=HeqKaz7AaK3VmTNj7u76ZeRxRIN2HKdX",
            img_url=f"{image_url}",
            service_url="https://aip.baidubce.com/rpc/2.0/ai_custom/v1/segmentation/tooth_seg_roi?input_type=url&access_token=",
            access_token=f'{access_token}',
            image_coordinate=image_box,
            image_class=image_class
        )

        # whitetooth_image, inverse_mask, mask01 = processor.get_image()
        IOUs_bool,error_code = processor.get_image()

        # return flag, whitetooth_image, inverse_mask, mask01
        return IOUs_bool,error_code

    # #左中右龋齿
    # elif image_class == 1 and image_position == False:
    #     token_url = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=SB6ljXvwQhWv8936PdC3g8Gg&client_secret=HeqKaz7AaK3VmTNj7u76ZeRxRIN2HKdX'
    #     access_token = get_access_token(token_url)
    #     processor = ToothProcessor_seg_left_right(
    #         token_request_url="https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=SB6ljXvwQhWv8936PdC3g8Gg&client_secret=HeqKaz7AaK3VmTNj7u76ZeRxRIN2HKdX",
    #         img_url=f"{image_url}",
    #         service_url="https://aip.baidubce.com/rpc/2.0/ai_custom/v1/segmentation/tooth_gingiva?input_type=url&access_token=",
    #         access_token=f'{access_token}',
    #         image_coordinate=image_box,
    #         image_class=image_class,
    #         image_position=image_position
    #     )
    #
    #     # whitetooth_image, inverse_mask, mask01 = processor.get_image()
    #     flag, error_code = processor.get_image()
    #
    #     # return flag, whitetooth_image, inverse_mask, mask01
    #     return flag, error_code

#获取楔缺百度接口返回
def get_tartar_image(data):
    image_url = data['url']
    # image_box = data['boxes']
    # image_class = data['healthType']
    # image_position = data['maxillofacial']

    #https://aip.baidubce.com/rpc/2.0/ai_custom/v1/detection/defect_detect
    #https://aip.baidubce.com/rpc/2.0/ai_custom/v1/detection/dilapidation
    token_url = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=SB6ljXvwQhWv8936PdC3g8Gg&client_secret=HeqKaz7AaK3VmTNj7u76ZeRxRIN2HKdX'
    access_token = get_access_token(token_url)
    processor = ToothProcessor_tartar_defect_seg(
        token_request_url="https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=SB6ljXvwQhWv8936PdC3g8Gg&client_secret=HeqKaz7AaK3VmTNj7u76ZeRxRIN2HKdX",
        img_url=f"{image_url}",
        service_url="https://aip.baidubce.com/rpc/2.0/ai_custom/v1/detection/dilapidation?input_type=url&access_token=",
        access_token=f'{access_token}'
        )

        # image_coordinate=image_box,
        # image_class=image_class,
        # image_position=image_position


    # whitetooth_image, inverse_mask, mask01 = processor.get_image()
    tartar_seg_result,error_code = processor.get_image()
    # return flag, whitetooth_image, inverse_mask, mask01
    return tartar_seg_result,error_code

#获得龋齿图片处理结果
def get_decay_image(data):

    image_url = data['url']

    processor = ToothProcessor_decay_detection(
        img_url=f"{image_url}",
    )

    # image_coordinate=image_box,
    # image_class=image_class,
    # image_position=image_position


    # whitetooth_image, inverse_mask, mask01 = processor.get_image()
    decay_tooth_result, error_code = processor.get_image()
    # return flag, whitetooth_image, inverse_mask, mask01
    return decay_tooth_result, error_code



# 在蓝图上定义路由和视图函数
@blueprint.route('/tooth_gingiva_seg', methods=['GET', 'POST'])
def decay_gingiva_area_limited():
    if request.method == 'POST':
        # 获取前端发送的JSON数据
        data = request.get_json()

        IOUs_bool, error_code = get_source_image(data)
        if error_code == 2001:
            response = {
                'code': '2001',
                'msg': '百度接口未正常返回'
            }
            return jsonify(response)
        elif error_code == 2002:
            response = {
                'code': '2002',
                'msg': '图像未能正常进行分割，请检查图像疾病类别healthType是否正确,如果是龋齿，请传入上下颌角度的图片。如果是牙龈炎，请上传左中右角度的图片。'
            }
            return jsonify(response)

        else:
            response = {
                'bools': IOUs_bool
            }
            return jsonify(response)

    elif request.method == 'GET':
        # 在这里处理GET请求
        return 'Hello World'



#tartar牙垢牙结石分割模型
@blueprint.route('/tartar_seg', methods=['GET', 'POST'])
def tartar_segment():
    if request.method == 'POST':
        # 获取前端发送的JSON数据
        data = request.json
        resluts, error_code = get_tartar_image(data)

        if error_code == 2001:
            response = {
                'code': '2001',
                'msg': '百度接口未正常返回'
            }
            return jsonify(response)

        else:
            response = {
                'resluts': resluts
            }
            return jsonify(response)

    elif request.method == 'GET':
        # 在这里处理GET请求
        return 'Hello World'

#龋齿检测
@blueprint.route('/tooth_decay', methods=['GET', 'POST'])
def decay_tooth():
    if request.method == 'POST':
        # 获取前端发送的JSON数据
        data = request.json
        resluts, error_code = get_decay_image(data)

        if error_code == 2001:
            response = {
                'code': '2001',
                'msg': '百度接口未正常返回'
            }
            return jsonify(response)

        else:
            response = {
                'resluts': resluts
            }
            return jsonify(response)

    elif request.method == 'GET':
        # 在这里处理GET请求
        return 'Hello World'


# 在应用程序上注册蓝图
app.register_blueprint(blueprint)
if __name__ == '__main__':
    # 运行Flask应用
    app.run(host="0.0.0.0",port="8888")
