from flask import Flask, request, jsonify
from modules.petsnal_colors import *

app = Flask(__name__)

@app.route('/',methods=['GET'])
def test():
    return "hello flask"

@app.route('/petsnal_color')
def process_image():
    try: 
        image_url = request.args.get('image_url')  # 이미지 URL을 파라미터로 받기
        prefer_id = request.args.get('prefer_id')
        print(image_url)
        print(prefer_id)
        image = process_image_from_url(image_url)
        result = petsnal_color(image, prefer_id)
        print(result)
        data = {
            'isSuccess' : result
        }
    except:
        data = {
            'isSuccess' : False
        }


    return jsonify(data)

@app.route('/fitting', methods=['GET'])
def process_fitting():
    try:
        image_url = request.args.get('image_url')  # 이미지 URL을 파라미터로 받기
        clothes_id = request.args.get('clothes_id')
        image = process_image_from_url(image_url)
        folder_path = 'assets/clothes'  # 경로를 'assets/clothes'로 수정
        clothes = str(clothes_id) + ".png"
        file_path = os.path.join(folder_path, clothes)
        fitting_image = Image.open(file_path)

        result = fitting_img(image, fitting_image)
        data = {
            'img_url': result
        }
    except:
        data = {
            'img_url': ""
        }

    return jsonify(data)

    

if __name__ == '__main__':
    #  FLASK_APP=app.py flask run 으로 실행
    app.run(port=5000,host="0.0.0.0")