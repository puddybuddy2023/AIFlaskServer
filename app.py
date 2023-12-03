from flask import Flask, request

app = Flask(__name__)

@app.route('/',methods=['GET'])
def test():
    return "hello flask"

@app.route('/petsnal_color')
def process_image():
    image_url = request.args.get('image_url')  # 이미지 URL을 파라미터로 받기
    prefer_id = request.args.get('prefer_id')

    return "Image processing completed!"
    

if __name__ == '__main__':
    #  FLASK_APP=app.py flask run 으로 실행
    app.run(port=5000,host="0.0.0.0")