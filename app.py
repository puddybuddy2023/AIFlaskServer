from flask import Flask, request

app = Flask(__name__)

@app.route('/',methods=['GET'])
def test():
    return "hello flask"

if __name__ == '__main__':
    #  FLASK_APP=app.py flask run 으로 실행
    app.run(port=5000,host="0.0.0.0")