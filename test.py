from flask import Flask
from modules.petsnal_colors import *

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'hello'

@app.route('/url_image')
def url_image_test():
    image = process_image_from_url("https://puddybuddybucket.s3.amazonaws.com/images/ac6d4434-ac3a-435b-b8aa-31a5989da144_%E1%84%80%E1%85%B3%E1%84%85%E1%85%B5%E1%86%B71.png-1.png")
    return "good"

@app.route('/virtual_fit')
def virtual_fit_test():
    image = Image.open("assets/images/1.jpeg")
    insert = Image.open("assets/clothes/76.png")
    res = virtual_fit_process_with_img(image, insert)
    return res

@app.route('/s3_upload')
def s3_upload_test():
    image = Image.open("assets/images/1.jpeg")
    res = upload_to_s3(image)
    print(res)
    return "good"

@app.route('/petsnal')
def petsnal_test():
    image = Image.open("assets/images/1.jpeg")
    preferId = 2
    res = petsnal_color(image, preferId)
    return "good"


if __name__ == '__main__':
    
    app.run(debug=True)