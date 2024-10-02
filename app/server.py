from flask import Flask, request, jsonify, abort
from flask_cors import CORS, cross_origin
import numpy as np
import datetime
import logging 
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import os
import sys
import image_stylization
from PIL import Image
import io
from flask_session import Session


# Инициализируем приложение Flask
app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)

# Создадим список стилей
styles = ['anime.jpg',
        'Aquarelle.jpg',
        'coloured_pencil.jpeg',
        'Pablo_Picasso.jpg',
        'pastel_painting.jpg',
        'pencil1.jpg',
        'pencil2.jpg',
        'Vincent_Van_Gogh.jpg',
        'vitrage_1.jpg',
        'vitrage2.jpg']


@app.route('/stylize', methods=['POST'])
def stylize():
    print(request, request.form, request.files)
    if not request.files or 'style' not in request.form: 
        abort(400)
    file = request.files.get('image')

    request_object_content = file.read()
    img = Image.open(io.BytesIO(request_object_content))
    style_transferer = image_stylization.Style_transferer()
    out = style_transferer.stylize_from_pil(img, f'Style_images/{styles[int(request.form.get("style"))]}', verbose=True)
    tmpfile = BytesIO()
    out.save(tmpfile, "JPEG")
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    im_tag = '<img src="data:image/jpeg;base64,'+ encoded + '"/>'
    result_dict = {'output': im_tag}
    return jsonify(result_dict)
# Функция для отоображения основной страницы
@app.route('/', methods=['GET'])
def index():
  return '''
<html>
    <head>

    </head>
    <body>
        <form id="send_form"  enctype="multipart/form-data" action="/stylize" method="POST">
            <span>Choose style</span>
            <select name="style">
                <option value=0>Anime</option>
                <option value=1>Aquarelle</option>
                <option value=2>Coloured pencil.jpeg</option>
                <option value=3>Pablo Picasso</option>
                <option value=4>Pastel painting</option>
                <option value=5>Pencil 1</option>
                <option value=6>Pencil 2</option>
                <option value=7>Vincent Van Gogh</option>
                <option value=8>Vitrage 1</option>
                <option value=9>Vitrage 2</option>
            </select>
            </br>
            <span>Choose image file to stylize</span><input name="image" type="file" /><br />
            <input type="submit" value='Stylize image'/>
            </br>
        </form>
        <div id='wait' style="visibility:hidden"><span>Please wait... Style transfer in progress</span></div>
        <div id="output_holder" style="clear: both; visibility:hidden">
            <div id="image_holder"></div>
            </br>
            <span>For download stylized image you can raght click on image and choose "save image as"</span>
        </div>
        <script>
            document.forms['send_form'].addEventListener('submit', (event) => {
                event.preventDefault();
                //const formData  = new FormData();
      
                //for(const name in data) {
                //    formData.append(name, data[name]);
                //}
                document.getElementById('wait').style.visibility = 'visible'
                document.getElementById('output_holder').style.visibility = 'hidden'
                fetch(event.target.action, {
                    method: 'POST',
                    body: new FormData(event.target),
                    keepalive: true,
                }).then((response) => {
                    if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                    }
                    return response.json(); // or response.text() or whatever the server sends
                }).then((body) => {
                    oh = document.getElementById('output_holder')
                    oh.style.visibility = 'visible'
                    ih = document.getElementById('image_holder')
                    ih.innerHTML = body.output
                    document.getElementById('wait').style.visibility = 'hidden'
                }).catch((error) => {
                    alert(error)
                });
            });
        </script>
    </body>
</html>'''

# Функция для добалвения заголовков в ответ
@app.after_request
def add_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    return response

# Запуск веб приложения
if __name__ == '__main__':
    style_transferer = image_stylization.Style_transferer()
    app.run(host = '0.0.0.0', port=5000)