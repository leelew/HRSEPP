from flask import Flask, render_template
from PIL import Image
import base64
import io

app = Flask(__name__)


@app.route("/")
def show_image():

    im = Image.open("dyj.png")
    data = io.BytesIO()
    im.save(data, "png")
    encoded_img_data = base64.b64encode(data.getvalue())

    return render_template("image.html", img_data=encoded_img_data.decode('utf-8'))

if __name__ == '__main__':
    app.run(host='0.0.0.0')
