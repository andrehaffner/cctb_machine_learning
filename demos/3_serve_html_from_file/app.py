from flask import Flask, render_template
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('quadratic.html')
    # a convention to find template file in `templates` folder

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
