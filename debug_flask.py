from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    print("Index route called")
    try:
        result = render_template('index.html')
        print(f"Template rendered, length: {len(result)}")
        return result
    except Exception as e:
        print(f"Error rendering template: {e}")
        return f"Error: {e}"

@app.route('/test')
def test():
    return "Test route works!"

if __name__ == '__main__':
    print(f"Templates folder: {app.template_folder}")
    print(f"Static folder: {app.static_folder}")
    app.run(debug=True, host='127.0.0.1', port=5001)
