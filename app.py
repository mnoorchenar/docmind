from flask import Flask, render_template_string
app = Flask(__name__)
HTML = """<!DOCTYPE html>
<html><head><title>docmind</title></head>
<body style="font-family:Arial;max-width:800px;margin:50px auto;padding:20px">
  <h1>docmind</h1>
  <p>Running on port 7860.</p>
  <span style="background:#28a745;color:#fff;padding:5px 15px;border-radius:15px">Running</span>
</body></html>"""
@app.route('/')
def home(): return render_template_string(HTML)
if __name__ == '__main__': app.run(host='0.0.0.0', port=7860)
