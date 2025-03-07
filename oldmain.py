from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def homepage():
    return render_template("page.html", title="HOME PAGE")

@app.route("/docs")
def docs():
    return render_template("page.html", title="docs page")

@app.route("/about")
def about():
    return render_template("page.html", title="about page")

if __name__ == "__main__":
    app.run(debug=True,port=int(os.environ.get("PORT", 8080)),host='0.0.0.0')