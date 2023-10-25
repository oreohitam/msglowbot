from flask import Flask, render_template, request, jsonify
from process import preparation, botResponse
preparation()
app = Flask(__name__)

@app.route("/")
def home():
	return render_template("index.html")

@app.route("/lokasi")
def lokasi():
	return render_template("lokasi.html")

@app.route("/aboutus")
def aboutus():
	return render_template("about.html")

@app.route("/team")
def team():
	return render_template("team.html")
	
@app.route("/contoh")
def contoh():
	return render_template("fitur.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
	text = request.get_json().get("message")
	response = botResponse(text)
	message = {"answer": response}
	return jsonify(message)

if __name__ == "__main__":
	app.run(debug=True)