from flask import Flask, render_template, request
import model

app = Flask(__name__)

@app.route("/", methods = ["GET", "POST"])
def home():
    if request.method == "GET":
        return render_template("index.html")
    if request.method == "POST":
        disease = request.form.get('disease')
        state = request.form.get('state')
        county = request.form.get('county')
        uninsured = request.form.get('uninsured')
        mask_never = request.form.get('mask_never')
        mask_always = request.form.get('mask_always')
        poverty = request.form.get('poverty')
        unemployment = request.form.get('unemployment')
        income = request.form.get('income')

        if disease.lower() == 'covid':
            return render_template("index.html", output = model.predict_covid(state,county,uninsured,mask_never,mask_always,poverty,unemployment,income))
        elif disease.lower() == 'stroke':
            return render_template("index.html", output = model.predict_stroke(state,county,uninsured,mask_never,mask_always,poverty,unemployment,income))
        elif disease.lower() == 'coronary':
            return render_template("index.html", output = model.predict_cor(state,county,uninsured,mask_never,mask_always,poverty,unemployment,income))
        else:
            return render_template("index.html", output = 'Invalid Name of Disease')

if __name__ == "__main__":
    app.run()