from flask import Flask, render_template, url_for,request
import pickle

model = pickle.load(open(r"food delivey estimation/rf.pkl",'rb'))
scaler=pickle.load(open(r"food delivey estimation/ss.pkl",'rb'))
encoder1=pickle.load(open(r"food delivey estimation/Time_Orderd.pkl",'rb'))
encoder2=pickle.load(open(r"food delivey estimation/Time_Order_picked.pkl",'rb'))
encoder3=pickle.load(open(r"food delivey estimation/Weatherconditions.pkl",'rb'))
encoder4=pickle.load(open(r"food delivey estimation/Road_traffic_density.pkl",'rb'))
encoder5=pickle.load(open(r"food delivey estimation/Type_of_order.pkl",'rb'))
encoder6=pickle.load(open(r"food delivey estimation/Type_of_vehicle.pkl",'rb'))
encoder7=pickle.load(open(r"food delivey estimation/Festival.pkl",'rb'))
encoder8=pickle.load(open(r"food delivey estimation/City.pkl",'rb'))
app = Flask(__name__)

@app.route('/')
def welcome():
    return render_template('index.html') 

@app.route('/predict')
def predict():
    return render_template("predict.html")

@app.route('/output',methods=['GET','POST'])
def output():
    if request.method=='POST':
        Delivery_person_Age=int(request.form['Delivery_person_Age'])
        Delivery_person_Ratings=float(request.form['Delivery_person_Ratings'])
        Time_Orderd=request.form['Time_Orderd']
        Time_Orderd = encoder1.transform([Time_Orderd])
        Time_Order_picked=request.form['Time_Order_picked']
        Time_Order_picked = encoder2.transform([Time_Order_picked])
        Weatherconditions=request.form['Weatherconditions']
        Weatherconditions = encoder3.transform([Weatherconditions])
        Road_traffic_density=request.form['Road_traffic_density']
        Road_traffic_density = encoder4.transform([Road_traffic_density])
        Vehicle_condition=int(request.form['Vehicle_condition'])
        Type_of_order=request.form['Type_of_order']  
        Type_of_order = encoder5.transform([Type_of_order])
        Type_of_vehicle=request.form['Type_of_vehicle']  
        Type_of_vehicle = encoder6.transform([Type_of_vehicle])
        multiple_deliveries=int(request.form['multiple_deliveries'])
        Festival=request.form['Festival']
        Festival = encoder7.transform([Festival])
        city=request.form['City']
        city = encoder8.transform([city])
        distance=float(request.form['distance'])
        total=[[Delivery_person_Age,Delivery_person_Ratings,Time_Orderd[0],Time_Order_picked[0],Weatherconditions[0],Road_traffic_density[0],Vehicle_condition,Type_of_order[0],Type_of_vehicle[0],multiple_deliveries,Festival[0],city[0],distance]]
        prediction = model.predict(scaler.transform(total))
        prediction = int(prediction[0])
        return render_template('Output.html',predict=prediction)

if __name__ == '__main__':
    app.run(debug=True)
