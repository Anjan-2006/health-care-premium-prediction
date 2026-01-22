from joblib import load
import pandas as pd

model_rest=load('artifacts/model_rest.joblib')
model_young=load('artifacts/model_young.joblib')
scaler_young=load('artifacts/scaler_young.joblib')
scaler_rest=load('artifacts/scaler_rest.joblib')


def calculate_normalized_risk(medical_history):
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0
    }
    # Split the medical history into potential two parts and convert to lowercase
    diseases = medical_history.lower().split(" & ")

    # Calculate the total risk score by summing the risk scores for each part
    total_risk_score = sum(risk_scores.get(disease, 0) for disease in diseases)  # Default to 0 if disease not found

    max_score = 14 # risk score for heart disease (8) + second max risk score (6) for diabetes or high blood pressure
    min_score = 0  # Since the minimum score is always 0

    # Normalize the total risk score
    normalized_risk_score = (total_risk_score - min_score) / (max_score - min_score)

    return normalized_risk_score




def preprocessing_input(input_dict):
    feature_columns = ["age", "number_of_dependants", "income_lakhs", "insurance_plan",
                   "genetical_risk", "normalized_risk_score", "gender_Male",
                   "region_Northwest", "region_Southeast", "region_Southwest",
                   "marital_status_Unmarried", "bmi_category_Obesity",
                   "bmi_category_Overweight", "bmi_category_Underweight",
                   "smoking_status_Occasional", "smoking_status_Regular",
                   "employment_status_Salaried", "employment_status_Self-Employed"]

    insurance_encoding={"Bronze":0,"Silver":1,"Gold":2}
    df=pd.DataFrame(0,columns=feature_columns,index=[0])


    for key, value in input_dict.items():

        if key == "gender" and value == "Male":
            df["gender_Male"] = 1

        elif key == "region":
            if value == "Northwest":
                df["region_Northwest"] = 1
            elif value == "Southeast":
                df["region_Southeast"] = 1
            elif value == "Southwest":
                df["region_Southwest"] = 1

        elif key == "marital_status" and value == "Unmarried":
            df["marital_status_Unmarried"] = 1

        elif key == "bmi_category":
            if value == "Obesity":
                df["bmi_category_Obesity"] = 1
            elif  value == "Overweight":
                df["bmi_category_Overweight"] = 1
            elif value == "Underweight":
                df["bmi_category_Underweight"] = 1

        elif key == "smoking_status":
            if value == "Occasional":
                df["smoking_status_Occasional"] = 1
            elif value == "Regular":
                df["smoking_status_Regular"] = 1

        elif key == "employment_status":
            if value == "Salaried":
                df["employment_status_Salaried"] = 1
            elif value == "Self-Employed":
                df["employment_status_Self-Employed"] = 1

        elif key == "insurance_plan":
            df["insurance_plan"] = insurance_encoding.get(value, 1)

        elif key == "age":
            df["age"] = value

        elif key == "number_of_dependants":
            df["number_of_dependants"] = value

        elif key == "income_lakhs":
            df["income_lakhs"] = value

        elif key == "genetical_risk":
            df["genetical_risk"] = value

    # Assuming the 'normalized_risk_score' needs to be calculated based on the 'age'
    df['normalized_risk_score'] = calculate_normalized_risk(input_dict['medical_history'])
    df = handle_scaling(input_dict['age'], df)
           
    return df


def handle_scaling(age,df):
    if age<=25:
        scaler_object=scaler_young
    elif age>25:
        scaler_object=scaler_rest

    
    cols_scale=scaler_object['cols_to_scale']
    scaler=scaler_object['scaler']

    df['income_level']=None
    df[cols_scale]=scaler.transform(df[cols_scale])

    df.drop('income_level',axis=1,inplace=True)

    return df

def predict(input_dict):
    input_df=preprocessing_input(input_dict)

    if input_df['age'][0]<=25:
       model=model_young
    else:
       model=model_rest

    prediction=model.predict(input_df)

    return int(prediction[0])