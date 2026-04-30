import gradio as gr
import pandas as pd
import pickle
import numpy as np

# 1. Load the Model
with open("best_model_pipeline.pkl", "rb") as f:
    model = pickle.load(f)


# 2. The Logic Function
def predict_insurance_cost(age, bmi, children, sex, smoker, region):

    # Pack inputs into a DataFrame
    # The column names must match your CSV file exactly
    input_df = pd.DataFrame(
        [[age, bmi, children, sex, smoker, region]],
        columns=["age", "bmi", "children", "sex", "smoker", "region"],
    )

    # Predict
    prediction = model.predict(input_df)[0]

    # Return formatted result
    return f"Predicted Insurance Cost: {np.clip(prediction, 0, None):,.2f} USD"


# 3. The App Interface
inputs = [
    gr.Slider(minimum=18, maximum=64, step=1, value=30, label="Age"),
    gr.Slider(minimum=10.0, maximum=55.0, step=0.1, value=25.0, label="BMI"),
    gr.Slider(minimum=0, maximum=5, step=1, value=0, label="Number of Children"),
    gr.Radio(["male", "female"], label="Sex", value="male"),
    gr.Radio(["yes", "no"], label="Smoker", value="no"),
    gr.Dropdown(
        ["northeast", "northwest", "southeast", "southwest"],
        label="Region",
        value="southeast",
    ),
]

app = gr.Interface(
    fn=predict_insurance_cost,
    inputs=inputs,
    outputs="text",
    title="Medical Insurance Cost Predictor",
    description=(
        "Enter your details below to predict your annual medical insurance cost. "
        "The model is a GradientBoosting pipeline trained on the insurance dataset."
    ),
)

app.launch(share=False)
