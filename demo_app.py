import gradio as gr
from models.baseline_model import BaselineModel

# Initialize models
baseline_model = BaselineModel()

def no_model(prompt: str) -> str:
    return prompt

def process_baseline(prompt: str) -> str:
    return baseline_model.generate(prompt)

def process_all(prompt: str):
    no_model_result = no_model(prompt)
    baseline_result = process_baseline(prompt)
    
    return no_model_result, baseline_result

# Create interface with single input and multiple outputs
demo = gr.Interface(
    fn=process_all,
    inputs=gr.Textbox(label="Prompt"),
    outputs=[
        gr.Textbox(label="No Model"),
        gr.Textbox(label="Baseline Model")
    ],
    title="HoudiniLM Demo"
)

demo.launch()
