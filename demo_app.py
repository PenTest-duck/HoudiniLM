import gradio as gr
from CODE.models.baseline_model import BaselineModel

# Initialize models
baseline_model_syn = BaselineModel(use_dan=False)
baseline_model_syn_prefix = BaselineModel(use_dan=True)

def no_model(prompt: str) -> str:
    return prompt

def process_baseline_syn(prompt: str) -> str:
    return baseline_model_syn.generate(prompt)

def process_baseline_syn_prefix(prompt: str) -> str:
    return baseline_model_syn_prefix.generate(prompt)

def process_all(prompt: str):
    no_model_result = no_model(prompt)
    baseline_result_syn = process_baseline_syn(prompt)
    baseline_result_syn_prefix = process_baseline_syn_prefix(prompt)
    
    return no_model_result, baseline_result_syn, baseline_result_syn_prefix

# Create interface with single input and multiple outputs
demo = gr.Interface(
    fn=process_all,
    inputs=gr.Textbox(label="Prompt"),
    outputs=[
        gr.Textbox(label="No Model"),
        gr.Textbox(label="Baseline Model (synonyms)"),
        gr.Textbox(label="Baseline Model (synonyms + prefix/suffix)"),
    ],
    title="HoudiniLM Demo"
)

demo.launch()
