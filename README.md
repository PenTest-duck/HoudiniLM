# HoudiniLM - Next-Gen Adversarial Prompt Generation with RL

**Team Name**: How To Rob A Bank

**Project Title**: HoudiniLM - Next-Gen Adversarial Prompt Generation with RL

**Team members**: Chris Yoo (z5589635), Rahul Markasserithodi (z5456750), Ishmanbir Singh (z5480281), Alan Niu (z5604369)

## Installation
0. You must have Python 3.13 installed. Enter the `CODE` directory.
1. Create a Python virtual environment using `python3.13 -m venv .venv`.
2. Enter the virtual environment using `source .venv/bin/activate`.
3. Install the `uv` package manager using `pip install uv`.
4. Install all required packages using `uv sync`.
5. Set your OpenAPI key in `CODE/.env`. An example is provided.

## Usage
1. All Python scripts should be run from the project root directory (e.g. `python CODE/reward_functions/strongreject_rubric.py`)
2. Download the RL fine-tuned LoRAs from [Sharepoint Download Link](https://unsw-my.sharepoint.com/:f:/g/personal/z5480281_ad_unsw_edu_au/Emy4maxOthdCv0PqicZwPNcBmn7mJHCJH8Ec7ljH66LrBQ?e=bRLBxe) and place them inside `CODE/models/loras/`.
3. Please do note that many of the scripts will not run on a typical computer as they involve 8B models. Most of our scripts have been run in a RunPod environment with a high-performance GPU.
