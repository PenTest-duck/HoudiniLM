install:
	python3.13 -m venv .venv
	. .venv/bin/activate
	pip install uv
	uv sync

venv:
	. .venv/bin/activate

demo: venv
	python demo/demo.py

