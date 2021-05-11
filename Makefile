.PHONY: install examples

install:
	pipenv install --two

examples:
	pipenv run python noise.py
