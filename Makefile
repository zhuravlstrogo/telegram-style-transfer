.PHONY: check format test

check:
	python -m black --check src tests
	python -m isort --check-only src tests
	python -m pylint src tests

format:
	python -m black src tests
	python -m isort src tests

test:
	python -m pytest -q
