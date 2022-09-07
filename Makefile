create-venv:
	python -m venv venv

update-pip:
	python -m pip install --upgrade pip

reqs: update-pip
	pip install -r requirements.txt

reqs-dev: reqs
	pip install -r requirements_dev.txt

test-cov-html: install-reqs-dev
	pytest --cov-report html --cov=tests