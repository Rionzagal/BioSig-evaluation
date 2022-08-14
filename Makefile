test-cov-html:
	pytest --cov-report html --cov=tests

gen-dist:
	python setup.py sdist bdist_wheel