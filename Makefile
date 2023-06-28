zen:
	python -c "import this"

#UTILS
install_requirements :
	pip install -U pip wheel
	pip install -r requirements.txt

# TESTS
test_gcp :
	python test/test_bq_access.py
