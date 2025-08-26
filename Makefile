PYTHON ?= python3

test: test-data test-model

# pytest returns error code 5 if no test is collected.
test-data:
	OMP_NUM_THREADS=4 ${PYTHON} -m pytest -vv -n 4 --dist load --disable-warnings ./byteff2/tests/data

test-model:
	OMP_NUM_THREADS=4 ${PYTHON} -m pytest -vv -n 4 --dist load --disable-warnings ./byteff2/tests/model

lint:
	pylint --disable=R,C --ignore-paths=./scripts

lint-v:
	pylint --ignore-paths=./scripts
