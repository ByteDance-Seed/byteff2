UV ?= uv

test: test-data test-model test-toolkit test-train

# pytest returns error code 5 if no test is collected.
test-data:
	OMP_NUM_THREADS=4 ${UV} run pytest -vv -n 4 --dist load --disable-warnings ./byteff2/tests/data

test-model:
	OMP_NUM_THREADS=4 ${UV} run pytest -vv -n 4 --dist load --disable-warnings ./byteff2/tests/model

test-toolkit:
	OMP_NUM_THREADS=4 ${UV} run pytest -vv -n 4 --dist load --disable-warnings ./byteff2/tests/toolkit

test-train:
	OMP_NUM_THREADS=4 ${UV} run pytest -vv -n 4 --dist load --disable-warnings ./byteff2/tests/train

lint:
	${UV} run --no-project --with ruff ruff check .

format:
	${UV} run --no-project --with ruff ruff format .

format-check:
	${UV} run --no-project --with ruff ruff format --check .
