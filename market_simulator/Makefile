.PHONY: build dist redist install install-from-source clean uninstall

build:
	CYTHONIZE=1 python setup.py build

dist:
	python setup.py bdist --cythonize

redist: clean dist

install:
	pip install dist/market_simulator-0.0.1-cp310-cp310-linux_x86_64.whl

clean:
	$(RM) -r build dist *.egg-info

uninstall:
	pip uninstall -y market_simulator