install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

install-gcp:
	pip install --upgrade pip &&\
		pip install -r requirements-gcp.txt

install-azure:
	pip install --upgrade pip &&\
		pip install -r requirements-azure.txt
		
install-aws:
	pip install --upgrade pip &&\
		pip install -r requirements-aws.txt


lint:
	pylint --disable=R,C hello.py
	
format:
	black *.py

clean:
	rm -rf __pycache__
	rm -f *.pyc
	rm -f *.log