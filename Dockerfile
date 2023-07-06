####### 👇 OPTIMIZED SOLUTION (x86)👇 #######

# tensorflow base-images are optimized: lighter than python-buster + pip install tensorflow
FROM python:3.10.6-buster
WORKDIR /prod

# We strip the requirements from useless packages like `ipykernel`, `matplotlib` etc...
COPY requirements_prod.txt requirements.txt
RUN pip install -U pip
RUN pip install -r requirements.txt
RUN mkdir models

COPY sarcasme sarcasme
COPY setup.py setup.py
RUN pip install .

COPY Makefile Makefile
# RUN make reset_local_files

CMD uvicorn sarcasme.api.fast:app --host 0.0.0.0 --port $PORT
