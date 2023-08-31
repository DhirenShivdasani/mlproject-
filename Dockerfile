FROM python:3.11.3
COPY . /application
WORKDIR /application
RUN pip install -r requirements.txt
CMD python application.py