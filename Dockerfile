FROM python:3.8.0
COPY . /app
WORKDIR /app
RUN pip3 install -r req.txt
RUN wget https://bucket.asim.id/invisor2.h5
EXPOSE 5000
CMD [“python3”, “main.py”]
