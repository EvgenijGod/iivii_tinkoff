FROM ubuntu:20.04

COPY . .
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y build-essential python3 python3-pip
RUN pip install --upgrade pip
RUN pip install -r req.txt

CMD /bin/sh -c "python3 main.py"