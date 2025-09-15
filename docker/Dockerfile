RUN apt-get update && apt-get install -y git libgl1-mesa-glx
COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt
WORKDIR /workspace