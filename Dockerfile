FROM tensorflow/tensorflow:1.2.1-py3
COPY . /uia-workshop
WORKDIR /uia-workshop
RUN pip install -r req.txt
ENTRYPOINT [ "python","-u","front_end.py", "-C", "topic" ]