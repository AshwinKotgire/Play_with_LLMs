FROM python:3.10
WORKDIR /basic_llm
ADD server.py .
COPY Requirements .
RUN pip install -r Requirements
CMD [ "python","./server.py" ]