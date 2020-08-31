FROM python:3

WORKDIR /usr/src/app
ENV PORT 80

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD gunicorn --bind 0.0.0.0:$PORT main:server
