FROM python:3.11-slim

ENV VIRTUAL_ENV=venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY ./requirements.txt /app/requirements.txt
WORKDIR /app

#RUN pip install --upgrade pip
RUN pip install --trusted-host pypi.python.org -r requirements.txt

COPY . /app

EXPOSE 8080

CMD [ "python", "manage.py", "run" ]

#ENTRYPOINT ["python", "webhook.py"]
