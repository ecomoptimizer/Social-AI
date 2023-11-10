# Use the Python3.8 image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt and install the dependencies
COPY requirements.txt requirements.txt

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn && gunicorn --version

COPY ./app /app

ENV FLASK_APP=app.py
ENV FLASK_ENV=production

EXPOSE 5065

CMD ["gunicorn", "-b", "0.0.0.0:5065", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-", "app:app"]