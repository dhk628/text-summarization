FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

COPY docker/requirements-docker.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app /app
COPY outputs/models /outputs/models

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.fileWatcherType=none", "--server.port=8501", "--server.address=0.0.0.0"]