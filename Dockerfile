FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml ./
COPY app.py ./

RUN uv pip install . --system

EXPOSE 8000

CMD ["python", "app.py"]
