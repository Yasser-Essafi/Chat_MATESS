FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    STATOUR_RUNTIME_ROOT=/home/statour \
    STATOUR_CHARTS_DIR=/home/statour/charts \
    STATOUR_REPORTS_DIR=/home/statour/reports \
    STATOUR_LOGS_DIR=/home/statour/logs \
    STATOUR_VECTORSTORE_DIR=/home/statour/vectorstore

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gnupg \
        unixodbc \
        unixodbc-dev \
    && curl -fsSL https://packages.microsoft.com/config/debian/12/packages-microsoft-prod.deb -o packages-microsoft-prod.deb \
    && dpkg -i packages-microsoft-prod.deb \
    && rm packages-microsoft-prod.deb \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 \
    && rm -rf /var/lib/apt/lists/*

COPY chatbotfinal/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt gunicorn

COPY . /app

RUN mkdir -p /home/statour/charts /home/statour/reports /home/statour/logs /home/statour/chat_history /home/statour/vectorstore

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "6", "--timeout", "300", "server:create_app()"]
