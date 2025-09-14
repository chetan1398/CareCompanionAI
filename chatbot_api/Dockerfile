# chatbot_api/Dockerfile

FROM python:3.11-slim

WORKDIR /app

# Copy source code
COPY ./src/ /app

# Copy dependency file
COPY ./pyproject.toml /code/pyproject.toml

# Install dependencies (but don’t try to package-install your code)
RUN pip install /code/.

# Ensure Python can resolve imports like `from agents...`
ENV PYTHONPATH=/app

# Copy entrypoint script
COPY ./src/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8000
CMD ["sh", "entrypoint.sh"]
