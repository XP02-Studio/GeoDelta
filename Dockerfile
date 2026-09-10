# Start with a standard Python computer
FROM python:3.10-slim

# Hugging Face requires a special user setup for security
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Copy your requirements and install them
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your backend and ml-engineer folders into the container
COPY --chown=user backend/ ./backend/
COPY --chown=user ml-engineer/ ./ml-engineer/

# Start the FastAPI server on port 7860
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]