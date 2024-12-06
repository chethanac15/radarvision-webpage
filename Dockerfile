# Use a smaller base image for the build stage
FROM python:3.9-slim as build

# Set the working directory
WORKDIR /app

# Copy only the necessary files
COPY requirements.txt ./

# Install dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc curl \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu \
    && apt-get remove -y gcc \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the rest of the application files
COPY . .

# Download the model
RUN MODEL_PATH="best_model_CustomVGG.pt" && \
    if [ ! -f "$MODEL_PATH" ]; then \
      echo "Downloading model from $MODEL_URL..."; \
      curl -o "$MODEL_PATH" "$MODEL_URL"; \
    else \
      echo "Model already exists at $MODEL_PATH, skipping download."; \
    fi

# Install Gunicorn
RUN pip install gunicorn

# Expose the port
EXPOSE 5000

# Start the Gunicorn server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app", "--w=2"]
