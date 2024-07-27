# Use an official Python runtime as the base image
FROM python:3.11.4-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

ARG GROQ_API_KEY

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port that the backend will listen on
EXPOSE 8000

# Set the command to run the backend server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]