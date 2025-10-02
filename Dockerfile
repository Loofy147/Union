# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY ./app/requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container at /app
# We will mount the 'app' and 'models' directories using docker-compose,
# so we only need to copy the entrypoint logic.
COPY ./app /app/app

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run app/main.py when the container launches
# Use gunicorn as the process manager with uvicorn workers for performance
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "-b", "0.0.0.0:8000", "app.main:app"]