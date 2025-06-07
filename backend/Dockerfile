# 1. Use an official, lightweight Python base image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the dependency file and install dependencies
# This is done in a separate step to leverage Docker's layer caching.
# If requirements.txt doesn't change, this layer won't be rebuilt.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of the application code into the container
COPY . .

# 5. The command to run the application
# Use 0.0.0.0 to make the server accessible from outside the container.
# The internal port will be 8000. We will map this to a different host port.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]