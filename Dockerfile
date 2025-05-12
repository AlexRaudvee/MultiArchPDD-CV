# Use a base image with Python installed
FROM python:3.12

# Set working directory inside the container
WORKDIR /MultiArchPDD-CV

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code, including setup.sh
COPY . .

# Make sure setup.sh is executable
# RUN chmod +x setup.sh

# Run setup.sh ONCE during build (e.g., to rename config or prep things)
# RUN ./setup.sh

# Set the default command for container runtime
CMD ["bash"]  