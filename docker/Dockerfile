FROM rayproject/ray:2.9.0-py310

# Set working directory
WORKDIR /home/ray

# Install TensorFlow and additional dependencies
RUN pip install --no-cache-dir \
    tensorflow==2.15.0 \
    numpy==1.24.3 \
    pandas==2.0.3 \
    matplotlib==3.7.2 \
    scikit-learn==1.3.0

# Copy training scripts
COPY training/ /home/ray/training/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

# Default command
CMD ["/bin/bash"]
