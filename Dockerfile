# Reproducible CPU image for the local research prototype.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONPATH=/app
# Default: run the test-suite. Override to run the pipeline or the dashboard, e.g.
#   docker run -p 8501:8501 IMAGE streamlit run src/dashboard/app.py --server.address 0.0.0.0
CMD ["pytest", "-q"]
