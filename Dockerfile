FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 8000
COPY data/Student_performance_data.parquet /data/Student_performance_data.parquet
#CMD ["python", "Auto_ML.py"]
CMD ["uvicorn", "Auto_ML:app","--host","0.0.0.0","--port", "8000"]
