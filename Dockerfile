FROM python:3.10-slim

# Installa dipendenze di sistema necessarie
RUN apt-get update && apt-get install -y \
    libopenexr-dev \
    build-essential \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgtk2.0-dev \
    tk \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Crea directory di lavoro
WORKDIR /app

# Copia e installa requisiti
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia il codice dell'applicazione
COPY record3d_viewer.py .

# Crea directory per i dati
RUN mkdir -p /data

# Configurazione per interfaccia grafica
ENV DISPLAY=host.docker.internal:0

# Comando di avvio
CMD ["python", "record3d_viewer.py"]