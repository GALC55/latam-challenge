# syntax=docker/dockerfile:1.2
# put you docker configuration here
# Usa una imagen base de Python
FROM python:3.11

# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos de requerimientos
COPY requirements.txt .

# Instala las dependencias
RUN pip install -r requirements.txt

# Copia el resto de la aplicación
COPY . .

# Expone el puerto 8080 para Uvicorn
EXPOSE 8080

# Comando para ejecutar la aplicación
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]