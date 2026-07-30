FROM python:3.10-slim


WORKDIR /workspace


# Copy project files
COPY . .

# Run installations.py
RUN python installations.py

EXPOSE 5000


CMD [ "python3", "-m" , "flask", "--app", "app", "run", "--host=0.0.0.0"]
