# ALPR
## Execute in dev enviroment
 - uvicorn server.api:app --host 0.0.0.0 --port 8000 --reload --ssl-keyfile=./key.pem --ssl-certfile=./cert.pem