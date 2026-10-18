# Web έκδοση του βοηθού (app/server.py). ΔΕΝ έχει δοκιμαστεί ακόμα σε build.
#
# Τα βάρη του μοντέλου ΔΕΝ είναι στο git (430 MB): πριν το build, το
# models/intent-model-234/model.safetensors πρέπει να υπάρχει τοπικά.
#
#   docker build -t heraklion-bot .
#   docker run -p 8000:8000 heraklion-bot
FROM python:3.9-slim

WORKDIR /bot
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    BOT_HOST=0.0.0.0 BOT_PORT=8000 BOT_LOG=0 PYTHONPATH=/bot/app

COPY requirements.txt .
# torch μόνο για CPU: πολύ μικρότερη εικόνα, και το BERT δεν χρειάζεται GPU.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

COPY app/ app/
COPY web/ web/
COPY data/ data/
COPY mappings/ mappings/
COPY models/intent-model-234/ models/intent-model-234/

RUN useradd --create-home bot && mkdir -p logs && chown -R bot /bot
USER bot
EXPOSE 8000
CMD ["python", "app/server.py"]
