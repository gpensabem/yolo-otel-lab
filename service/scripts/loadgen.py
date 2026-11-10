"""Gerador de carga simples (só stdlib) para popular as dashboards.

Mix de tráfego:  ~85% detecção em amostra | ~7% upload inválido (400) | ~4% lento | ~4% erro 500
Uso: python scripts/loadgen.py   (ou o serviço `loadgen` do compose: docker compose --profile load up)
"""
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
import uuid

API = os.getenv("API_URL", "http://localhost:8000")
WORKERS = int(os.getenv("WORKERS", "3"))
RPS_PER_WORKER = float(os.getenv("RPS_PER_WORKER", "1.5"))


def post(path, body=None, headers=None):
    req = urllib.request.Request(API + path, data=body or b"", headers=headers or {}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return 0


def bad_upload():
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"nota.txt\"\r\n"
            f"Content-Type: text/plain\r\n\r\nisso não é uma imagem\r\n--{boundary}--\r\n").encode()
    return post("/detect", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})


def worker(samples, stats):
    while True:
        p = random.random()
        if p < 0.85:
            conf = random.choice([0.15, 0.25, 0.4, 0.6])
            code = post(f"/detect/sample/{random.choice(samples)}?conf={conf}")
        elif p < 0.92:
            code = bad_upload()
        elif p < 0.96:
            code = post(f"/simulate/slow?ms={random.randint(400, 2000)}")
        else:
            code = post("/simulate/error")
        stats[code] = stats.get(code, 0) + 1
        time.sleep(random.expovariate(RPS_PER_WORKER))


def main():
    while True:
        try:
            samples = json.load(urllib.request.urlopen(API + "/samples", timeout=5))
            if samples:
                break
        except Exception:
            pass
        print("aguardando API...", flush=True)
        time.sleep(2)

    stats = {}
    for _ in range(WORKERS):
        threading.Thread(target=worker, args=(samples, stats), daemon=True).start()
    while True:
        time.sleep(10)
        print("status codes:", dict(sorted(stats.items())), flush=True)


if __name__ == "__main__":
    main()
