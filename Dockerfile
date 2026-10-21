FROM python:3.13.16-slim-bookworm@sha256:b4409e44a3e971621ee4bc2f3dc7a53859980685d261ab7a3a7bd4f0b84445c7

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    EXPECTED_PYTHON_VERSION=3.13.16 \
    MPLBACKEND=Agg \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory /workspace

WORKDIR /workspace

COPY requirements.lock /opt/tad/requirements.lock
RUN python -m pip install --no-cache-dir --require-hashes -r /opt/tad/requirements.lock \
    && python -m pip check

COPY scripts/check_environment.py /opt/tad/scripts/check_environment.py

CMD ["python", "/opt/tad/scripts/check_environment.py"]
