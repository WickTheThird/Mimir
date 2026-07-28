# See deploy/docker/docker-compose.yml for why launchd is the recommended path.
FROM python:3.13-slim AS base

# ripgrep and git are the two binaries the repository helpers genuinely want.
# kubectl, sdm, and psql are deliberately NOT installed: a container that can
# reach the cluster needs credentials mounted into it, which the ADR avoids.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ripgrep ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir ".[postgres,search]"

# Runs as a non-root user; the sandbox helper relies on rlimits, not on root.
RUN useradd --create-home --uid 10001 mimir && mkdir -p /data && chown mimir /data
USER mimir
ENV MIMIR_HOME=/data
EXPOSE 8756

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8756/api/health', timeout=3).status==200 else 1)"

CMD ["mimir-api"]
