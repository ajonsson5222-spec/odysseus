# syntax=docker/dockerfile:1
FROM node:22-slim AS builder

WORKDIR /app

# Copy dependency manifests
COPY package*.json ./

# Install dependencies for build
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi

# Copy source files and frontend assets
COPY server.ts ./
COPY static ./static

# Build server bundle to dist/server.cjs
RUN npm run build

# Production runner stage
FROM node:22-slim AS runner

WORKDIR /app

ENV NODE_ENV=production
ENV PORT=3000
ENV HOST=0.0.0.0
ENV DATA_DIR=/app/data

# Install production dependencies only
COPY package*.json ./
RUN if [ -f package-lock.json ]; then npm ci --omit=dev; else npm install --omit=dev; fi && npm cache clean --force

# Copy compiled backend and static frontend from builder
COPY --from=builder /app/dist ./dist
COPY static ./static

# Copy entrypoint script to initialize data directories and ensure volume permissions
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh && mkdir -p /app/data/uploads /app/data/documents

EXPOSE 3000

VOLUME ["/app/data"]

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["node", "dist/server.cjs"]
