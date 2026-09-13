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

# Create data directory with permissions for node user
RUN mkdir -p /app/data && chown -R node:node /app/data

USER node

EXPOSE 3000

VOLUME ["/app/data"]

CMD ["node", "dist/server.cjs"]
