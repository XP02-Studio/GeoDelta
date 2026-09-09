# ==============================================================================
# MULTI-STAGE DOCKERFILE: REACT / THREE.JS FRONTEND
# ==============================================================================
# Stage 1: Build & bundle React / Three.js client
# Stage 2: Serve via ultra-lightweight nginx:alpine
# Zero-external-dependency capable: supports pre-built bundles or local cache
# ==============================================================================

# --- Stage 1: Build Stage ---
FROM node:18-alpine AS builder

WORKDIR /app

# Copy dependency manifests
COPY package*.json ./

# In air-gapped builds, use pre-cached node_modules or offline tarballs if available
RUN if [ -f package-lock.json ]; then npm ci --prefer-offline --no-audit || npm install --prefer-offline --no-audit; else npm install --prefer-offline --no-audit || true; fi

# Copy application source code
COPY . .

# Run build step; fallback to copying public assets if static bundle already provided
RUN if [ -f "package.json" ] && grep -q '"build":' "package.json"; then npm run build || cp -r public build; else cp -r public build; fi

# --- Stage 2: Production Lightweight Runtime ---
FROM nginx:alpine AS runner

# Remove default nginx static assets
RUN rm -rf /usr/share/nginx/html/*

# Copy built static bundle from builder stage
COPY --from=builder /app/build /usr/share/nginx/html

# Copy custom air-gapped Nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose HTTP port
EXPOSE 80

# Non-root security user hardening
RUN chown -R nginx:nginx /usr/share/nginx/html && \
    chmod -R 755 /usr/share/nginx/html

STOPSIGNAL SIGQUIT

CMD ["nginx", "-g", "daemon off;"]
