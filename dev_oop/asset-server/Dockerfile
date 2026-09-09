# ==============================================================================
# LOCAL ASSET SERVER DOCKERFILE
# ==============================================================================
# Purpose: Delivers heavy 3D globe textures and Leaflet XYZ map tiles locally
# with zero external WAN requests and aggressive immutable caching headers.
# ==============================================================================

FROM nginx:alpine

# Remove default configuration and assets
RUN rm -rf /etc/nginx/conf.d/default.conf /usr/share/nginx/html/*

# Create dedicated read-only target mount paths
RUN mkdir -p /usr/share/nginx/html/tiles \
             /usr/share/nginx/html/textures \
             /var/cache/nginx

# Copy hardened, performance-tuned Nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Set permissions
RUN chown -R nginx:nginx /usr/share/nginx/html /var/cache/nginx && \
    chmod -R 755 /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=10s --timeout=3s --retries=3 \
  CMD wget -q --spider http://127.0.0.1:80/health || exit 1

STOPSIGNAL SIGQUIT

CMD ["nginx", "-g", "daemon off;"]
