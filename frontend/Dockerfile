# ============================================
# MCMT-ReID Frontend - Multi-stage Dockerfile
# ============================================

# ============================================
# Base stage - shared dependencies
# ============================================
FROM node:20-alpine AS base
WORKDIR /app

# Install dependencies only when needed
FROM base AS deps
# Check https://github.com/nodejs/docker-node/tree/b4117f9333da4138b03a546ec926ef50a31506c3#nodealpine to understand why libc6-compat might be needed.
RUN apk add --no-cache libc6-compat

# Copy package files
COPY package.json package-lock.json* ./

# Install dependencies
RUN npm ci

# ============================================
# Development stage - hot-reload enabled
# ============================================
FROM base AS development

# Install curl for healthcheck
RUN apk add --no-cache curl

# Copy node_modules from deps stage
COPY --from=deps /app/node_modules ./node_modules

# Copy configuration files
COPY package.json package-lock.json* ./
COPY next.config.ts tsconfig.json ./
COPY components.json ./
COPY postcss.config.mjs ./
COPY eslint.config.mjs ./

# Create src and public directories (will be mounted)
RUN mkdir -p src public

# Expose port
EXPOSE 3000

# Set environment variables
ENV NODE_ENV=development
ENV NEXT_TELEMETRY_DISABLED=1
ENV HOSTNAME="0.0.0.0"

# Start Next.js in development mode
CMD ["npm", "run", "dev"]

# ============================================
# Production builder stage
# ============================================
FROM base AS builder

COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Build the application
RUN npm run build

# ============================================
# Production stage - optimized for deployment
# ============================================
FROM base AS production

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

# Copy built assets
COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

CMD ["node", "server.js"]
