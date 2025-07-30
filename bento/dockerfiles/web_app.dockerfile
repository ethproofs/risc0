FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY bento/db-web-app/package*.json ./
RUN npm install

# Copy application files
COPY bento/db-web-app/ .

EXPOSE 3001

CMD ["npm", "start"]
