FROM node:20

WORKDIR /app

COPY package*.json ./

RUN npm install

COPY . .

ENV PORT=${FRONTEND_PORT}

EXPOSE ${FRONTEND_PORT}

CMD [ "npm", "run", "dev"]



