# Uses node ver 22 as our base image
FROM node:22

# Create /app inside the Docker host container
WORKDIR /app

# Copy package.json file code into folder "/app". 
# Format: COPY <source> <destination>
# destination can be ./
# ./ == /app
COPY package*.json ./ 
RUN npm install

COPY src ./src

# After creating app folder and files: 
# /app
# ├── package.json
# ├── src/
# │   ├── index.js
# │   └── ...

# EXPOSE tells Docker to listen to port 3000
EXPOSE 3000

# if use `RUN npm start`, docker will run while image is still building, which may cause the server to fail. 
# CMD will run AFTER building image.
CMD ["npm", "start"]