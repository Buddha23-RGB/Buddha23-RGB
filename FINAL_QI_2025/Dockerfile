# Use an official Node.js runtime as the base image
FROM node:14

# Set the working directory in the container to /app
WORKDIR /FINAL_QI_2025

# Copy package.json and package-lock.json to the working directory
COPY package*.json ./

# Install any needed packages
RUN npm install

# Copy the rest of the application to the working directory
COPY . .

# Make port 3000 available to the outside world
EXPOSE 3000

# Run the application when the container launches
CMD ["npm", "start"]