# Express-Blueprint
This is a RESTful API blueprint for managing user authentication and authorization using Node.js, Express, and SQLite.

### Getting Started
To get started with this API, you need to have Node.js. You can download Node.js from the official website (https://nodejs.org/)

Once you have Node.js installed, you can clone this repository and install the dependencies using npm:
```
git clone https://github.com/Frier03/Express-Blueprint
cd express-blueprint
npm install
```

Next, you need to create a configuration file called *config.json* in the root directory of the project. The configuration file should contain the following properties:
```
{
  "issuer": "Express-Blueprint",
  "audience": "https://my-api.com/",
  "port": 8888
}
```
Replace the values of *issuer*, *audience* and *port* with your own values. The *issuer* property should be the name of the authentication and authorization provider. The *issuer* property should be the URL of your API. The *port* property should be the port number that the API will listen on.

# Running the API
To run the API with watchdog, you can use the following command:
```
npm run start:watch
```
