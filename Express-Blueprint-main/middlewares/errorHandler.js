const logger = require('../utils/logger')

function errorHandler(err, req, res, next) {
  const statusCode = err.statusCode || 500; // default if no status code provided

  logger.error(`Error occured with status code ${statusCode}`, { err, action: 'Error'})
  res.status(statusCode).json({ message: err.message })
  }
  
  module.exports = { errorHandler };