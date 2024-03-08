const winston = require('winston')
const { format, transports } = winston
const path = require('path')

const logFormat = format.printf(info => `${info.timestamp} ${info.level} [${info.label}]: ${info.message}`)

const logger = winston.createLogger({
  level: process.env.NODE_ENV === 'error' ? 'info' : 'debug',
  format: format.combine(
    format.label({ label: path.basename(process.mainModule.filename) }),
    format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),

    // Format the metadata object
    format.metadata({ fillExcept: ['message', 'level', 'timestamp', 'label'] })
  ),
  transports: [
    new transports.Console({
        format: format.combine(
        format.colorize(),
        logFormat
        )
    }),
    new transports.File({
        filename: 'logs/debug.log',
        level: 'debug',
        format: format.combine(format.prettyPrint())
      }),
    new transports.File({
        filename: 'logs/info.log',
        level: 'info',
        format: format.combine(format.prettyPrint())
      }),
    new transports.File({
        filename: 'logs/error.log',
        level: 'error',
        format: format.combine(format.prettyPrint())
      }),
    new transports.File({
        filename: 'logs/combined.log',
        format: format.combine(format.prettyPrint())
    })
  ],
  exitOnError: false
})

module.exports = logger;