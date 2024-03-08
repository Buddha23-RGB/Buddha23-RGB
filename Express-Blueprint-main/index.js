const express = require('express')

const { errorHandler } = require('./middlewares/errorHandler')
const { authMiddleware } = require('./middlewares/authMiddleware')
const { authzMiddleware } = require('./middlewares/authzMiddleware')
const { validateInput } = require('./middlewares/inputValidation')
const { connect, runSql } = require('./utils/database')
const logger = require('./utils/logger')

const auth0Router = require('./routes/auth0')
const usersRouter = require('./routes/users')

// Connect to the database
connect()

// Run SQL command to create the users table if not exists
runSql('create-users-table.sql')
    .catch((err) => {
        console.error(`Error creating users table: ${err.message}`)
})

// Run SQL command to create the token blacklist table if not exists
runSql('create-token-blacklist-table.sql')
    .catch((err) => {
        console.error(`Error creating token blacklist table: ${err.message}`)
})

const app = express()

app.use(express.json());

app.use('/api/auth0/login', validateInput(['username', 'password']), authMiddleware)
app.use('/api/auth0/register', validateInput(['mail', 'username', 'password']))
app.use('/api/auth0/logout', authzMiddleware)
app.use('/api/auth0/', auth0Router)

app.use('/api/users/:uuid', validateInput(['uuid']), authzMiddleware)
app.use('/api/users', usersRouter)

app.use(errorHandler)

app.listen(8888, () => {
    logger.info(`Server started listening`, { port: 8888, action: 'Started server' })
})