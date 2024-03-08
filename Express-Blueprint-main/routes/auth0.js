const express = require('express');
const uuid = require('uuid');

const logger = require('../utils/logger');
const jwt = require('../utils/jwt');
const { runSql } = require('../utils/database');
const { hashPassword } = require('../utils/password');

const router = express.Router();

router.post('/login', (req, res) => {
  const username = req.user.name
  const userId = req.user.id

  // Create and sign a JWT
  const payload = { 'sub': username, 'subId': userId };
  const options = { expiresIn: '2h' };
  const token = jwt.createToken(payload, options);

  logger.debug(`Signing token for user ${username}`, { ...options, action: 'Signing token' });

  // Send the JWT in the Authorization header
  res.set('Authorization', `Bearer ${token}`);

  logger.info(`User ${username} logged in successfully (ID: ${userId})`, { action: 'Logged in' });
  res.status(200).json({ userId: userId })
});

router.post('/register', async (req, res) => {
  const { mail, username, password } = req.body;

  // Generate a new UUID for the user
  const userId = uuid.v4();

  // Hash the user's password using bcrypt
  const passwordHash = await hashPassword(password);

  // Register user in SQL
  try {
    await runSql('register-user.sql', [userId, username, mail, passwordHash]);
    logger.info(`User ${username} registered successfully (ID: ${userId})`);
    return res.status(200).json({ message: `User ${username} registered successfully` });
  } catch (err) {
    if (err.code === 'SQLITE_CONSTRAINT') {
      // The user already exists in the database
      logger.debug(`Failed to register user ${username}: username or mail already in use`);
      return res.status(400).json({ message: 'Username or mail already in use' });
    }
    logger.error(`Failed to register user ${username}: ${err.message}`);
    return res.status(500).json({ message: 'Internal server error' });
  }
});

router.post('/logout', async (req, res) => {
  const username = req.user.name
  const userId = req.user.id
  const token = req.user.token

  try {
    // Invalidate/blacklist the token by registering it to the blacklist
    await runSql('register-blacklisted-token.sql', [token]);

    logger.info(`User ${username} (ID: ${userId}) logged out successfully`, { action: 'Logged out' });
    return res.status(200).json({ message: 'Logout successful' });
  } catch(err){
    return res.status(500).json({ message: 'Internal server error' });
  }
});

module.exports = router;