const logger = require('../utils/logger');
const { runSql } = require('../utils/database');
const { verifyPassword } = require('../utils/password');

async function authMiddleware(req, res, next) {
  const { username, password } = req.body;

  logger.debug(`Validating login credentials for user ${username}`, { action: 'Authentication', username });

  const user = await runSql('get-user.sql', [username]);

  if (!user) {
    logger.debug(`User not found: ${username}`);
    return res.status(401).json({ message: 'Invalid username or password' });
  }

  const isPasswordValid = await verifyPassword(password, user.password);

  if (!isPasswordValid) {
    logger.debug(`Incorrect password for user ${username}`);
    return res.status(401).json({ message: 'Invalid username or password' });
  }

  logger.debug(`User ${username} authenticated successfully (ID: ${user.id})`);
  req.user = { id: user.id, name: username }; // Set the authenticated user in the request object
  return next();
}

module.exports = { authMiddleware };