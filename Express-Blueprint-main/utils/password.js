const bcrypt = require('bcrypt');
require('dotenv').config()

const SALT_ROUNDS = 10
function hashPassword(password) {
  return bcrypt.hash(password, SALT_ROUNDS);
}

async function verifyPassword(password, hash) {
    return await bcrypt.compare(password, hash);
}

module.exports = {
  hashPassword,
  verifyPassword,
};