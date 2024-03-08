const sqlite3 = require('sqlite3');
const path = require('path');
const fs = require('fs');

const config = require('../config.json')
const logger = require('../utils/logger')

let db;

function connect() {
    db = new sqlite3.Database(config.sql.database_file, (err) => {
      if (err) {
        logger.error(`Unable to connect to database.`)
      } else {
        logger.info('Connected to the database.');
      }
    });
  }

function runSql(sqlFileName, params = []) {
    const sqlFilePath = path.join(config.sql.sql_directory, sqlFileName)
    const sql = fs.readFileSync(sqlFilePath, 'utf8')
  
    return new Promise((resolve) => {
      db.get(sql, params, function(err, row) {
        if (err) {
          logger.error(`Unable to execute SQL command '${sqlFileName}' | Error '${err.message}'.`)
        } else {
          logger.debug(`SQL command '${sqlFileName}' executed successfully.`);
          resolve(row);
        }
      });
    });
  }

module.exports = {
    connect,
    runSql
}