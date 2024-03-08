function validateInput(requiredFields) {
  const numRequiredFields = requiredFields.length;
  return (req, res, next) => {
    const input = req.body;
    const numInputFields = Object.keys(input).length;

    if (!input || typeof input !== 'object') {
      return res.status(400).json({ message: `Invalid request body` });
    }
    if (numInputFields !== numRequiredFields) {
      return res.status(400).json({ message: `Expected ${numRequiredFields} fields, but received ${numInputFields}` });
    }
    for (const field of requiredFields) {
      if (!input.hasOwnProperty(field)) {
        return res.status(400).json({ message: `Missing '${field}' field` });
      }
      if (typeof input[field] !== 'string' || input[field].trim() === '') {
        return res.status(400).json({ message: `Invalid '${field}' field type` });
      }
    }
    return next();
  };
}

module.exports = { validateInput };