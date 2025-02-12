# lettuce-inspector
A lightweight Lettuce for performing evaluations
## Prerequisites
- git
- uv to handle dependencies
- credentials to connect to an OMOP-CDM database with embeddings.

First, clone this repository.
There are examples of how to run evaluations in `/examples`.
These can be tested using `uv run examples/compare-llms.py`, for example.
If these work, then your lettuce-inspector works!

## Database connection
Running RAG pipelines and some of the metrics included in evaluations requires a connection to a database.
The database connection credentials are read from the environment, and the easiest way to do this is to make a .env file with this format:

```
DB_HOST = <your database host>
DB_USER = <your username>
DB_PASSWORD = <your password>
DB_NAME = <the database name>
DB_PORT = <the port the database is served on>
DB_SCHEMA = <the database schema>
DB_VECTABLE = <the name of the table containing embeddings>
DB_VECSIZE = <the length of your embedding vectors>
```

Then this can be included when you run scripts with `uv run --env-file .env your_script.py`
