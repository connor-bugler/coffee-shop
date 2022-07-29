import os
import pandas as pd

from snowflake.snowpark import Session

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    ...


connection_parameters = {
    "account": os.environ["snowflake_account"],
    "user": os.environ["snowflake_user"],
    "password": os.environ["snowflake_password"],
    "role": os.environ["snowflake_user_role"],
    "warehouse": os.environ["snowflake_warehouse"],
    "database": os.environ["snowflake_database"],
    "schema": os.environ["snowflake_schema"]
}

session = Session.builder.configs(connection_parameters).create()


print(session.sql(
    'select current_warehouse(), current_database(), current_schema()').collect())
