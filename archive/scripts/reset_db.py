"""Drop and recreate the database."""
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from backend.core.config import settings

# Connect to postgres database
conn = psycopg2.connect(
    host=settings.postgres_host,
    port=settings.postgres_port,
    user=settings.postgres_user,
    password=settings.postgres_password,
    database="postgres"
)
conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
cursor = conn.cursor()

# Drop database if exists
cursor.execute(f"DROP DATABASE IF EXISTS {settings.postgres_db}")
print(f"✅ Dropped database '{settings.postgres_db}'")

# Create fresh database
cursor.execute(f"CREATE DATABASE {settings.postgres_db}")
print(f"✅ Created database '{settings.postgres_db}'")

cursor.close()
conn.close()

print("\nNow run: python scripts/init_db.py")
