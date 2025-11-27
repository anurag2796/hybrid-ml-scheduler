"""Simple test to check PostgreSQL connection."""
import psycopg2

try:
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        user="postgres",
        password="Coloreal@1",
        database="postgres"
    )
    print("✅ Successfully connected to PostgreSQL!")
    cursor = conn.cursor()
    cursor.execute("SELECT version();")
    version = cursor.fetchone()
    print(f"PostgreSQL version: {version[0]}")
    cursor.close()
    conn.close()
except Exception as e:
    print(f"❌ Failed to connect: {e}")
    print("\nPlease check:")
    print("1. PostgreSQL is running")
    print("2. Credentials are correct (user: postgres, password: Coloreal@1)")
    print("3. PostgreSQL is listening on localhost:5432")
