"""Check what tables exist in the database."""
import psycopg2

# Connect directly
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    user="postgres",
    password="Coloreal@1",
    database="hybrid_scheduler_db"
)

cursor = conn.cursor()

# List all tables
cursor.execute("""
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public'
""")

tables = cursor.fetchall()
print(f"Found {len(tables)} tables:")
for table in tables:
    print(f"  - {table[0]}")

# List all indexes
cursor.execute("""
    SELECT indexname, tablename
    FROM pg_indexes
    WHERE schemaname = 'public'
""")

indexes = cursor.fetchall()
print(f"\nFound {len(indexes)} indexes:")
for idx in indexes:
    print(f"  - {idx[0]} on {idx[1]}")

cursor.close()
conn.close()
