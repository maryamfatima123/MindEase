# dev_db_setup.py
import os
import sys
import sqlite3
import importlib.util
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime

ROOT = os.path.abspath(os.path.dirname(__file__))

# Names / paths to ignore when searching (virtualenvs, caches, system packages)
IGNORE_DIR_PARTS = (
    "site-packages",
    os.path.join("lib", "python"),
    "dist-packages",
    ".venv",
    "venv",
    "env",
    "easenv",
    ".env",
    "__pycache__",
    "node_modules",
)

def looks_like_ignored(path):
    lp = path.lower()
    return any(part in lp for part in IGNORE_DIR_PARTS)

def find_file(root, name, maxdepth=5):
    root = os.path.abspath(root)
    for dirpath, dirs, files in os.walk(root):
        # compute depth from root
        depth = dirpath[len(root):].count(os.sep)
        if depth > maxdepth:
            # prune deeper dirs
            dirs[:] = []
            continue

        if looks_like_ignored(dirpath):
            dirs[:] = []  # don't descend into ignored dirs
            continue

        if name in files:
            candidate = os.path.join(dirpath, name)
            # prefer a candidate that is under the project root (not under env/lib/site-packages)
            if "site-packages" not in candidate and "dist-packages" not in candidate:
                return candidate
            # keep searching for a better candidate
            return candidate
    return None

def import_module_from_path(path, modname="project_models"):
    spec = importlib.util.spec_from_file_location(modname, path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"❌ Error executing {path}: {e}")
        raise
    return module

def main():
    # optional override: pass path to models.py
    if len(sys.argv) > 1:
        models_path = sys.argv[1]
        if not os.path.isabs(models_path):
            models_path = os.path.abspath(models_path)
        if not os.path.exists(models_path):
            print("❌ Provided path does not exist:", models_path)
            sys.exit(1)
    else:
        models_path = find_file(ROOT, "models.py", maxdepth=6)

    if not models_path:
        print("❌ Could not find models.py in project tree. Try passing it explicitly:")
        print("   python dev_db_setup.py /full/path/to/your/backend/models.py")
        sys.exit(1)

    print("Found models.py at:", models_path)

    # Import it
    try:
        models = import_module_from_path(models_path)
    except Exception:
        print("❌ Failed to import models.py (see traceback above).")
        sys.exit(1)

    # Ensure the expected attributes exist
    required = ("Base", "User", "Score")
    missing = [r for r in required if not hasattr(models, r)]
    if missing:
        print("❌ models.py does not expose the required names:", missing)
        print("Please open the file and ensure it defines: Base, User, Score (class names or variables).")
        # print some helpful info: top-level names
        print("\nTop-level names found in that module:")
        print(sorted([n for n in dir(models) if not n.startswith("_")])[:200])
        sys.exit(1)

    Base = getattr(models, "Base")
    User = getattr(models, "User")
    Score = getattr(models, "Score")
    print("✅ Imported models: Base, User, Score")

    # Determine DB path (same default as your main.py)
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./mental_health.db")
    print("Using DATABASE_URL:", DATABASE_URL)

    # normalize sqlite path
    if DATABASE_URL.startswith("sqlite:///"):
        db_file = DATABASE_URL.replace("sqlite:///", "")
    else:
        db_file = DATABASE_URL
    db_file = os.path.abspath(db_file)
    print("Resolved DB file path:", db_file)

    engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    print("Creating tables (if not already present)...")
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        print("❌ Base.metadata.create_all failed:", e)
        raise
    print("Done creating tables.")

    # Show existing tables
    if not os.path.exists(db_file):
        print("⚠️ DB file was not created at expected path:", db_file)
        print("Check DATABASE_URL and permissions.")
        sys.exit(1)

    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cur.fetchall()]
    if not tables:
        print("⚠️ No tables found in DB (empty).")
    else:
        print("\nExisting tables:")
        for t in tables:
            print(" -", t)

    # Print schema for known tables
    for t in tables:
        print(f"\nSchema for `{t}`:")
        try:
            cur.execute(f"PRAGMA table_info({t});")
            for r in cur.fetchall():
                cid, name, type_, notnull, dflt, pk = r
                print(f"  - {name} ({type_}) pk={pk} notnull={notnull} default={dflt}")
        except Exception as e:
            print("  (could not read schema):", e)

    # Insert a sample PHQ-9 row if a score table exists
    score_table_candidates = [getattr(Score, "__tablename__", None), "score", "scores"]
    score_table_candidates = [c for c in score_table_candidates if c]
    score_table = None
    for cand in score_table_candidates:
        if cand in tables:
            score_table = cand
            break

    if score_table:
        try:
            print(f"\nInserting sample PHQ-9 row into `{score_table}` (user_id=1)...")
            cur.execute(
                f"INSERT INTO {score_table} (user_id, score_type, score_value, timestamp) VALUES (?, ?, ?, ?)",
                (1, "PHQ-9", 8, datetime.utcnow().isoformat()),
            )
            conn.commit()
            print("Insert succeeded.")
        except Exception as e:
            print("Insert failed (schema mismatch or constraints). Error:", e)
    else:
        print("\nNo score table found (candidates tried):", score_table_candidates)

    # Show last 5 rows from score table
    if score_table:
        try:
            print(f"\nLast 5 rows from `{score_table}`:")
            cur.execute(f"SELECT * FROM {score_table} ORDER BY rowid DESC LIMIT 5;")
            for r in cur.fetchall():
                print(r)
        except Exception as e:
            print("Failed to query rows:", e)

    conn.close()
    print("\n✅ dev_db_setup.py finished. Restart uvicorn if stopped and test API endpoints.")
    print("Example:")
    print("  curl -X POST -d 'username=YOUR&password=PASS' http://127.0.0.1:8000/login")
    print('  curl -H "Authorization: Bearer <token>" http://127.0.0.1:8000/scores')

if __name__ == "__main__":
    main()
