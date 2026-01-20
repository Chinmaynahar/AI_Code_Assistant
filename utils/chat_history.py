import sqlite3
import time

class ChatHistoryStore:
    def __init__(self, db_path="chat_history.db", ttl_seconds=3600):
        self.db_path = db_path
        self.ttl = ttl_seconds
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT,
                answer TEXT,
                created_at INTEGER
            )
        """)
        conn.commit()
        conn.close()

    def save(self, question, answer):
     conn = sqlite3.connect(self.db_path)
     cur = conn.cursor()
     cur.execute(
        "INSERT INTO chat_history (question, answer, created_at) VALUES (?, ?, ?)",
        (question, answer, int(time.time()))
    )
     conn.commit()
     conn.close()
    

    def load_recent(self, limit=10):
     cutoff = int(time.time()) - self.ttl

     conn = sqlite3.connect(self.db_path)
     cur = conn.cursor()
     cur.execute("""
        SELECT question, answer
        FROM chat_history
        WHERE created_at > ?
        ORDER BY created_at DESC
        LIMIT ?
     """, (cutoff, limit))
    
     rows = cur.fetchall()
     conn.close()
     return rows[::-1]  # oldest first
    
    def cleanup(self):
     cutoff = int(time.time()) - self.ttl
     conn = sqlite3.connect(self.db_path)
     cur = conn.cursor()
     cur.execute("DELETE FROM chat_history WHERE created_at <= ?", (cutoff,))
     conn.commit()
     conn.close()


