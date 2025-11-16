import sqlite3
from datetime import datetime
from dataclasses import dataclass
import os

@dataclass
class ScanHistory:
    scan_id: str
    source_file: str
    destination_file: str
    timestamp: str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def as_tuple(self):
        return (self.scan_id, self.source_file, self.destination_file, self.timestamp)

class ScanHistoryDB:
    def __init__(self, path: str):
        self.path = path

    def _connect_db(self):
        connection = sqlite3.connect(self.path, detect_types=sqlite3.PARSE_DECLTYPES)
        connection.row_factory = sqlite3.Row
        return connection

    def init_db(self):
        schema = """
        CREATE TABLE IF NOT EXISTS scan_history(
        scan_id TEXT,
        source_file TEXT,
        destination_file TEXT,
        timestamp TEXT
        PRIMARY KEY (scan_id, source_file, destination_file)
        )
        """

        connection = self._connect_db(self.path)
        connection.execute(schema)
        connection.commit()
        connection.close()

    def add(self, scans: list[ScanHistory]):
        self.init_db()
        connection = self._connect_db()
        rows = [scan.as_tuple() for scan in scans]

        connection.executemany(
            'INSERT INTO scan_history('
            'scan_id, source_file, destination_file, timestamp) '
            'VALUES (?, ?, ?, ?)',
            rows
        )

        connection.commit()
        connection.close()

    def _build_date_filter(self, params: list, start_date=None, end_date=None):
        clauses = []
        if start_date:
            clauses.append("timestamp >= ?")
            params.append(start_date)
        if end_date:
            clauses.append("timestamp <= ?")
            params.append(end_date)
        return " AND ".join(clauses) if clauses else ""

    def _build_other_filters(self, params: list, source=None, destination=None, scan=None):
        clauses = []
        if source:
            clauses.append("source_file = ?")
            params.append(source)
        if destination:
            clauses.append("destination_file = ?")
            params.append(destination)
        if scan:
            clauses.append("scan_id = ?")
            params.append(scan)
        return " AND ".join(clauses) if clauses else ""

    def get(self, start_date=None, end_date=None, limit=None, source=None, destination=None, scan=None):
        connection = self._connect_db()
        params = []

        filters = []
        date_filter = self._build_date_filter(params, start_date, end_date)
        if date_filter:
            filters.append(date_filter)
        other_filter = self._build_other_filters(params, source, destination, scan)
        if other_filter:
            filters.append(other_filter)

        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

        query = f"SELECT * FROM scan_history {where_clause}"
        if limit:
            query += f" LIMIT {limit}"

        cursor = connection.execute(query, params)
        results = [dict(row) for row in cursor.fetchall()]
        connection.close()
        return results
    
    def delete(self, start_date=None, end_date=None, source=None, destination=None, scan=None):
        connection = self._connect_db()
        params = []

        filters = []
        date_filter = self._build_date_filter(params, start_date, end_date)
        if date_filter:
            filters.append(date_filter)
        other_filter = self._build_other_filters(params, source, destination, scan)
        if other_filter:
            filters.append(other_filter)

        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

        query = f"DELETE FROM scan_history {where_clause}"

        connection.execute(query, params)
        connection.commit()
        connection.close()

    def clear(self):
        connection = self._connect_db()
        connection.execute("DELETE FROM scan_history")
        connection.commit()
        os.remove(self.path)