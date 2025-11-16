import sqlite3
from datetime import datetime
from dataclasses import dataclass, field
import os

@dataclass
class ScanHistory:
    scan_id: str
    file_id: str
    source_file: str
    destination_file: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    def as_tuple(self):
        return (self.scan_id, self.file_id, self.source_file, self.destination_file, self.timestamp)

@dataclass 
class ScanHistoryFilterOpts:
    scan_id: str | None = None
    file_id: str | None = None
    source_file: str | None = None
    destination_file: str | None = None
    start_date: str | None = None
    end_date: str | None = None


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
        file_id TEXT,
        source_file TEXT,
        destination_file TEXT,
        timestamp TEXT,
        PRIMARY KEY (scan_id, source_file, destination_file)
        )
        """

        connection = self._connect_db()
        connection.execute(schema)
        connection.commit()
        connection.close()

    def add(self, scans: list[ScanHistory]):
        self.init_db()
        connection = self._connect_db()
        rows = [scan.as_tuple() for scan in scans]
        connection.executemany(
            'INSERT INTO scan_history('
            'scan_id, file_id, source_file, destination_file, timestamp) '
            'VALUES (?, ?, ?, ?, ?)',
            rows
        )
        connection.commit()
        connection.close()

    def _build_other_filters(self, params: list, filter_opts: ScanHistoryFilterOpts | None = None):
        clauses = []
        filter_opts = filter_opts or ScanHistoryFilterOpts()
        if filter_opts.start_date:
            clauses.append("timestamp >= ?")
            params.append(filter_opts.start_date)
        if filter_opts.end_date:
            clauses.append("timestamp <= ?")
            params.append(filter_opts.end_date)
        if filter_opts.source_file:
            clauses.append("source_file = ?")
            params.append(filter_opts.source_file)
        if filter_opts.destination_file:
            clauses.append("destination_file = ?")
            params.append(filter_opts.destination_file)
        if filter_opts.scan_id:
            clauses.append("scan_id = ?")
            params.append(filter_opts.scan_id)
        if filter_opts.file_id:
            clauses.append("file_id = ?")
            params.append(filter_opts.file_id)
        return " AND ".join(clauses) if clauses else ""

    def get(self, filter_opts: ScanHistoryFilterOpts | None = None, limit=None):
        connection = self._connect_db()
        params = []
        query_filter = self._build_other_filters(params, filter_opts)
        where_clause = f"WHERE {query_filter}" if query_filter else ""

        query = f"SELECT * FROM scan_history {where_clause}"
        if limit:
            query += f" LIMIT {limit}"

        cursor = connection.execute(query, params)
        results = [dict(row) for row in cursor.fetchall()]
        connection.close()
        return results
    
    def delete(self, filter_opts: ScanHistoryFilterOpts | None = None):
        connection = self._connect_db()
        params = []
        query_filter = self._build_other_filters(params, filter_opts)
        where_clause = f"WHERE {query_filter}" if query_filter else ""
        query = f"DELETE FROM scan_history {where_clause}"
        connection.execute(query, params)
        connection.commit()
        connection.close()

    def clear(self):
        connection = self._connect_db()
        connection.execute("DELETE FROM scan_history")
        connection.commit()
        os.remove(self.path)
