---
name: psql-readonly-investigation
version: 0.1.0
description: STUB. Read-only database investigation. Generic technique is filled in; connection specifics are not.
when_to_use: A question needs evidence from a database. Read the stub warning about connection details.
specialist: sdm_investigator
max_risk: R2
author: mimir
updated_at: 2026-07-27
tags: [psql, postgres, database, sql, query, connections, locks, slow]
allowed_tools:
  - get_sdm_status
  - resolve_sdm_resource
  - inspect_database_schema
  - run_readonly_query
  - explain_query
  - search_memory
inputs:
  - name: resource
    description: The SDM resource that reaches the database.
    required: true
  - name: database
    description: Database name.
    required: true
outputs:
  - name: findings
    description: Query results with the SQL that produced them.
tests:
  - name: never-handles-credentials
    input: Show me the active connections on the billing database.
    assertions:
      - "contains: never handles credentials"
      - "contains: read-only transaction"

---

# Read-only database investigation (PARTIAL STUB)

The technique below is generic PostgreSQL and safe to rely on. What is **not**
filled in, per ADR 5.4 and 9.3, is how you reach your databases: SDM resource
names, ports, database names, roles, and which instances are production.

## Connection

MIMIR never handles credentials (ADR NG3). Authentication comes from your
existing `PGPASSFILE`, `PGSERVICE`, environment, or the SDM-provided local port.
If a connection needs a password typed, you type it.

Record here: which resource reaches which database, the role you connect as, and
which instances are production so the safety rules can recognise them.

## Safety, which is enforced regardless

`run_readonly_query` refuses anything the SQL classifier rates above read-only,
**and** wraps execution in a read-only transaction. Both, deliberately: the
classifier can be fooled by a crafted statement, the read-only transaction
cannot. A statement timeout and a row cap are always applied.

Writes go through `prepare_database_mutation`, which never executes, previews the
affected row count where it can, and requires an explicit transaction.

## Useful read-only queries

Active work and what is blocking:

```sql
SELECT pid, state, wait_event_type, wait_event,
       now() - query_start AS duration, left(query, 120) AS query
FROM pg_stat_activity
WHERE state <> 'idle' AND pid <> pg_backend_pid()
ORDER BY duration DESC LIMIT 20;
```

Connection pressure, which is a common hidden cause of upstream timeouts:

```sql
SELECT count(*) AS total,
       count(*) FILTER (WHERE state = 'active') AS active,
       count(*) FILTER (WHERE state = 'idle in transaction') AS idle_in_txn,
       current_setting('max_connections')::int AS max_connections
FROM pg_stat_activity;
```

A high `idle in transaction` count is usually an application forgetting to
commit, and it holds locks while it does nothing.

Blocking chains:

```sql
SELECT blocked.pid AS blocked_pid, blocking.pid AS blocking_pid,
       left(blocked.query, 80) AS blocked_query,
       left(blocking.query, 80) AS blocking_query
FROM pg_stat_activity blocked
JOIN pg_stat_activity blocking ON blocking.pid = ANY(pg_blocking_pids(blocked.pid))
WHERE cardinality(pg_blocking_pids(blocked.pid)) > 0;
```

Table and index sizes, and unused indexes:

```sql
SELECT relname, pg_size_pretty(pg_total_relation_size(relid)) AS total
FROM pg_catalog.pg_statio_user_tables
ORDER BY pg_total_relation_size(relid) DESC LIMIT 20;
```

## Method

1. Establish the question. "Is the database slow" is not one; "are checkout
   queries waiting on locks between 10:00 and 10:05" is.
2. Check connections and blocking before reading query plans. Saturation and
   lock waits explain more incidents than plan regressions.
3. `EXPLAIN` before `EXPLAIN ANALYZE`. `ANALYZE` executes the query.
4. Report the SQL alongside every result, so it can be re-run and checked.
