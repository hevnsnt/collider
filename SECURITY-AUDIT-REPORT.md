# Collision Protocol Security Audit Report

**Date:** January 5, 2026
**Auditor:** Claude Opus 4.5 Security Analysis
**Scope:** JLP Pool Server (Python), C++ Client, Website Frontend, HTTP API
**Live Server:** 207.244.230.197:17403

---

## Executive Summary

This security audit of the Collision Protocol pool infrastructure identified **4 CRITICAL**, **7 HIGH**, **8 MEDIUM**, and **6 LOW** severity vulnerabilities. The most severe issues involve authentication bypass, denial of service vectors, and information disclosure that could allow attackers to disrupt pool operations, steal work credit, or compromise the pool server.

**Immediate action is required on CRITICAL findings before public launch.**

---

## Architecture Overview

The Collision Protocol consists of:

1. **JLP Pool Server** (`/collision-protocol/pool_server.py`) - Python asyncio TCP server on port 17403
2. **HTTP API** - aiohttp server on port 17404 for website integration
3. **JLP Client** (`/thepuzzler/src/pool/jlp_pool_client.cpp`) - C++ worker client
4. **Website** (`/thepuzzler/website/`) - Next.js frontend with mock data
5. **Supporting modules** - DP store (SQLite + Bloom filter), collision detector, work manager

---

## CRITICAL Severity Findings

### CRITICAL-01: No Authentication/Authorization for Workers

**Location:** `/collision-protocol/pool_server.py` lines 316-366, `/collision-protocol/src/jlp_protocol.py`

**Description:** The worker authentication mechanism (`CLIENT_HELLO`) accepts any worker name without password verification. The config file has `password: ""` (empty) as default, and even when set, authentication is optional based on code analysis.

**Code Evidence:**
```python
# pool_server.py line 116
config.password = security.get('password', '')

# No password check in _authenticate() - only worker_name is extracted and used
hello = JLPProtocol.decode_client_hello(payload)
# ... proceeds to register worker without any credential verification
```

**Impact:**
- Attackers can impersonate any worker name
- Credit theft: Connect as existing worker name to steal DP contribution credit
- Worker ID enumeration through sequential assignment
- Payout manipulation by claiming another worker's identity

**Remediation:**
1. Implement mandatory authentication with HMAC-based challenge-response
2. Require worker registration with unique API keys
3. Add IP allowlisting as defense-in-depth
4. Implement worker signature verification on DP submissions

---

### CRITICAL-02: Unbounded Message Size Allows Memory Exhaustion DoS

**Location:** `/collision-protocol/src/jlp_protocol.py` lines 319-321

**Description:** While MAX_MESSAGE_SIZE is set to 1MB, the server reads the entire payload into memory based on the header's payload_size field. An attacker can send a header claiming 1MB payload and cause memory allocation before any validation.

**Code Evidence:**
```python
# jlp_protocol.py line 319-321
if length > MAX_MESSAGE_SIZE:
    raise ValueError(f"Message too large: {length}")
payload = await self.reader.readexactly(length)  # Allocates 'length' bytes
```

**Attack Scenario:**
1. Attacker sends thousands of connections with headers claiming ~1MB payloads
2. Each connection allocates 1MB buffer waiting for data
3. Attacker sends data slowly (slowloris-style) or not at all
4. Server runs out of memory, crashes, or becomes unresponsive

**Impact:**
- Complete pool server denial of service
- Memory exhaustion crash
- Other workers unable to connect/submit DPs

**Remediation:**
1. Implement connection-level rate limiting
2. Add per-connection memory budgets
3. Use streaming reads with chunk limits
4. Implement connection timeout for incomplete reads
5. Add max_connections enforcement (currently just a config, not enforced)

---

### CRITICAL-03: SQL Injection via Worker Name

**Location:** `/collision-protocol/src/dp_store.py` lines 230-259

**Description:** Worker names from `CLIENT_HELLO` are directly interpolated into SQL queries without parameterization in all locations.

**Code Evidence:**
```python
# dp_store.py lines 235-246
async with self.db.execute(
    "SELECT id FROM workers WHERE name = ?", (name,)  # This one is parameterized
) as cursor:
    # BUT the INSERT is also parameterized, so this is ACTUALLY SAFE

# HOWEVER - checking collision-protocol version vs kangaroo-pool version
```

**Upon further analysis:** The code actually uses parameterized queries correctly. Downgrading this finding.

**REVISED:** This is NOT a vulnerability - the code uses proper parameterization. Removing from CRITICAL.

---

### CRITICAL-03 (REVISED): No Rate Limiting on DP Submissions Enables Work Inflation Attack

**Location:** `/collision-protocol/pool_server.py` lines 453-493

**Description:** Workers can submit unlimited DPs without any validation that they performed actual cryptographic work. The `max_dp_rate` config option is defined but never implemented.

**Code Evidence:**
```python
# config.yaml lines 66-68 define rate limiting:
# max_dp_rate: 10000
# max_invalid_dps: 100

# BUT pool_server.py never checks these values!
# _handle_dp_batch() just accepts and stores all DPs:
async def _handle_dp_batch(self, worker: WorkerConnection, payload: bytes):
    dps = JLPProtocol.decode_client_dp(payload)
    # ... NO RATE LIMITING CHECK
    for dp in dps:
        self.total_dps_received += 1
        worker.total_dps += 1
        # ... stores without verification
```

**Impact:**
- **Payout theft:** Attacker submits millions of fake DPs to inflate their contribution share
- **Storage exhaustion:** SQLite database grows unbounded
- **Bloom filter poisoning:** Fake DPs pollute the bloom filter, causing false positive collisions
- **Honest worker dilution:** Real contributors' share is diluted by fake submissions

**Remediation:**
1. **IMMEDIATELY implement rate limiting:**
```python
async def _handle_dp_batch(self, worker, payload):
    now = time.time()
    window = now - 60  # 1-minute window
    worker_rate = [t for t in worker.dp_times if t > window]
    if len(worker_rate) > self.config.max_dp_rate:
        await worker.conn.send_error("Rate limit exceeded")
        return
    worker.dp_times.append(now)
    # ... continue processing
```
2. Implement DP verification (check that submitted X coordinates are valid curve points)
3. Add proof-of-work verification based on DP bits

---

### CRITICAL-04: Unvalidated DP Submissions Enable Collision Manipulation

**Location:** `/collision-protocol/src/dp_store.py` lines 285-316, `/collision-protocol/pool_server.py` lines 495-520

**Description:** Distinguished Points are stored without cryptographic validation. An attacker can submit crafted DPs designed to create false collisions or manipulate the collision detection algorithm.

**Code Evidence:**
```python
# dp_store.py add_dp() - no validation of X coordinate being on curve
async def add_dp(self, x: bytes, d: bytes, dp_type: int, dp_bits: int,
                 worker_id: int) -> Optional[StoredDP]:
    # Check for collision first
    existing = await self.check_collision(x)
    # ... stores X without verifying it's a valid secp256k1 point
    self.bloom.add(x)  # Adds to bloom filter without validation
```

**Attack Scenarios:**

1. **False Collision Attack:**
   - Submit DP with X coordinate matching a known stored DP but opposite type
   - Server attempts key recovery with attacker-controlled D values
   - Could cause incorrect solution broadcasts or waste server resources

2. **Bloom Filter Poisoning:**
   - Submit millions of fake X coordinates
   - Bloom filter fills with garbage
   - Legitimate collisions missed due to filter saturation

3. **Database Pollution:**
   - Fill database with invalid DPs
   - Slow down collision checks
   - Waste storage resources

**Remediation:**
1. Validate X coordinate is on secp256k1 curve:
```python
from fastecdsa import point, curve
def validate_dp(x_bytes: bytes) -> bool:
    x = int.from_bytes(x_bytes, 'big')
    try:
        # Check if point exists on curve
        point.Point(x, curve.secp256k1.y_from_x(x), curve.secp256k1)
        return True
    except:
        return False
```
2. Verify DP bits constraint (X must have required leading zeros)
3. Track invalid DP counts per worker and ban repeat offenders

---

## HIGH Severity Findings

### HIGH-01: HTTP API Exposes Sensitive Worker Information

**Location:** `/collision-protocol/pool_server.py` lines 634-656

**Description:** The `/api/workers` endpoint returns detailed information about all connected workers without authentication.

**Code Evidence:**
```python
async def _api_get_workers(self, request: web.Request) -> web.Response:
    workers_list = []
    async with self._workers_lock:
        for worker in self.workers.values():
            workers_list.append({
                'id': worker.worker_id,
                'name': worker.name,  # Could be Bitcoin address!
                'gpu_count': worker.gpu_count,
                'speed': worker.speed,
                'connected_at': int(worker.connected_at),
                'last_activity': int(worker.last_activity),
                'total_dps': worker.total_dps,
                'uptime_seconds': int(now - worker.connected_at),
            })
```

**Impact:**
- Exposes worker Bitcoin addresses (often used as worker names)
- Reveals hardware capabilities of pool participants
- Activity timing information enables targeted attacks
- Competitive intelligence gathering

**Remediation:**
1. Add API authentication (API keys or OAuth)
2. Mask sensitive worker names in public responses
3. Rate limit API endpoints
4. Consider making worker list admin-only

---

### HIGH-02: Wildcard CORS Allows Cross-Origin Attacks

**Location:** `/collision-protocol/pool_server.py` lines 585-591

**Description:** The HTTP API sets `Access-Control-Allow-Origin: *` allowing any website to make API requests.

**Code Evidence:**
```python
async def _cors_middleware(self, request: web.Request, handler):
    response = await handler(request)
    response.headers['Access-Control-Allow-Origin'] = '*'  # DANGEROUS
```

**Impact:**
- Malicious websites can query pool API from visitor browsers
- Could be combined with authenticated endpoints for CSRF
- Enables data harvesting from pool visitors

**Remediation:**
```python
ALLOWED_ORIGINS = ['https://collisionprotocol.com', 'https://www.collisionprotocol.com']

async def _cors_middleware(self, request, handler):
    origin = request.headers.get('Origin', '')
    response = await handler(request)
    if origin in ALLOWED_ORIGINS:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Vary'] = 'Origin'
    return response
```

---

### HIGH-03: No TLS/SSL on Pool Connection

**Location:** `/collision-protocol/pool_server.py` line 226-230

**Description:** The JLP protocol connection uses plain TCP without encryption.

**Code Evidence:**
```python
server = await asyncio.start_server(
    self._handle_connection,
    self.config.host,
    self.config.port
)
# No SSL context provided
```

**Impact:**
- Network eavesdroppers can see all pool traffic
- Man-in-the-middle attacks can modify DP submissions
- Private keys transmitted in SOLUTION messages are exposed
- Worker identification information exposed

**Remediation:**
1. Add TLS support:
```python
import ssl
ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
ssl_ctx.load_cert_chain('server.crt', 'server.key')
server = await asyncio.start_server(
    self._handle_connection,
    self.config.host,
    self.config.port,
    ssl=ssl_ctx
)
```
2. Update C++ client to support TLS connections
3. Consider certificate pinning in client

---

### HIGH-04: Solution Broadcast Leaks Private Keys in Plaintext

**Location:** `/collision-protocol/pool_server.py` lines 530-539, `/collision-protocol/src/jlp_protocol.py` lines 270-278

**Description:** When a solution is found, the private key is broadcast to ALL connected workers in plaintext over unencrypted connections.

**Code Evidence:**
```python
async def _broadcast_solution(self, private_key: bytes):
    logger.info("BROADCASTING SOLUTION TO ALL WORKERS")
    async with self._workers_lock:
        for w in self.workers.values():
            try:
                await w.conn.send_solution_found(private_key)  # PRIVATE KEY SENT IN PLAINTEXT
```

**Impact:**
- Any network observer can capture the private key
- Malicious workers receive the key and could race to sweep funds
- No verification that legitimate pool operator controls funds first

**Remediation:**
1. Never broadcast raw private keys
2. Use encrypted channel per-worker with pre-shared keys
3. Implement time-lock: server sweeps funds before broadcasting solution found notification (without the actual key)
4. Consider threshold signatures for multi-party key recovery

---

### HIGH-05: Integer Overflow in C++ Client Payload Handling

**Location:** `/thepuzzler/src/pool/jlp_pool_client.cpp` lines 290-296

**Description:** The payload_size field in the header is a 32-bit integer that could overflow when allocating the receive buffer.

**Code Evidence:**
```cpp
// jlp_pool_client.cpp lines 290-296
if (header.payload_size > 0) {
    payload.resize(header.payload_size);  // potential std::bad_alloc or overflow
    received = recv(socket_, (char*)payload.data(), header.payload_size, MSG_WAITALL);
    if (received != (int)header.payload_size) {
        return false;
    }
}
```

**Impact:**
- Malicious server could crash clients
- Memory corruption if resize fails silently
- DoS against workers

**Remediation:**
```cpp
if (header.payload_size > MAX_PAYLOAD_SIZE) {  // e.g., 10MB limit
    return false;
}
try {
    payload.resize(header.payload_size);
} catch (const std::bad_alloc&) {
    return false;
}
```

---

### HIGH-06: No Payload Size Validation Before Allocation in Protocol

**Location:** `/collision-protocol/src/jlp_protocol.py` lines 206-225

**Description:** Header parsing allows any payload size up to 2^32 before the 1MB check in JLPConnection.

**Code Evidence:**
```python
# decode_header doesn't validate size
version, msg_type, length = struct.unpack('<BBI', data[4:10])
# ... returns length without checking

# Check is in JLPConnection.read_message but after parsing
if length > MAX_MESSAGE_SIZE:
    raise ValueError(f"Message too large: {length}")
```

**Impact:**
- Crafted headers can cause allocation before size check
- Defense-in-depth violation

**Remediation:**
Add size validation in decode_header:
```python
if length > MAX_MESSAGE_SIZE:
    raise ValueError(f"Header declares oversized payload: {length}")
```

---

### HIGH-07: Solution File Written with World-Readable Permissions

**Location:** `/collision-protocol/src/collision.py` lines 269-285

**Description:** The solution file containing the private key is written with default permissions.

**Code Evidence:**
```python
def save_solution(self, path: str):
    with open(path, 'w') as f:  # Default permissions (typically 0644)
        f.write(f"Private Key (hex): {self.solution.private_key_hex}\n")
```

**Impact:**
- Other users on shared server can read private key
- If server is compromised, key is easily accessible
- No encryption at rest

**Remediation:**
```python
import os
import stat

def save_solution(self, path: str):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(fd, 'w') as f:
        f.write(f"Private Key (hex): {self.solution.private_key_hex}\n")
    # Also consider encrypting with operator's public key
```

---

## MEDIUM Severity Findings

### MEDIUM-01: Worker Name Not Sanitized for Logging

**Location:** `/collision-protocol/pool_server.py` line 331

**Code Evidence:**
```python
logger.info(f"Worker hello: name={hello.worker_name}, GPUs={hello.gpu_count}")
```

**Impact:** Log injection attacks via malicious worker names containing newlines or ANSI escape codes.

**Remediation:** Sanitize before logging:
```python
safe_name = hello.worker_name.replace('\n', '').replace('\r', '')[:64]
```

---

### MEDIUM-02: No Connection Limit Enforcement

**Location:** `/collision-protocol/pool_server.py` lines 52, 226-230

**Description:** `max_connections` is defined in config but never enforced.

**Remediation:**
```python
async def _handle_connection(self, reader, writer):
    if len(self.workers) >= self.config.max_connections:
        writer.close()
        await writer.wait_closed()
        return
```

---

### MEDIUM-03: Bloom Filter Persistence Creates Recovery Attack Vector

**Location:** `/collision-protocol/src/dp_store.py` lines 81-100

**Description:** Bloom filter is saved to disk, allowing offline analysis and pre-computation attacks.

**Impact:** Attacker could analyze bloom filter file to predict collision candidates.

**Remediation:** Encrypt bloom filter at rest or regenerate on restart.

---

### MEDIUM-04: No Input Validation on Config File Paths

**Location:** `/collision-protocol/pool_server.py` lines 76-78

**Description:** YAML config file path traversal not validated.

**Remediation:** Validate config path doesn't escape expected directories.

---

### MEDIUM-05: Debug Logging Exposes Internal Protocol Details

**Location:** `/thepuzzler/src/pool/jlp_pool_client.cpp` lines 369-392

**Code Evidence:**
```cpp
std::cerr << "[DEBUG] Raw payload (first 128 bytes): ";
for (size_t i = 0; i < std::min(payload.size(), (size_t)128); i++) {
    char buf[4];
    snprintf(buf, sizeof(buf), "%02x", payload[i]);
```

**Impact:** Debug output in production reveals protocol internals.

**Remediation:** Remove debug statements or gate behind compile flag.

---

### MEDIUM-06: Website Uses Mock Data Instead of Real API

**Location:** `/thepuzzler/website/src/components/landing/PoolPreview.tsx` lines 8-24

**Description:** Frontend displays hardcoded mock data rather than fetching from pool API.

**Impact:** Users may be misled about actual pool statistics.

**Remediation:** Implement real API integration with proper error handling.

---

### MEDIUM-07: No Request Timeout on HTTP API

**Location:** `/collision-protocol/pool_server.py` lines 575-582

**Description:** HTTP API has no timeout configuration, allowing slowloris attacks.

**Remediation:** Configure aiohttp timeouts:
```python
app = web.Application(
    middlewares=[self._cors_middleware],
    client_max_size=1024*1024,
)
# Set timeouts in runner
```

---

### MEDIUM-08: Async Task Exceptions Not Handled

**Location:** `/collision-protocol/pool_server.py` lines 239-240

**Code Evidence:**
```python
asyncio.create_task(self._stats_loop())
asyncio.create_task(self._checkpoint_loop())
# No exception handling - silent failures
```

**Remediation:**
```python
async def _safe_task(self, coro):
    try:
        await coro
    except Exception as e:
        logger.error(f"Background task failed: {e}")

asyncio.create_task(self._safe_task(self._stats_loop()))
```

---

## LOW Severity Findings

### LOW-01: Python Version Not Specified
No `python_requires` in setup or `pyproject.toml`.

### LOW-02: Missing Security Headers on HTTP API
No X-Content-Type-Options, X-Frame-Options, etc.

### LOW-03: Database Path Creates Parent Directories
Could create unexpected directory structures.

### LOW-04: Hardcoded Puzzle 135 References
Magic numbers scattered through codebase.

### LOW-05: No Health Check Authentication
`/health` endpoint accessible without auth.

### LOW-06: Worker Disconnect Logging Reveals Internal IDs
Sequential worker IDs reveal connection patterns.

---

## Recommendations Summary

### Immediate (Before Launch)

1. **Implement worker authentication** with API keys or challenge-response
2. **Add rate limiting** on DP submissions with the configured limits
3. **Validate DPs** are valid secp256k1 points
4. **Enable TLS** on pool connections
5. **Restrict CORS** to actual domain
6. **Fix solution broadcast** to not leak private keys

### Short-term (Within 30 Days)

1. Add HTTP API authentication
2. Implement connection limits
3. Add comprehensive input validation
4. Set secure file permissions for solution files
5. Integrate real API data into website

### Long-term

1. Consider hardware security modules for key management
2. Implement audit logging with tamper detection
3. Add intrusion detection for anomalous DP patterns
4. Deploy behind load balancer with DDoS protection
5. Regular penetration testing

---

## Files Analyzed

| Path | Purpose |
|------|---------|
| `/collision-protocol/pool_server.py` | Main pool server |
| `/collision-protocol/src/jlp_protocol.py` | Protocol implementation |
| `/collision-protocol/src/dp_store.py` | DP storage + bloom filter |
| `/collision-protocol/src/collision.py` | Key recovery |
| `/collision-protocol/src/work_manager.py` | Work distribution |
| `/collision-protocol/config.yaml` | Server configuration |
| `/thepuzzler/src/pool/jlp_pool_client.cpp` | C++ client |
| `/thepuzzler/src/pool/jlp_pool_client.hpp` | C++ client headers |
| `/thepuzzler/website/src/**` | Website components |
| `/kangaroo-pool/*` | Alternative pool implementation |

---

## Conclusion

The Collision Protocol pool infrastructure has significant security vulnerabilities that must be addressed before public launch. The most critical issues are:

1. **Lack of authentication** allows anyone to impersonate workers and steal credit
2. **No rate limiting** enables contribution inflation attacks
3. **Unvalidated DPs** allow collision manipulation
4. **Plaintext private key broadcast** over unencrypted connections

These findings represent real risks to the pool operator's reputation and participants' expected payouts. I recommend engaging professional penetration testers after remediating these issues to verify the fixes.

---

*Report generated by Claude Opus 4.5 security analysis*
