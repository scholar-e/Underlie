# TRIBE v2 Brain Comparison API — Setup Guide

## Quick Start

```powershell
# 1. Start the server (from auxiliary/)
& "..\everythingelse\tribev2\.env\Scripts\python" tribev2_api.py

# 2. Test it (separate terminal)
Invoke-RestMethod -Uri http://localhost:8766/health
```

## Endpoints

### `GET /health`
Returns `{"status": "ok"}`.

### `POST /compare`
Compare how two sentences stimulate the brain.

**Request body:**
```json
{
  "sentence_a": "The cat sat on the mat.",
  "sentence_b": "A dog ran in the park."
}
```

**Response:**
```json
{
  "mse": 0.042,
  "mae": 0.163,
  "correlation": 0.87,
  "cosine_similarity": 0.92,
  "n_segments_a": 12,
  "n_segments_b": 10,
  "n_segments_compared": 10
}
```

| Field | Meaning | Lower/higher? |
|-------|---------|---------------|
| `mse` | Mean squared error | Closer to 0 = more similar |
| `mae` | Mean absolute error | Closer to 0 = more similar |
| `correlation` | Pearson correlation | Closer to 1 = more similar |
| `cosine_similarity` | Cosine similarity | Closer to 1 = more similar |

---

## Allowing Others on Your Network

### 1. Find your LAN IP

```powershell
ipconfig
# Look for: IPv4 Address. . . . . . . . . . . : 192.168.x.x
```

### 2. Start the server on `0.0.0.0` (all interfaces)

```powershell
& "..\everythingelse\tribev2\.env\Scripts\python" tribev2_api.py --host 0.0.0.0 --port 8766
```

`0.0.0.0` means "listen on every network interface."

### 3. Allow the port through Windows Firewall

Run this **as Administrator**:

```powershell
New-NetFirewallRule -DisplayName "TRIBE API 8766" -Direction Inbound `
  -Protocol TCP -LocalPort 8766 -Action Allow
```

To remove later:
```powershell
Remove-NetFirewallRule -DisplayName "TRIBE API 8766"
```

### 4. Others call it using your LAN IP

```bash
curl -X POST http://192.168.1.100:8766/compare \
  -H "Content-Type: application/json" \
  -d '{"sentence_a":"Hello.","sentence_b":"Goodbye."}'
```

---

## Making It Public (Internet)

**Warning:** The API has no auth. Anyone who reaches it can run inference.

### Option A: SSH tunnel (easiest, no open ports)

On the server machine, nothing special needed. The client connects via SSH:

```bash
ssh -L 8766:localhost:8766 user@your-server-ip
# Then open http://localhost:8766 locally
```

### Option B: ngrok (no config)

```powershell
ngrok http 8766
# Get a public URL like https://abc123.ngrok.io
```

### Option C: Cloud VM with public IP

Start the server binding to `0.0.0.0` and open port 8766 in the cloud firewall (AWS security group, GCP firewall rule, etc.).

---

## Performance Notes

- **First request is slow** (~minutes) — whisperx downloads model files on first run
- Subsequent requests are faster because the model is cached
- The server uses CPU inference only (~2-5 min per sentence pair)
- Each request is processed sequentially (no concurrent requests)

## Stopping the Server

Press `Ctrl+C` in the terminal where it's running.
