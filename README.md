# SAFETYVISION AI
### Intelligent PPE Compliance, Workplace Risk Monitoring & Automated Safety Inspection Platform

**Final Year B.Tech Capstone Project — Artificial Intelligence & Data Science**

---

## 📌 Executive Summary

**SafetyVision AI** is an enterprise-grade occupational safety intelligence and video analytics platform engineered to monitor, verify, and enforce personal protective equipment (PPE) compliance and workplace hazard safety in real-time.

Going beyond naive object detection, SafetyVision AI integrates:
1. **YOLOv8 Edge Deep Learning**: Detects 10 domain-specific classes.
2. **Centroid & Spatial IoU Multi-Object Tracking**: Assigns persistent session IDs (`Person #1`, `Person #2`) without biometric face identification.
3. **Anatomical Spatial Association**: Uses head, face, and torso containment boundaries to link PPE items to individual workers.
4. **Safety Zone Geofencing**: Ray-casting point-in-polygon logic enforcing zone-specific PPE mandates and hazard multipliers.
5. **Context-Aware Risk Matrix**: Formulates real-time 0–100 dynamic risk scores compounding missing PPE with proximity to active machinery and industrial vehicles.
6. **Temporal Persistence Confirmation**: Filters optical jitter through configurable time thresholds and debounced alerts.
7. **Automated Inspection Audits & PDF Certification**: Compiles verified quantitative telemetry into formal vector PDF reports via ReportLab.
8. **Multi-Channel Outbound Notifications**: Push alerts to Telegram and transactional decision emails via Brevo.
9. **Interactive Telegram Approval System**: Real-time Approve/Reject inline button decision flow for public demo requests with signed HMAC payloads and automatic account provisioning.

---

## 🏗️ System Architecture: Edge Device vs Cloud Web Platform

SafetyVision AI separates heavy computer-vision edge workloads from cloud web operations:

```
+-------------------------------------------------------------+
|               LOCAL EDGE INFERENCE (NVIDIA RTX 3050)        |
| - RTSP / USB / Industrial CCTV Video Feeds                  |
| - High-FPS YOLOv8 Real-Time TensorRT/CUDA Inference         |
| - Spatial Association, Geofencing, Temporal Debounce        |
+-------------------------------------------------------------+
                               |
                               | (HTTP / REST API Alerts & Snapshots)
                               v
+-------------------------------------------------------------+
|               CLOUD WEB PLATFORM (RENDER / WSGI)            |
| - Centralized Dashboard, Heatmaps & Audit Logs              |
| - User Authentication, RBAC & Demo Request Workflows        |
| - Telegram Outbound Bot & Inbound Webhook Approval Engine   |
| - Brevo Email Notification Transport                        |
| - Vector PDF Report Generation & Evidence Management        |
+-------------------------------------------------------------+
```

*Note: The cloud web dashboard runs on standard lightweight CPU instances on Render without requiring a GPU. YOLO inference on live CCTV streams runs locally on the edge machine.*

---

## 🏷️ Trained Model & Supported Classes

SafetyVision AI uses an explicitly trained custom YOLOv8 model (`models/best.pt`, ~6.25 MB). The model detects strictly the following **10 classes**:

| Class ID | Class Name | Category | Color Signature | Regulatory Description |
|:---:|:---|:---|:---:|:---|
| `0` | **Hardhat** | Compliant PPE | `#10B981` (Green) | ANSI Z89.1 Industrial protective safety helmet |
| `1` | **Mask** | Compliant PPE | `#06B6D4` (Cyan) | Particulate / respiratory protective face mask |
| `2` | **NO-Hardhat** | Safety Violation | `#EF4444` (Red) | Unprotected head in active hazard zone |
| `3` | **NO-Mask** | Safety Violation | `#F97316` (Orange) | Absence of mandated respiratory protection |
| `4` | **NO-Safety Vest** | Safety Violation | `#DC2626` (Dark Red) | High-collision visibility infraction |
| `5` | **Person** | Human Worker | `#8B5CF6` (Violet) | Tracked worker subject with session ID |
| `6` | **Safety Cone** | Hazard Demarcation | `#F59E0B` (Amber) | Perimeter barrier marker for equipment corridors |
| `7` | **Safety Vest** | Compliant PPE | `#3B82F6` (Blue) | ANSI/ISEA 107 Class 2/3 high-visibility vest |
| `8` | **machinery** | Active Hazard | `#EC4899` (Magenta) | Stamping presses, CNC mills, cutting apparatus |
| `9` | **vehicle** | Mobile Hazard | `#6366F1` (Indigo) | Forklifts, pallet jacks, automated guided vehicles (AGVs) |

---

## 📐 Mathematical Formulations & Algorithms

### 1. Person-Level PPE Compliance Rate
For tracked individual $i$:
$$\text{Compliance}_i = \sum_{g \in \{\text{Hardhat}, \text{Mask}, \text{Vest}\}} w_g \cdot \mathbb{I}(\text{status}(g) = \text{Present})$$
- Default Weights: Hardhat ($33.33\%$), Mask ($33.33\%$), Safety Vest ($33.34\%$).

### 2. Dynamic Risk Score Formula ($0 - 100$)
$$R_i = \min\left(100, \max\left(0, (P_{\text{ppe}} + P_{\text{context}} + P_{\text{zone}} + P_{\text{duration}}) \times M_{\text{zone}}\right)\right)$$

Where:
- **Base PPE Penalties ($P_{\text{ppe}}$)**:
  - Missing Hardhat: $+25$
  - Missing Safety Vest: $+25$
  - Missing Mask: $+15$
  - Multiple Missing PPE Synergistic Penalty: $+10$ (if $\ge 2$ missing)
- **Environmental Context Penalties ($P_{\text{context}}$)**:
  - Near Active Machinery: $+20$ (if PPE missing), $+10$ (if compliant)
  - Near Vehicle Path: $+20$ (if PPE missing), $+10$ (if compliant)
- **Zone Breach Penalty ($P_{\text{zone}}$)**:
  - Restricted Area Entry: $+30$
  - Zone Mandated Gear Absence: $+15$ per item
- **Temporal Duration Escalation ($P_{\text{duration}}$)**:
  - Continuous breach $> 5$s: $+10$
  - Continuous breach $> 15$s: $+20$
  - Continuous breach $> 30$s: $+30$
- **Zone Risk Multiplier ($M_{\text{zone}}$)**:
  - Walkway: $1.0\times$
  - Machinery Bay: $1.3\times - 1.5\times$
  - High-Voltage Substation: $2.0\times$

### 3. Composite Site Safety Score
$$\text{Safety Score} = 100 - \left(0.35 \times \overline{\text{Risk}} + 0.35 \times (100 - \overline{\text{Compliance}}) + 0.30 \times \min(100, N_{\text{crit}} \times 15)\right)$$

---

## 🛠️ Technology Stack

- **Deep Learning & Computer Vision**: PyTorch, Ultralytics YOLOv8, OpenCV
- **Backend Framework**: Python 3.10+, Flask 3.1, Werkzeug
- **WSGI Production Server**: Gunicorn
- **Database Engine**: SQLite3 with WAL mode support
- **Audit & Compliance Reports**: ReportLab 5.0 (Vector PDF)
- **Messaging & Notifications**: Telegram Bot API, Brevo Transactional Email API
- **Frontend & UI**: HTML5 Canvas HUD, CSS3 Custom Properties, JavaScript ES6+, Chart.js 4.4, face-api.js

---

## 🚀 Local Installation & Quick Start

### 1. Prerequisites
- Python 3.10 or newer
- Git

### 2. Clone and Setup Environment
```bash
git clone https://github.com/rehan30105-cpu/SafetyVision-AI.git
cd SafetyVision-AI

# Create and activate virtual environment
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your keys:
```bash
copy .env.example .env
```

### 4. Create Initial Administrator Account
```bash
python create_admin.py --username admin --email admin@company.com --password YourStrongPassword123 --fullname "System Admin"
```

### 5. Run the Local Development Server
```bash
python -m flask --app app run --host 127.0.0.1 --port 5000
```
Open your browser at `http://127.0.0.1:5000`.

### 6. Run the Telegram Approval Poller (Local Development)
When testing Telegram approval buttons on localhost without a public domain, run the dedicated long-poll bridge:
```bash
python run_telegram_approval_poller.py
```

---

## 🌐 Render Cloud Deployment Guide

SafetyVision AI is pre-configured for one-click or manual deployment on [Render](https://render.com).

### Method A: Deploy via Render Blueprint (Recommended)
1. In your Render Dashboard, click **New +** &rarr; **Blueprint**.
2. Connect your repository: `https://github.com/rehan30105-cpu/SafetyVision-AI.git`.
3. Render reads `render.yaml` automatically and configures:
   - **Environment**: Python
   - **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command**: `gunicorn wsgi:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120`
   - **Health Check Path**: `/health`
4. Under Environment Variables in Render, provide your secrets:
   - `PUBLIC_BASE_URL`: `https://<your-render-subdomain>.onrender.com`
   - `TELEGRAM_BOT_TOKEN`: Your Telegram Bot token
   - `TELEGRAM_CHAT_ID`: Your Telegram group or channel ID
   - `TELEGRAM_APPROVAL_BOT_TOKEN`: Your approval Bot token
   - `TELEGRAM_APPROVAL_CHAT_ID`: Admin private chat ID
   - `TELEGRAM_APPROVAL_ADMIN_USER_ID`: Admin numeric Telegram ID
   - `BREVO_API_KEY`: Brevo API key
   - `MAIL_FROM_EMAIL`: Verified sender email
   - `ADMIN_EMAIL`: Administrator notification email
5. Click **Apply**.

### Method B: Manual Web Service Setup
1. Click **New +** &rarr; **Web Service**.
2. Select your repository `rehan30105-cpu/SafetyVision-AI`.
3. Configure settings:
   - **Name**: `safetyvision-ai`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command**: `gunicorn wsgi:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120`
   - **Health Check Path**: `/health`
4. Add the environment variables from `.env.example`.
5. Deploy the service.

### Setting Up the Production Telegram Approval Webhook
Because Render provides a public HTTPS domain, polling is **not needed** on Render. Use Telegram's official webhook delivery:
```bash
# Register the webhook with Telegram (replace with your values):
curl -F "url=https://<your-app>.onrender.com/telegram/demo-approval/webhook" \
     -F "secret_token=<YOUR_TELEGRAM_APPROVAL_WEBHOOK_SECRET>" \
     -F "allowed_updates=[\"message\",\"callback_query\"]" \
     https://api.telegram.org/bot<YOUR_TELEGRAM_APPROVAL_BOT_TOKEN>/setWebhook
```
Telegram will forward all Approve/Reject button clicks directly to your Render instance over HTTPS!

### Provision Admin on Render
Once deployed, open the Render **Shell** tab and run:
```bash
python create_admin.py --username admin --email admin@company.com --password YourStrongPassword123 --fullname "Production Admin"
```

---

## 🔑 Environment Variables Reference

| Variable | Required | Default | Description |
|---|:---:|:---:|---|
| `APP_ENV` | Yes | `production` | Environment mode (`production` enforces HTTPS cookies) |
| `SECRET_KEY` | Yes | - | High-entropy secret for Flask session signing |
| `PUBLIC_BASE_URL` | Yes | - | Fully qualified public URL (e.g. `https://app.onrender.com`) |
| `PORT` | No | `5000` | Port for Gunicorn / Flask server |
| `SESSION_COOKIE_SECURE` | No | `true` | Enforces `Secure` flag on session cookies |
| `DEMO_ACTIVATION_MAX_AGE` | No | `86400` | Expiration window for demo tokens (seconds) |
| `TELEGRAM_BOT_TOKEN` | Optional | - | Token for outbound PPE violation alert bot |
| `TELEGRAM_CHAT_ID` | Optional | - | Target chat ID for violation notifications |
| `TELEGRAM_ENABLED` | No | `true` | Master toggle for PPE Telegram alerts |
| `TELEGRAM_APPROVAL_BOT_TOKEN` | Optional | - | Dedicated bot token for demo approval buttons |
| `TELEGRAM_APPROVAL_CHAT_ID` | Optional | - | Numeric private chat ID of the reviewing admin |
| `TELEGRAM_APPROVAL_ADMIN_USER_ID`| Optional | - | Numeric Telegram user ID of authorized admin |
| `TELEGRAM_APPROVAL_WEBHOOK_SECRET`| Optional | - | Secret header for verifying webhook authenticity |
| `TELEGRAM_APPROVAL_ENABLED` | No | `true` | Enables interactive demo approval flow |
| `TELEGRAM_APPROVAL_POLLING_ENABLED` | No | `false` | Enable only for local polling; leave `false` on Render |
| `BREVO_API_KEY` | Optional | - | Brevo v3 REST API key for transactional emails |
| `MAIL_FROM_EMAIL` | Optional | - | Sender email address verified in Brevo |
| `MAIL_FROM_NAME` | No | `SafetyVision AI` | Sender display name for emails |
| `ADMIN_EMAIL` | Optional | - | Admin inbox receiving new demo request alerts |

---

## 🧪 Test Suite & Verification

Run the complete regression test suite:
```bash
pytest tests/
```
All 51 test scenarios test:
- Brevo email delivery & failure fallbacks
- Public demo submission, password hashing, and token generation
- Private media access control & authorization guards
- Telegram callback query polling, exponential backoff, and offset durability
- Telegram approval & rejection database commit transactions
- Multi-admin conflict safety & duplicate click idempotency

---

## 🔒 Security & Data Integrity

- **Password Security**: Passwords hashed with PBKDF2:SHA256 via Werkzeug.
- **Strict Role-Based Access Control**: `ADMIN`, `SAFETY_OFFICER`, and `OPERATOR` roles.
- **HMAC Callback Signing**: Telegram approval callback data contains an HMAC signature preventing button tampering.
- **Atomic Database Commit**: Reviews use `BEGIN IMMEDIATE` transactions to prevent duplicate review races.
- **Audit Logging**: Comprehensive logging of user status changes, session reviews, and incident handling.
- **No Biometric Storage**: Centroid tracking operates solely on bounding box coordinates; no facial recognition embeddings or biometric identities are stored.

---

## 🎓 Capstone Project Credentials
- **Program**: B.Tech Artificial Intelligence & Data Science
- **Platform**: SafetyVision AI Enterprise Workplace Safety Platform
