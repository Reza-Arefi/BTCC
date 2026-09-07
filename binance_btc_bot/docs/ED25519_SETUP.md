# Binance Ed25519 API key setup

Authentication is **Ed25519 only** (not HMAC). Do not set `BINANCE_API_SECRET`.

## Environment (repo `.env`)

```bash
BINANCE_API_KEY=<api_key_id_from_binance_after_upload>
BINANCE_ED25519_PRIVATE_KEY_PATH=/home/ubuntu/.config/binance_btc_bot/ed25519_private.pem
# optional if PEM encrypted:
# BINANCE_ED25519_PRIVATE_KEY_PASSPHRASE=
DRY_RUN=true
```

## Generate keys on Ubuntu (do this on the bot server)

```bash
mkdir -p ~/.config/binance_btc_bot
chmod 700 ~/.config/binance_btc_bot

# PKCS#8 private key (keep on server only)
openssl genpkey -algorithm Ed25519 -out ~/.config/binance_btc_bot/ed25519_private.pem
chmod 600 ~/.config/binance_btc_bot/ed25519_private.pem

# Public key in SPKI PEM — upload THIS to Binance
openssl pkey -in ~/.config/binance_btc_bot/ed25519_private.pem -pubout -out ~/.config/binance_btc_bot/ed25519_public.pem
cat ~/.config/binance_btc_bot/ed25519_public.pem
```

### Public key format Binance expects

Paste the **entire** PEM block (including headers), which looks like:

```text
-----BEGIN PUBLIC KEY-----
MCowBQYDK2VwAyEA................................
-----END PUBLIC KEY-----
```

That is **SubjectPublicKeyInfo (SPKI)** PEM — `-----BEGIN PUBLIC KEY-----` / `-----END PUBLIC KEY-----`.

In Binance → API Management → create **Self-generated** API key → choose **Ed25519** → paste the public PEM → enable **Spot & Margin Trading** only (no withdrawals).

Copy the resulting **API Key** id into `BINANCE_API_KEY`.

## User-data WebSocket (Spot WebSocket API)

Legacy `POST /api/v3/userDataStream` listenKey flow was discontinued (HTTP 410).
Authenticated account events use:

```text
wss://ws-api.binance.com:443/ws-api/v3
method: userDataStream.subscribe.signature
```

Signed with the same Ed25519 private key as REST (`BINANCE_ED25519_PRIVATE_KEY_PATH`).
Credentials are sent in the JSON request body — never in the WebSocket URL.

## Verify (still dry / live disabled)

```bash
set -a && source /home/ubuntu/LCADAME/BTCC/.env && set +a
python -m binance_btc_bot.main --status
```

You should see `ed25519=SET signer=READY` and `Live trading: DISABLED`.

Never commit the private PEM. Never paste the private key into chat.
