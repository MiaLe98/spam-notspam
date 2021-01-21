mkdir -p ~/.streamlit/
echo "[general]
email = \"mia@ebbot.ai\"
" > ~/.streamlit/credentials.toml
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
