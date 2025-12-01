#!/bin/bash
# Start the Flask server

cd "$(dirname "$0")"

# Kill any existing server
pkill -f "python.*app.py" 2>/dev/null
sleep 1

# Use port 8080 (new port to avoid cache issues)
PORT=8080
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "Port 8080 is in use. Using port 8081 instead."
    PORT=8081
fi

echo "Starting server on port $PORT..."
export PORT=$PORT
python3 app.py > server.log 2>&1 &
SERVER_PID=$!
echo $SERVER_PID > server.pid

sleep 3

if ps -p $SERVER_PID > /dev/null; then
    echo "✓ Server started successfully!"
    echo "  PID: $SERVER_PID"
    echo "  Port: $PORT"
    echo "  URL: http://localhost:$PORT"
    echo ""
    echo "Server logs: tail -f server.log"
    echo "Stop server: kill $SERVER_PID"
else
    echo "✗ Server failed to start. Check server.log for errors."
    tail -20 server.log
    exit 1
fi

