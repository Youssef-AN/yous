import http.server
import ssl

# Set up server address and handler
server_address = ('0.0.0.0', 8080)
handler = http.server.SimpleHTTPRequestHandler

# Create the HTTP server instance
httpd = http.server.HTTPServer(server_address, handler)

# Configure SSL context
#ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
#ssl_context.load_cert_chain(certfile="./opticrop-key.pem", keyfile="./opticrop.pem")

# Wrap the server's socket with SSL
#httpd.socket = ssl_context.wrap_socket(httpd.socket, server_side=True)

print("Serving on https://localhost:8080")
httpd.serve_forever()
