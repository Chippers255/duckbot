import sys
import socket


def check():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(10)  # seconds
        sock.connect(("127.0.0.1", 8008))
        data = sock.recv(1024)
    except Exception as e:
        print(f"{e.__class__.__name__}: {e}")
        data = None
    finally:
        sock.close()

    if data == b"healthy":
        print("healthy")
        sys.exit(0)
    else:
        print("unhealthy")
        sys.exit(1)


if __name__ == "__main__":
    check()
