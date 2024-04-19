from src.roles.worker import Worker


if __name__ == "__main__":
    ip = ""
    port = 5025
    worker = Worker(ip, port, "5HDxH5ntpmr7U3RjEz5g84Rikr93kmtqUWKQum3p3Kdot4Qh", True)

    worker.run()
