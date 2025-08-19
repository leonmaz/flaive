from edgefl.utils.config import load_config
from edgefl.server.sim_server import SimServer

def main():
    cfg = load_config("config/config.yaml")
    SimServer(cfg).run()

if __name__ == "__main__":
    main()
