from platform.platform_connect import PlatformServiceClientSync


def main():
    print("Hello from python!")


    client = PlatformServiceClientSync(address="http://127.0.0.1:8080")

    client.health(HealthRequest())

    vm = AgentMiniHarnessClientSync


if __name__ == "__main__":
    main()
