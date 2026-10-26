from ultron.agent import UltronAgent


def main() -> None:
    agent = UltronAgent()
    print("Ultron online. Type 'exit' to quit.")

    while True:
        try:
            prompt = input("ultron> ")
        except KeyboardInterrupt:
            print("\nUltron going offline.")
            break

        if prompt.strip().lower() in {"exit", "quit", "bye"}:
            print("Ultron going offline.")
            break

        if not prompt.strip():
            continue

        print(agent.handle_task(prompt))


if __name__ == "__main__":
    import sys

    if "--desktop" in sys.argv:
        from ultron.desktop_app import run_desktop_app

        run_desktop_app()
    elif "--demo" in sys.argv:
        agent = UltronAgent()
        demo_prompts = [
            "status",
            "new task Build the project dashboard",
            "skills",
        ]
        for prompt in demo_prompts:
            print(f"\nUser: {prompt}")
            print(agent.handle_task(prompt))
    else:
        main()
