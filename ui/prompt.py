from typing import Callable, Dict


class UserPrompt:

    @staticmethod
    def prompt(prompt_string: str, options: Dict[str, Callable]):
        user_option = None
        while user_option not in options.keys():
            user_option = input(prompt_string)
            if user_option not in options.keys():
                print("Invalid option. Please try again.")
                continue
            for option, method in options.items():
                if user_option == option:
                    return method

    @staticmethod
    def prompt_continue():
        from IPython import embed

        prompt_string = str(
            "Choose one of the following options:\n"
            "  (1) Continue with the experiment\n"
            "  (2) Abort the experiment\n"
            "  (3) Open an interactive shell\n"
            "Enter your choice: "
        )
        options = {
            "1": lambda: print("Continuing with the experiment..."),
            "2": lambda: (print("Experiment aborted."), exit()),
            "3": embed,
        }
        return UserPrompt.prompt(prompt_string, options)
