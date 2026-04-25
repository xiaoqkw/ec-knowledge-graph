__all__ = ["DialogueService"]


def __getattr__(name: str):
    if name == "DialogueService":
        from dialogue.service import DialogueService

        return DialogueService
    raise AttributeError(name)
