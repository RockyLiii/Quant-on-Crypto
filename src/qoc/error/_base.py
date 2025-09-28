class UnreachableError(RuntimeError):
    def __init__(self, *args: object) -> None:
        if not args:
            args = ("unreachable!",)
        super().__init__(*args)
