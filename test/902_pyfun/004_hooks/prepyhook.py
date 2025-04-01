
def log_phase(runner) -> int:
    with open("cape-prepyfunc.log", 'a+') as fp:
        fp.write(f"{runner.get_phase()}\n")

