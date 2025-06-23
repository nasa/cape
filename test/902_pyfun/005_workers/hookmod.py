
def log_phase(runner) -> int:
    with open("cape-workerpyfunc.log", 'a+') as fp:
        fp.write(f"{runner.getx_iter()}\n")

